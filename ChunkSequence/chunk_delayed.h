#ifndef CHUNK_DELAYED_H
#define CHUNK_DELAYED_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/chunk_filter.h"  // FILTER_BATCH_SIZE
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"

// ─────────────────────────────────────────────────────────────────────────────
// Block-delayed sequences over chunk_seq  (recursive read-plan model).
//
// A port of parlaylib's "block iterable delayed" design to the out-of-core (SSD)
// setting.  Where the eager primitives round-trip every intermediate through the
// SSDs, a delayed sequence fuses an operation chain so intermediates never touch
// disk: `map` is lazy, `reduce`/`force` consume in one read pass, and `scan` is
// partially delayed (one read pass for block offsets, then a new lazy sequence).
//
// A "block" is one CHUNK_SIZE chunk: parlay workers process chunks in parallel;
// iteration is sequential within a chunk.
//
// ── Representation: a delayed sequence is a *tree of nodes* ──────────────────
// Every node (leaf_source, leaf_index, map_node, scan_node, zip_node) exposes a
// small uniform interface so one generic driver can execute *any* composition —
// including nested/N-ary zips and zips of maps/scans:
//
//   size_t length()      const;             // logical element count
//   size_t num_chunks()  const;             // ceil(length / ELEMS_PER_CHUNK)
//   size_t chunk_len(i)  const;             // elements in logical chunk i
//
//   // (1) READ PLAN: append the physical reads this node needs for logical
//   //     chunk i.  Each read is a `chunk` (filename+begin_addr+used) tagged
//   //     with a sequential read-id (its position in `refs`).  Internal nodes
//   //     forward to children *left-to-right*; a node past its own range (a
//   //     padded child) appends nothing.
//   template<class Refs> void plan(size_t i, Refs& refs) const;
//
//   // (2) BUILD: construct the fused forward-iterator for logical chunk i.
//   //     `bufs[cursor]` are the resolved buffers in plan() order; each leaf
//   //     consumes exactly one and advances `cursor`.  build() MUST visit
//   //     children in the SAME left-to-right order as plan() so the positional
//   //     cursor lines up with the read-ids plan() assigned.  For i beyond this
//   //     node's range it returns a dummy iterator (consuming no buffer); such
//   //     an iterator is always wrapped by an enclosing pad_iter with
//   //     remaining==0, so it is never dereferenced.
//   template<class Bufs> auto build(size_t i, const Bufs& bufs, size_t& cur) const;
//
// One logical chunk of a *leaf* is one physical read (a chunk_seq stores each
// logical chunk as one contiguous region on one drive).  The "one logical chunk
// -> several physical reads" case is the zip_node union.  Read-sharing for a
// C=f(A,B) operand (deduping A/B's reads) is a *future* step; today each leaf
// occurrence gets its own read-id (so C re-reads A,B).
//
// The driver (for_each_window / for_each_chunk) walks logical chunks in windows
// of FILTER_BATCH_SIZE (a memory bound), plans the window's reads, issues them
// through the async ChunkSequenceReader, and once they land builds+runs each
// chunk.  Everything is templated (no std::function) so the fused chain inlines.
//
// LIFETIME: a leaf_source holds a pointer to its chunk_seq; every source in the
// tree must outlive every terminal call.  force on a sequence whose value_type
// exceeds 8 B is unsupported (the on-disk grid assumes ≤8 B elements) — zip's
// std::pair elements stay transient and are meant to be map-ed to a scalar
// before force.
// ─────────────────────────────────────────────────────────────────────────────

namespace ChunkSequenceOps {
namespace delayed {

// ── lazy forward iterators (sequential within a chunk) ───────────────────────

// Generates f(cur), f(cur+1), …  Base iterator for `tabulate` (no buffer).
template<class F>
struct counting_value_iter {
    size_t cur;
    F f;
    auto operator*() const { return f(cur); }
    counting_value_iter& operator++() { ++cur; return *this; }
};
template<class F>
counting_value_iter<F> make_counting(size_t cur, F f) { return {cur, f}; }

// Lazily applies g to the element under `it`.  Composes a `map` onto any chain.
template<class It, class G>
struct map_iter {
    It it;
    G g;
    auto operator*() const { return g(*it); }
    map_iter& operator++() { ++it; return *this; }
};
template<class It, class G>
map_iter<It, G> make_map_iter(It it, G g) { return {it, g}; }

// Exclusive prefix scan iterator: *iter is the running accumulator (seeded per
// chunk with that chunk's offset); ++ folds the underlying element in.
template<class It, class F, class V>
struct scan_iter {
    It it;
    F f;
    V acc;
    V operator*() const { return acc; }
    scan_iter& operator++() { acc = f(acc, *it); ++it; return *this; }
};
template<class It, class F, class V>
scan_iter<It, F, V> make_scan_iter(It it, F f, V acc) { return {it, f, acc}; }

// Yields the first `remaining` elements of `it`, then `pad` forever.  Used by
// zip to fill a shorter operand's tail (remaining==0 for a chunk it never
// reaches — its inner iterator is then never dereferenced, so a null/dummy inner
// is safe).
template<class It, class V>
struct pad_iter {
    It it;
    size_t remaining;
    V pad;
    V operator*() const { return remaining ? (V)(*it) : pad; }
    pad_iter& operator++() { if (remaining) { ++it; --remaining; } return *this; }
};
template<class It, class V>
pad_iter<It, V> make_pad_iter(It it, size_t remaining, V pad) {
    return {it, remaining, pad};
}

// Zips two (already pad-wrapped) iterators into a sequence of std::pair.
template<class ItA, class ItB>
struct zip_iter {
    ItA a;
    ItB b;
    auto operator*() const { return std::pair{*a, *b}; }
    zip_iter& operator++() { ++a; ++b; return *this; }
};
template<class ItA, class ItB>
zip_iter<ItA, ItB> make_zip_iter(ItA a, ItB b) { return {a, b}; }

// ── nodes ────────────────────────────────────────────────────────────────────
// Shared helper: elements in logical chunk i of a sequence of `len` elements.
inline size_t grid_chunk_len(size_t len, size_t i) {
    const size_t base = i * ELEMS_PER_CHUNK;
    return base >= len ? 0 : std::min(ELEMS_PER_CHUNK, len - base);
}
inline size_t grid_num_chunks(size_t len) {
    return (len + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK;
}

// Leaf backed by a materialized chunk_seq on SSD (from `delay`).  One logical
// chunk == one physical read (chunks[i]); a chunk_seq stores each logical chunk
// contiguously on one drive.  T is the stored element type.
template<class T>
struct leaf_source {
    using value_type = T;
    const chunk_seq* src;
    size_t len;                                   // total element count

    size_t length()     const { return len; }
    size_t num_chunks() const { return grid_num_chunks(len); }
    size_t chunk_len(size_t i) const { return grid_chunk_len(len, i); }

    template<class Refs>
    void plan(size_t i, Refs& refs) const {
        if (i >= num_chunks()) return;            // padded/out-of-range: no read
        chunk c = src->chunks[i];                 // index-ordered: chunks[i].index == i
        c.index = refs.size();                    // repurpose .index as the read-id tag
        refs.push_back(c);
    }
    template<class Bufs>
    const T* build(size_t i, const Bufs& bufs, size_t& cursor) const {
        if (i >= num_chunks()) return nullptr;    // dummy; enclosing pad never derefs it
        return reinterpret_cast<const T*>(bufs[cursor++]);
    }
};

// Leaf that generates element i as f(i), with no source files (from `tabulate`).
template<class F>
struct leaf_index {
    using value_type = std::decay_t<std::invoke_result_t<F, size_t>>;
    size_t n;
    F f;

    size_t length()     const { return n; }
    size_t num_chunks() const { return grid_num_chunks(n); }
    size_t chunk_len(size_t i) const { return grid_chunk_len(n, i); }

    template<class Refs> void plan(size_t, Refs&) const {}   // no reads
    template<class Bufs>
    auto build(size_t i, const Bufs&, size_t&) const {
        return make_counting(i * ELEMS_PER_CHUNK, f);        // padded if out-of-range
    }
};

// Lazily map g over every element of child D.
template<class D, class G>
struct map_node {
    using value_type = std::decay_t<std::invoke_result_t<G, typename D::value_type>>;
    D d;
    G g;

    size_t length()     const { return d.length(); }
    size_t num_chunks() const { return d.num_chunks(); }
    size_t chunk_len(size_t i) const { return d.chunk_len(i); }

    template<class Refs> void plan(size_t i, Refs& refs) const { d.plan(i, refs); }
    template<class Bufs>
    auto build(size_t i, const Bufs& bufs, size_t& cursor) const {
        return make_map_iter(d.build(i, bufs, cursor), g);
    }
};

// Exclusive prefix scan of child D under monoid M.  Per-chunk offsets are
// precomputed at construction (see `scan`); build seeds scan_iter per chunk.
template<class D, class M>
struct scan_node {
    using value_type = typename D::value_type;
    D d;
    M m;
    std::shared_ptr<std::vector<value_type>> offsets;

    size_t length()     const { return d.length(); }
    size_t num_chunks() const { return d.num_chunks(); }
    size_t chunk_len(size_t i) const { return d.chunk_len(i); }

    template<class Refs> void plan(size_t i, Refs& refs) const { d.plan(i, refs); }
    template<class Bufs>
    auto build(size_t i, const Bufs& bufs, size_t& cursor) const {
        value_type seed = (i < offsets->size()) ? (*offsets)[i] : m.identity;
        return make_scan_iter(d.build(i, bufs, cursor), m, seed);
    }
};

// Element-wise pairing of two child nodes; element i = {A[i], B[i]}.  The
// shorter child is padded with its pad value up to len = max(lenA, lenB).
// Nesting (zip(zip(A,B), C)) and delayed operands (zip(map(A), scan(...))) work
// because plan/build simply recurse into the children.
template<class DA, class DB>
struct zip_node {
    using value_type = std::pair<typename DA::value_type, typename DB::value_type>;
    DA a;
    DB b;
    typename DA::value_type padA;
    typename DB::value_type padB;
    size_t lenA, lenB, len;

    size_t length()     const { return len; }
    size_t num_chunks() const { return grid_num_chunks(len); }
    size_t chunk_len(size_t i) const { return grid_chunk_len(len, i); }

    template<class Refs>
    void plan(size_t i, Refs& refs) const {
        a.plan(i, refs);                           // union of children's reads,
        b.plan(i, refs);                           // left-to-right (matches build)
    }
    template<class Bufs>
    auto build(size_t i, const Bufs& bufs, size_t& cursor) const {
        const size_t eb = i * ELEMS_PER_CHUNK;
        const size_t n  = grid_chunk_len(len, i);
        const size_t rA = eb >= lenA ? 0 : std::min(n, lenA - eb);   // A's real count
        const size_t rB = eb >= lenB ? 0 : std::min(n, lenB - eb);   // B's real count
        // Sequence the two child builds explicitly: they both advance `cursor`,
        // and C++ leaves function-argument evaluation order unspecified.
        auto ia = a.build(i, bufs, cursor);
        auto ib = b.build(i, bufs, cursor);
        return make_zip_iter(make_pad_iter(ia, rA, padA),
                             make_pad_iter(ib, rB, padB));
    }
};

// Public total element count of any delayed sequence.
template<class D> size_t size(const D& d) { return d.length(); }

// ── constructors / combinators (all lazy, no I/O) ────────────────────────────

// Wrap an on-SSD chunk_seq as a delayed sequence (identity transform).
template<class T = uint64_t>
auto delay(const chunk_seq& seq) {
    const size_t nc = seq.chunks.size();
    const size_t len = nc == 0 ? 0
        : (nc - 1) * ELEMS_PER_CHUNK + seq.chunks[nc - 1].used / sizeof(T);
    return leaf_source<T>{&seq, len};
}

// A delayed sequence whose element i is f(i), with no source files.
template<class F>
auto tabulate(size_t n, F f) { return leaf_index<F>{n, f}; }

// Lazily map g over every element (no temp buffer, no I/O).
template<class D, class G>
auto map(D d, G g) { return map_node<D, G>{d, g}; }

// Strict zip: element i = {a[i], b[i]}.  Both operands must have equal length.
template<class DA, class DB>
auto zip(DA a, DB b) {
    const size_t lenA = a.length(), lenB = b.length();
    CHECK(lenA == lenB) << "zip: length mismatch " << lenA << " vs " << lenB
                        << " (use zip(a, b, pad) to pad the shorter side)";
    return zip_node<DA, DB>{a, b, typename DA::value_type{},
                            typename DB::value_type{}, lenA, lenB, std::max(lenA, lenB)};
}

// Padded zip: if operands differ in length the shorter is padded with `pad` up
// to max(lenA, lenB).  A single pad value requires a shared element type.
template<class DA, class DB, class Pad>
auto zip(DA a, DB b, Pad pad) {
    using VA = typename DA::value_type;
    using VB = typename DB::value_type;
    static_assert(std::is_same_v<VA, VB>,
        "zip(a, b, pad): a single pad value requires both operands to share a value_type");
    const size_t lenA = a.length(), lenB = b.length();
    return zip_node<DA, DB>{a, b, (VA)pad, (VB)pad, lenA, lenB, std::max(lenA, lenB)};
}

// ── driver ───────────────────────────────────────────────────────────────────
//
// Walk logical chunks in windows of FILTER_BATCH_SIZE.  Per window: plan every
// chunk's physical reads (recording each chunk's read-id slice), issue them all
// through the async ChunkSequenceReader (byte reads; each ref's .index tags its
// buffer), wait for the whole window, then hand the window to `wbody` which may
// build any chunk's fused iterator via `build_chunk(local_b)`.  Buffers are
// freed after wbody returns, so a consumer must copy anything it keeps.
template<class D, class WindowBody>
void for_each_window(const D& d, WindowBody&& wbody, size_t reader_threads = 8) {
    const size_t nc = d.num_chunks();
    for (size_t base = 0; base < nc; base += FILTER_BATCH_SIZE) {
        const size_t w = std::min(FILTER_BATCH_SIZE, nc - base);

        // 1. Plan the window's reads; slice[b] = read-id of chunk b's first read.
        std::vector<chunk> refs;
        std::vector<size_t> slice(w);
        for (size_t b = 0; b < w; b++) {
            slice[b] = refs.size();
            d.plan(base + b, refs);
        }
        const size_t total = refs.size();
        std::vector<char*> bufs(total, nullptr);

        // build_chunk reconstructs chunk (base+b)'s fused iterator; both plan and
        // build traverse the tree left-to-right, so the positional cursor at
        // slice[b] lines up with the read-ids planned for that chunk.
        auto build_chunk = [&](size_t b) {
            size_t cursor = slice[b];
            return d.build(base + b, bufs, cursor);
        };

        if (total == 0) {                          // pure index/tabulate: no I/O
            wbody(base, w, build_chunk);
            continue;
        }

        // 2. Issue all the window's reads; place each buffer by its read-id tag.
        chunk_seq rs;
        rs.chunks = std::move(refs);
        ChunkSequenceReader<char> reader;
        reader.PrepChunks(rs);
        reader.Start(reader_threads, 32, 16);
        for (size_t k = 0; k < total; k++) {
            auto [buf, n, rid] = reader.Poll();
            (void)n;
            CHECK(buf != nullptr) << "delayed: short read";
            bufs[rid] = buf;
        }

        // 3. The window is fully resident: build + compute all its chunks.
        wbody(base, w, build_chunk);

        for (char* p : bufs) if (p) reader.allocator.Free(p);
    }
}

// Convenience wrapper: invoke body(chunk_index, n_elements, iterator) for every
// chunk, parallel across a window.  body must consume exactly [it, it+n).
template<class D, class Body>
void for_each_chunk(const D& d, Body&& body, size_t reader_threads = 8) {
    for_each_window(d, [&](size_t base, size_t w, auto build_chunk) {
        parlay::parallel_for(0, w, [&](size_t b) {
            auto it = build_chunk(b);
            body(base + b, d.chunk_len(base + b), it);
        }, 1);
    }, reader_threads);
}

// ── terminals ────────────────────────────────────────────────────────────────

// Per-chunk monoid reduction: sums[i] = reduction of chunk i.  Shared by reduce
// and scan's first pass (c == num_chunks accumulators fit in RAM).
template<class D, class Monoid>
std::vector<typename D::value_type>
per_chunk_reduce(const D& d, Monoid m) {
    using R = typename D::value_type;
    std::vector<R> sums(d.num_chunks());
    for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
        R s = m.identity;
        for (size_t i = 0; i < n; i++) { s = m(s, *it); ++it; }
        sums[ci] = s;
    });
    return sums;
}

// reduce: fold the whole sequence under the monoid (one read pass).
template<class D, class Monoid>
typename D::value_type reduce(const D& d, Monoid m) {
    using R = typename D::value_type;
    std::vector<R> sums = per_chunk_reduce(d, m);
    R acc = m.identity;                              // c is small: sequential combine
    for (const R& s : sums) acc = m(acc, s);
    return acc;
}

// scan: exclusive prefix scan (parlay convention), partially delayed.
//   Pass 1 (one read pass): per-chunk reductions -> block offsets + total.
//   Pass 2 (lazy): a scan_node that, when consumed, re-reads the source and runs
//   the seeded within-chunk scan.  Returns {scan_node, total}.
template<class D, class Monoid>
auto scan(const D& d, Monoid m) {
    using R = typename D::value_type;
    std::vector<R> sums = per_chunk_reduce(d, m);
    const size_t nc = sums.size();

    auto offsets = std::make_shared<std::vector<R>>(nc);
    R run = m.identity;
    for (size_t i = 0; i < nc; i++) { (*offsets)[i] = run; run = m(run, sums[i]); }
    const R total = run;

    return std::pair{scan_node<D, Monoid>{d, m, offsets}, total};
}

// force: materialize a delayed sequence to a real chunk_seq on SSD (one file per
// drive, balls-in-bins).  Returns an index-ordered chunk_seq.  Allocates a fresh
// output buffer per chunk (a scan chain reads the source element on ++ *after*
// the accumulator is emitted, so in-place reuse would corrupt it).
template<class D>
chunk_seq force(const D& d, const std::string& result_prefix) {
    using R = typename D::value_type;
    static_assert(CHUNK_SIZE % sizeof(R) == 0,
        "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
    static_assert(sizeof(R) <= sizeof(uint64_t),
        "force: the on-disk chunk grid assumes <=8B elements; map wider values "
        "(e.g. zip's std::pair) down to a scalar before force");

    const size_t nc = d.num_chunks();
    if (nc == 0) return {};
    const size_t num_drives = GetSSDList().size();

    // Randomly assign each output chunk to a drive; insertion order within a
    // drive gives its CHUNK_SIZE-aligned slot.
    std::vector<size_t> drive_of(nc);
    {
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
        for (size_t i = 0; i < nc; i++) drive_of[i] = dist(rng);
    }
    std::vector<std::vector<size_t>> drive_chunks(num_drives);
    for (size_t i = 0; i < nc; i++) drive_chunks[drive_of[i]].push_back(i);
    std::vector<size_t> slot_of(nc);
    for (size_t dr = 0; dr < num_drives; dr++)
        for (size_t s = 0; s < drive_chunks[dr].size(); s++)
            slot_of[drive_chunks[dr][s]] = s;

    // Pre-fallocate each drive file to its exact final size.
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t dr) {
        filenames[dr] = GetFileName(result_prefix, dr);
        const size_t file_size = drive_chunks[dr].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[dr].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
            SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
    }, /*granularity=*/1);

    // Output chunk descriptors are fully determined up front (index-ordered).
    std::vector<chunk> out_chunks(nc);
    for (size_t i = 0; i < nc; i++)
        out_chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                         d.chunk_len(i) * sizeof(R), i};

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
        R* out = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(out != nullptr) << "delayed::force: allocation failed";
        for (size_t i = 0; i < n; i++) { out[i] = *it; ++it; }
        memset((char*)out + n * sizeof(R), 0, CHUNK_SIZE - n * sizeof(R));
        writer.Push(std::shared_ptr<R>(out, free), CHUNK_SIZE / sizeof(R),
                    drive_of[ci], slot_of[ci] * CHUNK_SIZE);
    });

    writer.Wait();
    return {out_chunks};
}

// filter: pack survivors (pred over the fused elements) into a dense chunk_seq.
// Modeled on ChunkFilter — index-contiguous windows, per-chunk survivor
// compaction, prefix sums, parallel scatter, dense CHUNK_SIZE packing with a
// cross-window carry — but each chunk's elements come from walking the fused
// node iterator, so preceding maps/zips fuse into this read pass.  Returns an
// index-ordered chunk_seq.
template<class D, class Pred>
chunk_seq filter(const D& d, const std::string& result_prefix, Pred pred) {
    using R = typename D::value_type;
    static_assert(CHUNK_SIZE % sizeof(R) == 0,
        "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
    static_assert(sizeof(R) <= sizeof(uint64_t),
        "filter: the on-disk chunk grid assumes <=8B elements");

    if (d.num_chunks() == 0) return {};
    const size_t num_drives = GetSSDList().size();
    const size_t epct       = CHUNK_SIZE / sizeof(R);  // elements per output chunk

    // Create/truncate one output file per drive (writer opens with O_CREAT only).
    std::vector<std::string> filenames(num_drives);
    parlay::parallel_for(0, num_drives, [&](size_t dr) {
        filenames[dr] = GetFileName(result_prefix, dr);
        int fd = open(filenames[dr].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
    }, 1);

    std::vector<size_t> next_slot(num_drives, 0);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

    std::vector<R> carry;            // survivors not yet filling a full chunk (< epct)
    carry.reserve(epct);
    std::vector<chunk> out_chunks;
    size_t out_idx = 0;

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    // One window == one FILTER_BATCH_SIZE batch.  for_each_window runs windows
    // sequentially, so the cross-window `carry` threads correctly.
    for_each_window(d, [&](size_t base, size_t w, auto build_chunk) {
        std::vector<R*>     surv(w, nullptr);   // compacted survivors of chunk base+b
        std::vector<size_t> scount(w, 0);

        parlay::parallel_for(0, w, [&](size_t b) {
            auto it = build_chunk(b);
            const size_t n = d.chunk_len(base + b);
            R* sb = (R*)malloc(std::max<size_t>(1, n) * sizeof(R));
            CHECK(sb != nullptr) << "delayed::filter: allocation failed";
            size_t s = 0;
            for (size_t j = 0; j < n; j++) { R v = *it; ++it; if (pred(v)) sb[s++] = v; }
            surv[b] = sb; scount[b] = s;
        }, 1);

        // Prefix sums: offset[b] = absolute position of chunk b's first survivor.
        std::vector<size_t> offset(w + 1);
        offset[0] = carry.size();
        for (size_t b = 0; b < w; b++) offset[b + 1] = offset[b] + scount[b];
        const size_t total         = offset[w];
        const size_t num_out       = total / epct;
        const size_t new_carry_cnt = total % epct;

        // Allocate output buffers (full chunks + 1 overflow for the new carry).
        const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
        std::vector<R*> obuf(num_alloc, nullptr);
        for (size_t k = 0; k < num_alloc; k++) {
            obuf[k] = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(obuf[k] != nullptr) << "delayed::filter: output allocation failed";
            memset(obuf[k], 0, CHUNK_SIZE);
        }
        if (!carry.empty() && num_alloc > 0)
            memcpy(obuf[0], carry.data(), carry.size() * sizeof(R));

        // Parallel scatter (non-overlapping ranges by construction).
        parlay::parallel_for(0, w, [&](size_t b) {
            if (scount[b] == 0) return;
            const R* src = surv[b];
            size_t pos = offset[b], rem = scount[b], src_o = 0;
            while (rem > 0) {
                const size_t k   = pos / epct;
                const size_t off = pos % epct;
                const size_t can = std::min(rem, epct - off);
                memcpy(obuf[k] + off, src + src_o, can * sizeof(R));
                pos += can; src_o += can; rem -= can;
            }
        }, 1);

        for (size_t b = 0; b < w; b++) free(surv[b]);

        // Push full output chunks with balls-in-bins drive assignment.
        for (size_t k = 0; k < num_out; k++) {
            const size_t dr   = drive_dist(rng);
            const size_t slot = next_slot[dr]++;
            writer.Push(std::shared_ptr<R>(obuf[k], free),
                        CHUNK_SIZE / sizeof(R), dr, slot * CHUNK_SIZE);
            out_chunks.push_back({filenames[dr], slot * CHUNK_SIZE, CHUNK_SIZE, out_idx++});
        }

        carry.resize(new_carry_cnt);
        if (new_carry_cnt > 0) {
            memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(R));
            free(obuf[num_out]);
        }
    });

    // Flush the final partial chunk.
    if (!carry.empty()) {
        R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "delayed::filter: final allocation failed";
        memset(buf, 0, CHUNK_SIZE);
        memcpy(buf, carry.data(), carry.size() * sizeof(R));
        const size_t dr   = drive_dist(rng);
        const size_t slot = next_slot[dr]++;
        writer.Push(std::shared_ptr<R>(buf, free), CHUNK_SIZE / sizeof(R), dr, slot * CHUNK_SIZE);
        out_chunks.push_back({filenames[dr], slot * CHUNK_SIZE,
                              carry.size() * sizeof(R), out_idx++});
    }

    writer.Wait();
    return {out_chunks};
}

} // namespace delayed
} // namespace ChunkSequenceOps

#endif // CHUNK_DELAYED_H
