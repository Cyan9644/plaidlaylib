#ifndef CHUNK_DELAYED_H
#define CHUNK_DELAYED_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
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
//   // (1) READ PLAN: register the physical reads this node needs for logical
//   //     chunk i via Planner::need (keyed by source chunk_seq*, so a source
//   //     appearing in several leaves of the chunk collapses to one read).
//   //     Internal nodes forward to children *left-to-right*; a node past its
//   //     own range (a padded child) registers nothing.
//   template<class Planner> void plan(size_t i, Planner& p) const;
//
//   // (2) BUILD: construct the fused forward-iterator for logical chunk i,
//   //     pulling each in-range leaf's buffer from Resolver::next.  build() MUST
//   //     visit children in the SAME left-to-right order as plan() so the
//   //     resolver lines up with the reads plan() registered.  For i beyond this
//   //     node's range it returns a dummy iterator (consuming no buffer); such
//   //     an iterator is always wrapped by an enclosing pad_iter with
//   //     remaining==0, so it is never dereferenced.
//   template<class Resolver> auto build(size_t i, Resolver& r) const;
//
// One logical chunk of a *leaf* is one physical read (a chunk_seq stores each
// logical chunk as one contiguous region on one drive).  The "one logical chunk
// -> several physical reads" case is the zip_node union; identical reads from a
// shared source (e.g. A,B in both zip(A,B) and a scan of zip(A,B)) are deduped,
// so C=f(A,B) reuses A,B's buffers instead of re-reading them.
//
// Three drivers execute a tree.  for_each_chunk streams: one read pass whose
// dispatcher releases each chunk to a parlay worker the instant that chunk's own
// reads land, so reads and compute overlap (used by reduce/scan/force).
// for_each_window collects a FILTER_BATCH_SIZE window before computing, needed
// by filter's sequential cross-chunk carry.  sequential_for_each_chunk reads via
// blocking pread on the calling thread instead of a ChunkSequenceReader, for
// small ranges consumed from inside an already-parallel outer loop.  Everything
// is templated (no std::function) so the fused chain inlines.
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

// ── read planning: dedup + resolve ───────────────────────────────────────────
//
// A node's plan() calls Planner::need once per in-range leaf, keyed by the
// source chunk_seq*, so a source that appears in several leaves of one logical
// chunk (e.g. A and B in both zip(A,B) and a scan of zip(A,B)) collapses to a
// single physical read.  build() then calls Resolver::next once per in-range
// leaf, in the same left-to-right order, to get that leaf's resolved buffer.
struct Planner {
    std::vector<chunk>            unique_reads;  // this chunk's deduped reads (slot order)
    std::vector<uint32_t>         leaf_slots;    // one local slot per in-range leaf occurrence
    std::vector<const chunk_seq*> src_of;        // dedup key per unique read (parallel array)

    void need(const chunk_seq* src, const chunk& c) {
        for (uint32_t s = 0; s < src_of.size(); s++)          // fanout is tiny: linear scan
            if (src_of[s] == src) { leaf_slots.push_back(s); return; }
        leaf_slots.push_back((uint32_t)unique_reads.size());
        src_of.push_back(src);
        unique_reads.push_back(c);
    }
};
struct Resolver {
    const std::vector<char*>*    bufs;           // this chunk's buffers, by local slot
    const std::vector<uint32_t>* leaf_slots;     // the list Planner produced (same order)
    size_t cursor = 0;
    char* next() { return (*bufs)[(*leaf_slots)[cursor++]]; }
};

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

    // Elements a full physical CHUNK_SIZE chunk holds for this T -- see
    // cut_source::epc; using the global (uint64_t-sized) ELEMS_PER_CHUNK here
    // instead only matches physical layout when sizeof(T) == 8.
    static constexpr size_t epc = CHUNK_SIZE / sizeof(T);

    size_t length()     const { return len; }
    size_t num_chunks() const { return (len + epc - 1) / epc; }
    size_t chunk_len(size_t i) const {
        const size_t base = i * epc;
        return base >= len ? 0 : std::min(epc, len - base);
    }

    template<class Planner>
    void plan(size_t i, Planner& p) const {
        if (i < num_chunks())                     // index-ordered: chunks[i].index == i
            p.need(src, src->chunks[i]);          // one read (shared if the src repeats)
    }                                             // else padded/out-of-range: no read
    template<class Resolver>
    const T* build(size_t i, Resolver& r) const {
        if (i >= num_chunks()) return nullptr;    // dummy; enclosing pad never derefs it
        return reinterpret_cast<const T*>(r.next());
    }
};

// Walks a re-windowed slice of a source: the first `lo_remaining` elements come
// from `lo` (already offset into the source's physical chunk), then the rest
// from `hi` (the start of the next physical chunk).  `hi` is never dereferenced
// when a chunk's slice lands fully inside one physical chunk (lo_remaining never
// hits 0 before the caller stops iterating).
template<class T>
struct cut_iter {
    const T* lo;
    size_t lo_remaining;
    const T* hi;
    T operator*() const { return lo_remaining ? *lo : *hi; }
    cut_iter& operator++() {
        if (lo_remaining) { ++lo; --lo_remaining; } else { ++hi; }
        return *this;
    }
};

// Leaf over an arbitrary [start_index, end_index) slice of a source chunk_seq
// (from `cut`), re-indexed to its own CHUNK_SIZE/sizeof(T)-element grid (`epc`
// below) starting at 0.  Because the slice's grid origin is offset from the
// source's own physical chunk boundaries by `start_index % epc`, every output
// logical chunk lands at the same offset into a physical chunk and so spans at
// most two physical reads (the tail of one, the head of the next) -- never
// more, since one output chunk holds <= epc elements and physical chunks
// (besides the source's last) are exactly epc elements.
template<class T>
struct cut_source {
    using value_type = T;
    const chunk_seq* src;
    size_t start_index;                           // offset into src, in elements
    size_t len;                                    // slice length

    // Elements a full physical CHUNK_SIZE chunk holds for this T -- NOT the
    // global (uint64_t-sized) ELEMS_PER_CHUNK, which only matches physical
    // layout when sizeof(T) == 8.  For any other element size (e.g. the
    // 32-byte weighted_edge), using ELEMS_PER_CHUNK here picks the wrong
    // physical chunk / offset once the cut range passes the true per-chunk
    // element count and reads out of bounds.
    static constexpr size_t epc = CHUNK_SIZE / sizeof(T);

    size_t length()     const { return len; }
    size_t num_chunks() const { return (len + epc - 1) / epc; }
    size_t chunk_len(size_t i) const {
        const size_t base = i * epc;
        return base >= len ? 0 : std::min(epc, len - base);
    }

    // Physical layout of output chunk i: up to two (chunk-index, take-count)
    // segments.  phys_hi == (size_t)-1 means the chunk fits in one physical read.
    struct Seg { size_t phys_lo, offset_lo, take_lo, phys_hi, take_hi; };
    Seg segments(size_t i) const {
        Seg s{};
        const size_t cl = chunk_len(i);
        s.phys_hi = (size_t)-1;
        if (cl == 0) return s;
        const size_t g0 = start_index + i * epc;
        s.phys_lo   = g0 / epc;
        s.offset_lo = g0 % epc;
        const size_t avail_lo = src->chunks[s.phys_lo].used / sizeof(T) - s.offset_lo;
        s.take_lo = std::min(cl, avail_lo);
        const size_t rem = cl - s.take_lo;
        if (rem > 0) { s.phys_hi = s.phys_lo + 1; s.take_hi = rem; }
        return s;
    }

    template<class Planner>
    void plan(size_t i, Planner& p) const {
        if (i >= num_chunks()) return;
        Seg s = segments(i);
        // Keyed by the physical chunk's own address (not `src`), since one
        // output chunk can need two distinct physical reads from the same
        // source -- src-level dedup (as leaf_source uses) would collapse them.
        p.need(reinterpret_cast<const chunk_seq*>(&src->chunks[s.phys_lo]), src->chunks[s.phys_lo]);
        if (s.phys_hi != (size_t)-1)
            p.need(reinterpret_cast<const chunk_seq*>(&src->chunks[s.phys_hi]), src->chunks[s.phys_hi]);
    }
    template<class Resolver>
    cut_iter<T> build(size_t i, Resolver& r) const {
        if (i >= num_chunks()) return {nullptr, 0, nullptr};
        Seg s = segments(i);
        const T* lo = reinterpret_cast<const T*>(r.next()) + s.offset_lo;
        const T* hi = (s.phys_hi != (size_t)-1) ? reinterpret_cast<const T*>(r.next()) : nullptr;
        return {lo, s.take_lo, hi};
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

    template<class Planner> void plan(size_t, Planner&) const {}   // no reads
    template<class Resolver>
    auto build(size_t i, Resolver&) const {
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

    template<class Planner> void plan(size_t i, Planner& p) const { d.plan(i, p); }
    template<class Resolver>
    auto build(size_t i, Resolver& r) const {
        return make_map_iter(d.build(i, r), g);
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

    template<class Planner> void plan(size_t i, Planner& p) const { d.plan(i, p); }
    template<class Resolver>
    auto build(size_t i, Resolver& r) const {
        value_type seed = (i < offsets->size()) ? (*offsets)[i] : m.identity;
        return make_scan_iter(d.build(i, r), m, seed);
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

    template<class Planner>
    void plan(size_t i, Planner& p) const {
        a.plan(i, p);                              // union of children's reads,
        b.plan(i, p);                              // left-to-right (matches build)
    }
    template<class Resolver>
    auto build(size_t i, Resolver& r) const {
        const size_t eb = i * ELEMS_PER_CHUNK;
        const size_t n  = grid_chunk_len(len, i);
        const size_t rA = eb >= lenA ? 0 : std::min(n, lenA - eb);   // A's real count
        const size_t rB = eb >= lenB ? 0 : std::min(n, lenB - eb);   // B's real count
        // Sequence the two child builds explicitly: both advance the resolver,
        // and C++ leaves function-argument evaluation order unspecified.
        auto ia = a.build(i, r);
        auto ib = b.build(i, r);
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
    constexpr size_t epc = CHUNK_SIZE / sizeof(T);
    const size_t len = nc == 0 ? 0
        : (nc - 1) * epc + seq.chunks[nc - 1].used / sizeof(T);
    return leaf_source<T>{&seq, len};
}

// A delayed [start_index, end_index) slice of an on-SSD chunk_seq, re-indexed
// to start at 0.  Unlike `delay`, which is an identity view (chunk i of the
// view IS chunk i of the source), this re-windows the source, so consuming it
// (e.g. via `reduce`, `force`, or ExternalPrimitives' delayed-source
// `materialize`) never writes the slice back to disk the way
// sequential_cut_no_compression does.
template<class T = uint64_t>
auto cut(const chunk_seq& seq, size_t start_index, size_t end_index) {
    CHECK(start_index <= end_index) << "cut: start_index " << start_index
        << " > end_index " << end_index;
    const size_t nc = seq.chunks.size();
    // Physical chunks (besides the last) hold CHUNK_SIZE/sizeof(T) elements of
    // T, not the global (uint64_t-sized) ELEMS_PER_CHUNK -- see cut_source::epc.
    const size_t total = nc == 0 ? 0
        : (nc - 1) * (CHUNK_SIZE / sizeof(T)) + seq.chunks[nc - 1].used / sizeof(T);
    CHECK(end_index <= total) << "cut: end_index " << end_index
        << " exceeds source length " << total;
    return cut_source<T>{&seq, start_index, end_index - start_index};
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

// ── drivers ──────────────────────────────────────────────────────────────────
//
// for_each_window and for_each_chunk both plan each logical chunk with a
// Planner (deduping shared reads), issue the reads through the async
// ChunkSequenceReader, and — once a chunk's buffers are resident — build its
// fused iterator with a Resolver.  They differ only in *scheduling*:
// for_each_window collects a whole window before computing (needed by
// filter's sequential carry); for_each_chunk streams — it releases a chunk to
// a worker the instant that chunk's own reads land, so reads and compute
// overlap continuously with no window barrier.  sequential_for_each_chunk
// (below, after for_each_chunk) plans and builds each chunk the same way but
// reads via blocking pread on the calling thread instead of a
// ChunkSequenceReader — no reader/thread/io_uring setup at all — for small
// ranges consumed from inside an already-parallel outer loop.

// Windowed: collect a FILTER_BATCH_SIZE window, then hand it to `wbody`, which
// may build any chunk via build_chunk(local_b).  Buffers are freed after wbody
// returns, so a consumer must copy anything it keeps.  Used by filter (whose
// dense-packing carry threads sequentially across chunks in index order).
template<class D, class WindowBody>
void for_each_window(const D& d, WindowBody&& wbody, size_t reader_threads = 8) {
    const size_t nc = d.num_chunks();
    for (size_t base = 0; base < nc; base += FILTER_BATCH_SIZE) {
        const size_t w = std::min(FILTER_BATCH_SIZE, nc - base);

        // Plan each chunk (deduped) into the window's flat read list.
        std::vector<chunk>    refs;                       // .index = window read-id
        std::vector<uint32_t> owner;                      // read-id -> local chunk b
        std::vector<std::vector<char*>>    cbufs(w);      // per-chunk buffers (slot order)
        std::vector<std::vector<uint32_t>> cslots(w);     // per-chunk leaf_slots
        std::vector<size_t> first(w);                     // chunk b -> first window read-id
        for (size_t b = 0; b < w; b++) {
            Planner pl;
            d.plan(base + b, pl);
            first[b] = refs.size();
            cbufs[b].assign(pl.unique_reads.size(), nullptr);
            cslots[b] = std::move(pl.leaf_slots);
            for (chunk& c : pl.unique_reads) {
                c.index = refs.size();
                refs.push_back(c);
                owner.push_back((uint32_t)b);
            }
        }
        const size_t total = refs.size();

        auto build_chunk = [&](size_t b) {
            Resolver r{&cbufs[b], &cslots[b], 0};
            return d.build(base + b, r);
        };

        if (total == 0) { wbody(base, w, build_chunk); continue; }   // pure index: no I/O

        chunk_seq rs;
        rs.chunks = std::move(refs);
        ChunkSequenceReader<char> reader;
        reader.PrepChunks(rs);
        reader.Start(reader_threads, 32, 16);
        for (size_t k = 0; k < total; k++) {              // completions arrive out of order
            auto [buf, n, rid] = reader.Poll(); (void)n;
            CHECK(buf != nullptr) << "delayed: short read";
            const size_t b = owner[rid];
            cbufs[b][rid - first[b]] = buf;               // slot = offset from chunk's first read
        }

        wbody(base, w, build_chunk);

        for (size_t b = 0; b < w; b++)
            for (char* p : cbufs[b]) if (p) reader.allocator.Free(p);
    }
}

// Streaming: one read pass over the whole sequence with per-chunk async release.
// A dispatcher thread assembles chunks from the reader's out-of-order
// completions and hands each finished chunk to a parlay worker, so body runs
// while later chunks are still being read (no window barrier).  body must be
// chunk-disjoint and order-independent — true for reduce (writes sums[ci]) and
// force (writes chunk ci's precomputed drive/slot).  reader_threads defaults to
// 10 to match the eager ChunkReduce reader (the config that reaches device-read
// speed); one reader serves the whole pass, so there is no per-window setup cost.
template<class D, class Body>
void for_each_chunk(const D& d, Body&& body, size_t reader_threads = 10,
                    size_t compute_workers = 0) {
    const size_t nc = d.num_chunks();
    if (nc == 0) return;
    // 0 = "use the whole pool"; callers that share the pool with other tasks
    // (e.g. count_sort's writer I/O threads) pass P - kWriterIoThreads so the
    // scatter fork-join matches the available workers instead of oversubscribing.
    if (compute_workers == 0) compute_workers = parlay::num_workers();

    // Plan every chunk up front (metadata only): deduped reads + per-chunk state.
    std::vector<chunk>    refs;                            // .index = global read-id
    std::vector<uint32_t> owner;                           // read-id -> chunk
    std::vector<size_t>   first(nc);                       // chunk -> first read-id
    std::vector<size_t>   remaining(nc);                   // reads not yet landed
    std::vector<std::vector<char*>>    cbufs(nc);          // per-chunk buffers (slot order)
    std::vector<std::vector<uint32_t>> cslots(nc);         // per-chunk leaf_slots
    for (size_t ci = 0; ci < nc; ci++) {
        Planner pl;
        d.plan(ci, pl);
        first[ci]     = refs.size();
        remaining[ci] = pl.unique_reads.size();
        cbufs[ci].assign(pl.unique_reads.size(), nullptr);
        cslots[ci] = std::move(pl.leaf_slots);
        for (chunk& c : pl.unique_reads) {
            c.index = refs.size();
            refs.push_back(c);
            owner.push_back((uint32_t)ci);
        }
    }
    const size_t total = refs.size();

    auto run_chunk = [&](size_t ci) {
        Resolver r{&cbufs[ci], &cslots[ci], 0};
        auto it = d.build(ci, r);
        body(ci, d.chunk_len(ci), it);
    };

    if (total == 0) {                                     // pure index: no I/O, no reader
        parlay::parallel_for(0, nc, [&](size_t ci) { run_chunk(ci); }, 1);
        return;
    }

    chunk_seq rs;
    rs.chunks = std::move(refs);
    ChunkSequenceReader<char> reader;
    reader.PrepChunks(rs);
    reader.Start(reader_threads, 32, 16, /*buf_queue_sz=*/128);

    SimpleQueue<size_t> ready;                            // ready chunk ids (bounded backlog)
    ready.SetSizeLimit(FILTER_BATCH_SIZE);

    // Dispatcher: assemble chunks from out-of-order completions; release each the
    // moment its last read lands.  Single-threaded assembly ⇒ no atomics; the
    // ready queue's push/poll gives workers the happens-before on cbufs[ci].
    // When `ready` is full it blocks here, which back-pressures the reader, so
    // live buffers stay bounded (no window, but a budget).
    std::thread dispatcher([&] {
        for (size_t ci = 0; ci < nc; ci++)             // chunks needing no reads (e.g. a
            if (remaining[ci] == 0) ready.Push(ci);    // padded tail) are ready immediately
        for (size_t done = 0; done < total; done++) {
            auto [buf, n, rid] = reader.Poll(); (void)n;
            CHECK(buf != nullptr) << "delayed: short read";
            const size_t ci = owner[rid];
            cbufs[ci][rid - first[ci]] = buf;
            if (--remaining[ci] == 0) ready.Push(ci);
        }
        ready.Close();
    });

    // Workers: build + compute each ready chunk, then free its buffers.
    parlay::parallel_for(0, compute_workers, [&](size_t) {
        while (true) {
            auto [ci, code] = ready.Poll((size_t)0);
            if (code == QueueCode::FINISH) break;
            run_chunk(ci);
            for (char* p : cbufs[ci]) if (p) reader.allocator.Free(p);
        }
    }, 1);

    dispatcher.join();
}

// Reusable read state for sequential_for_each_chunk / sequential_materialize: a
// caller-owned fd cache + a pool of persistent CHUNK_SIZE O_DIRECT buffers, so
// opens and CHUNK_SIZE allocations happen once across MANY calls instead of
// once per call.  Meant to be constructed once (e.g. one per parlay::worker_id()
// slot, held for an algorithm's whole lifetime -- see chunk_partition.h's /
// count_sort.h's per-worker-slot idiom, valid because a parlay task runs
// uninterrupted on one worker) and threaded through every
// sequential_for_each_chunk / sequential_materialize call that would otherwise
// pay open()+aligned_alloc() per call (e.g. Bellman-Ford's per-vertex
// delayed::cut, called O(rounds*n) times).
//
// The buffer pool grows on demand -- slot s covers "unique read slot s" across
// calls -- and never shrinks; a cut_source needs at most 2 slots (one physical
// chunk, or a chunk-boundary-straddling two), but nothing here hardcodes that,
// so a wider-fanout node just grows the pool further.  Non-copyable (owns fds +
// raw buffers).
//
// fd_cache is bounded (LRU, MAX_CACHED_FDS entries) rather than growing forever:
// an algorithm like Bellman-Ford holds one context per parlay worker for its
// whole run and touches one fd per unique *physical chunk file* across
// O(rounds*n) calls, so an unbounded per-worker cache can accumulate enough
// open fds (across all workers) to blow past the process's RLIMIT_NOFILE --
// observed as open() failing with EMFILE. Rounds revisit the same vertices, so
// LRU still captures most of the reuse an unbounded cache would.
struct SequentialReadContext {
    static constexpr size_t MAX_CACHED_FDS = 256;

    std::list<std::string> lru;   // front = most recently used
    std::unordered_map<std::string, std::pair<int, std::list<std::string>::iterator>> fd_cache;
    std::vector<char*> buf_pool;  // persistent CHUNK_SIZE aligned buffers, grow-on-demand

    SequentialReadContext() = default;
    SequentialReadContext(const SequentialReadContext&) = delete;
    SequentialReadContext& operator=(const SequentialReadContext&) = delete;

    // Returns an open fd for `filename`, opening (and evicting the LRU entry if
    // the cache is full) on a miss. Unlike the old direct fd_cache.emplace, a
    // failed open() is fatal rather than being silently cached as -1 and fed to
    // a later pread -- that used to fail with EBADF on every subsequent access
    // to the same filename, silently leaving the caller's buffer untouched.
    int get_fd(const std::string& filename) {
        auto it = fd_cache.find(filename);
        if (it != fd_cache.end()) {
            lru.splice(lru.begin(), lru, it->second.second);
            return it->second.first;
        }
        if (fd_cache.size() >= MAX_CACHED_FDS) {
            const std::string& victim = lru.back();
            close(fd_cache.at(victim).first);
            fd_cache.erase(victim);
            lru.pop_back();
        }
        int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
        CHECK(fd >= 0) << "SequentialReadContext: open failed for " << filename
                        << ": " << std::strerror(errno);
        lru.push_front(filename);
        fd_cache.emplace(filename, std::make_pair(fd, lru.begin()));
        return fd;
    }

    ~SequentialReadContext() {
        for (char* p : buf_pool) free(p);
        for (auto& [name, entry] : fd_cache) close(entry.first);
    }
};

// Sequential streaming: one blocking, O_DIRECT pread per physical chunk, on
// the calling thread -- no ChunkSequenceReader, no io_uring rings, no
// dispatcher thread.  For use when D covers only a handful of logical chunks
// (e.g. one vertex's adjacency slice via delayed::cut) from *inside* an
// already-parallel outer loop: the async drivers above pay reader_threads
// io_uring rings + a dispatcher thread per call, a cost a few-hundred-byte
// payload never amortizes when paid once per outer-loop iteration.  Modeled
// on ExternalPrimitives/materialize.h's sequential_materialize(chunk_seq), but
// generalized to any delayed node D via the existing generic Planner/Resolver
// interface -- not specific to cut_source.  Unlike for_each_chunk there is no
// upfront across-chunk planning; each logical chunk is planned, read, and
// consumed one at a time, which is fine for the small-range case this is for.
//
// The SequentialReadContext overload reuses the caller-supplied fd cache and
// buffer pool across calls instead of opening/allocating fresh state every
// time -- fd/buffer lifetime is then caller-controlled (freed when ctx is
// destroyed, not at the end of this call).  The no-context overload keeps the
// original one-shot cost profile (its own private context, torn down here).
template<class D, class Body>
void sequential_for_each_chunk(const D& d, SequentialReadContext& ctx, Body&& body) {
    const size_t nc = d.num_chunks();
    if (nc == 0) return;

    for (size_t ci = 0; ci < nc; ci++) {
        Planner pl;
        d.plan(ci, pl);
        const size_t nr = pl.unique_reads.size();

        if (ctx.buf_pool.size() < nr) {                 // grow-on-demand, never shrink
            const size_t old = ctx.buf_pool.size();
            ctx.buf_pool.resize(nr, nullptr);
            for (size_t s = old; s < nr; s++) {
                ctx.buf_pool[s] = (char*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
                CHECK(ctx.buf_pool[s] != nullptr) << "sequential_for_each_chunk: allocation failed";
            }
        }

        for (size_t s = 0; s < nr; s++) {
            const chunk& c = pl.unique_reads[s];
            char* buf = ctx.buf_pool[s];
            if (c.used == 0) continue;
            int fd = ctx.get_fd(c.filename);
            SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
        }

        Resolver r{&ctx.buf_pool, &pl.leaf_slots, 0};
        auto it = d.build(ci, r);
        body(ci, d.chunk_len(ci), it);   // buffers are ctx-owned: reused by the next
    }                                     // chunk/call, not freed here.
}

template<class D, class Body>
void sequential_for_each_chunk(const D& d, Body&& body) {
    SequentialReadContext ctx;
    sequential_for_each_chunk(d, ctx, std::forward<Body>(body));
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

// segmented_reduce: `bounds` (size num_segments+1, exclusive prefix over D's own
// element indices, bounds[0]==0, bounds.back()==d.length()) partitions D into
// contiguous segments; returns one R per segment, monoid-reduced over every
// element in that segment.  One streaming pass (for_each_chunk) regardless of
// how many segments there are or how many chunks a segment spans: each chunk
// classifies every segment it touches as fully owned (no other chunk can touch
// it -> written directly) or boundary (touches the chunk's first or last
// element -> stashed per chunk index for an O(n_chunks) sequential merge
// afterward, chaining through segments spanning many consecutive chunks).
// Same mechanism as ChunkSegmentedReduce, generalized from a raw chunk_seq<T>
// to any composed delayed node (so a preceding map/zip/etc. fuses into this
// one pass instead of paying a separate read).
template<class D, class Monoid>
parlay::sequence<typename D::value_type>
segmented_reduce(const D& d, const parlay::sequence<size_t>& bounds, Monoid m,
                  size_t reader_threads = 10) {
    using R = typename D::value_type;
    const size_t nc = d.num_chunks();
    const size_t num_segments = bounds.size() - 1;

    std::vector<size_t> chunk_start(nc + 1, 0);       // global element offset of chunk i
    for (size_t i = 0; i < nc; i++) chunk_start[i + 1] = chunk_start[i] + d.chunk_len(i);

    parlay::sequence<R> out(num_segments, m.identity);
    std::vector<std::vector<std::pair<size_t, R>>> boundary(nc);

    for_each_chunk(d, [&](size_t ci, size_t n, auto it) {
        if (n == 0) return;
        const size_t global_start = chunk_start[ci];
        const size_t global_end = global_start + n;

        const size_t v_lo = (size_t)(std::upper_bound(bounds.begin(), bounds.end(), global_start) - bounds.begin()) - 1;
        const size_t v_hi = (size_t)(std::upper_bound(bounds.begin(), bounds.end(), global_end - 1) - bounds.begin()) - 1;

        auto finalize = [&](size_t v, R val) {
            const bool is_boundary =
                (v == v_lo && bounds[v_lo] < global_start) ||
                (v == v_hi && bounds[v_hi + 1] > global_end);
            if (is_boundary) boundary[ci].push_back({v, val});
            else out[v] = val;
        };

        size_t cur_v = v_lo;
        R cur_val = m.identity;
        for (size_t i = 0; i < n; i++) {
            const size_t g = global_start + i;
            while (g >= bounds[cur_v + 1]) {
                finalize(cur_v, cur_val);
                cur_v++;
                cur_val = m.identity;
            }
            cur_val = m(cur_val, *it);
            ++it;
        }
        finalize(cur_v, cur_val);
    }, reader_threads);

    bool have_open = false;
    size_t open_v = 0;
    R open_val = m.identity;
    for (size_t c = 0; c < nc; c++) {
        for (auto& [v, val] : boundary[c]) {
            if (have_open && v == open_v) {
                open_val = m(open_val, val);
            } else {
                if (have_open) out[open_v] = open_val;
                open_v = v;
                open_val = val;
                have_open = true;
            }
        }
    }
    if (have_open) out[open_v] = open_val;

    return out;
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
