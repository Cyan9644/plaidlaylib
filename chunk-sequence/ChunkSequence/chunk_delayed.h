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
// Block-delayed sequences over chunk_seq.
//
// A port of parlaylib's "block iterable delayed" (BID) design
// (deps/parlaylib/parlay/internal/{block_delayed,stream_delayed}.h) to the
// out-of-core (SSD) setting.  Where the eager primitives (ChunkMap/ChunkReduce/
// ChunkScan/ChunkFilter) round-trip every intermediate through the SSDs, a
// delayed sequence fuses an operation chain so that intermediates never touch
// disk:  `map` is lazy, `reduce`/`force` consume in a single read pass, and
// `scan` is partially delayed (one read pass for the block offsets, then a new
// lazy sequence).  For `reduce(map(map(delay(seq),f),g),m)` the eager path moves
// 3n reads + 2n writes; the delayed path moves 1n reads and 0 writes.
//
// A "block" is one 4 MiB chunk: parlay workers process chunks in parallel and
// iteration is sequential within a chunk — the same shape as the eager
// chunk_reduce.h / chunk_scan.h consumer loops.
//
// Representation: a delayed sequence is a (source descriptor) + a per-chunk
// iterator factory `make_iter`.  Given a chunk's input it returns a lightweight
// forward iterator that lazily yields the transformed element type R.  Two
// source kinds share all iterator wrappers and terminals:
//   * delayed_file  — backed by a chunk_seq on SSD; the reader streams raw
//                     buffers and the factory transforms them.
//   * delayed_index — backed by an index range (from `tabulate`); chunks are
//                     generated on the fly, so reduce/scan do zero source I/O.
//
// Everything is templated (not std::function) so the fused chain inlines.
//
// LIFETIME: a delayed_file (and any sequence derived from it, including the
// result of `scan`) holds a pointer to the source chunk_seq.  The source must
// outlive every terminal call made on the delayed sequence.
//
// flatten is intentionally omitted: chunks store plain uint64_t, so there is
// nothing nested to flatten.
// ─────────────────────────────────────────────────────────────────────────────

namespace ChunkSequenceOps {
namespace delayed {

// ── lazy forward iterators (sequential within a chunk) ───────────────────────

// Generates f(cur), f(cur+1), …  Used as the base iterator for `tabulate`
// (an index-backed source with no underlying buffer).
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

// ── the two delayed-sequence kinds ───────────────────────────────────────────

// File-backed: make_iter(const TSrc* raw, size_t n, size_t chunk_index) -> iter.
template<class TSrc, class MakeIter>
struct delayed_file {
    using source_type = TSrc;
    static constexpr bool is_file = true;

    const chunk_seq* source;
    MakeIter make_iter;

    using iterator   = std::invoke_result_t<MakeIter, const TSrc*, size_t, size_t>;
    using value_type = std::decay_t<decltype(*std::declval<iterator&>())>;

    size_t num_chunks() const { return source->chunks.size(); }
};

// Index-backed: make_iter(size_t base_index, size_t n, size_t chunk_index) -> iter.
template<class MakeIter>
struct delayed_index {
    static constexpr bool is_file = false;

    size_t n;
    MakeIter make_iter;

    using iterator   = std::invoke_result_t<MakeIter, size_t, size_t, size_t>;
    using value_type = std::decay_t<decltype(*std::declval<iterator&>())>;

    size_t num_chunks() const { return (n + ELEMS_PER_CHUNK - 1) / ELEMS_PER_CHUNK; }
};

// Number of elements in chunk i (map/scan preserve element counts).
template<class D>
size_t chunk_len(const D& d, size_t i) {
    if constexpr (D::is_file) {
        return d.source->chunks[i].used / sizeof(typename D::source_type);
    } else {
        const size_t base = i * ELEMS_PER_CHUNK;
        return std::min(ELEMS_PER_CHUNK, d.n - base);
    }
}

// ── constructors / combinators (all lazy, no I/O) ────────────────────────────

// Wrap an on-SSD chunk_seq as a delayed sequence (identity transform).
template<class TSrc = uint64_t>
auto delay(const chunk_seq& seq) {
    auto mk = [](const TSrc* raw, size_t, size_t) { return raw; };
    return delayed_file<TSrc, decltype(mk)>{&seq, mk};
}

// A delayed sequence whose element i is f(i), with no source files.
template<class F>
auto tabulate(size_t n, F f) {
    auto mk = [f](size_t base, size_t, size_t) { return make_counting(base, f); };
    return delayed_index<decltype(mk)>{n, mk};
}

// Lazily map g over every element.  Returns a delayed sequence of the same kind;
// composes with no temp buffer and no I/O.
template<class D, class G>
auto map(D d, G g) {
    auto inner = d.make_iter;
    if constexpr (D::is_file) {
        using TSrc = typename D::source_type;
        auto mk = [inner, g](const TSrc* raw, size_t n, size_t ci) {
            return make_map_iter(inner(raw, n, ci), g);
        };
        return delayed_file<TSrc, decltype(mk)>{d.source, mk};
    } else {
        auto mk = [inner, g](size_t base, size_t n, size_t ci) {
            return make_map_iter(inner(base, n, ci), g);
        };
        return delayed_index<decltype(mk)>{d.n, mk};
    }
}

// ── driver: the one place the two source kinds differ ────────────────────────
//
// Invokes body(chunk_index, n_elements, begin_iterator) for every chunk, in
// parallel across chunks.  body must consume exactly [it, it+n) sequentially.
template<class D, class Body>
void run_chunks(const D& d, size_t reader_threads, Body&& body) {
    if constexpr (D::is_file) {
        using TSrc = typename D::source_type;
        ChunkSequenceReader<TSrc> reader;
        reader.PrepChunks(*d.source);
        reader.Start(reader_threads, 32, 16);
        parlay::parallel_for(0, parlay::num_workers(), [&](size_t) {
            while (true) {
                auto [raw, n, ci] = reader.Poll();
                if (raw == nullptr) break;
                body(ci, n, d.make_iter(raw, n, ci));
                reader.allocator.Free(raw);
            }
        }, 1);
    } else {
        const size_t nc = d.num_chunks();
        parlay::parallel_for(0, nc, [&](size_t ci) {
            const size_t base = ci * ELEMS_PER_CHUNK;
            const size_t n    = std::min(ELEMS_PER_CHUNK, d.n - base);
            body(ci, n, d.make_iter(base, n, ci));
        }, 1);
    }
}

// ── terminals ────────────────────────────────────────────────────────────────

// Per-chunk monoid reduction: sums[i] = reduction of chunk i.  Shared by reduce
// and scan's first pass.  c == num_chunks accumulators fit in RAM (the same
// assumption chunk_scan.h relies on).
template<class D, class Monoid>
std::vector<typename D::value_type>
per_chunk_reduce(const D& d, Monoid m, size_t reader_threads) {
    using R = typename D::value_type;
    std::vector<R> sums(d.num_chunks());
    run_chunks(d, reader_threads, [&](size_t ci, size_t n, auto it) {
        R s = m.identity;
        for (size_t i = 0; i < n; i++) { s = m(s, *it); ++it; }
        sums[ci] = s;
    });
    return sums;
}

// reduce: fold the whole sequence under the monoid.  One read pass (file-backed)
// or zero source I/O (index-backed).
template<class D, class Monoid>
typename D::value_type reduce(const D& d, Monoid m) {
    using R = typename D::value_type;
    std::vector<R> sums = per_chunk_reduce(d, m, /*reader_threads=*/10);
    R acc = m.identity;                              // c is small: sequential combine
    for (const R& s : sums) acc = m(acc, s);
    return acc;
}

// scan: exclusive prefix scan (parlay convention), partially delayed.
//   Pass 1 (one read pass): per-chunk reductions -> block offsets + total.
//   Pass 2 (lazy): return a new delayed sequence that, when consumed, re-reads
//   the source and runs the seeded within-chunk scan.  No second read or any
//   write happens until the result is forced/reduced; further maps fuse on top.
// Returns {delayed_sequence, total}.
template<class D, class Monoid>
auto scan(const D& d, Monoid m) {
    using R = typename D::value_type;
    std::vector<R> sums = per_chunk_reduce(d, m, /*reader_threads=*/10);
    const size_t nc = sums.size();

    auto offsets = std::make_shared<std::vector<R>>(nc);
    R run = m.identity;
    for (size_t i = 0; i < nc; i++) { (*offsets)[i] = run; run = m(run, sums[i]); }
    const R total = run;

    auto inner = d.make_iter;
    if constexpr (D::is_file) {
        using TSrc = typename D::source_type;
        auto mk = [inner, m, offsets](const TSrc* raw, size_t n, size_t ci) {
            return make_scan_iter(inner(raw, n, ci), m, (*offsets)[ci]);
        };
        return std::pair{delayed_file<TSrc, decltype(mk)>{d.source, mk}, total};
    } else {
        auto mk = [inner, m, offsets](size_t base, size_t n, size_t ci) {
            return make_scan_iter(inner(base, n, ci), m, (*offsets)[ci]);
        };
        return std::pair{delayed_index<decltype(mk)>{d.n, mk}, total};
    }
}

// force: materialize a delayed sequence to a real chunk_seq on SSD, using the
// same one-file-per-drive balls-in-bins layout as ChunkMap / tabulate.  Returns
// an index-ordered chunk_seq.  Allocates a fresh output buffer per chunk (no
// in-place reuse): a scan chain reads the source element on ++ *after* the
// accumulator is emitted, so overwriting the source in place would corrupt it.
template<class D>
chunk_seq force(const D& d, const std::string& result_prefix) {
    using R = typename D::value_type;
    static_assert(CHUNK_SIZE % sizeof(R) == 0,
        "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");

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
                         chunk_len(d, i) * sizeof(R), i};

    UnorderedWriterConfig wcfg;
    wcfg.num_threads   = num_drives;
    wcfg.io_uring_size = 32;
    wcfg.queue_size    = 64;
    wcfg.num_files     = num_drives;
    UnorderedFileWriter<R> writer;
    writer.Start(filenames, wcfg);

    run_chunks(d, /*reader_threads=*/5, [&](size_t ci, size_t n, auto it) {
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

// filter: terminal that packs survivors (pred over the fused delayed elements)
// into a tightly packed chunk_seq.  Modeled on ChunkFilter — index-contiguous
// batches of FILTER_BATCH_SIZE chunks, sorted by index, per-chunk survivor
// compaction, prefix sums, parallel scatter, dense CHUNK_SIZE packing — but each
// chunk's elements are produced by walking the delayed iterator, so preceding
// maps fuse into this read pass.  Returns an index-ordered chunk_seq.
template<class D, class Pred>
chunk_seq filter(const D& d, const std::string& result_prefix, Pred pred) {
    using R = typename D::value_type;
    static_assert(CHUNK_SIZE % sizeof(R) == 0,
        "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");

    const size_t n_in = d.num_chunks();
    if (n_in == 0) return {};
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

    // Process input in index-contiguous batches.  surv[b]/scount[b] hold the
    // compacted survivors (type R) of chunk (base + b).
    for (size_t base = 0; base < n_in; base += FILTER_BATCH_SIZE) {
        const size_t batch_n = std::min(FILTER_BATCH_SIZE, n_in - base);
        std::vector<R*>     surv(batch_n, nullptr);
        std::vector<size_t> scount(batch_n, 0);

        // Extract + compact survivors for each chunk in the batch.
        if constexpr (D::is_file) {
            using TSrc = typename D::source_type;
            chunk_seq sub;
            sub.chunks.assign(d.source->chunks.begin() + base,
                              d.source->chunks.begin() + base + batch_n);
            ChunkSequenceReader<TSrc> reader;
            reader.PrepChunks(sub);
            reader.Start(5, 32, 16);

            std::vector<std::tuple<TSrc*, size_t, size_t>> raws(batch_n);
            for (size_t i = 0; i < batch_n; i++) raws[i] = reader.Poll();
            std::sort(raws.begin(), raws.end(), [](const auto& a, const auto& b) {
                return std::get<2>(a) < std::get<2>(b);   // by chunk index
            });

            parlay::parallel_for(0, batch_n, [&](size_t b) {
                auto [raw, n, ci] = raws[b];
                R* sb = (R*)malloc(std::max<size_t>(1, n) * sizeof(R));
                CHECK(sb != nullptr) << "delayed::filter: allocation failed";
                auto it = d.make_iter(raw, n, ci);
                size_t s = 0;
                for (size_t j = 0; j < n; j++) { R v = *it; ++it; if (pred(v)) sb[s++] = v; }
                surv[b] = sb; scount[b] = s;
            }, 1);

            for (auto& r : raws) reader.allocator.Free(std::get<0>(r));
        } else {
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                const size_t ci   = base + b;
                const size_t cbas = ci * ELEMS_PER_CHUNK;
                const size_t n    = std::min(ELEMS_PER_CHUNK, d.n - cbas);
                R* sb = (R*)malloc(std::max<size_t>(1, n) * sizeof(R));
                CHECK(sb != nullptr) << "delayed::filter: allocation failed";
                auto it = d.make_iter(cbas, n, ci);
                size_t s = 0;
                for (size_t j = 0; j < n; j++) { R v = *it; ++it; if (pred(v)) sb[s++] = v; }
                surv[b] = sb; scount[b] = s;
            }, 1);
        }

        // Prefix sums: offset[b] = absolute position of chunk b's first survivor.
        std::vector<size_t> offset(batch_n + 1);
        offset[0] = carry.size();
        for (size_t b = 0; b < batch_n; b++) offset[b + 1] = offset[b] + scount[b];
        const size_t total         = offset[batch_n];
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
        parlay::parallel_for(0, batch_n, [&](size_t b) {
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

        for (size_t b = 0; b < batch_n; b++) free(surv[b]);

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
    }

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
