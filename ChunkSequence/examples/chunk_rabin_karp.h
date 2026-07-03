#ifndef CHUNK_RABIN_KARP_H
#define CHUNK_RABIN_KARP_H

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#include "absl/log/check.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "ChunkSequence/dense_pack.h"
#include "utils/file_utils.h"
#include "configs.h"

namespace ChunkSequenceOps {
namespace detail {

// A finite field modulo a Mersenne prime, ported from parlaylib's rabin_karp
// example.  The prime fits in 32 bits so products fit in 64 bits.  The fast
// mod leaves val in [0, p] with p ≡ 0, so equality compares normalized values
// (the parlay original compared raw val and could false-negative on p vs 0).
struct field {
    static constexpr unsigned long p = 0x7fffffff;  // Mersenne prime 2^31 - 1
    unsigned int val;
    field() : val(0) {}
    template<typename Int>
    field(Int i) : val((unsigned int)i) {}
    field operator+(field a) const {
        unsigned long x = (unsigned long)val + a.val;
        return field((x & p) + (x >> 31));  // fast mod p
    }
    field operator*(field a) const {
        unsigned long x = (unsigned long)val * a.val;
        unsigned long y = (x & p) + (x >> 31);
        return field((y & p) + (y >> 31));  // fast mod p
    }
    field operator-(field a) const {        // val, a.val in [0, p]; p ≡ 0
        unsigned long x = (unsigned long)val + (p - a.val);
        return field((x & p) + (x >> 31));
    }
    bool operator==(field a) const { return val % p == a.val % p; }
};

/**
 * A produced batch for ChunkRabinKarp: owns the reader-pool text buffers
 * (still referenced as each chunk's overlap by its left neighbor, so they are
 * freed only at destruction) plus the per-chunk match-position sequences that
 * DensePack packs.  run(b) reads results[b] from the settled Batch, which is
 * move-stable (the outer vector's element storage is heap-allocated).
 */
template<typename CharT>
struct RkBatch {
    std::unique_ptr<ChunkSequenceReader<CharT>> reader;  // keeps the pool alive
    std::vector<CharT*> bufs;  // one per chunk, index-sorted order
    std::vector<parlay::sequence<uint64_t>> results;

    RkBatch() = default;
    RkBatch(RkBatch&&) = default;
    RkBatch& operator=(RkBatch&&) = default;
    RkBatch(const RkBatch&) = delete;
    RkBatch& operator=(const RkBatch&) = delete;
    ~RkBatch() {
        if (reader)
            for (CharT* b : bufs) reader->allocator.Free(b);
    }

    size_t size() const { return results.size(); }
    DensePackRun<uint64_t> run(size_t b) const {
        return {results[b].data(), results[b].size()};
    }
};

/**
 * Sequential rolling-hash Rabin-Karp over one chunk's text plus its overlap
 * into the next chunk, reporting matches that START in [0, n_b).  Hashes use
 * Horner orientation, H(window at i) = sum s[i+j] * x^(m-1-j), rolled with
 * H' = (H - s[i]*x^(m-1)) * x + s[i+m] — no modular inverse needed.  A hash
 * hit is double-checked against the pattern chars before reporting.  The
 * overlap lives in a separate buffer (the right neighbor's head), hence the
 * two-segment `at()`.
 */
template<typename CharT>
inline void RkScanChunk(const CharT* text, long n_b,
                        const CharT* overlap, long ov,
                        const CharT* pattern, long m,
                        field pattern_hash, field x, field x_m1,
                        uint64_t global_pos,
                        parlay::sequence<uint64_t>& out) {
    const long total = n_b + ov;
    if (total < m) return;
    auto at = [&](long i) { return i < n_b ? text[i] : overlap[i - n_b]; };
    auto fch = [&](long i) {
        return field((unsigned int)(std::make_unsigned_t<CharT>)at(i));
    };

    field h(0);
    for (long j = 0; j < m; j++) h = h * x + fch(j);

    const long last_start = std::min(n_b - 1, total - m);
    for (long i = 0; ; i++) {
        if (h == pattern_hash) {
            bool eq = true;
            for (long j = 0; j < m && eq; j++)
                if (at(i + j) != pattern[j]) eq = false;
            if (eq) out.push_back(global_pos + (uint64_t)i);
        }
        if (i == last_start) break;
        h = (h - fch(i) * x_m1) * x + fch(i + m);
    }
}

} // namespace detail

/**
 * Rabin-Karp search over an out-of-core text: find every occurrence of
 * `pattern` in the chunk_seq of CharT elements `seq`, returning the global
 * match-start positions as a tightly packed, index-ordered chunk_seq of
 * uint64_t (in text order).
 *
 * Same chunk structure as ChunkKmp (examples/chunk_kmp.h): each chunk is
 * scanned sequentially by one worker, chunks in parallel, with the m-1 char
 * overlap into chunk k+1 read from the right neighbor's buffer already in
 * DRAM in the same DensePack batch; only at batch seams (1 chunk in 128) is
 * the next chunk's head fetched with one small synchronous O_DIRECT read.
 * Within a chunk the search is the classic rolling-window Rabin-Karp rather
 * than parlaylib's prefix-hash scans, which out-of-core would mean writing an
 * 8x-blowup hash array to disk.
 *
 * Requires m <= CHUNK_SIZE/sizeof(CharT) (the pattern fits in a chunk, so a
 * match spans at most 2 chunks) and a dense input (every chunk but the last
 * full — the library invariant), so only the final chunk can be short.
 *
 * @tparam CharT    Element type of the text (must match the chunk_seq).
 * @tparam Pattern  Random-access container of CharT with size().
 */
template<typename CharT = char, typename Pattern>
chunk_seq ChunkRabinKarp(const chunk_seq& seq,
                         const std::string& result_prefix,
                         const Pattern& pattern) {
    const size_t n_in = seq.chunks.size();
    const long m = (long)pattern.size();
    const size_t epct = CHUNK_SIZE / sizeof(CharT);
    CHECK((size_t)m <= epct) << "ChunkRabinKarp: pattern must fit within one chunk";
    if (m == 0 || n_in == 0) return {};

    // Local pattern copy, hash base x (as in the parlay original), x^(m-1)
    // for rolling, and the pattern's Horner-orientation hash.
    std::vector<CharT> pat(m);
    for (long i = 0; i < m; i++) pat[i] = pattern[i];
    const detail::field x(500000000);
    detail::field x_m1(1);
    for (long i = 0; i < m - 1; i++) x_m1 = x_m1 * x;
    detail::field pattern_hash(0);
    for (long i = 0; i < m; i++)
        pattern_hash = pattern_hash * x +
            detail::field((unsigned int)(std::make_unsigned_t<CharT>)pat[i]);

    // Global text position of each chunk's first element (exclusive prefix sum
    // of element counts — metadata only, no I/O).
    std::vector<uint64_t> pos_of(n_in + 1);
    pos_of[0] = 0;
    for (size_t i = 0; i < n_in; i++)
        pos_of[i + 1] = pos_of[i] + seq.chunks[i].used / sizeof(CharT);

    // Bytes of chunk `c` a left neighbor may need as overlap: min(m-1, count).
    auto head_bytes = [&](const chunk& c) {
        return std::min((size_t)(m - 1) * sizeof(CharT), c.used);
    };

    return DensePack<uint64_t>(n_in, result_prefix,
        [&](size_t base, size_t batch_n) {
            // Read this batch's contiguous slice [base, base+batch_n) with its
            // own reader, so completions can only belong to this batch.
            chunk_seq sub;
            sub.chunks.assign(seq.chunks.begin() + base,
                              seq.chunks.begin() + base + batch_n);
            auto reader = std::make_unique<ChunkSequenceReader<CharT>>();
            reader->PrepChunks(sub);
            reader->Start(5, 32, 16);

            struct BC { CharT* buf; size_t n; size_t idx; };
            std::vector<BC> bc(batch_n);
            for (size_t i = 0; i < batch_n; i++) {
                auto [ptr, n, cidx] = reader->Poll();
                bc[i] = {ptr, n, cidx};
            }
            // Restore logical order so bc[b+1] is bc[b]'s right neighbor.
            std::sort(bc.begin(), bc.end(),
                      [](const BC& a, const BC& b) { return a.idx < b.idx; });

            // Seam overlap: the last chunk of this batch needs the head of the
            // next batch's first chunk — one small synchronous O_DIRECT read
            // (begin_addr is CHUNK_SIZE-aligned).
            CharT* seam = nullptr;
            size_t seam_count = 0;
            if (base + batch_n < n_in && m > 1) {
                const chunk& next = seq.chunks[base + batch_n];
                const size_t bytes = head_bytes(next);
                seam = (CharT*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                             AlignUp(bytes));
                CHECK(seam != nullptr) << "ChunkRabinKarp: seam allocation failed";
                int fd = open(next.filename.c_str(), O_DIRECT | O_RDONLY);
                SYSCALL(fd);
                SYSCALL(pread(fd, seam, AlignUp(bytes), (off_t)next.begin_addr));
                SYSCALL(close(fd));
                seam_count = bytes / sizeof(CharT);
            }

            detail::RkBatch<CharT> batch;
            batch.reader = std::move(reader);
            batch.bufs.resize(batch_n);
            batch.results.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t b) {
                batch.bufs[b] = bc[b].buf;
                const CharT* overlap;
                long ov;
                if (b + 1 < batch_n) {          // right neighbor is in-batch
                    overlap = bc[b + 1].buf;
                    ov = (long)std::min((size_t)(m - 1), bc[b + 1].n);
                } else {                        // batch seam (or last chunk)
                    overlap = seam;
                    ov = (long)seam_count;
                }
                detail::RkScanChunk<CharT>(bc[b].buf, (long)bc[b].n,
                                           overlap, ov, pat.data(), m,
                                           pattern_hash, x, x_m1,
                                           pos_of[base + b], batch.results[b]);
            }, /*granularity=*/1);

            free(seam);
            return batch;
        });
}

} // namespace ChunkSequenceOps

#endif // CHUNK_RABIN_KARP_H
