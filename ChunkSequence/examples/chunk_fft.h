#ifndef CHUNK_FFT_H
#define CHUNK_FFT_H
//
// Out-of-core 1-D FFT of a power-of-two complex<double> sequence, factored as an
// A x B matrix and computed in two passes with deliberately different I/O shapes:
//
//   Stage 1 (streaming):  a length-A FFT on each *contiguous* block of A elements,
//                          done with the existing streaming ExternalTransform
//                          engine (deep-queue async io_uring read -> transform ->
//                          async write).  In-place per block; no transpose.
//   Stage 2 (random):     a length-B FFT down each *strided* column, gathered and
//                          scattered as small 4 KiB random io_uring reads/writes at
//                          arbitrary offsets.  This is the pass that, in the
//                          external-memory literature, is normally preceded by an
//                          explicit transpose -- we skip it and pay the strided-read
//                          cost directly (cheap on SSDs, catastrophic on HDDs).
//
// The two stages give a crisp contrast under benchmarks/io_trace.py: stage 1 is
// large sequential streaming, stage 2 is many small random accesses fanned across
// all drives.
//
// -------------------------- The math (transpose-free four-step) --------------
// Let the logical input be x[0..N), N = A*B, W_N = exp(-2*pi*i/N).  We *store* x
// so that physical position p = b*A + a holds the logical element m = a*B + b
// (a in [0,A), b in [0,B)); equivalently physical block b (positions [b*A, b*A+A))
// holds x[b], x[b+B], x[b+2B], ...  (a column of the logical matrix).  Then the
// Cooley-Tukey split
//
//     X[k1 + A*k2] = SUM_b W_B^{b k2} * ( W_N^{b k1} * SUM_a x-block-b[a] W_A^{a k1} )
//
// factors as:
//   Stage 1:  U[k1][b]  = SUM_a  (block b)[a] * W_A^{a k1}          (length-A FFT)
//   twiddle:  U'[k1][b] = U[k1][b] * W_N^{b k1}
//   Stage 2:  V[k1][k2] = SUM_b  U'[k1][b]    * W_B^{b k2}          (length-B FFT)
//   output:   X[k1 + A*k2] = V[k1][k2].
//
// Because stage 1 operates on contiguous blocks it is the streaming pass; the
// twiddle is applied in stage 2 (a pre-multiply before the length-B transform).
// After stage 1, physical position b*A + k1 holds U[k1][b].  Stage 2, for a fixed
// within-block offset k1, gathers the strided set { b*A + k1 : b in [0,B) }
// (stride A -> random), twiddles, transforms, and scatters V[k1][k2] back to
// physical k2*A + k1.  So the final result lives at physical k2*A + k1 and the
// output permutation is X[k1 + A*k2] = phys[k2*A + k1] (a digit-swap transpose).
//
// Reads (stage 2 gather) use a small io_uring random reader added here -- the
// library's ChunkSequenceReader is chunk-sequential only.  Writes (stage 1 and
// the stage-2 scatter) reuse the standard engine: ExternalTransform for stage 1
// and UnorderedFileWriter's arbitrary-offset Push for the scatter.
//
// Requires the upstream FFT header (parlaylib-examples/fast_fourier_transform.h)
// only for the driver's in-memory baseline (complex_fft); the kernels here are a
// self-contained iterative radix-2 FFT so there is no nested-parallelism or
// name-clash coupling.

#include <complex>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <liburing.h>

#include "parlay/primitives.h"
#include "configs.h"
#include "utils/file_utils.h"
#include "utils/simple_queue.h"
#include "utils/unordered_file_writer.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/external_engine.h"

namespace ChunkFFT {

using cd = std::complex<double>;

// Complex elements per chunk (distinct from chunk_seq.h's ELEMS_PER_CHUNK, which
// is defined for 8-byte uint64_t).  16 divides CHUNK_SIZE, so blocks stay
// O_DIRECT-aligned.
constexpr size_t EPC = CHUNK_SIZE / sizeof(cd);
// Columns per 4 KiB O_DIRECT block = the smallest no-amplification random unit.
constexpr size_t COLS_PER_BLOCK = O_DIRECT_MULTIPLE / sizeof(cd);   // 256

inline double kPi() { return std::acos(-1.0); }

// ---- factorization ---------------------------------------------------------
struct Dims { size_t N, A, B; };

// A = streaming block length (contiguous, length-A FFT), capped at one chunk so a
// block never straddles a chunk and >= COLS_PER_BLOCK so a 4 KiB random read is a
// whole number of columns.  N is the largest power of two <= n.
inline Dims choose_dims(size_t n) {
    size_t k = 0;
    while ((size_t(1) << (k + 1)) <= n) k++;   // 2^k <= n < 2^{k+1}
    const size_t N = size_t(1) << k;
    size_t A = size_t(1) << (k / 2);           // ~sqrt(N)
    if (A > EPC) A = EPC;
    if (A < COLS_PER_BLOCK) A = COLS_PER_BLOCK;
    const size_t B = N / A;
    return {N, A, B};
}

// ---- self-contained iterative radix-2 FFT with precomputed twiddles ---------
// A plan holds w[j] = exp(-2*pi*i*j/n) for j in [0, n/2); one plan is built per
// transform length (A for stage 1, B for stage 2) and shared read-only across all
// worker threads.  Precomputing the roots removes the per-butterfly twiddle
// recurrence (the old w *= wlen -- a serial dependency that blocked pipelining),
// so stage len uses w[k * (n/len)] by direct index; the butterfly itself uses raw
// double arithmetic to sidestep std::complex<double>::operator*'s inf/nan-guarded
// slow path.  Together this is the compute win now that stage 2 is CPU-bound.
struct FFTPlan {
    size_t n = 0;
    std::vector<cd> w;                               // w[j] = exp(-2*pi*i*j/n)
    void init(size_t n_) {
        n = n_;
        w.resize(n / 2);
        const double base = -2.0 * kPi() / (double)n;
        parlay::parallel_for(0, n / 2, [&](size_t j) {
            w[j] = std::polar(1.0, base * (double)j);
        });
    }
};

// Forward transform: out[k] = SUM_j a[j] * exp(-2*pi*i*j*k/n).  In place, input in
// natural order (bit-reversal done here).  p.n must be a power of two.
inline void fft_inplace(cd* a, const FFTPlan& p) {
    const size_t n = p.n;
    for (size_t i = 1, j = 0; i < n; i++) {          // bit-reversal permutation
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    const cd* W = p.w.data();
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t h = len >> 1;
        const size_t step = n / len;                 // W[k*step] = exp(-2*pi*i*k/len)
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < h; k++) {
                const cd& wk = W[k * step];
                const double wr = wk.real(), wi = wk.imag();
                cd* lo = &a[i + k];
                cd* hi = &a[i + k + h];
                const double xr = hi->real(), xi = hi->imag();
                const double vr = xr * wr - xi * wi;  // v = *hi * w  (raw doubles)
                const double vi = xr * wi + xi * wr;
                const double ur = lo->real(), ui = lo->imag();
                *lo = cd(ur + vr, ui + vi);
                *hi = cd(ur - vr, ui - vi);
            }
        }
    }
}

// ---- stage 1: streaming length-A block FFTs (reuse ExternalTransform) -------
inline chunk_seq stage1_rows(const chunk_seq& input, const Dims& d,
                             const std::string& prefix) {
    const size_t A = d.A;
    FFTPlan planA;
    planA.init(A);                                         // shared read-only across workers
    return ChunkSequenceOps::ExternalTransform<cd, cd>(
        input, prefix,
        [A, &planA](const cd* in, size_t n, size_t index,
            const ChunkSequenceOps::ChunkEmitter<cd>& emit) {
            cd* out = emit.alloc();
            const size_t nblocks = n / A;                  // n is a multiple of A
            for (size_t blk = 0; blk < nblocks; blk++) {
                cd* o = out + blk * A;
                std::memcpy(o, in + blk * A, A * sizeof(cd));
                fft_inplace(o, planA);                     // in-place length-A FFT
            }
            // emit() always writes a full CHUNK_SIZE block; zero any tail past the
            // valid elements of a (possibly partial) last chunk.
            if (n < EPC) std::memset((char*)out + n * sizeof(cd), 0,
                                     CHUNK_SIZE - n * sizeof(cd));
            emit.emit(out, n, index);
        });
}

// ---- a tiny io_uring random-access ring (reads or writes) -------------------
// Transfers reqs.size() blocks of `block_bytes` between `buf` and the fds: block i
// moves between fds[reqs[i].first] at offset reqs[i].second and buf + i*block_bytes
// (a read when write=false, a write when write=true).  Up to `qd` outstanding, so
// the many small random ops overlap across drives.  One thread per ring; stage 2
// runs several rings (gather + scatter) concurrently for the aggregate depth.
class RandomRing {
public:
    explicit RandomRing(size_t qd) : qd_(qd) {
        SYSCALL(InitIoUringWithRetry((unsigned)qd, &ring_, 0));
    }
    ~RandomRing() { io_uring_queue_exit(&ring_); }

    void run(const std::vector<int>& fds,
             const std::vector<std::pair<int, size_t>>& reqs,
             char* buf, size_t block_bytes, bool write) {
        const size_t n = reqs.size();
        size_t submitted = 0, completed = 0;
        while (completed < n) {
            bool any = false;
            while (submitted < n && submitted - completed < qd_) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                if (sqe == nullptr) break;                 // ring full; drain first
                const auto& [d, off] = reqs[submitted];
                char* p = buf + submitted * block_bytes;
                if (write) io_uring_prep_write(sqe, fds[d], p, (unsigned)block_bytes, off);
                else       io_uring_prep_read (sqe, fds[d], p, (unsigned)block_bytes, off);
                io_uring_sqe_set_data(sqe, (void*)submitted);
                submitted++;
                any = true;
            }
            if (any) SYSCALL(io_uring_submit(&ring_));
            io_uring_cqe* cqe;
            SYSCALL(io_uring_wait_cqe(&ring_, &cqe));
            do {
                SYSCALL(cqe->res);                         // short op => negative/!=bytes
                io_uring_cqe_seen(&ring_, cqe);
                completed++;
            } while (io_uring_peek_cqe(&ring_, &cqe) == 0);
        }
    }

private:
    io_uring ring_;
    size_t qd_;
};

// ---- stage 2: strided column FFTs via random gather/scatter (in place on s1) -
// The COLS_PER_BLOCK-wide tiles are independent (disjoint physical offsets, each
// read-before-write within its own tile), so they run as a THREE-stage pipeline
// where both the random reads AND the random writes are issued from many rings:
//
//   * G gather threads (own RandomRing each) pull tiles off an atomic counter and
//     read each tile's B strided 4 KiB blocks into a pooled buffer.
//   * 1 consumer (this thread) runs the per-column twiddle + length-B FFT (parlay)
//     in place, then hands the buffer to the scatter stage.
//   * S scatter threads (own RandomRing each) write each computed tile's B blocks
//     back to their strided offsets DIRECTLY via io_uring -- straight from the tile
//     buffer, no per-row alloc/memcpy, and no shared writer queue -- then recycle
//     the buffer to the free pool.
//
// This removes the earlier single-threaded scatter that funnelled 16.7M tiny 4 KiB
// writes through one UnorderedFileWriter queue (a lock convoy that starved the
// drives and, by holding each tile's buffer until its whole serial scatter
// finished, back-pressured the gather too).  Now the write issue is parallel and
// off-lock, and buffer recycling is decoupled from the consumer.  Buffers flow
// free-pool -> gather -> ready -> consumer -> to_scatter -> scatter -> free-pool.
// DRAM is bounded to NBUF * (B * 4 KiB); NBUF (and the G,S split) is capped by a
// byte budget (env FFT_STAGE2_DRAM_BUDGET_BYTES, default min(8 GiB, phys/8)).
inline void stage2_cols(chunk_seq& s1, const Dims& d, const std::string& prefix) {
    const size_t A = d.A, B = d.B, N = d.N;
    const size_t num_drives = GetSSDList().size();
    const double pi = kPi();

    // Per-drive fds: one read set and one write set (both O_DIRECT).
    std::map<std::string, int> fidx;
    std::vector<int> rd_fds(num_drives), wr_fds(num_drives);
    for (size_t dd = 0; dd < num_drives; dd++) {
        const std::string fn = GetFileName(prefix, dd);
        fidx[fn] = (int)dd;
        rd_fds[dd] = open(fn.c_str(), O_DIRECT | O_RDONLY);  SYSCALL(rd_fds[dd]);
        wr_fds[dd] = open(fn.c_str(), O_DIRECT | O_WRONLY);  SYSCALL(wr_fds[dd]);
    }

    // Precompute per-chunk (drive, begin_addr) so address mapping is a vector
    // lookup, not a std::map probe, in the hot B-entry req-building loops.
    std::vector<int> chunk_drive(s1.chunks.size());
    std::vector<size_t> chunk_begin(s1.chunks.size());
    for (size_t ci = 0; ci < s1.chunks.size(); ci++) {
        chunk_drive[ci] = fidx.at(s1.chunks[ci].filename);
        chunk_begin[ci] = s1.chunks[ci].begin_addr;
    }
    auto locate = [&](size_t p, int& drive, size_t& offset) {
        const size_t ci = p / EPC;
        drive  = chunk_drive[ci];
        offset = chunk_begin[ci] + (p % EPC) * sizeof(cd);
    };

    const size_t ntiles = A / COLS_PER_BLOCK;
    const size_t block_bytes = COLS_PER_BLOCK * sizeof(cd);   // 4 KiB
    const size_t tile_bytes = B * block_bytes;
    const size_t qd = std::min<size_t>(B, 256);              // per ring

    FFTPlan planB;
    planB.init(B);                                           // shared read-only across workers

    // NBUF buffers, split into G gather rings + S scatter rings.  Cap by budget.
    const size_t phys = (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    size_t budget = std::min<size_t>(size_t(8) << 30, phys / 8);
    if (const char* e = getenv("FFT_STAGE2_DRAM_BUDGET_BYTES")) budget = std::stoull(e);
    size_t nbuf = std::min<size_t>(18, std::max<size_t>(1, budget / tile_bytes));
    nbuf = std::max<size_t>(nbuf, 4);                        // need a few to pipeline
    const size_t G = std::max<size_t>(1, (nbuf - 2) / 2);
    const size_t S = std::max<size_t>(1, (nbuf - 2) / 2);
    std::fprintf(stderr,
        "stage2: NBUF=%zu G=%zu S=%zu qd=%zu tile=%zu MiB (peak %zu MiB)\n",
        nbuf, G, S, qd, tile_bytes >> 20, (nbuf * tile_bytes) >> 20);

    struct Item { cd* buf; size_t tile; };
    SimpleQueue<cd*> free_pool;
    SimpleQueue<Item> ready, to_scatter;
    std::vector<cd*> all_bufs(nbuf);
    for (size_t i = 0; i < nbuf; i++) {
        all_bufs[i] = (cd*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, tile_bytes);
        CHECK(all_bufs[i] != nullptr) << "stage2: tile allocation failed";
        free_pool.Push(all_bufs[i]);
    }
    // Row i of a tile (tile t, offset col0 = t*COLS_PER_BLOCK) lives at physical
    // i*A + col0, for both the gather (read into buf+i*block) and the scatter
    // (write from buf+i*block) -- same offset set, so one builder serves both.
    auto build_reqs = [&](size_t tile, std::vector<std::pair<int, size_t>>& reqs) {
        const size_t col0 = tile * COLS_PER_BLOCK;
        for (size_t i = 0; i < B; i++) {
            int drive; size_t off;
            locate(i * A + col0, drive, off);
            reqs[i] = {drive, off};
        }
    };

    std::atomic<size_t> next_tile{0};
    std::vector<std::thread> gatherers, scatterers;
    for (size_t g = 0; g < G; g++) {
        gatherers.emplace_back([&]() {
            RandomRing ring(qd);
            std::vector<std::pair<int, size_t>> reqs(B);
            while (true) {
                const size_t t = next_tile.fetch_add(1);
                if (t >= ntiles) break;
                cd* buf = free_pool.Poll(nullptr).first;    // blocks until a buffer frees
                build_reqs(t, reqs);
                ring.run(rd_fds, reqs, (char*)buf, block_bytes, /*write=*/false);
                ready.Push(Item{buf, t});
            }
        });
    }
    for (size_t s = 0; s < S; s++) {
        scatterers.emplace_back([&]() {
            RandomRing ring(qd);
            std::vector<std::pair<int, size_t>> reqs(B);
            while (true) {
                auto [it, code] = to_scatter.Poll(Item{nullptr, 0});
                if (code != QueueCode::SUCCESS) break;       // closed + drained
                build_reqs(it.tile, reqs);
                ring.run(wr_fds, reqs, (char*)it.buf, block_bytes, /*write=*/true);
                free_pool.Push(it.buf);
            }
        });
    }

    // Consumer (this thread): compute exactly ntiles tiles (any order), hand each to
    // the scatter stage, then close the scatter queue so its threads drain and exit.
    for (size_t done = 0; done < ntiles; done++) {
        Item it = ready.Poll(Item{nullptr, 0}).first;
        cd* buf = it.buf;
        const size_t col0 = it.tile * COLS_PER_BLOCK;

        // Transform each of the COLS_PER_BLOCK columns down the b-axis: twiddle by
        // W_N^{b*k1} (geometric in b, ratio W_N^{k1}) then a length-B FFT.  Result
        // V[k1][k2] overwrites the buffer in place at row k2.
        parlay::parallel_for(0, COLS_PER_BLOCK, [&](size_t c) {
            thread_local std::vector<cd> col;            // per-worker scratch, reused
            col.resize(B);
            const size_t k1 = col0 + c;
            const cd ratio = std::polar(1.0, -2.0 * pi * (double)(k1 % N) / (double)N);
            cd tw(1.0, 0.0);
            for (size_t b = 0; b < B; b++) {
                col[b] = buf[b * COLS_PER_BLOCK + c] * tw;
                tw *= ratio;
            }
            fft_inplace(col.data(), planB);
            for (size_t k2 = 0; k2 < B; k2++)
                buf[k2 * COLS_PER_BLOCK + c] = col[k2];
        });

        to_scatter.Push(Item{buf, it.tile});
    }
    to_scatter.Close();

    for (auto& t : gatherers)  t.join();
    for (auto& t : scatterers) t.join();
    for (cd* b : all_bufs) free(b);
    for (int fd : rd_fds) close(fd);
    for (int fd : wr_fds) close(fd);
}

// Output permutation: logical index k -> physical position holding X[k].
// X[k1 + A*k2] lives at physical k2*A + k1.
inline size_t out_perm(size_t k, const Dims& d) {
    const size_t k1 = k % d.A;
    const size_t k2 = k / d.A;
    return k2 * d.A + k1;
}

// ---- in-memory baseline: the SAME transpose-free four-step, no I/O ----------
// The out-of-core-vs-in-mem comparison should isolate the I/O cost, not pit our
// four-step against a *different* FFT implementation.  These run the identical
// algorithm and kernel (fft_inplace + FFTPlan, same parallel structure as
// stage1_rows / stage2_cols) on a DRAM array, so timing them is the honest
// in-memory baseline.

// Column-major placement matching the out-of-core build (the untimed "build"
// step): W[b*A + a] = x[a*B + b], for logical input x[m].
inline parlay::sequence<cd> in_mem_place(const parlay::sequence<cd>& x, const Dims& d) {
    const size_t A = d.A, B = d.B;
    return parlay::tabulate(d.N, [&](size_t p) {
        const size_t a = p % A, b = p / A;
        return x[a * B + b];
    });
}

// The two transpose-free passes in place on the placed array W (the timed part,
// matching stage1_s + stage2_s): length-A block FFTs, then per-column twiddle +
// length-B FFT.  Afterwards W is X in logical order (out_perm is the identity for
// this factorization), i.e. the same physical layout the out-of-core result has.
inline void in_mem_transform(parlay::sequence<cd>& W, const Dims& d) {
    const size_t N = d.N, A = d.A, B = d.B;
    const double pi = kPi();
    FFTPlan planA;
    planA.init(A);
    parlay::parallel_for(0, B, [&](size_t b) {
        fft_inplace(W.data() + b * A, planA);              // length-A block FFT
    });
    FFTPlan planB;
    planB.init(B);
    parlay::parallel_for(0, A, [&](size_t k1) {
        thread_local std::vector<cd> col;
        col.resize(B);
        const cd ratio = std::polar(1.0, -2.0 * pi * (double)(k1 % N) / (double)N);
        cd tw(1.0, 0.0);
        for (size_t b = 0; b < B; b++) {                   // gather strided + twiddle
            col[b] = W[b * A + k1] * tw;
            tw *= ratio;
        }
        fft_inplace(col.data(), planB);                    // length-B column FFT
        for (size_t k2 = 0; k2 < B; k2++)
            W[k2 * A + k1] = col[k2];                       // scatter back
    });
}

}  // namespace ChunkFFT

#endif  // CHUNK_FFT_H
