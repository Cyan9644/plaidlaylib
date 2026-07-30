#ifndef CHUNK_EMIT_H
#define CHUNK_EMIT_H
//
// ChunkEmitService — a "task" primitive decoupled from any one sequence: a
// parallel-for over the input's chunk grid where each worker indexes FREELY into
// the sequence (in[g] / in.get(g) for any g via the global demand-driven IO
// service, chunk_indexed → indexed_io_service.h) and emits a variable number of
// output elements.  The emitted runs are dense-packed (DensePack) into an
// index-ordered chunk_seq, exactly like ChunkFilter / ChunkFlatTabulate.
//
// Unlike ChunkFlatMap (which hands the body a pre-read contiguous window + halo),
// the body here is plain imperative code — `for (g = in.lo(); ...) c = in[g]` —
// that may read anywhere (across chunk boundaries, data-dependent jumps, gathers).
// The per-worker ServiceView cursor amortizes the shared-cache lookup to one probe
// per block, so a sequential scan reaches streaming-reader parity (see
// service_spike.cpp), while irregular access still coalesces + overlaps through the
// one shared cache.  Reads only; imperative writes stay on IndexedChunkSeq.
//
// body signature:  parlay::sequence<R> body(ServiceView<T>& in)
//   in.lo()/in.hi() = this task's chunk index range (attribute emitted elements to
//                     starts in [lo,hi) if order/dedup matters); in.n() = total.

#include <algorithm>
#include <string>
#include <vector>

#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/dense_pack.h"
#include "ChunkSequence/indexed_io_service.h"

namespace ChunkSequenceOps {

namespace detail {
template<typename R>
struct EmitBatch {
    std::vector<parlay::sequence<R>> results;
    size_t size() const { return results.size(); }
    DensePackRun<R> run(size_t b) const { return {results[b].data(), results[b].size()}; }
};
}  // namespace detail

/**
 * Run `body` once per input chunk (grid = CHUNK_SIZE/sizeof(T) elements), letting it
 * index freely into `seq` via a ServiceView, and dense-pack the emitted R runs.
 *
 * @tparam T  on-disk element type of the input sequence.
 * @tparam R  emitted element type (must divide CHUNK_SIZE; see dense_pack.h).
 * @param block_bytes  service coalescing granularity: CHUNK_SIZE for sequential
 *   scans (default, streaming parity), O_DIRECT_MULTIPLE for scattered access.
 */
template<typename T, typename R, typename Body>
chunk_seq ChunkEmitService(const chunk_seq& seq, const std::string& result_prefix,
                           Body body, size_t block_bytes = CHUNK_SIZE) {
    if (seq.chunks.empty()) return {};
    auto& svc = IndexedIoService::instance();
    IndexedIoService::HandleState* h = svc.open<T>(seq, block_bytes);
    const size_t ept_in = h->epc;
    const size_t n = h->n;
    const size_t num_virtual = seq.chunks.size();   // one run per input chunk

    chunk_seq out = DensePack<R>(num_virtual, result_prefix,
        [&](size_t base, size_t batch_n) {
            detail::EmitBatch<R> batch;
            batch.results.resize(batch_n);
            parlay::parallel_for(0, batch_n, [&](size_t i) {
                const size_t vc = base + i;
                const size_t lo = vc * ept_in;
                const size_t hi = std::min(lo + ept_in, n);
                ServiceView<T> in(svc, h, lo, hi);
                batch.results[i] = body(in);
            }, /*granularity=*/1);
            return batch;
        });

    svc.close(h);
    return out;
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_EMIT_H
