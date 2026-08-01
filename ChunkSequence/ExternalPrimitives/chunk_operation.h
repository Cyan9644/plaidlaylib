#ifndef CHUNK_OPERATION_H
#define CHUNK_OPERATION_H

#include <parlay/primitives.h>
#include <parlay/random.h>

#include <cstddef>
#include <cstring>
#include <functional>
#include <vector>

#include "ChunkSequence/ExternalPrimitives/small_sequence_ops.h"
#include "ChunkSequence/chunk_seq.h"

namespace ChunkSequenceOps {

// Named in-place operations `apply` can dispatch to. Extend by adding
// an enumerator here and a matching `if constexpr` arm in both `apply`
// overloads below.
enum class ChunkOperation { Sort, Shuffle };

// Run a named operation over every sequence in `seqs`, in place over each
// sequence's own chunks, DRAM-budget-checked and wave-batched
// (process_inplace_budgeted) so `seqs` need not already be pre-sized to fit
// DRAM. This is the discoverable front door on top of process_inplace /
// process_inplace_budgeted for callers who don't want to hand-write a raw
// Processor lambda; write one directly against process_inplace_budgeted (or
// process_inplace) for anything not covered by ChunkOperation.
//
//   Op == Sort:    `less` is the comparator (default std::less<>); `seed` is
//                  ignored.
//   Op == Shuffle: `seed` seeds a parlay::random, forked per-bucket exactly
//                  as Permutation::Run's processor does (random_shuffle.h);
//                  `less` is ignored. Deterministic for a given seed,
//                  regardless of how the DRAM budget happens to split `seqs`
//                  into waves (process_inplace_budgeted translates each
//                  processor call's bucket index back to seqs's own global
//                  index).
template <ChunkOperation Op, typename T = uint64_t, typename Less = std::less<>>
void apply(std::vector<chunk_seq>& seqs, Less less = {},
                   size_t seed = 0) {
  static_assert(Op == ChunkOperation::Sort || Op == ChunkOperation::Shuffle,
               "apply: unsupported ChunkOperation");
  if constexpr (Op == ChunkOperation::Sort) {
    process_inplace_budgeted<T>(seqs, [&](size_t, T* buf, size_t nelem) {
      parlay::sort_inplace(parlay::make_slice(buf, buf + nelem), less);
    });
  } else {  // ChunkOperation::Shuffle
    parlay::random rng(seed);
    process_inplace_budgeted<T>(
        seqs, [&](size_t bucket, T* buf, size_t nelem) {
          auto shuffled = parlay::random_shuffle(
              parlay::make_slice(buf, buf + nelem), rng.fork(bucket));
          std::memcpy(buf, shuffled.data(), nelem * sizeof(T));
        });
  }
}

// Single-sequence form.
template <ChunkOperation Op, typename T = uint64_t, typename Less = std::less<>>
void apply(chunk_seq& seq, Less less = {}, size_t seed = 0) {
  std::vector<chunk_seq> tmp{std::move(seq)};
  apply<Op, T>(tmp, std::move(less), seed);
  seq = std::move(tmp[0]);
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_OPERATION_H
