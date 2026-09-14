// drive_policy.h -- runtime switch for how a chunk_seq's chunks are spread
// across the drives.
//
// TEMPORARY / EXPERIMENTAL (branch `sequence-ablation`).  The library's normal
// behaviour is to scatter chunks pseudo-randomly (balls-in-bins) so that a
// streaming pass hits every drive at once; this header exists to measure that
// choice against the two obvious alternatives.  Selected at run time by the
// PLAID_DRIVE_POLICY environment variable:
//
//   unset / "random"       balls-in-bins -- each call site's own existing draw
//                          (std::mt19937_64, or parlay::hash64) is used
//                          verbatim, so this path is bit-identical to the
//                          pre-ablation code.
//   "round_robin"          chunk i -> drive i % D.
//   "blocked"              the first N/D of the sequence on drive 0, the next
//                          N/D on drive 1, ...:  chunk i -> i*D/total.
//                          Proportional, so it spreads over all D drives
//                          without needing an input big enough to fill them.
//
// Why the *blocked* layout is expected to hurt: ChunkSequenceReader::Start
// hands reader thread t a strided slice of the chunk list (t, t+T, t+2T, ...)
// and each worker drains its slice in index order, so every reader thread sits
// on the same drive at the same instant.  UnorderedFileWriter has the same
// strided thread<->file affinity.  Measuring that is the point.
//
// A runtime env var rather than a -D define: the policy is consulted once per
// emitted chunk (one 4 MiB write), so the cost is a relaxed load against an
// SSD round trip, and one build serves all three arms.

#ifndef PLAID_DRIVE_POLICY_H
#define PLAID_DRIVE_POLICY_H

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace plaid {

enum class DrivePolicy { kRandom, kRoundRobin, kBlocked };

// The policy for this process, read once from PLAID_DRIVE_POLICY.  An
// unrecognized value is fatal rather than a silent fall back to kRandom: a
// typo in a sweep script would otherwise quietly produce a second copy of the
// baseline and look like "the policies make no difference".
inline DrivePolicy GetDrivePolicy() {
  static const DrivePolicy policy = [] {
    const char* e = getenv("PLAID_DRIVE_POLICY");
    if (e == nullptr || *e == '\0' || strcmp(e, "random") == 0)
      return DrivePolicy::kRandom;
    if (strcmp(e, "round_robin") == 0) return DrivePolicy::kRoundRobin;
    if (strcmp(e, "blocked") == 0) return DrivePolicy::kBlocked;
    LOG(FATAL) << "PLAID_DRIVE_POLICY=\"" << e
               << "\" is not one of random|round_robin|blocked";
    return DrivePolicy::kRandom;  // unreachable; silences -Wreturn-type
  }();
  return policy;
}

inline const char* DrivePolicyName() {
  switch (GetDrivePolicy()) {
    case DrivePolicy::kRoundRobin:
      return "round_robin";
    case DrivePolicy::kBlocked:
      return "blocked";
    case DrivePolicy::kRandom:
      break;
  }
  return "random";
}

/**
 * Pick the drive for one chunk.
 *
 * @param idx    logical chunk index within the sequence (NOT emission order --
 *               `blocked` is defined over sequence position, so a call site
 *               with both must pass the logical one).
 * @param total  total chunks in the sequence, or 0 when the producer does not
 *               know its output length yet (DensePack and friends).  `blocked`
 *               has no meaning without a total, so it degrades to round-robin
 *               there; no benchmarked path hits that case.
 * @param nd     number of drives, i.e. GetSSDList().size().
 * @param rnd    the call site's own existing random/hashed draw.  Returned
 *               unchanged under kRandom, which is what keeps the baseline arm
 *               byte-identical to the pre-ablation code.
 *
 * No overflow risk in `idx * nd`: 1 TiB of 4 MiB chunks is 2^18 chunks.
 */
inline size_t PickDrive(size_t idx, size_t total, size_t nd, size_t rnd) {
  if (nd <= 1) return 0;
  switch (GetDrivePolicy()) {
    case DrivePolicy::kRoundRobin:
      return idx % nd;
    case DrivePolicy::kBlocked:
      if (total == 0) return idx % nd;
      return std::min(nd - 1, idx * nd / total);
    case DrivePolicy::kRandom:
      break;
  }
  return rnd;
}

}  // namespace plaid

#endif  // PLAID_DRIVE_POLICY_H
