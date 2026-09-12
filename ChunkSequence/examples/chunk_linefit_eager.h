#ifndef EXTERNAL_LINEFIT_EAGER_H
#define EXTERNAL_LINEFIT_EAGER_H
#include <parlay/primitives.h>

#include <utility>

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/examples/chunk_linefit.h"

// Eager counterpart to plaid::linefit (chunk_linefit.h).  Same algorithm --
// two passes, zero disk writes, centered-sums least-squares fit -- but built
// directly on NRemoveWorker/NReader (chunk_seq.h) instead of the delayed
// zip/map/reduce fusion engine.  There is no eager ChunkZip primitive in the
// library, so this is the hand-rolled analogue: each pass polls x and y in
// lockstep and folds locally per worker, mirroring how RemoveWorker underlies
// ChunkReduce for the single-sequence case.  Comparing this against
// plaid::linefit isolates the overhead/benefit of the delayed engine's fused
// dispatch versus a direct explicit fold, since both do the same 2 reads / 0
// writes / same formula.

namespace plaid {

auto linefit_eager(const chunk_seq& x, const chunk_seq& y) {
  size_t n = plaid::get_used_bytes(x) / sizeof(double);

  // Pass 1: sum(x), sum(y) via a lockstep fold over x and y.
  auto partials1 = NRemoveWorker<double>(
      {&x, &y}, /*reader_threads=*/10, [](NReader<double>& reader) {
        point acc = add_points.identity;
        while (true) {
          auto m = reader.Poll();
          if (!m.valid()) break;
          double* xs = m.ptrs[0];
          double* ys = m.ptrs[1];
          for (size_t i = 0; i < m.sizes[0]; i++)
            acc = add_points(acc, point(xs[i], ys[i]));
          reader.Free(m);
        }
        return acc;
      });
  auto [xsum, ysum] = parlay::reduce(partials1, add_points);
  double xa = xsum / n;
  double ya = ysum / n;

  // Pass 2: Stt = sum((x-xa)^2), bb = sum((x-xa)*y).
  auto partials2 = NRemoveWorker<double>(
      {&x, &y}, /*reader_threads=*/10, [xa](NReader<double>& reader) {
        point acc = add_points.identity;
        while (true) {
          auto m = reader.Poll();
          if (!m.valid()) break;
          double* xs = m.ptrs[0];
          double* ys = m.ptrs[1];
          for (size_t i = 0; i < m.sizes[0]; i++) {
            double v = xs[i] - xa;
            acc = add_points(acc, point(v * v, v * ys[i]));
          }
          reader.Free(m);
        }
        return acc;
      });
  auto [Stt, bb] = parlay::reduce(partials2, add_points);

  double b = bb / Stt;
  double a = ya - xa * b;
  return point(a, b);
}

}  // namespace plaid
#endif
