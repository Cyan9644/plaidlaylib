#ifndef EXTERNAL_EVEN_SQUARES_H
#define EXTERNAL_EVEN_SQUARES_H
#include <parlay/delayed.h>
#include <parlay/primitives.h>

#include <optional>
#include <utility>

#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/filter.h"
#include "ChunkSequence/Primitives/chunk_seq.h"

namespace plaid {

struct AddMonoid {
  size_t identity = 0;
  size_t operator()(size_t a, size_t b) const { return a + b; }
};
inline constexpr AddMonoid add{};

template <typename T>
size_t sum_of_even_squares_delay(chunk_seq& seq) {
  return plaid::delayed::reduce(
      plaid::delayed::map(
          plaid::delayed::lazy_filter(
              plaid::delayed::delay<T>(seq),
              [&](T element) { return ((element % 2 == 0) ? 1 : 0); }),
          [&](T element) { return element * element; }),
      add);
}

template <typename T>
size_t sum_of_even_squares_eager(chunk_seq& seq) {
  chunk_seq filtered = plaid::ChunkFilter<T>(
      seq, "even_squares_tmp",
      [&](T element) { return ((element % 2 == 0) ? 1 : 0); });

  return plaid::delayed::reduce(
      plaid::delayed::map(
          plaid::delayed::delay<T>(filtered),
          [&](T element) { return element * element; }),
      add);
}

template <typename T>
size_t sum_of_even_squares_parlay_delayed(parlay::sequence<T> seq) {
  return parlay::delayed::reduce(parlay::delayed::map(
      parlay::delayed::filter(seq,
                              [&](T i) { return i % 2 == 0 ? true : false; }),
      [&](T k) { return k * k; }));
}

template <typename T>
size_t sum_of_even_squares_parlay_actually_delayed(parlay::sequence<T> seq) {
  return parlay::delayed::reduce(
      parlay::delayed::filter_op(seq, [&](T i) -> std::optional<T> {
        return (i % 2 == 0) ? std::optional<T>(i * i) : std::nullopt;
      }));
}

template <typename T>
size_t sum_of_even_squares_parlay_eager(parlay::sequence<T> seq) {
  return parlay::reduce(parlay::map(
      parlay::filter(seq, [&](T i) { return i % 2 == 0 ? true : false; }),
      [&](T k) { return k * k; }));
}

}  // namespace plaid

#endif
