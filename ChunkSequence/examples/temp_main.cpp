#include <iostream>
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_seq.h"

chunk_seq add(const chunk_seq& a, const chunk_seq& b, bool extra_one = false);
int main() {
    // chunk_seq a = ChunkSequenceOps::tabulate(1 << 24, "ones", [](int i) { return 1; });
    // chunk_seq b = ChunkSequenceOps::tabulate(1 << 24, "twos", [](int i) { return 2; });
    chunk_seq a = ChunkSequenceOps::tabulate(1 << 24, "ones", [](int i) { return (i == 0) ? 1 : 0; });
    chunk_seq b = ChunkSequenceOps::tabulate(1 << 24, "twos", [](int i) { return 0xffffffffffffffff; });
    chunk_seq res = add(a, b, false);
    // write to single files for debugging
    res.consolidate("result");
}

using digit = unsigned long;
using double_digit = unsigned __int128;
// number of bytes in a digit, 64 here
constexpr int digit_len = sizeof(digit) * 8;

chunk_seq add(const chunk_seq& a, const chunk_seq& b, bool extra_one) {
    size_t n_a = ChunkSequenceOps::size(a);
    size_t n_b = ChunkSequenceOps::size(b);


    // if a is shorter than b, flip the order (a should be the longer one)
    if (n_a < n_b) return add(b, a, extra_one);
    if (n_b == 0) return a;

    enum carry : char { no = 0, yes = 1, propagate = 2 };

    using namespace ChunkSequenceOps;

    bool a_sign = peek(a, n_a - 1) >> (digit_len - 1);
    bool b_sign = peek(b, n_b - 1) >> (digit_len - 1);
    double_digit mask = (static_cast<double_digit>(1) << digit_len) - 1;

    digit pad = b_sign ? mask : 0;
    auto A = delayed::delay(a);
    auto B = delayed::delay(b);
    auto ab_pairs = delayed::zip(A, B, pad);

    auto classifications = delayed::map(ab_pairs, [&](auto p) {
        auto [a_val, b_val] = p;
        double_digit sum = (double_digit) a_val + b_val;
        // are we definitely carrying, definitely not, or on the border
        return static_cast<enum carry>(2 * (sum == mask) + (sum >> digit_len));
    });

    auto carry_fn = [] (enum carry a, enum carry b) {return (b == propagate) ? a : b;};
    auto monoid = parlay::binary_op(carry_fn, static_cast<enum carry>(0));
    // do the scan over this associative carry_fn to get the carries in parallel
    auto scanned = delayed::scan(classifications, monoid).first;

    auto triple_zipped = delayed::zip(ab_pairs, scanned);
    // sum a + b + carry
    // triple is really pair<pair<digit, digit>, enum carry>
    auto result = (delayed::map(triple_zipped, [&](auto triple) {
        auto pair = std::get<0>(triple);
        auto [a,b] = pair;
        auto c = std::get<1>(triple);
        // TODO: this doesn't quite work I think, if all the states are 2 then you end up with no propagates anyways
        // maybe you do need some logic in the carry function itself
        return a + b + (c == propagate ? extra_one : c);
    }));
    chunk_seq result_seq = delayed::force(result, "added");
    size_t peeked = peek(result_seq, n_a - 1);
    if (a_sign == b_sign && (peeked >> (digit_len - 1) != a_sign)) {
        ChunkSequenceOps::push(result_seq, a_sign ? mask : 0);
    }
    return result_seq;
}

using bigint = parlay::sequence<digit>;
namespace delayed = parlay::delayed;

template <typename Bigint1, typename Bigint2>
bigint add(const Bigint1& a, const Bigint2& b, bool extra_one = false) {
    long na = a.size();
    long nb = b.size();

    // if a is shorter than b, flip the order (a should be the longer one)
    if (na < nb) return add(b, a, extra_one);
    if (nb == 0) return parlay::to_sequence(a);

    enum carry : char { no = 0, yes = 1, propagate = 2 };

    bool a_sign = a[na - 1] >> (digit_len - 1);
    bool b_sign = b[nb - 1] >> (digit_len - 1);
    double_digit mask = (static_cast<double_digit>(1) << digit_len) - 1;

    digit pad = b_sign ? mask : 0;
    bigint result(na);

    // b may be shorter than a — B(i) virtually sign-extends it
    auto B = delayed::tabulate(na, [&](size_t i) {
        return (i < nb) ? b[i] : pad;
    });
    // this could be created with a delayed tabulate via indexing also
    auto ab_pairs = delayed::zip(a, B);

    // for each position find if it is yes/no/maybe on carry in
    auto classifications = delayed::map(ab_pairs, [&](auto p) {
        auto [a_val, b_val] = p;
        double_digit sum = (double_digit) a_val + b_val;
        return static_cast<enum carry>(2 * (sum == mask) + (sum >> digit_len));
    });
    // if this is a propagate (borderline sum) then just take the prev
    auto carry_fn = [] (enum carry a, enum carry b) {return (b == propagate) ? a : b;};
    // I think this trick here to bring in the extra one as carry in won't work in general
    // I think the identity does actually need to be an additive identity depending on how scan works
    auto monoid = parlay::binary_op(carry_fn, static_cast<enum carry>(extra_one));
    auto scanned = delayed::scan(classifications, monoid).first;

    auto triple_zipped = delayed::zip(ab_pairs, scanned);
    result = delayed::to_sequence(delayed::map(triple_zipped, [](auto triple) {
        auto pair = std::get<0>(triple);
        auto [a,b] = pair;
        auto c = std::get<1>(triple);
        return a + b + c;
    }));

    if (a_sign == b_sign && (result[na - 1] >> (digit_len - 1) != a_sign)) {
        result.push_back(a_sign ? mask : 0);
    }
    return result;
}
