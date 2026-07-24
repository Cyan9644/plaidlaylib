#include <iostream>
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"

chunk_seq add(const chunk_seq& a, const chunk_seq& b, bool extra_one);

int main() {
    // just ten digits, but can be scaled up. The two seqs don't need to be equally long
    // the seq is interpreted as a sequence of digits of len 64 bits, in a 2^64 base number system
    chunk_seq a = ChunkSequenceOps::tabulate(10, "a", [](size_t i){return i;});
    chunk_seq b = ChunkSequenceOps::tabulate(10, "b", [](size_t i){return 1;});
    // call the hand rolled impl shown below
    chunk_seq c = add(a, b, 0);
    // the real version requires you to name the file for the output on the drives (maybe should be changed)
    // chunk_seq c = ChunkSequenceOps::ChunkBigIntAdd(a, b, "sum", false);
    // this combines the chunks back to one file dropped in current working directory, can be inspected in a hex editor
    c.consolidate("result");
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

    // this issues a random read so it's not meant to be used frequently
    // this example intereprets the most significant chunk as a signed number
    bool a_sign = a[n_a - 1] >> (digit_len - 1);
    bool b_sign = b[n_b - 1] >> (digit_len - 1);
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
        /*
         * this is equivalent to the above
        if (sum < mask) {
            return no;
        } else if (sum == mask) {
            return propagate;
        } else {
            return yes;
        }
        */
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
        // bring in the extra one if needed (c can only be propagate at the least significant digit I think)
        return a + b + (c == propagate ? extra_one : c);
    }));
    // materialize the delayed sequence
    chunk_seq result_seq = delayed::force(result, "added");
    size_t peeked = result_seq[n_a - 1];
    // handle signs
    if (a_sign == b_sign && (peeked >> (digit_len - 1) != a_sign)) {
        // this is a random write (and potentially allocates a new chunk)
        result_seq.push_back(a_sign ? mask : 0);
    }
    return result_seq;
}
