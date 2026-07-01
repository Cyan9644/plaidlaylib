#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "ChunkSequence/chunk_flat_tabulate.h"
#include "ChunkSequence/chunk_seq.h"

// **************************************************************
// In-memory sieve for small primes up to sqrt(n).
// Identical to the original parlay primes example.
// At n = 2^50, sqrt(n) ~ 33M: the small-primes list is ~2M entries (~16 MB) — fits in RAM.
// **************************************************************
parlay::sequence<long> in_mem_primes(long n) {
    if (n < 2) return {};
    long sqrt_n = (long)std::sqrt((double)n);
    auto sqrt_primes = in_mem_primes(sqrt_n);
    parlay::sequence<bool> flags(n + 1, true);
    parlay::parallel_for(0, n / sqrt_n + 1, [&](long i) {
        long start = sqrt_n * i;
        long end   = (std::min)(start + sqrt_n, n + 1);
        for (long j = 0; j < (long)sqrt_primes.size(); j++) {
            long p     = sqrt_primes[j];
            long first = (std::max)(2 * p, (((start - 1) / p) + 1) * p);
            for (long k = first; k < end; k += p)
                flags[k] = false;
        }
    }, 1);
    flags[0] = flags[1] = false;
    return parlay::filter(parlay::iota<long>(n + 1),
                          [&](long i) { return flags[i]; });
}

// **************************************************************
// Out-of-core primes sieve via ChunkFlatTabulate.
//
// Small primes (up to sqrt(n)) are computed in memory, then each
// 4 MB virtual chunk [start, end) is sieved independently in parallel:
//   - local bool array (512 KB at CHUNK_SIZE=4MB) stays cache-resident
//   - for each small prime p, cross off multiples in [start, end)
//   - collect surviving indices as uint64_t
//
// ChunkFlatTabulate packs the variable-length per-chunk survivor lists
// into a dense output chunk_seq (same output layout as ChunkFilter).
//
// Complexity: O(n log log n) work, O(c log n) span, c = n / ELEMS_PER_CHUNK.
// **************************************************************
chunk_seq chunk_primes(size_t n, const std::string& result_prefix) {
    long sqrt_n = (long)std::sqrt((double)n);
    // Correct for floating-point rounding: ensure sqrt_n^2 <= n < (sqrt_n+1)^2.
    while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;

    parlay::sequence<long> small = in_mem_primes(sqrt_n);

    return ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n + 1, result_prefix,
        [&](size_t start, size_t end) {
            std::vector<bool> flags(end - start, true);
            for (long p : small) {
                size_t first = std::max((size_t)(2 * p),
                                        (((start - 1) / p) + 1) * p);
                for (size_t k = first; k < end; k += (size_t)p)
                    flags[k - start] = false;
            }
            parlay::sequence<uint64_t> out;
            size_t lo = (start < 2) ? 2 : start;
            for (size_t i = lo; i < end; i++)
                if (flags[i - start]) out.push_back((uint64_t)i);
            return out;
        });
}

int main(int argc, char* argv[]) {
    size_t n = (argc > 1) ? std::stoull(argv[1]) : 1'000'000;
    const std::string out_path = (argc > 2) ? argv[2] : "primes_output.bin";

    chunk_seq primes_seq = chunk_primes(n, "primes");

    size_t count = 0;
    for (const auto& c : primes_seq.chunks)
        count += c.used / sizeof(uint64_t);

    std::cout << "pi(" << n << ") = " << count << "\n";

    if (count == 0) return 0;

    // Read the last output chunk and print the final few primes.
    const chunk& last = primes_seq.chunks.back();
    const size_t last_n = last.used / sizeof(uint64_t);
    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr);
    int fd = open(last.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, AlignUp(last.used), (off_t)last.begin_addr));
    close(fd);

    const uint64_t* elems = reinterpret_cast<const uint64_t*>(buf);
    const size_t show = std::min(last_n, size_t(10));
    std::cout << "last " << show << " prime(s):";
    for (size_t i = last_n - show; i < last_n; i++)
        std::cout << " " << elems[i];
    std::cout << "\n";

    free(buf);

    // Write the full prime sequence as packed uint64_t to out_path.
    primes_seq.consolidate(out_path);
    std::cout << "written to " << out_path << "\n";

    return 0;
}
