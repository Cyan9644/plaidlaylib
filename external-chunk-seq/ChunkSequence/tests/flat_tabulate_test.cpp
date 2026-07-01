#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_flat_tabulate.h"
#include "ChunkSequence/chunk_seq.h"

static void cleanup_prefix(const std::string& prefix) {
    const auto& ssds = GetSSDList();
    for (size_t d = 0; d < ssds.size(); d++)
        unlink(GetFileName(prefix, d).c_str());
}

// ── shared helpers ──────────────────────────────────────────────────────────

// Count elements stored in a chunk_seq from its metadata (no read needed).
static size_t count_elems(const chunk_seq& seq) {
    size_t total = 0;
    for (const auto& c : seq.chunks)
        total += c.used / sizeof(uint64_t);
    return total;
}

// Check that all chunks except the last have used == CHUNK_SIZE.
static bool check_packing(const chunk_seq& seq, const std::string& label) {
    for (size_t i = 0; i + 1 < seq.chunks.size(); i++) {
        if (seq.chunks[i].used != CHUNK_SIZE) {
            std::cout << "    FAIL packing: " << label << " chunk " << i
                      << " used=" << seq.chunks[i].used << " (expected CHUNK_SIZE)\n";
            return false;
        }
    }
    return true;
}

// Check the index-ordered invariant: chunks[i].index == i.
static bool check_index_order(const chunk_seq& seq, const std::string& label) {
    for (size_t i = 0; i < seq.chunks.size(); i++) {
        if (seq.chunks[i].index != i) {
            std::cout << "    FAIL index order: " << label << " chunks[" << i
                      << "].index=" << seq.chunks[i].index << "\n";
            return false;
        }
    }
    return true;
}

// Consolidate seq to a local file, read it back sequentially, compare each
// element to expected[j].  Returns true if all elements match.
static bool check_order(const chunk_seq& seq,
                        const std::vector<uint64_t>& expected,
                        const std::string& tmp_path) {
    seq.consolidate(tmp_path);
    int fd = open(tmp_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cout << "    FAIL open(" << tmp_path << "): " << strerror(errno) << "\n";
        return false;
    }
    constexpr size_t BUF_ELEMS = 1 << 20;
    std::vector<uint64_t> buf(BUF_ELEMS);
    size_t j = 0;
    bool ok = true;
    while (ok) {
        ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
        if (got < 0) { std::cout << "    FAIL read: " << strerror(errno) << "\n"; ok = false; break; }
        if (got == 0) break;
        size_t cnt = (size_t)got / sizeof(uint64_t);
        for (size_t i = 0; i < cnt && ok; i++, j++) {
            if (j >= expected.size() || buf[i] != expected[j]) {
                std::cout << "    FAIL order: element " << j
                          << " got=" << buf[i] << " expected=" << expected[j] << "\n";
                ok = false;
            }
        }
    }
    close(fd);
    unlink(tmp_path.c_str());
    if (ok && j != expected.size()) {
        std::cout << "    FAIL order: read " << j
                  << " elements, expected " << expected.size() << "\n";
        ok = false;
    }
    return ok;
}

// ── in-memory sieve (for reference counts / golden values) ──────────────────

parlay::sequence<long> in_mem_primes(long n) {
    if (n < 2) return {};
    long sqrt_n = (long)std::sqrt((double)n);
    auto sp = in_mem_primes(sqrt_n);
    parlay::sequence<bool> flags(n + 1, true);
    parlay::parallel_for(0, n / sqrt_n + 1, [&](long i) {
        long start = sqrt_n * i;
        long end   = (std::min)(start + sqrt_n, n + 1);
        for (long p : sp) {
            long first = (std::max)(2 * p, (((start - 1) / p) + 1) * p);
            for (long k = first; k < end; k += p) flags[k] = false;
        }
    }, 1);
    flags[0] = flags[1] = false;
    return parlay::filter(parlay::iota<long>(n + 1), [&](long i) { return flags[i]; });
}

// ── tests ───────────────────────────────────────────────────────────────────

// Test 1: identity — f returns every index in [start, end).
// Output should be [0, 1, ..., n-1] in order with n elements.
static bool test_identity() {
    std::cout << "test_identity\n" << std::flush;
    const size_t n = 3 * ELEMS_PER_CHUNK + 7;  // spans multiple chunks, non-aligned
    const std::string prefix = "ft_identity";

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n, prefix,
        [](size_t start, size_t end) {
            parlay::sequence<uint64_t> out(end - start);
            for (size_t i = 0; i < end - start; i++) out[i] = (uint64_t)(start + i);
            return out;
        });

    bool pass = true;

    // Count
    size_t got = count_elems(result);
    if (got != n) {
        std::cout << "  FAIL count: got=" << got << " expected=" << n << "\n";
        pass = false;
    } else {
        std::cout << "  count  OK\n";
    }

    // Packing
    if (!check_packing(result, "identity")) pass = false;
    else std::cout << "  packing OK\n";

    // Index order invariant
    if (!check_index_order(result, "identity")) pass = false;
    else std::cout << "  index  OK\n";

    // Element order: should be [0,1,...,n-1]
    std::vector<uint64_t> expected(n);
    for (size_t i = 0; i < n; i++) expected[i] = (uint64_t)i;
    if (!check_order(result, expected, "ft_identity_consolidated")) pass = false;
    else std::cout << "  order  OK\n";

    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

// Test 2: odds — f returns odd numbers in [start, end).
// Output should be [1, 3, 5, ..., largest odd < n] in order.
static bool test_odds() {
    std::cout << "test_odds\n" << std::flush;
    // Use 2 full + 1 partial virtual chunk so carry propagates across a boundary
    const size_t n = 2 * ELEMS_PER_CHUNK + 5;
    const std::string prefix = "ft_odds";

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n, prefix,
        [](size_t start, size_t end) {
            parlay::sequence<uint64_t> out;
            size_t s = (start % 2 == 0) ? start + 1 : start;
            for (size_t i = s; i < end; i += 2) out.push_back((uint64_t)i);
            return out;
        });

    bool pass = true;

    // Build expected list
    std::vector<uint64_t> expected;
    for (size_t i = 1; i < n; i += 2) expected.push_back((uint64_t)i);

    size_t got = count_elems(result);
    if (got != expected.size()) {
        std::cout << "  FAIL count: got=" << got << " expected=" << expected.size() << "\n";
        pass = false;
    } else {
        std::cout << "  count  OK\n";
    }

    if (!check_packing(result, "odds")) pass = false;
    else std::cout << "  packing OK\n";

    if (!check_index_order(result, "odds")) pass = false;
    else std::cout << "  index  OK\n";

    if (!check_order(result, expected, "ft_odds_consolidated")) pass = false;
    else std::cout << "  order  OK\n";

    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

// Test 3: empty — f returns nothing, output should be empty.
static bool test_empty_output() {
    std::cout << "test_empty_output\n" << std::flush;
    const std::string prefix = "ft_empty";

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(ELEMS_PER_CHUNK, prefix,
        [](size_t, size_t) { return parlay::sequence<uint64_t>{}; });

    bool pass = true;
    if (!result.chunks.empty()) {
        std::cout << "  FAIL: expected 0 output chunks, got " << result.chunks.size() << "\n";
        pass = false;
    } else {
        std::cout << "  OK: empty output\n";
    }
    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

// Test 4: primes count — compare chunk_primes output count against in_mem_primes.
static bool test_primes_count(size_t n) {
    std::cout << "test_primes_count  n=" << n << "\n" << std::flush;
    const std::string prefix = "ft_primes";

    // Compute small primes in memory
    long sqrt_n = (long)std::sqrt((double)n);
    while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
    parlay::sequence<long> small = in_mem_primes(sqrt_n);

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n + 1, prefix,
        [&](size_t start, size_t end) {
            std::vector<bool> flags(end - start, true);
            for (long p : small) {
                size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
                for (size_t k = first; k < end; k += (size_t)p)
                    flags[k - start] = false;
            }
            parlay::sequence<uint64_t> out;
            size_t lo = (start < 2) ? 2 : start;
            for (size_t i = lo; i < end; i++)
                if (flags[i - start]) out.push_back((uint64_t)i);
            return out;
        });

    bool pass = true;

    // Reference: in-memory sieve (trusted for small n)
    parlay::sequence<long> ref = in_mem_primes((long)n);
    const size_t expected_count = ref.size();

    size_t got_count = count_elems(result);
    if (got_count != expected_count) {
        std::cout << "  FAIL count: got=" << got_count
                  << " expected=" << expected_count << "\n";
        pass = false;
    } else {
        std::cout << "  count  OK  (pi(" << n << ")=" << expected_count << ")\n";
    }

    if (!check_packing(result, "primes")) pass = false;
    else std::cout << "  packing OK\n";

    if (!check_index_order(result, "primes")) pass = false;
    else std::cout << "  index  OK\n";

    // For small n also verify element order matches reference exactly
    if (n <= 10'000'000ULL) {
        std::vector<uint64_t> expected(ref.begin(), ref.end());
        if (!check_order(result, expected, "ft_primes_consolidated")) pass = false;
        else std::cout << "  order  OK\n";
    }

    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

// Test 5: chunk boundary — verify that 524287 (= 2^19 - 1, a Mersenne prime) is
// included and that 524288 (= 2^19, clearly composite) is absent.
// These values straddle the boundary of the first virtual chunk [0, 524288).
static bool test_chunk_boundary() {
    std::cout << "test_chunk_boundary\n" << std::flush;

    // Cover a range that includes both sides of the first chunk boundary.
    const size_t n = ELEMS_PER_CHUNK + 100;  // [0, 524388)
    const std::string prefix = "ft_boundary";

    long sqrt_n = (long)std::sqrt((double)n);
    while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
    parlay::sequence<long> small = in_mem_primes(sqrt_n);

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n + 1, prefix,
        [&](size_t start, size_t end) {
            std::vector<bool> flags(end - start, true);
            for (long p : small) {
                size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
                for (size_t k = first; k < end; k += (size_t)p)
                    flags[k - start] = false;
            }
            parlay::sequence<uint64_t> out;
            size_t lo = (start < 2) ? 2 : start;
            for (size_t i = lo; i < end; i++)
                if (flags[i - start]) out.push_back((uint64_t)i);
            return out;
        });

    bool pass = true;

    // Consolidate and search for the boundary values.
    const std::string consolidated = "ft_boundary_consol";
    result.consolidate(consolidated);

    int fd = open(consolidated.c_str(), O_RDONLY);
    CHECK(fd >= 0) << "could not open consolidated file";

    constexpr size_t BUF_ELEMS = 1 << 20;
    std::vector<uint64_t> buf(BUF_ELEMS);
    std::set<uint64_t> found;
    while (true) {
        ssize_t got = read(fd, buf.data(), BUF_ELEMS * sizeof(uint64_t));
        if (got <= 0) break;
        size_t cnt = (size_t)got / sizeof(uint64_t);
        for (size_t i = 0; i < cnt; i++) found.insert(buf[i]);
    }
    close(fd);
    unlink(consolidated.c_str());

    // 524287 = 2^19 - 1 is a Mersenne prime — must be present.
    if (found.count(524287) == 0) {
        std::cout << "  FAIL: 524287 (2^19-1, Mersenne prime) missing from output\n";
        pass = false;
    } else {
        std::cout << "  OK: 524287 present\n";
    }

    // 524288 = 2^19 is composite — must be absent.
    if (found.count(524288) != 0) {
        std::cout << "  FAIL: 524288 (composite) incorrectly present in output\n";
        pass = false;
    } else {
        std::cout << "  OK: 524288 absent\n";
    }

    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    cleanup_prefix(prefix);
    return pass;
}

// Test 6: consolidate writes packed raw uint64_t with no padding or header.
// Runs chunk_primes(1000), consolidates to a temp file, reads it back, and
// checks element-by-element against in_mem_primes(1000).
static bool test_consolidate_output() {
    std::cout << "test_consolidate_output\n" << std::flush;
    const size_t n = 1000;
    const std::string prefix = "ft_consol_primes";
    const std::string out_path = "ft_consol_primes_output.bin";

    long sqrt_n = (long)std::sqrt((double)n);
    while ((long long)(sqrt_n + 1) * (sqrt_n + 1) <= (long long)n) sqrt_n++;
    parlay::sequence<long> small = in_mem_primes(sqrt_n);

    chunk_seq result = ChunkSequenceOps::ChunkFlatTabulate<uint64_t>(n + 1, prefix,
        [&](size_t start, size_t end) {
            std::vector<bool> flags(end - start, true);
            for (long p : small) {
                size_t first = std::max((size_t)(2 * p), (((start - 1) / p) + 1) * p);
                for (size_t k = first; k < end; k += (size_t)p)
                    flags[k - start] = false;
            }
            parlay::sequence<uint64_t> out;
            size_t lo = (start < 2) ? 2 : start;
            for (size_t i = lo; i < end; i++)
                if (flags[i - start]) out.push_back((uint64_t)i);
            return out;
        });

    result.consolidate(out_path);

    bool pass = true;
    parlay::sequence<long> ref = in_mem_primes((long)n);

    // Verify file size: must be exactly ref.size() * 8 bytes, no padding or header.
    struct stat st;
    if (stat(out_path.c_str(), &st) != 0) {
        std::cout << "  FAIL: could not stat " << out_path << "\n";
        pass = false;
    } else {
        const size_t expected_bytes = ref.size() * sizeof(uint64_t);
        if ((size_t)st.st_size != expected_bytes) {
            std::cout << "  FAIL file size: got=" << st.st_size
                      << " expected=" << expected_bytes << "\n";
            pass = false;
        } else {
            std::cout << "  file size OK (" << expected_bytes << " bytes)\n";
        }
    }

    // Read back and compare element-by-element.
    int fd = open(out_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cout << "  FAIL open: " << strerror(errno) << "\n";
        pass = false;
    } else {
        std::vector<uint64_t> got_vals(ref.size());
        ssize_t bytes = read(fd, got_vals.data(), ref.size() * sizeof(uint64_t));
        close(fd);
        if (bytes != (ssize_t)(ref.size() * sizeof(uint64_t))) {
            std::cout << "  FAIL short read: got=" << bytes << "\n";
            pass = false;
        } else {
            bool ok = true;
            for (size_t i = 0; i < ref.size() && ok; i++) {
                if (got_vals[i] != (uint64_t)ref[i]) {
                    std::cout << "  FAIL element " << i
                              << ": got=" << got_vals[i] << " expected=" << ref[i] << "\n";
                    ok = pass = false;
                }
            }
            if (ok) std::cout << "  values OK (" << ref.size() << " primes)\n";
        }
    }

    std::cout << "  => " << (pass ? "PASS" : "FAIL") << "\n\n";
    unlink(out_path.c_str());
    cleanup_prefix(prefix);
    return pass;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    bool all_pass = true;

    all_pass &= test_identity();
    all_pass &= test_odds();
    all_pass &= test_empty_output();

    // Known pi(n) values; element-order check runs for n <= 10^7.
    all_pass &= test_primes_count(100);                    // pi = 25
    all_pass &= test_primes_count(ELEMS_PER_CHUNK - 1);   // includes 524287 (Mersenne prime)
    all_pass &= test_primes_count(ELEMS_PER_CHUNK);       // exactly one virtual chunk
    all_pass &= test_primes_count(ELEMS_PER_CHUNK + 1);   // one elem in second chunk
    all_pass &= test_primes_count(1'000'000);              // pi = 78498
    all_pass &= test_primes_count(10'000'000);             // pi = 664579

    all_pass &= test_chunk_boundary();
    all_pass &= test_consolidate_output();

    // Optional large-n check from command line (count only, no order check).
    if (argc > 1) {
        size_t n = std::stoull(argv[1]);
        all_pass &= test_primes_count(n);
    }

    std::cout << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
