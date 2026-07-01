#include <iostream>
#include <atomic>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_scan.h"

// ── monoids ──────────────────────────────────────────────────────────────────

struct SumMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};

struct XorMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a ^ b; }
};

// XOR(0 ^ 1 ^ … ^ k) using the 4-cycle closed form.
static uint64_t xor_prefix(uint64_t k) {
    switch (k % 4) {
        case 0: return k;
        case 1: return 1;
        case 2: return k + 1;
        default: return 0;
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

// Read one output chunk from disk via pread and verify every element against
// f_expected, which maps the global element index (in [0,n)) to the expected
// scanned value.  No assumption about begin_addr or which file the chunk lives
// in — we use the chunk descriptor as-is.
template<typename R>
bool verify_chunk(const chunk& c,
                  size_t elems_per_input_chunk,
                  std::function<R(uint64_t)> f_expected) {
    const size_t count     = c.used / sizeof(R);
    const size_t start_idx = c.index * elems_per_input_chunk; // global start in [0,n)
    const size_t read_size = AlignUp(count == 0
                                     ? (size_t)O_DIRECT_MULTIPLE
                                     : c.used);

    if (count == 0) return true;

    int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
        std::cerr << "  FAIL chunk " << c.index << ": open(" << c.filename
                  << "): " << strerror(errno) << "\n";
        return false;
    }

    R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
    CHECK(buf != nullptr);

    const ssize_t got = pread(fd, buf, read_size, (off_t)c.begin_addr);
    close(fd);

    if (got < 0 || (size_t)got < c.used) {
        std::cerr << "  FAIL chunk " << c.index << ": pread got " << got
                  << " expected at least " << c.used << "\n";
        free(buf);
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < count && ok; i++) {
        const R expected = f_expected(start_idx + i);
        if (buf[i] != expected) {
            std::cerr << "  FAIL chunk " << c.index << " element " << i
                      << ": got " << (uint64_t)buf[i]
                      << " expected " << (uint64_t)expected
                      << " (global idx " << start_idx + i << ")\n";
            ok = false;
        }
    }

    free(buf);
    return ok;
}

// Run one named scan test case, return true iff all chunks verify correctly.
// f_expected(global_idx) gives the expected exclusive-scan value at global_idx;
// expected_total is the grand total ChunkScan must also return.
template<typename Monoid, typename R = uint64_t>
bool run_test(const std::string& name,
              const chunk_seq& input,
              const std::string& result_prefix,
              size_t elems_per_input_chunk,
              Monoid monoid,
              std::function<R(uint64_t)> f_expected,
              R expected_total) {
    std::cout << "  " << name << " ... " << std::flush;

    // Plain variables (not a structured binding) so `output` can be captured by
    // the parlay::parallel_for lambda below under -std=c++17.
    const auto scan_result =
        ChunkSequenceOps::ChunkScan<uint64_t, R>(input, result_prefix, monoid);
    const chunk_seq& output = scan_result.first;
    const R total = scan_result.second;

    // Verify the returned grand total.
    bool total_ok = (total == expected_total);
    if (!total_ok)
        std::cerr << "\n  FAIL total: got " << (uint64_t)total
                  << " expected " << (uint64_t)expected_total << "\n";

    // Assert output files are on configured SSD mounts.
    {
        const auto& ssds = GetSSDList();
        size_t ssd_count = 0;
        for (const auto& c : output.chunks)
            for (const auto& ssd : ssds)
                if (c.filename.rfind(ssd, 0) == 0) { ssd_count++; break; }
        if (ssd_count != output.chunks.size()) {
            std::cout << "FAIL (only " << ssd_count << "/" << output.chunks.size()
                      << " output chunks on SSDs)\n";
            return false;
        }
    }

    // The output must describe the same number of chunks as the input.
    if (output.chunks.size() != input.chunks.size()) {
        std::cout << "FAIL (chunk count: got " << output.chunks.size()
                  << " want " << input.chunks.size() << ")\n";
        return false;
    }

    // Every chunk index [0, n_chunks) must appear exactly once.
    const size_t n_chunks = output.chunks.size();
    std::vector<bool> seen(n_chunks, false);
    bool coverage_ok = true;
    for (const auto& c : output.chunks) {
        if (c.index >= n_chunks || seen[c.index]) {
            std::cerr << "  FAIL: duplicate/out-of-range index " << c.index << "\n";
            coverage_ok = false;
        } else {
            seen[c.index] = true;
        }
    }

    // Verify element values in parallel.
    std::atomic<size_t> pass_count{0}, fail_count{0};
    parlay::parallel_for(0, n_chunks, [&](size_t ci) {
        if (verify_chunk<R>(output.chunks[ci], elems_per_input_chunk, f_expected))
            pass_count++;
        else
            fail_count++;
    }, /*granularity=*/1);

    const bool ok = coverage_ok && total_ok && (fail_count == 0);
    std::cout << (ok ? "PASS" : "FAIL")
              << "  chunks=" << n_chunks
              << "  verified=" << pass_count.load()
              << "  failed="  << fail_count.load()
              << "  total="   << (total_ok ? "OK" : "BAD") << "\n";

    // Free the per-drive output files now that they're verified.  At GiB-scale
    // inputs the configured "SSDs" may share one filesystem, so keeping every
    // test case's output around would exhaust space.
    std::set<std::string> out_files;
    for (const auto& c : output.chunks) out_files.insert(c.filename);
    for (const auto& fname : out_files) unlink(fname.c_str());

    return ok;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    // Default to 128 Mi uint64_t elements = exactly 1 GiB = 256 chunks, so the
    // chunk count comfortably exceeds the drive count (one file per SSD).
    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 134'217'728ULL;

    std::cout << "Building perm(" << n << ")...\n" << std::flush;
    const chunk_seq input = ChunkSequenceOps::perm(n);
    std::cout << input.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n\n";

    bool all_pass = true;

    // exclusive sum scan over perm(n): out[i] = 0+1+…+(i-1) = i*(i-1)/2
    // (out[0] = 0); total = 0+1+…+(n-1) = n*(n-1)/2.
    all_pass &= run_test<SumMonoid>(
        "sum  exclusive prefix",
        input, "scan_sum", ELEMS_PER_CHUNK, SumMonoid{},
        std::function<uint64_t(uint64_t)>(
            [](uint64_t i) -> uint64_t { return i * (i - 1) / 2; }),
        /*expected_total=*/(uint64_t)(n - 1) * n / 2);

    // exclusive xor scan over perm(n): out[i] = 0^1^…^(i-1) = xor_prefix(i-1)
    // (out[0] = 0); total = 0^1^…^(n-1) = xor_prefix(n-1).
    all_pass &= run_test<XorMonoid>(
        "xor  exclusive prefix",
        input, "scan_xor", ELEMS_PER_CHUNK, XorMonoid{},
        std::function<uint64_t(uint64_t)>(
            [](uint64_t i) -> uint64_t { return i == 0 ? 0 : xor_prefix(i - 1); }),
        /*expected_total=*/xor_prefix(n - 1));

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
