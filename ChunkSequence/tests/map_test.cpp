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
#include "ChunkSequence/chunk_map.h"

// ── helpers ──────────────────────────────────────────────────────────────────

// Read one output chunk from disk via pread and verify every element.
//
// f_expected maps the global input index (= value written by iota) to the
// expected output value.  No assumption is made about begin_addr or which
// file the chunk lives in — we use the chunk descriptor as-is.
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

// Run one named map test case, return true iff all chunks verify correctly.
template<typename R>
bool run_test(const std::string& name,
              const chunk_seq& input,
              const std::string& result_prefix,
              size_t elems_per_input_chunk,
              std::function<R(uint64_t)> f) {
    std::cout << "  " << name << " ... " << std::flush;

    const chunk_seq output = ChunkSequenceOps::ChunkMap<uint64_t, R>(input, result_prefix, f);

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
        if (verify_chunk<R>(output.chunks[ci], elems_per_input_chunk, f))
            pass_count++;
        else
            fail_count++;
    }, /*granularity=*/1);

    const bool ok = coverage_ok && (fail_count == 0);
    std::cout << (ok ? "PASS" : "FAIL")
              << "  chunks=" << n_chunks
              << "  verified=" << pass_count.load()
              << "  failed="  << fail_count.load() << "\n";

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

    using T = uint64_t;

    std::cout << "Building iota(" << n << ")...\n" << std::flush;
    const chunk_seq input = ChunkSequenceOps::iota(n);
    std::cout << input.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n\n";

    bool all_pass = true;

    // ── in-place cases (T == R == uint64_t) ──────────────────────────────────

    // identity: output[i] == input[i] == global_index
    all_pass &= run_test<T>("identity      x -> x",
                            input, "map_id", ELEMS_PER_CHUNK,
                            [](T x) -> T { return x; });

    // increment: output[i] == global_index + 1
    all_pass &= run_test<T>("increment     x -> x+1",
                            input, "map_incr", ELEMS_PER_CHUNK,
                            [](T x) -> T { return x + 1; });

    // double: output[i] == global_index * 2
    all_pass &= run_test<T>("double        x -> x*2",
                            input, "map_double", ELEMS_PER_CHUNK,
                            [](T x) -> T { return x * 2; });

    // complement: output[i] == ~global_index
    all_pass &= run_test<T>("complement    x -> ~x",
                            input, "map_compl", ELEMS_PER_CHUNK,
                            [](T x) -> T { return ~x; });

    // ── type-changing case (T=uint64_t, R=uint32_t) ───────────────────────────
    // Tests the non-in-place allocation path; for n < 2^32 the low 32 bits
    // are lossless.
    all_pass &= run_test<uint32_t>("narrow u64->u32 x -> uint32_t(x)",
                                   input, "map_narrow", ELEMS_PER_CHUNK,
                                   [](T x) -> uint32_t { return (uint32_t)x; });

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
