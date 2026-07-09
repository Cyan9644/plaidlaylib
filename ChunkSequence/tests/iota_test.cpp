#include <iostream>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"

/**
 * Verify that iota(n) writes the identity sequence correctly.
 *
 * For every chunk in the returned chunk_seq, this test opens chunk.filename
 * directly with open(O_DIRECT|O_RDONLY) and reads chunk.used bytes from
 * chunk.begin_addr using pread().  It then checks that element i of the
 * chunk equals  chunk.index * ELEMS_PER_CHUNK + i  for all i in [0, count).
 *
 * No ChunkSequenceReader, no UnorderedFileReader — raw syscalls only.
 */
int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    using T = uint64_t;

    // ── write ────────────────────────────────────────────────────────────────
    std::cout << "iota(" << n << ") writing...\n" << std::flush;
    const chunk_seq seq = ChunkSequenceOps::iota(n);
    std::cout << "wrote " << seq.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n" << std::flush;

    // ── assert all chunks live on configured SSD mounts ──────────────────────
    {
        const auto& ssds = GetSSDList();
        size_t ssd_count = 0;
        for (const auto& c : seq.chunks)
            for (const auto& ssd : ssds)
                if (c.filename.rfind(ssd, 0) == 0) { ssd_count++; break; }
        if (ssd_count != seq.chunks.size()) {
            std::cerr << "FAIL: only " << ssd_count << "/" << seq.chunks.size()
                      << " chunks are on configured SSDs\n";
            return 1;
        }
        std::cout << "SSD check OK (sample: " << seq.chunks[0].filename << ")\n"
                  << std::flush;
    }

    // ── verify each chunk via direct syscall reads ────────────────────────────
    std::atomic<size_t> pass_count{0};
    std::atomic<size_t> fail_count{0};

    parlay::parallel_for(0, seq.chunks.size(), [&](size_t ci) {
        const chunk& c = seq.chunks[ci];

        // Chunk metadata sanity checks.
        CHECK(c.used % sizeof(T) == 0) << "chunk " << c.index << ": used not T-aligned";
        CHECK(c.begin_addr % CHUNK_SIZE == 0) << "chunk " << c.index << ": begin_addr misaligned";

        const size_t count     = c.used / sizeof(T);
        const size_t start     = c.index * ELEMS_PER_CHUNK;
        const size_t read_size = AlignUp(c.used == 0 ? (size_t)O_DIRECT_MULTIPLE : c.used);

        if (count == 0) { pass_count++; return; }

        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        if (fd < 0) {
            std::cerr << "FAIL chunk " << c.index << ": open(" << c.filename
                      << ") failed: " << strerror(errno) << "\n";
            fail_count++;
            return;
        }

        T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_size);
        CHECK(buf != nullptr);

        const ssize_t got = pread(fd, buf, read_size, (off_t)c.begin_addr);
        close(fd);

        if (got < 0 || (size_t)got < c.used) {
            std::cerr << "FAIL chunk " << c.index << ": pread returned " << got
                      << " (expected " << c.used << ")\n";
            free(buf);
            fail_count++;
            return;
        }

        // Check each element.
        bool ok = true;
        for (size_t i = 0; i < count && ok; i++) {
            if (buf[i] != static_cast<T>(start + i)) {
                std::cerr << "FAIL chunk " << c.index << " element " << i
                          << ": got " << buf[i] << " expected " << (start + i)
                          << " (start=" << start << ")\n";
                fail_count++;
                ok = false;
            }
        }
        if (ok) pass_count++;

        free(buf);
    }, /*granularity=*/1);

    // ── coverage check ────────────────────────────────────────────────────────
    // Every chunk index [0, num_chunks) must appear exactly once.
    const size_t num_chunks = seq.chunks.size();
    std::vector<bool> seen(num_chunks, false);
    bool coverage_ok = true;
    for (const auto& c : seq.chunks) {
        if (c.index >= num_chunks || seen[c.index]) {
            std::cerr << "FAIL: duplicate or out-of-range chunk index " << c.index << "\n";
            coverage_ok = false;
        }
        seen[c.index] = true;
    }
    for (size_t i = 0; i < num_chunks; i++) {
        if (!seen[i]) {
            std::cerr << "FAIL: chunk index " << i << " missing from chunk_seq\n";
            coverage_ok = false;
        }
    }

    // ── report ────────────────────────────────────────────────────────────────
    const bool all_pass = (fail_count == 0) && coverage_ok;
    std::cout << (all_pass ? "PASS" : "FAIL")
              << "  chunks=" << num_chunks
              << "  verified=" << pass_count.load()
              << "  failed=" << fail_count.load()
              << "\n";

    return all_pass ? 0 : 1;
}
