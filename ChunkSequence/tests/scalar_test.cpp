#include <iostream>
#include <cstdint>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"

/**
 * Verify the scalar element ops on a materialized chunk_seq:
 *   ChunkSequenceOps::size  — total element count (not chunk count)
 *   ChunkSequenceOps::peek  — read one element at a logical index
 *   ChunkSequenceOps::push  — append one element (in place)
 *
 * Uses perm(n) as ground truth (element i == i), then exercises both push
 * paths: appending into a partial last chunk (read-modify-write of one block)
 * and appending past a full last chunk (new-chunk allocation).  Correctness is
 * cross-checked with peek and, finally, consolidate() to a local file.
 */
int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    namespace ops = ChunkSequenceOps;
    using T = uint64_t;

    int fails = 0;
    auto expect = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "FAIL: " << msg << "\n"; fails++; }
    };

    // ── size + peek on a partial-last-chunk perm ───────────────────────────────
    const size_t n = (argc > 1) ? std::stoull(argv[1])
                                : (3 * ELEMS_PER_CHUNK + 12345);  // partial last chunk
    std::cout << "perm(" << n << "), ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << "\n";
    chunk_seq seq = ops::perm(n);

    expect(ops::size(seq) == n, "size(perm(n)) != n");

    for (size_t i : {(size_t)0, n / 2, n - 1, 3 * ELEMS_PER_CHUNK,      // last-chunk head
                     3 * ELEMS_PER_CHUNK + 6000}) {                     // mid last chunk
        if (i >= n) continue;
        T got = ops::peek(seq, i);
        expect(got == (T)i, "peek(" + std::to_string(i) + ")=" + std::to_string(got));
    }

    // ── push into a partial last chunk (read-modify-write path) ────────────────
    for (size_t k = 0; k < 5; k++) {
        const size_t before = ops::size(seq);
        const T v = 1'000'000'000ULL + k;
        ops::push(seq, v);
        expect(ops::size(seq) == before + 1, "size did not grow by 1 after push");
        expect(ops::peek(seq, before) == v, "peek(new tail) != pushed value");
        // identity neighbors in the same RMW block/chunk stay untouched
        expect(ops::peek(seq, n - 1) == (T)(n - 1), "push corrupted neighbor n-1");
        expect(ops::peek(seq, 0) == (T)0, "push corrupted element 0");
    }

    // ── push that spills into a brand-new chunk ────────────────────────────────
    // Fill the current last chunk exactly, then push once more.
    {
        size_t last_used = seq.chunks.back().used;   // bytes
        size_t room = (CHUNK_SIZE - last_used) / sizeof(T);
        for (size_t j = 0; j < room; j++) ops::push(seq, (T)0xABCD0000 + j);
        expect(seq.chunks.back().used == CHUNK_SIZE, "last chunk not full after filling");

        const size_t nc_before = seq.chunks.size();
        const size_t idx = ops::size(seq);
        const T v = 0xFEED1234ULL;
        ops::push(seq, v);
        expect(seq.chunks.size() == nc_before + 1, "push did not allocate a new chunk");
        expect(seq.chunks.back().index == nc_before, "new chunk index wrong");
        expect(seq.chunks.back().used == sizeof(T), "new chunk used != sizeof(T)");
        expect(ops::size(seq) == idx + 1, "size wrong after new-chunk push");
        expect(ops::peek(seq, idx) == v, "peek(new-chunk element) != pushed value");
    }

    // ── consolidate + verify the whole thing matches an in-memory model ────────
    const std::string out = "/tmp/scalar_test_consolidated.bin";
    seq.consolidate(out);
    {
        FILE* f = fopen(out.c_str(), "rb");
        CHECK(f != nullptr);
        const size_t total = ops::size(seq);
        std::vector<T> buf(total);
        size_t got = fread(buf.data(), sizeof(T), total, f);
        fclose(f);
        expect(got == total, "consolidated file wrong length");
        // Every index we can predict: [0, n) is the identity.
        bool ok = true;
        for (size_t i = 0; i < n && ok; i++)
            if (buf[i] != (T)i) { expect(false, "consolidated[" + std::to_string(i) + "] wrong"); ok = false; }
    }

    // ── delayed::size (file / map / index / zip) ───────────────────────────────
    {
        namespace d = ChunkSequenceOps::delayed;
        chunk_seq base = ops::perm(n);                       // fresh, exactly n elems
        auto del = d::delay(base);
        expect(d::size(del) == n, "delayed::size(delay(seq)) != n");
        auto m = d::map(del, [](uint64_t x) { return x + 1; });
        expect(d::size(m) == n, "delayed::size(map) != n");   // map preserves count
        auto tab = d::tabulate(n + 7, [](size_t i) { return (uint64_t)i; });
        expect(d::size(tab) == n + 7, "delayed::size(tabulate) != n+7");
        auto z = d::zip(del, tab, (uint64_t)0);               // padded zip
        expect(d::size(z) == std::max(n, n + 7), "delayed::size(zip) != max(lenA,lenB)");
    }

    std::cout << (fails == 0 ? "PASS" : "FAIL") << "  size=" << ops::size(seq)
              << "  chunks=" << seq.chunks.size() << "  fails=" << fails << "\n";
    return fails == 0 ? 0 : 1;
}
