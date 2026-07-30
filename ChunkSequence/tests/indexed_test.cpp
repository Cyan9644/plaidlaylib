#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <unistd.h>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_indexed.h"
#include "ChunkSequence/examples/chunk_transpose.h"

/**
 * Correctness of the direct-indexing view (ChunkSequence/chunk_indexed.h):
 *   1. Session::get reads the right element (scattered indices) on iota(n).
 *   2. Batch gather of a known permutation (reverse) matches.
 *   3. Session::set (default RMW config) mutates the written element and leaves
 *      block neighbours untouched; verified via chunk_seq::operator[].
 *   4. transpose(transpose(A)) == A (round-trip identity) on a small square.
 *
 * Ground truth is iota (element i == i), cross-checked with operator[]/to_vector.
 */
int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);
    namespace ops = ChunkSequenceOps;
    using T = uint64_t;

    int fails = 0;
    auto expect = [&](bool ok, const std::string& msg) {
        if (!ok) { std::cerr << "FAIL: " << msg << "\n"; fails++; }
    };
    auto cleanup = [](const std::string& prefix) {
        const auto& ssds = GetSSDList();
        for (size_t d = 0; d < ssds.size(); d++) unlink(GetFileName(prefix, d).c_str());
    };

    const size_t n = (argc > 1) ? std::stoull(argv[1])
                                : (3 * ELEMS_PER_CHUNK + 12345);   // partial last chunk
    std::cout << "indexed_test: iota(" << n << "), ELEMS_PER_CHUNK=" << ELEMS_PER_CHUNK << "\n";

    // ── 1. Session::get on scattered indices ───────────────────────────────────
    {
        chunk_seq seq = ops::iota(n);
        ops::IndexedChunkSeq<T> view(seq);
        auto s = view.session();
        for (size_t i : {(size_t)0, (size_t)1, (size_t)511, (size_t)512, n / 3,
                         n / 2, n - 1, ELEMS_PER_CHUNK, 2 * ELEMS_PER_CHUNK + 7}) {
            if (i >= n) continue;
            T got = s.get(i);
            expect(got == (T)i, "get(" + std::to_string(i) + ")=" + std::to_string(got));
        }
        // Repeated gets on the same block (cache hits) still correct.
        for (size_t i = 100; i < 140; i++)
            expect(s.get(i) == (T)i, "cached get(" + std::to_string(i) + ")");
        cleanup("iota");
    }

    // ── 2. Batch gather of a reverse permutation ───────────────────────────────
    {
        chunk_seq seq = ops::iota(n);
        ops::IndexedChunkSeq<T> view(seq);
        const size_t k = std::min<size_t>(n, 100000);
        std::vector<size_t> idx(k);
        for (size_t j = 0; j < k; j++) idx[j] = n - 1 - j;       // reverse
        std::vector<T> out(k);
        view.gather(idx.data(), out.data(), k);
        bool ok = true;
        for (size_t j = 0; j < k && ok; j++) ok = out[j] == (T)(n - 1 - j);
        expect(ok, "gather(reverse) mismatch");
        cleanup("iota");
    }

    // ── 3. Session::set (RMW) mutates one element, preserves block neighbours ───
    {
        chunk_seq seq = ops::iota(n);
        ops::IndexedChunkSeq<T> view(seq);            // default cfg: write_full_blocks=false
        {
            auto s = view.session();
            s.set(1000, 0xAAAA0000ULL);               // write within a block
            s.set(1000 + 512, 0xBBBB0000ULL);         // next block
            s.flush();
        }
        // Read back through a fresh path (operator[] opens its own fd).
        expect(seq[1000] == 0xAAAA0000ULL, "set(1000) not persisted");
        expect(seq[1000 + 512] == 0xBBBB0000ULL, "set(1000+512) not persisted");
        expect(seq[1001] == (T)1001, "set corrupted block neighbour 1001");
        expect(seq[999]  == (T)999,  "set corrupted block neighbour 999");
        cleanup("iota");
    }

    // ── 4. transpose round-trip identity on a small square ─────────────────────
    {
        const size_t M = 3 * 512;                     // on the block grid (TILE=512), ~4.5 chunks
        const size_t NN = M * M;
        chunk_seq A = ops::tabulate<T>(NN, "tr_a", [](size_t i) { return (T)i; });
        chunk_seq B = ChunkTranspose::transpose<T>(A, M, "tr_b");
        chunk_seq C = ChunkTranspose::transpose<T>(B, M, "tr_c");

        // B[q] should be A[(q%M)*M + q/M] == (q%M)*M + q/M; C should equal A (iota).
        auto vb = B.to_vector<T>();
        auto vc = C.to_vector<T>();
        expect(vb.size() == NN && vc.size() == NN, "transpose size wrong");
        bool okB = vb.size() == NN, okC = vc.size() == NN;
        for (size_t q = 0; q < NN && okB; q++)
            okB = vb[q] == (T)((q % M) * M + (q / M));
        for (size_t q = 0; q < NN && okC; q++) okC = vc[q] == (T)q;
        expect(okB, "single transpose wrong");
        expect(okC, "double transpose != identity");
        cleanup("tr_a"); cleanup("tr_b"); cleanup("tr_c");
    }

    std::cout << (fails == 0 ? "PASS" : "FAIL") << "  fails=" << fails << "\n";
    return fails == 0 ? 0 : 1;
}
