#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>

#include "parlay/primitives.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_find_if.h"

// perm(n) holds the sequence 0, 1, …, n-1 in order, so the element equal to
// `target` sits at logical position `target`.  find_if(== target) must return
// exactly `target`, and a predicate that never fires must return n.

static bool report(const std::string& name, size_t got, size_t expected) {
    const bool ok = (got == expected);
    std::cout << "  " << std::left << std::setw(36) << name
              << (ok ? "PASS" : "FAIL")
              << "  got=" << got << " expected=" << expected << "\n";
    return ok;
}

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const size_t n = (argc > 1) ? std::stoull(argv[1]) : 5'000'000ULL;

    std::cout << "Building perm(" << n << ")...\n" << std::flush;
    const chunk_seq input = ChunkSequenceOps::perm(n);
    std::cout << input.chunks.size() << " chunks across "
              << GetSSDList().size() << " drives\n\n";

    bool all_pass = true;

    // Match at several positions: first, last, and a few interior/boundary ones.
    std::vector<size_t> targets = {0, 1, n / 2, n - 1};
    if (n > 524'288) targets.push_back(524'288);       // first element of chunk 1
    if (n > 524'289) targets.push_back(524'289);
    for (size_t t : targets) {
        all_pass &= report("find_if(== " + std::to_string(t) + ")",
                           ChunkSequenceOps::ChunkFindIf<uint64_t>(
                               input, [t](uint64_t x) { return x == t; }),
                           t);
    }

    // Predicate that is never satisfied -> not found -> returns n.
    all_pass &= report("find_if(no match)",
                       ChunkSequenceOps::ChunkFindIf<uint64_t>(
                           input, [n](uint64_t x) { return x >= n; }),
                       n);

    // Predicate satisfied by many elements -> first satisfying index.
    // x >= n/2 first holds at x == n/2, i.e. position n/2.
    all_pass &= report("find_if(>= n/2)",
                       ChunkSequenceOps::ChunkFindIf<uint64_t>(
                           input, [n](uint64_t x) { return x >= n / 2; }),
                       n / 2);

    std::cout << "\n" << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
    return all_pass ? 0 : 1;
}
