#include <iostream>
#include "ChunkSequence/chunk_delayed.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/examples/chunk_bigint_add.h"

int main() {
    chunk_seq a = ChunkSequenceOps::tabulate(10, "a", [](size_t i){return 1;});
    chunk_seq b = ChunkSequenceOps::tabulate(10, "a", [](size_t i){return 2;});
    // should be a bunch of threes, all the intermediates in here are delayed and the function itself forces back to ssds
    chunk_seq c = ChunkSequenceOps::ChunkBigIntAdd(a, b, "sum");
    // this combines the chunks back to one file, can be inspected in a hex editor
    c.consolidate("result");
}
