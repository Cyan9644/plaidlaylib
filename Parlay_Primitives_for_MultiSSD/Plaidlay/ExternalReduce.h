#ifndef EXTERNAL_REDUCE_H
#define EXTERNAL_REDUCE_H
#include "config_threads.h"
#include <pthread.h>
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include "chunk_header.h"
#include <liburing.h>
#include <cstring>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/monoid.h>
#include "utils/unordered_file_reader_modified.h"


#ifndef NUM_SSDS
#define NUM_SSDS 30
#endif

// Reduce an External_Sequence (a list of chunk_headers describing on-disk
// blocks) down to a single value. This is the chunk-model analogue of
// sequence_algorithms/reduce.h: a pure streaming read with no writes.
//
// `op` must be associative with `identity` as its unit (it is wrapped in a
// parlay::monoid so the final fold over per-worker partials stays correct). T is
// the on-disk element type; R is the accumulator type (defaults to T but may
// differ, e.g. summing `int` elements into a `size_t`).
template<typename T, typename R = T, typename BinOp>
R ExternalReduce(External_Sequence &seq, BinOp op, R identity) {
    parlay::monoid monoid(op, identity);

    UnorderedChunkReader<T, CHUNK_SIZE> reader;
    reader.PrepFiles(seq.ordered_underlying_sequence);
    reader.Start();

    // One partial accumulator per worker. Each worker drains the reader's shared
    // buffer queue (Poll() is thread-safe) until it sees the drained sentinel,
    // then we fold the partials with the same monoid. This mirrors the in-memory
    // Reduce and keeps the IO-bound path saturated without a hand-rolled batch
    // loop.
    return parlay::reduce(parlay::tabulate(parlay::num_workers(), [&](size_t) {
        R acc = identity;
        while (true) {
            auto [ptr, size, fidx, index, which, fname] = reader.Poll();
            if (ptr == nullptr) break;  // reader drained and closed
            for (size_t i = 0; i < size; i++) {
                acc = monoid(acc, ptr[i]);  // empty chunks (size == 0) loop 0x
            }
            reader.allocator.Free(ptr);
        }
        return acc;
    }, 1), monoid);
}

#endif
