#ifndef EXTERNAL_SAMPLE_H
#define EXTERNAL_SAMPLE_H
#include <parlay/primitives.h>
#include <parlay/random.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ChunkSequence/ExternalPrimitives/chunk_operation.h"
#include "ChunkSequence/ExternalPrimitives/count_sort.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/small_sequence_ops.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/scan_find.h"
#include "ChunkSequence/ExternalPrimitives/sort_buckets.h"
#include "ChunkSequence/chunk_histogram_by_index.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_pack.h"
#include "ChunkSequence/examples/external/primitive_quicksort.h"
#include "configs.h"



namespace ChunkSequenceOps {

template <typename T>
parlay::sequence<T> sample(const chunk_seq& seq, size_t number_elements){

const size_t total = size<T>(seq);

parlay::sequence<size_t> scan_seq(seq.chunks.size());
scan_size<T>(seq, scan_seq);

parlay::random_generator gen;
std::uniform_int_distribution<size_t> dis(0, total -1);

return parlay::tabulate(number_elements, [&](long i){

auto r = gen[i];
auto index = dis(r);
return scan_find<T>(seq, scan_seq, index);

});

}

}


#endif