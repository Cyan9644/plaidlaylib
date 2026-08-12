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

#include "ChunkSequence/Primitives/operation.h"
#include "ChunkSequence/Primitives/count_sort.h"
#include "ChunkSequence/Primitives/flatten.h"
#include "ChunkSequence/Primitives/small_sequence_ops.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/scan_find.h"
#include "ChunkSequence/Primitives/sort_buckets.h"
#include "ChunkSequence/Primitives/histogram_by_index.h"
#include "ChunkSequence/Primitives/map.h"
#include "ChunkSequence/Primitives/pack.h"
#include "ChunkSequence/examples/external_TODO/primitive_quicksort.h"
#include "configs.h"



namespace plaid {

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