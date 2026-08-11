#ifndef CHUNK_REVERSE_H
#define CHUNK_REVERSE_H

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "utils/logger.h"
#include "ChunkSequence/Primitives/count_sort.h"
#include "ChunkSequence/Primitives/flatten.h"
#include "ChunkSequence/Primitives/materialize.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "absl/log/check.h"
#include "configs.h"


namespace small_sequence_ops{

template<typename T>
chunk reverse(chunk& p){


int fd = open(p.filename.c_str(), O_DIRECT | O_RDWR, 0644);

T buffer[CHUNK_SIZE/sizeof(T)];

pread(fd, buffer, CHUNK_SIZE, (off_t)p.begin_addr); //this is going to be rounded up anyway so we might as well read the whole chunk


// std::vector<T> vec_tor(std::begin(buffer), std::end(buffer));
// vec_tor = std::reverse(vec_torvector);
size_t end = CHUNK_SIZE/sizeof(T) - 1;
size_t temp;
for(size_t i = 0; i < sizeof(buffer); i++, end--){

temp = buffer[i];
buffer[i] = buffer[end];
buffer[end] = temp;

}

pwrite(fd, buffer, CHUNK_SIZE, (off_t)p.begin_addr);

close(fd);

return p;

}




}


namespace ChunkSequenceOps{


//the good news is that this becomes extremely simple with the chunk_seq.
//basically the idea is that we reverse the ordering within a sequence and then reverse the order of the actual blocks

template<typename T>
chunk_seq reverse(chunk_seq& seq){

parlay::parallel_for(0, seq.chunks.size(), [&](size_t i){

seq.chunks[i] = ::small_sequence_ops::reverse<T>(seq.chunks[i]);

});

std::reverse(seq.chunks.begin(), seq.chunks.end());

return seq;

}

}  // namespace ChunkSequenceOps



#endif
