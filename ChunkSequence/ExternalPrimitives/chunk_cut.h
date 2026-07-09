#ifndef CHUNK_CUT_H
#define CHUNK_CUT_H
#include <pthread.h>
#include "ChunkSequence/external_engine.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>
#include <cstring>
#include <random>
#include <array>
#include <algorithm>
#include <parlay/parallel.h>
#include <parlay/primitives.h>


#define NUM_SSDS 30
#ifndef PRACTICAL_SSDS 
#define PRACTICAL_SSDS 8
#define BUFFER_SIZE 512
#endif
    
//logic of the chunk cut method:

//we take an external sequence and two indices. we return an external sequence that is a copy of the requested range of the original sequence.
//key to this is that there are two points of difficulty: 
//1. The start of the requested chunk may not be aligned, in which case we'll read that chunk and create a new chunk header for it that is aligned
//2. The end of the requested chunk may not be aligned, in which case we'll again read that chunk and create a new chunk header for it


namespace ChunkSequenceOps{


//maybe we'll want to make this callable directly on an external sequence in the future

//a couple of notes: cut does not modify the original sequence
//even if the read locations are already aligned, we currently rewrite them to not share data.
//slice methods typically return a new sequence anyway, so I think this is not an issue
//currently we rewrite to a new buffer for each start/end read because we don't know whether they'll be aligned properly
//this is not the case for the middle reads because we know that our index runs through them, but we might start/end in the middle of a chunk
//but we know that we're going to copy everything anyway, so perhaps we can shave off the write to a new alignment
template<typename T>
chunk_seq sequential_cut_no_compression(const chunk_seq& seq, size_t start_index, size_t end_index){


//if(end_index > (ChunkSequenceOps::size(seq) * CHUNK_SIZE / sizeof(T)) || start_index >= end_index){ //start_index is size_t unsigned
if(end_index > (size<T>(seq)) || start_index >= end_index){ //start_index is size_t unsigned
    return {};
}
//we could allocate this perfectly if we computed a scan over the chunk headers, but I don't think it's worth it
parlay::sequence<chunk> chunk_headers;
size_t tracker = start_index;
size_t counter = 0;
size_t index_counter = 0;
while(index_counter < seq.chunks.size() && counter + (seq.chunks[index_counter].used/sizeof(T)) < start_index){

//this could be more simply implemented with a single tracker
counter += seq.chunks[index_counter].used/sizeof(T);
tracker-=seq.chunks[index_counter].used/sizeof(T); 
index_counter++;

}
//we have now found the correct chunk for the start
T* buff = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
int fd1 = open(seq.chunks[index_counter].filename.c_str(), O_RDONLY | O_DIRECT);
//seq.chunks[index_counter].used/sizeof(T) - tracker is the #of elements used in the block - the expected start position of the first
//index we want to see, which means we need to read size of the difference between the two to get all the data from start to end
SYSCALL(pread(fd1, buff, AlignUp(seq.chunks[index_counter].used), (off_t) seq.chunks[index_counter].begin_addr));
memmove(buff, buff + tracker, (seq.chunks[index_counter].used/sizeof(T) - tracker) * sizeof(T));
std::string start_cut = seq.chunks[index_counter].filename + "_cut";

int fd1_filename1 = open(start_cut.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
//we know that the O_DIRECT alignment below us is already full of the original data, so we're trying the one above
// SYSCALL(pwrite(fd1_filename,buff, (seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T), (off_t)(AlignUp(seq.chunks[index_counter].begin_addr + tracker*sizeof(T)))));
SYSCALL(pwrite(fd1_filename1,buff, AlignUp((seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T)), (off_t)0));

close(fd1);
close(fd1_filename1);
chunk start_chunk;
// start_chunk.filename = seq.chunks[index_counter].filename; start_chunk.begin_addr =(AlignUp(seq.chunks[index_counter].begin_addr + tracker*sizeof(T)));
start_chunk.filename = start_cut; start_chunk.begin_addr = (0);
start_chunk.used = (seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T); start_chunk.index = 0;


counter += seq.chunks[index_counter].used/sizeof(T);
index_counter++;
// tracker-=seq.chunks[index_counter].used/sizeof(T); 


tracker = end_index-counter;
chunk_headers.push_back(start_chunk);
while(index_counter < seq.chunks.size() && counter + (seq.chunks[index_counter].used/sizeof(T)) < end_index){

//this could be more simply implemented with a single tracker
counter += seq.chunks[index_counter].used/sizeof(T);
tracker-=seq.chunks[index_counter].used/sizeof(T);
chunk_headers.push_back(seq.chunks[index_counter]);
index_counter++;


}
//we have now reached the end chunk, so we need potentially just the first part of it.

//we're allocating a new buffer because eventually we want these start/end chunks things to be in a parallel do 
T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
int fd = open(seq.chunks[index_counter].filename.c_str(), O_RDONLY | O_DIRECT);


// SYSCALL(pread(fd, buf, AlignUp(seq.chunks[index_counter].used/sizeof(T) - tracker) * sizeof(T), (off_t) (seq.chunks[index_counter].begin_addr + tracker*sizeof(T))));
// SYSCALL(pread(fd, buf, AlignUp(seq.chunks[index_counter].used/sizeof(T) - tracker) * sizeof(T), (off_t) AlignDown((seq.chunks[index_counter].begin_addr + tracker*sizeof(T)))));
SYSCALL(pread(fd, buf, AlignUp(tracker * sizeof(T)), (off_t) seq.chunks[index_counter].begin_addr));

std::string end_cut = seq.chunks[index_counter].filename + "_cut";
int fd_filename = open(end_cut.c_str(), O_WRONLY | O_DIRECT | O_CREAT, 0644);
// SYSCALL(pwrite(fd_filename,buf, (seq.chunks[index_counter].used/sizeof(T)-tracker) * sizeof(T), (off_t)(AlignUp(seq.chunks[index_counter].begin_addr + tracker*sizeof(T)))));
SYSCALL(pwrite(fd_filename,buf, AlignUp(tracker * sizeof(T)), (off_t)0));
close(fd);
close(fd_filename);
chunk end_chunk;
end_chunk.filename = end_cut; end_chunk.begin_addr = 0;
end_chunk.used = tracker * sizeof(T); end_chunk.index = chunk_headers.size();

chunk_headers.push_back(end_chunk);

//maybe don't return a local variable
chunk_seq sequence = from_chunks(chunk_headers); //use the constructor for the chunk sequence, doesn't exist yet
free(buf);
free(buff);
return sequence;



}



}


#endif