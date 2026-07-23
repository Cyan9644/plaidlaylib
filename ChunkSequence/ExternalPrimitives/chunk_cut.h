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
#include <mutex>
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

inline chunk_seq cut_by_chunk(const chunk_seq& seq, size_t chunk_begin,
                              size_t chunk_end) {
    chunk_seq out;
    if (chunk_begin >= chunk_end || chunk_begin >= seq.chunks.size()) return out;
    chunk_end = std::min(chunk_end, seq.chunks.size());
    out.chunks.reserve(chunk_end - chunk_begin);
    for (size_t i = chunk_begin; i < chunk_end; i++) {
        chunk c = seq.chunks[i];
        c.index = out.chunks.size();
        out.chunks.push_back(c);
    }
    return out;
}


// ── metadata shift / guard-limb append (zero-chunk aliasing) ─────────────────
// A shift (multiply a big integer by base^(k*ELEMS_PER_CHUNK)) and a guard-limb
// append both need chunks that *read as zeros* without moving any data.  We keep
// one zero-filled CHUNK_SIZE block per drive, written once, and alias it as many
// times as needed — safe because these chunks are only ever read, never written.
inline const std::string& zero_chunk_prefix() {
    static const std::string p = "bimul_zero";
    return p;
}

// Write one zero-filled CHUNK_SIZE block per drive (idempotent for the run).
inline void ensure_zero_chunks() {
    static std::once_flag once;
    std::call_once(once, [] {
        const size_t nd = GetSSDList().size();
        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "ensure_zero_chunks: buffer allocation failed";
        memset(buf, 0, CHUNK_SIZE);
        for (size_t d = 0; d < nd; d++) {
            const std::string fn = GetFileName(zero_chunk_prefix(), d);
            int fd = open(fn.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
            SYSCALL(fd);
            SYSCALL(pwrite(fd, buf, CHUNK_SIZE, (off_t)0));
            close(fd);
        }
        free(buf);
    });
}

// One chunk header aliasing drive d's shared zero block, exposing `used` bytes.
inline chunk zero_chunk_header(size_t d, size_t used, size_t index) {
    ensure_zero_chunks();
    const size_t nd = GetSSDList().size();
    return {GetFileName(zero_chunk_prefix(), d % nd), 0, used, index};
}

// seq * base^(k * ELEMS_PER_CHUNK), as pure metadata: prepend k full zero chunks
// (spread across drives) and reindex seq's chunks by +k.  Preserves the
// dense-except-last invariant (the prepended chunks are full; seq's own tail
// stays the tail) and leaves the top limb untouched (a canonical non-negative
// operand stays canonical).  No data is written.
inline chunk_seq prepend_zero_chunks(const chunk_seq& seq, size_t k) {
    if (k == 0) return seq;
    chunk_seq out;
    out.chunks.reserve(k + seq.chunks.size());
    for (size_t j = 0; j < k; j++)
        out.chunks.push_back(zero_chunk_header(j, CHUNK_SIZE, j));
    for (const chunk& c : seq.chunks) {
        chunk nc = c;
        nc.index = out.chunks.size();
        out.chunks.push_back(nc);
    }
    return out;
}

// Append one chunk holding `used` zero bytes (aliased, no write).  The caller
// must guarantee seq's current last chunk is full, so the result stays
// dense-except-last.  Used to attach a zero guard limb (used == sizeof(limb)) to
// a chunk-aligned cut half whose top limb has its sign bit set.
inline chunk_seq append_zero_chunk(const chunk_seq& seq, size_t used) {
    chunk_seq out = seq;
    out.chunks.push_back(zero_chunk_header(out.chunks.size(), used,
                                           out.chunks.size()));
    return out;
}


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
// Distinct suffix from the end seam below: if the start and end chunks land on
// the same drive their base filenames are identical, so a shared "_cut" suffix
// would make both seams write offset 0 of the same file and clobber each other.
std::string start_cut = seq.chunks[index_counter].filename + "_cut_start";

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

std::string end_cut = seq.chunks[index_counter].filename + "_cut_end";
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