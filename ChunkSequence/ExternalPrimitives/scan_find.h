#ifndef EXTERNAL_FIND_H
#define EXTERNAL_FIND_H
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"


//the goal of this method is to compute a scan over the number of elements actually used in a sequence, which allows us to 
//more efficiently search the external sequence
//this is just sequential for now but I'll parallelize it later on
namespace ChunkSequenceOps {
template<typename T>
void scan_size(const chunk_seq& seq, parlay::sequence<size_t>& pseq){
    size_t acc = 0;
    for(size_t i = 0; i < seq.chunks.size(); i++){
        pseq[i] = acc;
        acc += seq.chunks[i].used / sizeof(T);
    }
}

template<typename T>
T scan_find(const chunk_seq& seq, const parlay::sequence<size_t>& pseq, size_t g){

    const size_t res = std::upper_bound(pseq.begin(), pseq.end(), g) - pseq.begin() - 1;
    const auto& c = seq.chunks[res];
    const size_t local = g - pseq[res];

    // We want one element, so read only the O_DIRECT-aligned block that holds it
    // rather than the whole CHUNK_SIZE chunk.  This cuts the per-pivot I/O by
    // ~CHUNK_SIZE/O_DIRECT_MULTIPLE (e.g. 4 MiB -> 4 KiB); with 31*8 pivot probes
    // that is the difference between ~GiB and ~MiB of reads for the sample phase.
    const size_t byte_off    = c.begin_addr + local * sizeof(T);
    const size_t aligned_off = AlignDown(byte_off);
    const size_t delta       = byte_off - aligned_off;  // element's offset in block
    const size_t read_len    = AlignUp(delta + sizeof(T));

    char* buf = (char*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, read_len);
    CHECK(buf != nullptr) << "allocation wrong";
    int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, read_len, (off_t)aligned_off));
    close(fd);
    T val;
    std::memcpy(&val, buf + delta, sizeof(T));
    free(buf);
    return val;
}

}  


#endif
