#ifndef EXTERNAL_FIND_H
#define EXTERNAL_FIND_H
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include <parlay/primitives.h>

#include "ChunkSequence/chunk_seq.h"


//the goal of this method is to compute a scan over the number of elements actually used in a sequence, which allows us to 
//more efficiently search the external sequence
//this is just sequential for now but I'll parallelize it later on
template<typename T>
void ChunkSequenceOps::scan_size(const chunk_seq& seq, parlay::sequence<size_t>& pseq){
    size_t count = 0;
    pseq[count] = 0;
    for(const auto& c: seq.chunks) {

        const size_t cnt = c.used / sizeof(T);  
        if(!count == 0){ 
        pseq[count] = pseq[count-1] + cnt;
        }
        count++;

    }
}

template<typename T>
T ChunkSequenceOps::scan_find(const chunk_seq& seq, parlay::sequence<size_t> pseq, size_t g) {
    
    auto res = parlay::find_upper(pseq, g);
    auto c = seq.chunk_seq[res];
    //pretty sure you don't need an aligned buffer for reading 
    T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "allocation wrong";
    int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fd);
    offset = AlignDown(begin_addr + g*sizeof(T))
    SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
    close(fd);
    T val = buf[g];
    free(buf);
    return val;
}


#endif
