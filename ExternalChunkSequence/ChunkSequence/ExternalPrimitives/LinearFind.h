#ifndef EXTERNAL_LINEAR_H
#define EXTERNAL_LINEAR_H
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include <parlay/primitives.h>

// Brings in chunk_seq, CHUNK_SIZE, O_DIRECT_MEMORY_ALIGNMENT (configs.h),
// AlignUp (utils/file_utils.h), and SYSCALL/CHECK (utils/logger.h + absl).
#include "ChunkSequence/chunk_seq.h"

// template<typename T>
// T ChunkSequenceOps::LinearFind(chunk_seq& seq, size_t index){
// size_t top = seq.chunks.size();
// for(int i = 0; i < top; i++){
//     if(index < seq.chunks[i].used){
//         int filedes = open(seq.chunks[i].filename.c_str(), O_DIRECT | O_RDONLY);
//         T* buffer = calloc(seq.chunks[i].used * sizeof(T));
//         lseek(filedes, seq.chunks[i].begin_addr, SEEK_SET);
//         read(filedes, buffer, seq.chunks[i].used);
//         T val = buffer[index];
//         free(buffer);
//         return val;
//     }
//     else{
//         index -= seq.chunks[i].used;
//     }}}

template<typename T>
T LinearFind(const chunk_seq& seq,size_t g) {
    for(const auto& c: seq.chunks) {
        const size_t cnt = c.used / sizeof(T);   
        if (g < cnt) {
            //pretty sure you don't need an aligned buffer for reading 
            T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(buf != nullptr) << "allocation wrong";
            int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
            SYSCALL(fd);
            SYSCALL(pread(fd, buf, AlignUp(c.used), (off_t)c.begin_addr));
            close(fd);
            T val = buf[g];
            free(buf);
            return val;
        }
        g -= cnt;
    }
    CHECK(false) << "out of range";
    return T{};
}
#endif