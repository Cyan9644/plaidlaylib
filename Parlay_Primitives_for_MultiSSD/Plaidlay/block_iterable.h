

#include "externalSeq.h"
#include <iostream>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

#define BLOCKSIZE 2048


//generic view for block iterable sequence 
template <typename T>
struct flat_view{
const T* data;
size_t n;
using iterator = const T*;

size_t num_blocks(){
    size_t num;
    n % BLOCKSIZE == 0 ? num = n/BLOCKSIZE : num = n/BLOCKSIZE + 1;
    return num;
}

size_t begin_block(size_t block_num){

}


}