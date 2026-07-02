#ifndef CHUNK_HEADER_H
#define CHUNK_HEADER_H

#include <pthread.h>
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>
#include <cstring>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <string>
#include <vector>


struct chunk_header{

std::string filename;
size_t begin_address;
size_t used;
size_t index;


};

std::string get_filename(chunk_header* chunk);

// size_t get_begin_address(chunk_header* chunk);
size_t get_bytes_used(chunk_header* chunk);
size_t get_index(chunk_header* chunk);


struct External_Sequence{

    // std::vector<chunk_header> ordered_underlying_sequence;
    parlay::sequence<chunk_header> ordered_underlying_sequence;

    External_Sequence(size_t length) : ordered_underlying_sequence(length){}
    
    size_t size() const{
        return this->ordered_underlying_sequence.size();
    }


  parlay::sequence<chunk_header>& getSeq(External_Sequence&);


    parlay::sequence<chunk_header>::iterator begin() {return ordered_underlying_sequence.begin();}
    parlay::sequence<chunk_header>::iterator end(){return ordered_underlying_sequence.end();}

    chunk_header& operator[](size_t i) {return ordered_underlying_sequence[i];}
    const chunk_header& operator[](size_t i) const {return ordered_underlying_sequence[i];}
    
    size_t get_begin_address(chunk_header* chunk);

    chunk_header& get_val(size_t index, chunk_header& chunker) {

        //we want to return the chunk header that contains the requested index in the logical sequence
        size_t running_total = 0;
        size_t ctr = 0;
        chunker = this->ordered_underlying_sequence[ctr];
        while(running_total + chunker.used <= index){
            running_total += chunker.used;
            chunker = this->ordered_underlying_sequence[++ctr];
        }
        //we have now found the block where this index lives in the logical sequence
        //okay, but the chunk header doesn't actually contain any data. 
        //let's just return it to the user and have them figure out what to do with it.
        return chunker;
    }

    //we need to be able to insert a chunk header in-place in order to build an external sequence
    //but note that the underlying datatype
    void add_header(chunk_header& chunk){
        // parlay::sequence<chunk_header> chunk1 = this->ordered_underlying_sequence.split(0, index);
        // chunk1.append(chunk);
        // parlay::sequence<chunk_header> chunk2 = this->ordered_underlying_sequence.split(index, this->ordered_underlying_sequence.size());
        this->ordered_underlying_sequence.push_back(chunk);

    }

};


// std::vector<std::vector<chunk_header>>& getSeq(External_Sequence* seq);
// std::vector<chunk_header>& getSeq(External_Sequence&);


#endif