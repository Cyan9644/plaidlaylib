//this file is kept for reference but should not be used
#ifndef count_sort_H
#define count_sort_H
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
#endif



 


            



namespace ChunkSequenceOps{


//ChunkCountSort should take in an external chunk sequence and reorder physically or logically such that 
//(stably) all elements with ID 0 come before all elements with ID 1, etc.
//the problem is that the IDs themselves are potentially so large that they come in an external sequence themselves --
//this means we require a minimum of a read for every chunk header to find the IDs
//and another to figure out what values those correspond to
//another, more concerning point: do we need to skip values? e.g. if we see a bucket index 30 on the 0th pass,
//do we ignore it and continue looking for ID 0 elements? I think no becuase then we have far too many reads
//a better approach is probably to have an array of bucket_number size with each element being an external sequence
//but then eventually we need to fuse these back into a single sequence, which shouldn't be too bad
template<typename T>
chunk_seq& count_sort(const chunk_seq& seq, const chunk_seq& ids, std::vector<chunk_seq&> externalSequenceVector){
    //what we want to do here is poll from both seq and ids to match elements to bucket indices
    //the current strategy is to make separate buffers for each bucket and push these to the external sequence corresponding to the
    //bucket in externalSequenceVector when they become full
    std::vector<chunk_seq*> buffers(externalSequenceVector.size());
    for(int i = 0; i < externalSequenceVector.size(); i++){
        buffers[i] = (chunk_seq*) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    }
    std::vector<size_t> buffer_counters(externalSequenceVector.size(), 0);

    //problem: we want these two readers to be going at the same time, but not nested
    //also we can't hold the full sequences in DRAM at once, so I hope this doesn't try to do that
    //this is what we implemneted NRemoveWorker for

    // auto remove_from_ids = ChunkSequenceOps::RemoveWorker<T>(seq,  10, [&](ChunkSequenceReader<T>& reader1){
    //     // parlay::sequence<size_t> remove(num_unique, 0);
    //     while(true){
    //         auto [ptr1, size1,index1] = reader1.Poll();
            
    //         if(ptr1 == nullptr) break;
    //         for(size_t k=0; k <size1; k++){
    //             //some logic to do something, obviously the current structure is wrong
                
    //         }
    //         reader.allocator.Free(ptr1);
    //     }
    //     return remove;
    // });



    auto remove_from_queue = ChunkSequenceOps::NRemoveWorker<T>(ids, seq,  10, [&](ChunkSequenceReader<T>& reader){
        // parlay::sequence<size_t> remove(num_unique, 0);
        while(true){
            auto match = reader.Poll();
            if(!match.valid()) break;
            uint64_t* id_pointer = match.ptrs[0];
            uint64_t* sequence_pointer = match.ptrs[1];
            size_t size = match.sizes[0];
            
            if(ptr2 == nullptr) break;
            for(size_t k=0; k <size; k++){
                size_t j = id_pointer[k];
                buffers[j][buffer_counters[j]] = sequence_pointer[k]; //we want the buffer with index of the bucket to have the element added to it
                if(buffer[j][buffer_counters[j]] == BUFFER_SIZE-1){
                    //buffer is full, so we need to push it to the corresponding chunk sequence
                }
                //I don't think we have an efficient chunk append method yet, but I'll implement this later
                //probably we'd want to do this with an io-uring like structure where we have another
                //buffer of buffers to avoid doing single-chunk writes
                externalSequenceVector[j].chunk_append(buffer[j]);
                //could perhaps just zero the memory instead of reallocing
                free(buffer[j]);
                buffer[j] = (chunk_seq*) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            }
            reader.free(match);
        }
        return remove;

    });

    return remove_from_queue;


    // parlay::sequence<size_t> total(num_unique, 0);
    // for(auto& remove : remove_from_queue){
    //     for(size_t j = 0; j < num_unique; j++){
    //         total[j] += remove[j];
    //     }
    // }
    // for(int i = 0; i < externalSequenceVector.size(); i++){
    //     free(buffers[i]);
    // }
    // return total;

}


}


parlay::sequence<size_t> fuse(parlay::sequence<chunk_seq&>& seq){


}

#endif