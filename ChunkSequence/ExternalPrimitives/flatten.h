#ifndef FLATTEN_H
#define FLATTEN_H
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/n_reader.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "configs.h"


namespace ChunkSequenceOps{


    inline chunk_seq flatten(const std::vector<chunk_seq>& chunk_sequences){
        size_t count = 0;
        for(size_t i =0; i<chunk_sequences.size(); i++){
            count +=chunk_sequences[i].chunks.size();
        }
        // size_t count = 0;

        // for(int i = 0; i < chunk_sequences.size(); i++){
        //     count+= chunk_sequences[i].size();
        // }
        // chunk_seq chunker(count);
        // // for(int i = 1; i < chunk_sequences.size(); i++){
        // //     chunk_sequences[0]chunk_seq.app

        // // }
        chunk_seq chunker;
        chunker.chunks.resize(count);
        size_t overall_counter = 0;
        for(size_t i = 0; i <chunk_sequences.size(); i++){
            for(size_t k = 0; k <chunk_sequences[i].chunks.size(); k++){
                auto c = chunk_sequences[i].chunks[k];
                c.index = overall_counter;              
                chunker.chunks[overall_counter] =c;
                overall_counter++;
            }
        }
        return chunker;
    }
    inline chunk_seq flatten(const parlay::sequence<chunk_seq>& chunk_sequences){
    size_t count = 0;
    for(size_t i =0; i<chunk_sequences.size(); i++){
        count +=chunk_sequences[i].chunks.size();
    }
    // size_t count = 0;

    // for(int i = 0; i < chunk_sequences.size(); i++){
    //     count+= chunk_sequences[i].size();
    // }
    // chunk_seq chunker(count);
    // // for(int i = 1; i < chunk_sequences.size(); i++){
    // //     chunk_sequences[0]chunk_seq.app

    // // }
    chunk_seq chunker;
    chunker.chunks.resize(count);
    size_t overall_counter = 0;
    for(size_t i = 0; i <chunk_sequences.size(); i++){
        for(size_t k = 0; k <chunk_sequences[i].chunks.size(); k++){
            auto c = chunk_sequences[i].chunks[k];
            c.index = overall_counter;              
            chunker.chunks[overall_counter] =c;
            overall_counter++;
        }
    }
    return chunker;
}



}

#endif
