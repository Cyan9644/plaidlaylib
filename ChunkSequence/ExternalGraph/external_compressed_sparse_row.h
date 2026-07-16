#ifndef EXTERNAL_CSR_H
#define EXTERNAL_CSR_H

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/ExternalPrimitives/flatten.h"
#include "ChunkSequence/ExternalPrimitives/materialize.h"
#include "ChunkSequence/ExternalPrimitives/chunk_cut.h"


#define vertex size_t
#define weight long double

struct weighted_edge {
    vertex connecting_vertex;
    weight edge_weight;
};

//the compressed sparse row represents a graph with two arrays: N and F
//N is the exclusive prefix sum of node degrees, while F is the actual list of adjacency nodes
//we are assuming that the degree prefix sum can fit in-memory -- the edges are the memory bottleneck
//i.e. we are processing a dense graph or a sparse graph that doesn't have a ridiculous number of nodes
//we have 512 GB DRAM on the remote machine


//for csr, to access the nodes of vertex i, we compute N[]
//
struct chunk_csr{
    parlay::sequence<size_t> degree_scan; //this will be the degree scan
    // chunk_seq adjacent_weights;
    // chunk_seq edge_weights;
    chunk_seq edges; //this was originally two sequences but I think it's better to 

    // void set_scan_adj_matrix(T graph){//let's assume that we're accepting a graph adjacency matrix that implements some interface that gives us access to the degree of each node
    //     //any practical graph is probably stored in a file
    //     for(size_t i : grp)
    // }

    //takes a file and constructs the chunk_csr in parallel, assuming that we can store the vertex list purely in-memory
    void from_file(){

    }


    //merge the two adjacencies, degree_scan will need to be calculated again
    void merge(chunk_csr& other_chunk_csr){
        std::vector<chunk_seq> vec(2);
        vec[0] = this->edges;
        vec[1] = other_chunk_csr.edges;
        this->edges = ChunkSequenceOps::flatten(vec);
        size_t offset = this->degree_scan[this->degree_scan.size()-1];
        parlay::parallel_for(0, other_chunk_csr.degree_scan.size(), [&](long i){

            other_chunk_csr.degree_scan[i] += offset;


        });
        this->degree_scan = parlay::append(this->degree_scan,
            parlay::sequence<size_t>(other_chunk_csr.degree_scan.begin() + 1, other_chunk_csr.degree_scan.end()));
    }


    //this method should accept a vertex ID n and return a parlay sequence of the (destination, weight) edges of n
    parlay::sequence<weighted_edge> get_adjacent(size_t n){

        return ChunkSequenceOps::materialize<weighted_edge>(ChunkSequenceOps::sequential_cut_no_compression<weighted_edge>(this->edges, this->degree_scan[n], this->degree_scan[n+1]));
    }

    //method to return a delayed external sequence of adjacent. yeah, right.
    chunk_seq delay_get_adjacent(size_t n){
        return ChunkSequenceOps::sequential_cut_no_compression<weighted_edge>(this->edges, this->degree_scan[n], this->degree_scan[n+1]);
    }

    bool edge_exist(size_t n, size_t edge_id){
        parlay::sequence<weighted_edge> inter = ChunkSequenceOps::materialize<weighted_edge>(ChunkSequenceOps::sequential_cut_no_compression<weighted_edge>(this->edges, this->degree_scan[n], this->degree_scan[n+1]));
        auto iterator = parlay::find_if(inter, [&](const weighted_edge& e){
            return e.connecting_vertex == edge_id;
        });
        if(iterator != inter.end()){
            return true;
        }
        return false;
    }
    size_t degree_of(size_t node_id){
        return (this->degree_scan[node_id+1] - this->degree_scan[node_id]);
    }

    //one thing that might be useful is conversion to compressed sparse column format


};

// struct weighted_chunk_csr{
//     parlay::sequence<size_t> degree_scan; //this will be the degree scan
//     chunk_seq<size_t> adjacent;
//     chunk_seq<size_t> edge_weights;

//     // void set_scan_adj_matrix(T graph){//let's assume that we're accepting a graph adjacency matrix that implements some interface that gives us access to the degree of each node
//     //     //any practical graph is probably stored in a file
//     //     for(size_t i : grp)
//     // }

//     //this method should accept a vertex ID n and return a parlay sequence of the vertices connected to n
//     parlay::sequence<size_t> get_adjacent(size_t n){

//         size_t degrees = this->degree_scan[n+1] - this->degree_scan[n];
//         return ChunkSequenceOps::materialize(ChunkSequenceOps::sequential_cut_no_compression<size_t>(this->adjacent, this->degree_scan[n], degrees));
//     }

//     //method to return a delayed external sequence of adjacent 
//     chunk_seq& delay_get_adjacent(size_t n){
//         size_t degrees = this->degree_scan[n+1] - this->degree_scan[n];
//         return ChunkSequenceOps::delayed::sequential_cut_no_compression<size_t>(this->adjacent, this->degree_scan[n], degrees);
//     }

//     bool edge_exist(size_t edge_id){
//         size_t degrees = this->degree_scan[n+1] - this->degree_scan[n];
//         parlay::sequence<size_t> inter = ChunkSequenceOps::materialize(ChunkSequenceOps::sequential_cut_no_compression<size_t>(this->adjacent, this->degree_scan[n], degrees));
//         auto iterator = parlay::find_if(inter, [&](size_t id){
//             return inter[id] == edge_id;
//         });
//         if(iterator != inter.end()){
//             return true;
//         }
//         return false;
//     }
//     size_t degree_of(size_t node_id){
//         return (this->degree_scan[node_id+1] - this->degree_scan[node_id]);
//     }

//     size_t size(){

//         return this->adjacent.size();
//     }


//     //one thing that might be useful is conversion to compressed sparse column format

// }



#endif