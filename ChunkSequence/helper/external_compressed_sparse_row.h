#ifndef EXTERNAL_CSR_H
#define EXTERNAL_CSR_H

#include <fcntl.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>

// cut.h / flatten.h / materialize.h / segmented_reduce.h have all since been
// merged into secondary_primitives.h; delayed::cut lives in delayed.h.
#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/delayed.h"
#include "ChunkSequence/Primitives/secondary_primitives.h"
#include "utils/file_utils.h"

using vertex = size_t;
using weight = long double;

struct weighted_edge {
  vertex connecting_vertex;
  weight edge_weight;
};

// the compressed sparse row represents a graph with two arrays: N and F
// N is the exclusive prefix sum of node degrees, while F is the actual list of
// adjacency nodes we are assuming that the degree prefix sum can fit in-memory
// -- the edges are the memory bottleneck i.e. we are processing a dense graph
// or a sparse graph that doesn't have a ridiculous number of nodes we have 512
// GB DRAM on the remote machine

// for csr, to access the nodes of vertex i, we compute N[]
//
struct chunk_csr {
  parlay::sequence<size_t> degree_scan;  // this will be the degree scan
  chunk_seq edges;

  // merge the two adjacencies, degree_scan will need to be calculated again
  void merge(chunk_csr& other_chunk_csr) {
    std::vector<chunk_seq> vec(2);
    vec[0] = this->edges;
    vec[1] = other_chunk_csr.edges;
    this->edges = plaid::flatten(vec);
    size_t offset = this->degree_scan[this->degree_scan.size() - 1];
    parlay::parallel_for(0, other_chunk_csr.degree_scan.size(), [&](long i) {
      other_chunk_csr.degree_scan[i] += offset;
    });
    this->degree_scan = parlay::append(
        this->degree_scan,
        parlay::sequence<size_t>(other_chunk_csr.degree_scan.begin() + 1,
                                 other_chunk_csr.degree_scan.end()));
  }

  // this method should accept a vertex ID n and return a parlay sequence of the
  // (destination, weight) edges of n
  parlay::sequence<weighted_edge> get_adjacent(size_t n) const {
    return plaid::materialize<weighted_edge>(
        plaid::sequential_cut_no_compression<weighted_edge>(
            this->edges, this->degree_scan[n], this->degree_scan[n + 1]));
  }

  // method to return a delayed external sequence of adjacent
  auto delay_get_adjacent(size_t n) const {
    return plaid::delayed::cut<weighted_edge>(
        this->edges, this->degree_scan[n], this->degree_scan[n + 1]);
  }

  bool edge_exist(size_t n, size_t edge_id) {
    parlay::sequence<weighted_edge> inter =
        plaid::materialize<weighted_edge>(
            plaid::sequential_cut_no_compression<weighted_edge>(
                this->edges, this->degree_scan[n], this->degree_scan[n + 1]));
    auto iterator = parlay::find_if(inter, [&](const weighted_edge& e) {
      return e.connecting_vertex == edge_id;
    });
    if (iterator != inter.end()) {
      return true;
    }
    return false;
  }
  size_t degree_of(size_t node_id) {
    return (this->degree_scan[node_id + 1] - this->degree_scan[node_id]);
  }

  template <typename R, typename ElemFn, typename Monoid>
  parlay::sequence<R> segmented_reduce_over_edges(ElemFn elem_to_val,
                                                  Monoid monoid,
                                                  size_t reader_threads = 10) {
    return plaid::ChunkSegmentedReduce<weighted_edge, R>(
        this->edges, this->degree_scan, elem_to_val, monoid, reader_threads);
  }

  // Small fast field parsers for from_file(): advance p past one
  // whitespace-separated token (and any leading whitespace) and return its
  // parsed value.
  inline vertex parse_vertex(const char*& p) {
    while (*p == ' ' || *p == '\t') p++;
    char* end;
    vertex v = (vertex)strtoull(p, &end, 10);
    p = end;
    return v;
  }
  inline weight parse_weight(const char*& p) {
    while (*p == ' ' || *p == '\t') p++;
    char* end;
    weight w = strtold(p, &end);
    p = end;
    return w;
  }

  // Load a graph from a text file of "v1 v2 weight" lines (one edge per line,
  // directed).  Not used by the RMAT-generated synthetic graphs the
  // bellman_ford driver builds, but kept as part of the struct's API.
  void from_file(const std::string& filename,
                 const std::string& result_prefix = "csr_edges") {
    int fd = open(filename.c_str(), O_RDONLY);
    SYSCALL(fd);
    struct stat st;
    SYSCALL(fstat(fd, &st));
    const size_t file_size = (size_t)st.st_size;
    std::vector<char> buf(file_size);
    for (size_t done = 0; done < file_size;) {
      ssize_t r = read(fd, buf.data() + done, file_size - done);
      SYSCALL(r);
      done += (size_t)r;
    }
    close(fd);
    const char* data = buf.data();

    parlay::sequence<size_t> line_start =
        parlay::filter(parlay::iota<size_t>(file_size),
                       [&](size_t i) { return i == 0 || data[i - 1] == '\n'; });
    const size_t m = line_start.size();
    CHECK(m > 0) << "from_file: no edges found in " << filename;

    struct edge_input {
      vertex v1, v2;
      weight w;
    };
    parlay::sequence<edge_input> parsed(m);
    std::atomic<vertex> max_v{0};
    parlay::parallel_for(0, m, [&](size_t i) {
      const char* p = data + line_start[i];
      vertex v1 = parse_vertex(p);
      vertex v2 = parse_vertex(p);
      weight w = parse_weight(p);
      parsed[i] = {v1, v2, w};
      vertex local = std::max(v1, v2);
      vertex prev = max_v.load(std::memory_order_relaxed);
      while (local > prev && !max_v.compare_exchange_weak(
                                 prev, local, std::memory_order_relaxed)) {
      }
    });
    const size_t num_vertices =
        (size_t)max_v.load(std::memory_order_relaxed) + 1;

    std::vector<std::atomic<size_t>> degree_count(num_vertices);
    for (auto& c : degree_count) c.store(0, std::memory_order_relaxed);
    parlay::parallel_for(0, m, [&](size_t i) {
      degree_count[parsed[i].v1].fetch_add(1, std::memory_order_relaxed);
    });
    this->degree_scan = parlay::sequence<size_t>(num_vertices + 1);
    this->degree_scan[0] = 0;
    for (size_t v = 0; v < num_vertices; v++)
      this->degree_scan[v + 1] =
          this->degree_scan[v] +
          degree_count[v].load(std::memory_order_relaxed);
    std::vector<std::atomic<size_t>> cursor(num_vertices);
    parlay::parallel_for(0, num_vertices, [&](size_t v) {
      cursor[v].store(this->degree_scan[v], std::memory_order_relaxed);
    });
    parlay::sequence<weighted_edge> placed(m);
    parlay::parallel_for(0, m, [&](size_t i) {
      const size_t pos =
          cursor[parsed[i].v1].fetch_add(1, std::memory_order_relaxed);
      placed[pos] = {parsed[i].v2, parsed[i].w};
    });
    this->edges = plaid::to_chunk_seq(placed, result_prefix);
  }
};

#endif
