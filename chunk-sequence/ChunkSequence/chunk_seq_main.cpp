#include <cstdint>
#include <iostream>
#include <chrono>

#include "ChunkSequence/chunk_filter.h"
#include "ChunkSequence/chunk_map.h"
#include "absl/log/log.h"
#include "absl/log/initialize.h"

#include "utils/command_line.h"
#include "utils/file_utils.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_reduce.h"

struct AddMonoid {
    uint64_t identity = 0;
    uint64_t operator()(uint64_t a, uint64_t b) const { return a + b; }
};
struct DoubleMonoid {
    double identity = 0;
    double operator()(double a, double b) const { return a + b; }
};

// static double now() {
//     using namespace std::chrono;
//     return duration<double>(high_resolution_clock::now().time_since_epoch()).count();
// }

struct point {
    float x,y;
};

int main(int argc, char* argv[]) {
    absl::InitializeLog();
    ParseGlobalArguments(argc, argv);

    size_t tot = 1'00'000'000;
    chunk_seq rand_points_in_unit_square = ChunkSequenceOps::tabulate<point>(tot, "rand_points_in_unit_square", [](size_t i) {
        return point{static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX};
    });
    chunk_seq vals = ChunkSequenceOps::ChunkMap<point, double>(rand_points_in_unit_square, "vals", [](point val) {
        point p = reinterpret_cast<point&>(val);
        double norm = p.x * p.x + p.y * p.y;
        return norm <= 1.0 ? 1.0 : 0.0;
    });
    double pi = 4.0 * ChunkSequenceOps::ChunkReduce<double>(vals, DoubleMonoid()) / tot;
    #include <iomanip>
    std::cout << "Monte Carlo: π ≈ " << std::setprecision(30) << pi << std::endl;

     chunk_seq terms = ChunkSequenceOps::tabulate<double>(1'00'000'000, "terms", [](size_t i) {
         return (i % 2 == 0 ? 1 : -1) / (2.0 * i + 1);
     });
     pi = 4.0 * ChunkSequenceOps::ChunkReduce<double>(terms, DoubleMonoid());
     #include <iomanip>
     std::cout << "Summation: π ≈ " << std::setprecision(30) << pi << std::endl;
}
