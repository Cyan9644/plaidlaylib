#include <iostream>
#include <fstream>

#include "utils/command_line.h"
#include "ChunkSequence/ExternalGraph/external_compressed_sparse_row.h"

int main(int argc, char* argv[]) {
    ParseGlobalArguments(argc, argv);

    const std::string path = "/tmp/tmp_csr_check_input.txt";
    {
        std::ofstream f(path);
        // vertex 0 -> 1 (w=1.5), 0 -> 2 (w=2.5), 1 -> 2 (w=3.5), 3 -> 0 (w=4.5)
        f << "0 1 1.5\n0 2 2.5\n1 2 3.5\n3 0 4.5\n";
    }

    chunk_csr g;
    g.from_file(path, "tmp_csr_check_edges");

    bool ok = true;
    auto check = [&](bool cond, const std::string& msg) {
        if (!cond) { std::cerr << "FAIL: " << msg << "\n"; ok = false; }
    };

    check(g.degree_scan.size() == 5, "degree_scan size (expected 5 for 4 vertices)");
    check(g.degree_of(0) == 2, "degree_of(0) == 2");
    check(g.degree_of(1) == 1, "degree_of(1) == 1");
    check(g.degree_of(2) == 0, "degree_of(2) == 0");
    check(g.degree_of(3) == 1, "degree_of(3) == 1");

    auto adj0 = g.get_adjacent(0);
    check(adj0.size() == 2, "adj(0) has 2 edges");
    bool has1 = false, has2 = false;
    for (auto& e : adj0) {
        if (e.connecting_vertex == 1) { check(e.edge_weight == (weight)1.5, "0->1 weight"); has1 = true; }
        if (e.connecting_vertex == 2) { check(e.edge_weight == (weight)2.5, "0->2 weight"); has2 = true; }
    }
    check(has1 && has2, "adj(0) contains both 1 and 2");

    auto adj3 = g.get_adjacent(3);
    check(adj3.size() == 1 && adj3[0].connecting_vertex == 0, "adj(3) == {0}");

    check(g.edge_exist(0, 1), "edge_exist(0,1)");
    check(!g.edge_exist(0, 3), "!edge_exist(0,3)");

    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
