//VERIFICATION SCRIPT WRITTEN BY CLAUDE, DO NOT TRUST
//not yet verified by me 
// --Alex


#include "externalSeq.h"
#include <iostream>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

static void nop_del(size_t *) {}


static void write_constant(const std::string &prefix, size_t n, size_t val,
                           size_t num_files) {
    const size_t step = 1UL << 20; 
    auto *buf = (size_t *)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                        n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) buf[i] = val;

    UnorderedWriterConfig cfg;
    cfg.num_threads = 1;
    cfg.num_files   = num_files;
    UnorderedFileWriter<size_t> writer(prefix, cfg);
    for (size_t i = 0; i < n; i += step) {
        size_t chunk = std::min(step, n - i);
        writer.Push(std::shared_ptr<size_t>(buf + i, nop_del), chunk);
    }
    writer.Close();
    writer.Wait();
    free(buf);
}

static std::vector<size_t> read_output_file(const FileInfo &fi) {
    size_t n = fi.true_size / sizeof(size_t);
    std::vector<size_t> out(n);
    int fd = open(fi.file_name.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(("open " + fi.file_name).c_str());
        return out;
    }
    size_t total = n * sizeof(size_t);
    size_t off   = 0;
    while (off < total) {
        ssize_t r = read(fd, (char *)out.data() + off, total - off);
        if (r <= 0) break;
        off += (size_t)r;
    }
    close(fd);
    return out;
}

static void cleanup(const std::vector<FileInfo> &files) {
    for (const auto &fi : files) {
        unlink(fi.file_name.c_str());
    }
}

static bool test_ones(size_t n, size_t num_files) {
    std::cout << "test_ones: n=" << n << " num_files=" << num_files << "\n";

    write_constant("sv_in", n, 1, num_files);

    auto in_files = FindFiles("sv_in");
    GetFileInfo(in_files); 

    auto out_files = Scan<size_t>(in_files, "sv_out");


    std::sort(out_files.begin(), out_files.end(),
              [](const FileInfo &a, const FileInfo &b) {
                  return a.file_index < b.file_index;
              });

    size_t expected = 0;
    size_t errors = 0;
    size_t total= 0;

    for (const auto &fi : out_files) {
        auto elems = read_output_file(fi);
        for (size_t v : elems) {
            if (v != expected) {
                if (errors < 10) {
                    std::cout << "  MISMATCH at global pos " << total
                              << ": got " << v << ", expected " << expected << "\n";
                }
                ++errors;
            }
            ++expected;
            ++total;
        }
    }

    cleanup(in_files);
    cleanup(out_files);

    if (errors == 0 && total == n) {
        std::cout << "  PASS (" << total << " elements verified)\n";
        return true;
    }
    std::cout << "  FAIL: " << errors << " mismatches, "
              << total << "/" << n << " elements read\n";
    return false;
}


static bool test_randperm(size_t power_of_two) {
    size_t n = 1UL << power_of_two;
    std::cout << "test_randperm: n=2^" << power_of_two << "=" << n << "\n";

    externalSeq<size_t> input = externalSeqOps::randPerm<size_t>("rp_in", power_of_two);


    struct QD { size_t *ptr; size_t size; size_t index; };
    auto pq_cmp = [](const QD &a, const QD &b) { return a.index > b.index; };

    std::vector<size_t> ref;
    ref.reserve(n);
    auto in_files_sorted = input.files;
    std::sort(in_files_sorted.begin(), in_files_sorted.end(),
              [](const FileInfo &a, const FileInfo &b) {
                  return a.file_index < b.file_index;
              });
    for (const auto &fi : in_files_sorted) {
        UnorderedFileReader<size_t> rdr;
        rdr.PrepFiles({fi});
        rdr.Start();
        std::priority_queue<QD, std::vector<QD>, decltype(pq_cmp)> q(pq_cmp);
        size_t next = 0;
        while (true) {
            auto [ptr, sz, _fi, idx] = rdr.Poll();
            if (ptr == nullptr) { CHECK(q.empty()); break; }
            q.push({ptr, sz, idx});
            while (!q.empty() && q.top().index == next) {
                auto top = q.top(); q.pop();
                next += top.size;
                for (size_t i = 0; i < top.size; i++) ref.push_back(top.ptr[i]);
            }
        }
    }


    std::vector<size_t> expected(ref.size());
    size_t running = 0;
    for (size_t j = 0; j < ref.size(); j++) {
        expected[j] = running;
        running += ref[j];
    }


    auto out_files = Scan<size_t>(input.files.to_vector(), "rp_out");
    std::sort(out_files.begin(), out_files.end(),
              [](const FileInfo &a, const FileInfo &b) {
                  return a.file_index < b.file_index;
              });

   
    size_t pos= 0;
    size_t errors = 0;
    for (const auto &fi : out_files) {
        auto elems = read_output_file(fi);
        for (size_t v : elems) {
            if (pos < expected.size() && v != expected[pos]) {
                if (errors < 10) {
                    std::cout << "  MISMATCH at pos " << pos
                              << ": got " << v << ", expected " << expected[pos] << "\n";
                }
                ++errors;
            }
            ++pos;
        }
    }

    cleanup(input.files.to_vector());
    cleanup(out_files);

    if (errors == 0 && pos == n) {
        std::cout << "  PASS (" << pos << " elements verified)\n";
        return true;
    }
    std::cout << "  FAIL: " << errors << " mismatches, "
              << pos << "/" << n << " elements read\n";
    return false;
}

int main() {
    
    PopulateSSDList(30, false, false);

    bool all_pass = true;

    // Small: 4 input files, 512K elements each → 2M total.
    all_pass &= test_ones(4UL * (1UL << 19), 4);

    // Larger: 4 input files, 1M elements each → 4M total.
    all_pass &= test_ones(4UL * (1UL << 20), 4);

    // Random-permutation scan: 2^20 = 1M elements.
    all_pass &= test_randperm(20);

    std::cout << (all_pass ? "\nAll tests PASSED\n" : "\nSome tests FAILED\n");
    return all_pass ? 0 : 1;
}
