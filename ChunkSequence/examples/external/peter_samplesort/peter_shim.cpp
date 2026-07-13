// Implementation of the plain-typed shim over Peter's external sample sort.
// See peter_shim.h for why this is the *only* TU that includes Peter's headers.
//
// Compiled with this directory first on the include path so Peter's own
// configs.h / utils/*.h resolve here (not the main repo's clashing copies), and
// linked against the main repo's file_utils.o (byte-identical implementation)
// for FindFiles / GetFileName / GetSSDList / Read / etc.

#include "peter_shim.h"

#include <chrono>
#include <cstdio>
#include <fcntl.h>
#include <functional>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/utilities.h"

#include "absl/log/check.h"

#include "configs.h"            // Peter's copy (SSD layout, O_DIRECT_MULTIPLE)
#include "utils/file_utils.h"   // FindFiles/GetFileName/GetSSDList (main repo's .o)
#include "peter_samplesort.h"   // SampleSort<T>

namespace peter_shim {

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Same deterministic, duplicate-free key the chunk-seq side uses, so both
// substrates (and the in-memory baseline) hold the identical multiset.
static uint64_t key_at(std::size_t i) { return parlay::hash64(i); }

double BuildInput(const std::string& prefix, std::size_t n) {
    // 512 uint64 == one 4096-byte O_DIRECT block; a whole-block multiple keeps
    // every file O_DIRECT-aligned and marker-free (true_size == file_size),
    // which is what Peter's Sort() assumes (GetFileInfo eof_marker=false).
    CHECK(n % 512 == 0)
        << "peter_shim::BuildInput needs n % 512 == 0 for O_DIRECT alignment; got " << n;
    const auto t0 = Clock::now();

    const std::size_t total_blocks = n / 512;
    const std::size_t num_drives = GetSSDList().size();
    // One file per drive, but never more files than blocks (tiny n).
    const std::size_t num_files = std::max<std::size_t>(1, std::min(num_drives, total_blocks));

    // Per-file element counts (each a 512-multiple) and their starting global
    // key index, so file f holds keys key_at(start_f .. start_f+count_f).
    std::vector<std::size_t> count(num_files), start(num_files);
    std::size_t running = 0;
    for (std::size_t f = 0; f < num_files; f++) {
        const std::size_t blocks_f = total_blocks / num_files + (f < total_blocks % num_files ? 1 : 0);
        count[f] = blocks_f * 512;
        start[f] = running;
        running += count[f];
    }
    CHECK(running == n);

    parlay::parallel_for(0, num_files, [&](std::size_t f) {
        const std::string name = GetFileName(prefix, f);
        int fd = open(name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        // Stream the file in bounded buffers rather than materializing it whole
        // (files can be many GiB at bench scale).
        constexpr std::size_t kBufElems = (4u << 20) / sizeof(uint64_t);  // 4 MiB
        std::vector<uint64_t> buf(std::min(kBufElems, count[f]));
        std::size_t written = 0;
        while (written < count[f]) {
            const std::size_t m = std::min(buf.size(), count[f] - written);
            const std::size_t base = start[f] + written;
            for (std::size_t j = 0; j < m; j++) buf[j] = key_at(base + j);
            Write(fd, buf.data(), m * sizeof(uint64_t));
            written += m;
        }
        SYSCALL(close(fd));
    }, 1);

    return elapsed(t0);
}

double Sort(const std::string& in_prefix, const std::string& out_prefix,
            std::vector<std::string>& result_files,
            std::vector<std::size_t>& result_true_sizes) {
    std::vector<FileInfo> input = FindFiles(in_prefix);
    CHECK(!input.empty()) << "peter_shim::Sort found no input files for prefix " << in_prefix;

    const auto t0 = Clock::now();
    SampleSort<uint64_t> sorter;
    std::vector<FileInfo> results =
        sorter.Sort(input, out_prefix, std::less<uint64_t>());
    const double sort_s = elapsed(t0);

    result_files.clear();
    result_true_sizes.clear();
    result_files.reserve(results.size());
    result_true_sizes.reserve(results.size());
    for (const auto& fi : results) {
        result_files.push_back(fi.file_name);
        // true_size, never file_size: every bucket file ends with a padding block
        // (OrderedFileWriter always writes one, to carry the end marker), and an
        // *empty* bucket -- which his sample can produce, since GetPivots draws a
        // garbage pivot for <= 3 pivots -- consists of nothing else.  Falling back
        // to file_size there would read that padding back as 512 garbage keys.
        // CleanUp fills true_size for every bucket and WorkerOnlyPhase2 carries it
        // onto the result, so it is always set.
        result_true_sizes.push_back(fi.true_size);
    }
    return sort_s;
}

std::vector<uint64_t> ReadBackSorted(const std::vector<std::string>& result_files,
                                     const std::vector<std::size_t>& result_true_sizes) {
    std::vector<uint64_t> out;
    std::size_t total = 0;
    for (std::size_t s : result_true_sizes) total += s / sizeof(uint64_t);
    out.reserve(total);
    std::vector<uint64_t> buf;
    for (std::size_t i = 0; i < result_files.size(); i++) {
        const std::size_t elems = result_true_sizes[i] / sizeof(uint64_t);
        if (elems == 0) continue;
        // Buffered (non-O_DIRECT) read of the logical bytes only.
        int fd = open(result_files[i].c_str(), O_RDONLY);
        SYSCALL(fd);
        buf.resize(elems);
        std::size_t got = 0;
        while (got < result_true_sizes[i]) {
            ssize_t r = read(fd, (char*)buf.data() + got, result_true_sizes[i] - got);
            SYSCALL(r);
            if (r == 0) break;
            got += (std::size_t)r;
        }
        SYSCALL(close(fd));
        out.insert(out.end(), buf.begin(), buf.end());
    }
    return out;
}

static void unlink_prefix(const std::string& prefix) {
    for (const auto& fi : FindFiles(prefix)) unlink(fi.file_name.c_str());
}

void Cleanup(const std::string& in_prefix, const std::string& out_prefix) {
    unlink_prefix(in_prefix);
    unlink_prefix(out_prefix);
    unlink_prefix("spfx_");  // Peter's hard-coded intermediate-bucket prefix
}

}  // namespace peter_shim
