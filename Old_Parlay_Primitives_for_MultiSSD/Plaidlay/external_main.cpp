//claude-built, probably should not trust

// Build & run (from the repo root, on a filesystem that supports O_DIRECT):
//
//   g++ -std=c++17 -O2 -I . -I Plaidlay -I Plaidlay/parlaylib/include \
//       -I deps/abseil-cpp/install/include \
//       Plaidlay/external_main.cpp utils/file_utils.cpp utils/logger.cpp \
//       utils/random_number_generator.cpp -o external_main_test \
//       -Wl,--start-group deps/abseil-cpp/install/lib/*.a -Wl,--end-group \
//       -lpthread -luring
//   ./external_main_test
//
// The tests create and delete their own scratch files in the working directory.

#include "externalFilter.h"
#include "ExternalIota.h"
#include "ExternalMap.h"
#include "ExternalReduce.h"

// The legacy randPerm/map/filter/scan demo (legacyDemo/mapThroughput, defined at
// the bottom of this file) depends on externalSeq.h, which currently pulls in the
// in-progress scan.h refactor and does not compile on its own. That demo is
// reference-only and is not run by main(), so it is compiled out by default.
// Define ENABLE_EXTERNAL_SEQ_DEMO to build it once the scan refactor lands.
#ifdef ENABLE_EXTERNAL_SEQ_DEMO
#include "externalSeq.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#ifdef ENABLE_EXTERNAL_SEQ_DEMO
auto add = [](size_t a, size_t b) {return a + b;};

void mapThroughput();
void legacyDemo();
#endif

// ===========================================================================
//  Correctness and memory tests for ExternalFilter (Plaidlay/externalFilter.h)
//
//  ExternalFilter reads an input External_Sequence (a list of chunk_headers
//  describing on-disk blocks), applies a predicate to every element, and writes
//  the surviving elements back out as a new External_Sequence. Each input block
//  maps 1:1 to an output block that keeps the input block's `index`, so sorting
//  the output by `index` reproduces a stable global order.
//
//  These tests build a known input on disk, run the filter, read the result
//  back, and compare against an in-memory reference filter. A separate test
//  checks that memory does not grow with the number of batches (i.e. the
//  per-batch buffers are actually freed).
// ===========================================================================

namespace {

int g_failures = 0;

void Check(bool cond, const std::string &name) {
    if (cond) {
        std::cout << "  [PASS] " << name << std::endl;
    } else {
        std::cout << "  [FAIL] " << name << std::endl;
        g_failures++;
    }
}

// Current resident set size (KB), from /proc/self/statm.
size_t CurrentRssKb() {
    std::ifstream statm("/proc/self/statm");
    size_t total_pages = 0, resident_pages = 0;
    statm >> total_pages >> resident_pages;
    long page_kb = sysconf(_SC_PAGESIZE) / 1024;
    return resident_pages * (size_t) page_kb;
}

// Peak resident set size (KB) over the process lifetime, from VmHWM.
size_t PeakRssKb() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmHWM:") {
            size_t value = 0;
            status >> value;  // value is already in kB
            return value;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

// Build an input External_Sequence on disk: one chunk per file. Each file holds
// that chunk's bytes at offset 0, padded up to an O_DIRECT-aligned length so the
// reader's aligned read is fully satisfied. Records every created file in
// `created_files` for later cleanup.
template<typename T>
External_Sequence BuildInput(const std::string &prefix,
                             const std::vector<std::vector<T>> &chunks,
                             std::vector<std::string> &created_files) {
    External_Sequence seq(chunks.size());
    for (size_t i = 0; i < chunks.size(); i++) {
        const std::string fname = prefix + "_in_" + std::to_string(i);
        const size_t used = chunks[i].size() * sizeof(T);
        const size_t padded = AlignUp(used);

        int fd = open(fname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        CHECK(fd >= 0) << "could not create input file " << fname;
        if (used > 0) {
            ssize_t w = pwrite(fd, chunks[i].data(), used, 0);
            CHECK(w == (ssize_t) used) << "short write building input " << fname;
        }
        // Zero-pad so an O_DIRECT read of AlignUp(used) bytes stays inside the file.
        CHECK(ftruncate(fd, (off_t) padded) == 0) << "ftruncate failed for " << fname;
        close(fd);
        created_files.push_back(fname);

        chunk_header h;
        h.filename = fname;
        h.begin_address = 0;
        h.used = used;
        h.index = i;  // global order key: chunk i contributes the i-th block
        seq.ordered_underlying_sequence[i] = h;
    }
    return seq;
}

// Read an output External_Sequence back into a flat vector, in index order.
template<typename T>
std::vector<T> ReadOutput(const External_Sequence &out) {
    parlay::sequence<chunk_header> headers = out.ordered_underlying_sequence;
    std::sort(headers.begin(), headers.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });

    std::vector<T> result;
    for (const auto &h : headers) {
        const size_t n = h.used / sizeof(T);
        if (n == 0) {
            continue;  // block produced no surviving elements
        }
        std::vector<T> buf(n);
        int fd = open(h.filename.c_str(), O_RDONLY);
        CHECK(fd >= 0) << "could not open output file " << h.filename;
        ssize_t r = pread(fd, buf.data(), h.used, (off_t) h.begin_address);
        CHECK(r == (ssize_t) h.used) << "short read of output " << h.filename;
        close(fd);
        result.insert(result.end(), buf.begin(), buf.end());
    }
    return result;
}

void RemoveFiles(const std::vector<std::string> &files) {
    for (const auto &f : files) {
        unlink(f.c_str());
    }
}

// The deterministic set of output filenames for a given prefix (no side effects).
std::vector<std::string> OutputNames(const std::string &prefix) {
    std::vector<std::string> names;
    for (int i = 0; i < NUM_SSDS; i++) {
        names.push_back(prefix + "_out_" + std::to_string(i));
    }
    return names;
}

std::vector<std::string> MakeOutputNames(const std::string &prefix) {
    std::vector<std::string> names = OutputNames(prefix);
    // Start fresh so stale blocks from a previous run can't be confused for output.
    RemoveFiles(names);
    return names;
}

constexpr size_t kBufferBytes = 4u << 20;  // ExternalFilter's per-block buffer size

// Default per-chunk length: varied and deliberately not block-aligned, but well
// under the 4 MiB block buffer so each input chunk maps to a single output block.
size_t VariedLen(size_t c) { return 300 + (c * 37) % 900; }
// Many tiny chunks, to stress the per-chunk header bookkeeping.
size_t TinyLen(size_t c) { return 1 + (c * 13) % 17; }

// Generate `num_chunks` chunks of T, numbering elements sequentially across all
// chunks via `make(global_index)`. `len(chunk_index)` sets each chunk's length.
// Also produces the in-memory reference: `predicate` applied to the flattened
// sequence, in order -- which is exactly what sorting the output by `index`
// should reproduce.
template<typename T, typename MakeFn, typename PredFn, typename LenFn>
void MakeChunks(size_t num_chunks, MakeFn make, PredFn predicate, LenFn len,
                std::vector<std::vector<T>> &chunks, std::vector<T> &reference) {
    size_t counter = 0;
    for (size_t c = 0; c < num_chunks; c++) {
        const size_t n = len(c);
        std::vector<T> chunk;
        chunk.reserve(n);
        for (size_t k = 0; k < n; k++) {
            const T v = make(counter++);
            chunk.push_back(v);
            if (predicate(v)) {
                reference.push_back(v);
            }
        }
        chunks.push_back(std::move(chunk));
    }
}

// Run the filter over `chunks` with `predicate` and return the output sequence
// (headers), so callers can both read the data back and inspect structure.
template<typename T>
External_Sequence RunFilter(const std::string &prefix,
                            const std::vector<std::vector<T>> &chunks,
                            const std::function<bool(const T)> &predicate,
                            std::vector<std::string> &all_files) {
    External_Sequence in = BuildInput<T>(prefix, chunks, all_files);
    std::vector<std::string> out_names = MakeOutputNames(prefix);
    all_files.insert(all_files.end(), out_names.begin(), out_names.end());

    return ExternalFilter<T>(in, predicate, out_names);
}

// Structural invariants that must hold for any successful filter, independent of
// the predicate: one output header per input chunk, every header's byte count is
// a whole number of T's and fits in a block buffer, output filenames are drawn
// only from the provided output set, and the surviving input `index` values form
// exactly the set {0, ..., num_input_chunks - 1} (each block preserved once).
template<typename T>
void CheckOutputInvariants(const External_Sequence &out, size_t num_input_chunks,
                           const std::vector<std::string> &out_names,
                           const std::string &name) {
    const parlay::sequence<chunk_header> &hs = out.ordered_underlying_sequence;
    std::unordered_set<std::string> allowed(out_names.begin(), out_names.end());

    bool ok = (hs.size() == num_input_chunks);
    std::vector<size_t> indices;
    indices.reserve(hs.size());
    for (const chunk_header &h : hs) {
        if (h.used % sizeof(T) != 0) ok = false;       // whole elements only
        if (h.used > kBufferBytes) ok = false;          // never exceeds a block
        if (allowed.find(h.filename) == allowed.end()) ok = false;  // valid sink
        indices.push_back(h.index);
    }
    std::sort(indices.begin(), indices.end());
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] != i) ok = false;  // a permutation of 0..n-1, no dup/gap
    }
    Check(ok, name + ": output header invariants (count/index/size/filename)");
}

// Run one size_t case end-to-end and assert structure + count + stable order.
template<typename PredFn, typename LenFn>
void RunSizeCase(const std::string &name, const std::string &prefix,
                 size_t num_chunks, PredFn pred, LenFn len) {
    std::cout << name << " (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batch(es))" << std::endl;
    std::function<bool(const size_t)> predicate = pred;
    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> reference;
    MakeChunks<size_t>(num_chunks, [](size_t i) { return i; }, predicate, len,
                       chunks, reference);

    std::vector<std::string> files;
    External_Sequence out = RunFilter<size_t>(prefix, chunks, predicate, files);
    std::vector<size_t> got = ReadOutput<size_t>(out);

    CheckOutputInvariants<size_t>(out, num_chunks, OutputNames(prefix), name);
    Check(got.size() == reference.size(), name + ": element count matches reference");
    Check(got == reference, name + ": values match reference in stable index order");
    RemoveFiles(files);
}

// --- Tests -----------------------------------------------------------------

// Batch-boundary coverage: a single chunk, an under-full batch, an exactly full
// batch, one element past a batch, and a multi-batch run with a partial tail.
// The under-full / past-boundary / partial-tail cases all exercise the
// bad_flags path (Poll() returns null for slots with no pending chunk).
void TestBatchBoundaries() {
    RunSizeCase("TestSingleChunk", "eftest_one", 1,
                [](size_t x) { return x % 2 == 0; }, VariedLen);
    RunSizeCase("TestUnderFullBatch", "eftest_under", NUM_SSDS - 1,
                [](size_t x) { return x % 2 == 0; }, VariedLen);
    RunSizeCase("TestExactBatch", "eftest_exact", NUM_SSDS,
                [](size_t x) { return x % 2 == 0; }, VariedLen);
    RunSizeCase("TestOverBatch", "eftest_over", NUM_SSDS + 1,
                [](size_t x) { return x % 2 == 0; }, VariedLen);
    RunSizeCase("TestMultiBatchPartial", "eftest_multi", 2 * NUM_SSDS + 5,
                [](size_t x) { return x % 3 == 0; }, VariedLen);
}

// Predicate-shape coverage: reject all, keep all, keep a sparse subset, and an
// alternating predicate (independent of value parity).
void TestPredicateShapes() {
    // Reject everything: every output block is empty (used == 0), output empty.
    RunSizeCase("TestEmptyResult", "eftest_empty", NUM_SSDS + 3,
                [](size_t) { return false; }, VariedLen);
    // Keep everything: output must equal the full input sequence.
    RunSizeCase("TestKeepAll", "eftest_keep", NUM_SSDS + 7,
                [](size_t) { return true; }, VariedLen);
    // Sparse: only a handful survive per block, many blocks contribute nothing.
    RunSizeCase("TestSparse", "eftest_sparse", 3 * NUM_SSDS + 11,
                [](size_t x) { return x % 97 == 0; }, VariedLen);
    // Position-based predicate (keep odd-indexed elements) to avoid leaning on
    // value parity; still a deterministic function of the global index value.
    RunSizeCase("TestAlternating", "eftest_alt", NUM_SSDS + 13,
                [](size_t x) { return (x & 1u) == 1u; }, VariedLen);
}

// Many tiny chunks across multiple batches: stresses per-chunk header
// bookkeeping and the stable-ordering sort with lots of small blocks.
void TestManyTinyChunks() {
    RunSizeCase("TestManyTinyChunks", "eftest_tiny", 4 * NUM_SSDS + 9,
                [](size_t x) { return x % 4 == 0; }, TinyLen);
}

// Non-size_t element type: a 16-byte record. The predicate looks only at `key`,
// but the reference compares the whole record, so any truncation, misalignment,
// or wrong sizeof(T) handling in the read/filter/write path is caught.
struct Record {
    uint64_t key;
    uint64_t tag;
    bool operator==(const Record &o) const { return key == o.key && tag == o.tag; }
};

void TestRecordType() {
    const std::string name = "TestRecordType";
    const size_t num_chunks = 2 * NUM_SSDS + 3;
    std::cout << name << " (" << num_chunks << " chunks, 16-byte records)" << std::endl;
    std::function<bool(const Record)> pred = [](const Record r) { return r.key % 5 == 0; };
    auto make = [](size_t i) { return Record{(uint64_t) i, (uint64_t) (i * 1000003ull + 7)}; };

    std::vector<std::vector<Record>> chunks;
    std::vector<Record> reference;
    MakeChunks<Record>(num_chunks, make, pred, VariedLen, chunks, reference);

    std::vector<std::string> files;
    External_Sequence out = RunFilter<Record>("eftest_rec", chunks, pred, files);
    std::vector<Record> got = ReadOutput<Record>(out);

    CheckOutputInvariants<Record>(out, num_chunks, OutputNames("eftest_rec"), name);
    Check(got.size() == reference.size(), name + ": record count matches reference");
    Check(got == reference, name + ": full records (key+tag) match reference in order");
    RemoveFiles(files);
}

// Determinism: the same input filtered twice must yield byte-identical output,
// even though writes are spread across SSDs with a random distribution.
void TestDeterminism() {
    const std::string name = "TestDeterminism";
    const size_t num_chunks = 2 * NUM_SSDS + 4;
    std::cout << name << " (" << num_chunks << " chunks, run twice)" << std::endl;
    std::function<bool(const size_t)> pred = [](size_t x) { return x % 7 == 0; };

    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> reference;
    MakeChunks<size_t>(num_chunks, [](size_t i) { return i; }, pred, VariedLen,
                       chunks, reference);

    std::vector<std::string> files_a, files_b;
    std::vector<size_t> got_a = ReadOutput<size_t>(RunFilter<size_t>("eftest_det_a", chunks, pred, files_a));
    std::vector<size_t> got_b = ReadOutput<size_t>(RunFilter<size_t>("eftest_det_b", chunks, pred, files_b));

    Check(got_a == reference, name + ": first run matches reference");
    Check(got_a == got_b, name + ": repeated run produces identical output");
    RemoveFiles(files_a);
    RemoveFiles(files_b);
}

// Memory test: run several batches and confirm resident memory returns to near
// baseline afterwards (the per-batch 4 MiB output buffers and the reader's
// buffer pool must be freed, not leaked). A regression where buffers leak would
// retain ~NUM_SSDS * 4 MiB per batch.
void TestMemoryBounded() {
    const size_t num_chunks = 6 * NUM_SSDS;  // several full batches
    std::cout << "TestMemoryBounded (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batches)" << std::endl;
    std::function<bool(const size_t)> pred = [](size_t x) { return x % 2 == 0; };
    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> reference;
    MakeChunks<size_t>(num_chunks, [](size_t i) { return i; }, pred, VariedLen,
                       chunks, reference);

    const size_t rss_before = CurrentRssKb();
    std::vector<std::string> files;
    External_Sequence out = RunFilter<size_t>("eftest_mem", chunks, pred, files);
    std::vector<size_t> got = ReadOutput<size_t>(out);
    const size_t rss_after = CurrentRssKb();
    const size_t peak = PeakRssKb();

    CheckOutputInvariants<size_t>(out, num_chunks, OutputNames("eftest_mem"), "TestMemoryBounded");
    Check(got == reference, "memory-run output still matches reference");

    // If buffers leaked, retained memory would scale with batch count
    // (~NUM_SSDS * 4 MiB * num_batches). Allow a generous constant slack.
    const size_t retained_kb = rss_after > rss_before ? rss_after - rss_before : 0;
    const size_t leak_threshold_kb = 256 * 1024;  // 256 MiB
    std::cout << "    rss before=" << rss_before / 1024 << " MiB, after="
              << rss_after / 1024 << " MiB, retained=" << retained_kb / 1024
              << " MiB, peak=" << peak / 1024 << " MiB" << std::endl;
    Check(retained_kb < leak_threshold_kb,
          "resident memory returns near baseline after filtering (no per-batch leak)");

    RemoveFiles(files);
}

// ===========================================================================
//  Correctness and memory tests for ExternalIota (Plaidlay/ExternalIota.h)
//
//  ExternalIota(n, seq, filenames) materializes the logical sequence
//  [0, 1, ..., n-1] onto disk as fixed-size 4 MiB chunks spread across the
//  given files (one chunk per SSD slot per batch). The value at global element
//  position g lives in chunk c = g / elems_per_chunk at offset g % elems_per_chunk,
//  i.e. it is simply g. Each chunk's header carries its global chunk number in
//  `index`, so sorting the output by `index` reproduces the iota in order. The
//  final chunk is written as a full 4 MiB buffer but its header's `used` counts
//  only the in-range elements; the trailing values past element n-1 are padding.
//
//  IMPORTANT CONTRACT: ExternalIota does NOT resize `seq` (the resize is
//  currently commented out, ExternalIota.h:170). It writes headers into slots
//  [0, num_chunks) and assumes the caller pre-sized `seq` to exactly
//  num_chunks = ceil(n / elems_per_chunk). A larger `seq` would leave trailing
//  default headers (index 0) that corrupt the final sort, so RunIota constructs
//  `seq` at exactly that size (see IotaChunkCount).
//
//  These tests run ExternalIota, then read every chunk's `used` bytes back from
//  disk at its recorded offset and confirm the reconstruction equals the iota,
//  plus structural invariants on the headers and a no-leak memory check.
// ===========================================================================

// 16-byte element type to exercise a non-size_t T. ExternalIota fills each slot
// with `buffer[i][k] = begin_val + k`, so T must be constructible from size_t;
// the second word is derived from the value so the full sizeof(T) bytes are
// checked for round-trip (not just the low 8).
struct Iota16 {
    uint64_t v;
    uint64_t tag;
    Iota16() = default;
    Iota16(size_t x) : v((uint64_t) x), tag((uint64_t) x * 2654435761ull + 1) {}  // implicit
    bool operator==(const Iota16 &o) const { return v == o.v && tag == o.tag; }
};

// Elements per chunk, mirroring ExternalIota's buffer_size = (4<<20)/sizeof(T).
template<typename T>
constexpr size_t IotaElemsPerChunk() { return (4u << 20) / sizeof(T); }

// Number of chunks ExternalIota will produce for n elements (ceil division).
template<typename T>
size_t IotaChunkCount(size_t n) {
    const size_t per = IotaElemsPerChunk<T>();
    return (n + per - 1) / per;
}

// Run ExternalIota for n elements into a fresh set of output files, sizing the
// input/output sequence to exactly num_chunks per the contract above.
template<typename T>
External_Sequence RunIota(const std::string &prefix, size_t n,
                          std::vector<std::string> &out_files) {
    std::vector<std::string> names = MakeOutputNames(prefix);  // fresh NUM_SSDS files
    out_files.insert(out_files.end(), names.begin(), names.end());
    External_Sequence seq(IotaChunkCount<T>(n));
    return ExternalIota<T>(n, seq, names);
}

// Structural invariants for any successful iota, independent of the random SSD
// placement: exactly num_chunks headers; indices are a permutation of
// {0, ..., num_chunks-1}; each header's `used` is a whole number of T's equal to
// that chunk's in-range element count (so the last chunk is partial when n is not
// a multiple of elems_per_chunk); `used` never exceeds a 4 MiB block; filenames
// come only from the provided set; offsets are O_DIRECT aligned; and no two
// chunks' 4 MiB on-disk footprints overlap within a file.
template<typename T>
void CheckIotaInvariants(const External_Sequence &out, size_t n,
                         const std::vector<std::string> &out_names,
                         const std::string &name) {
    const size_t per = IotaElemsPerChunk<T>();
    const size_t num_chunks = IotaChunkCount<T>(n);
    constexpr size_t kBlock = 4u << 20;

    parlay::sequence<chunk_header> hs = out.ordered_underlying_sequence;
    std::sort(hs.begin(), hs.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });
    std::unordered_set<std::string> allowed(out_names.begin(), out_names.end());

    bool ok = (hs.size() == num_chunks);
    size_t total_elems = 0;
    std::vector<std::pair<std::string, size_t>> regions;  // (filename, begin_address)
    for (size_t i = 0; i < hs.size(); i++) {
        const chunk_header &h = hs[i];
        if (h.index != i) ok = false;                                  // permutation 0..n-1
        if (allowed.find(h.filename) == allowed.end()) ok = false;     // valid sink
        if (h.used % sizeof(T) != 0) ok = false;                       // whole elements
        if (h.used == 0) ok = false;                                   // every chunk has data
        if (h.used > kBlock) ok = false;                               // within one block
        if (h.begin_address % O_DIRECT_MEMORY_ALIGNMENT != 0) ok = false;
        const size_t expect_valid = std::min(per, n - i * per);        // last chunk partial
        if (h.used != expect_valid * sizeof(T)) ok = false;
        total_elems += h.used / sizeof(T);
        regions.push_back({h.filename, h.begin_address});
    }
    if (total_elems != n) ok = false;                                  // exactly n elements
    std::sort(regions.begin(), regions.end());
    for (size_t i = 1; i < regions.size(); i++) {
        if (regions[i].first == regions[i - 1].first &&
            regions[i].second < regions[i - 1].second + kBlock) {
            ok = false;                                                // overlapping footprints
        }
    }
    Check(ok, name + ": iota header invariants (index/used/size/filename/offset/no-overlap)");
}

// Read each chunk's `used` bytes back from disk at its recorded offset and verify
// every element equals make(global_index). Streams one chunk at a time so host
// memory stays bounded. Returns true iff all values match; sets total_out to the
// number of elements seen.
template<typename T, typename MakeFn>
bool VerifyIotaData(const External_Sequence &out, MakeFn make, size_t &total_out) {
    const size_t per = IotaElemsPerChunk<T>();
    parlay::sequence<chunk_header> hs = out.ordered_underlying_sequence;
    std::sort(hs.begin(), hs.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });

    bool ok = true;
    size_t total = 0;
    for (const chunk_header &h : hs) {
        const size_t cnt = h.used / sizeof(T);
        if (cnt == 0) continue;
        std::vector<T> buf(cnt);
        int fd = open(h.filename.c_str(), O_RDONLY);
        CHECK(fd >= 0) << "could not open iota output " << h.filename;
        ssize_t r = pread(fd, buf.data(), h.used, (off_t) h.begin_address);
        CHECK(r == (ssize_t) h.used) << "short read of iota output " << h.filename;
        close(fd);
        const size_t base = h.index * per;
        for (size_t j = 0; j < cnt; j++) {
            if (!(buf[j] == make(base + j))) { ok = false; break; }
        }
        total += cnt;
    }
    total_out = total;
    return ok;
}

// Run one iota case end-to-end: structure + element count + values-in-order.
template<typename T, typename MakeFn>
void RunIotaCase(const std::string &name, const std::string &prefix,
                 size_t n, MakeFn make) {
    const size_t num_chunks = IotaChunkCount<T>(n);
    std::cout << name << " (n=" << n << ", " << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batch(es))" << std::endl;

    std::vector<std::string> files;
    External_Sequence out = RunIota<T>(prefix, n, files);

    CheckIotaInvariants<T>(out, n, OutputNames(prefix), name);
    size_t total = 0;
    bool data_ok = VerifyIotaData<T>(out, make, total);
    Check(total == n, name + ": element count == n");
    Check(data_ok && total == n, name + ": values equal iota in stable index order");
    RemoveFiles(files);
}

// --- Tests -----------------------------------------------------------------

// Batch-boundary coverage: a single element, a single full chunk, an under-full
// batch, an exactly full batch, one chunk past a batch, and a multi-batch run.
// The non-multiple sizes exercise the partial-tail path; the >NUM_SSDS sizes
// exercise the per-slot bad_flags path (slots with no chunk in the final batch).
void TestIotaBatchBoundaries() {
    const size_t per = IotaElemsPerChunk<size_t>();
    auto idv = [](size_t g) { return (size_t) g; };
    auto N = [per](size_t chunks, size_t tail) { return (chunks - 1) * per + tail; };

    RunIotaCase<size_t>("TestIotaSingleElement",  "eitest_one",   1,                            idv);
    RunIotaCase<size_t>("TestIotaSingleFullChunk","eitest_full1", per,                          idv);
    RunIotaCase<size_t>("TestIotaUnderFullBatch", "eitest_under", N(NUM_SSDS - 1, per / 3 + 1), idv);
    RunIotaCase<size_t>("TestIotaExactBatch",     "eitest_exact", N(NUM_SSDS,     per),         idv);
    RunIotaCase<size_t>("TestIotaOverBatch",      "eitest_over",  N(NUM_SSDS + 1, 7),           idv);
    RunIotaCase<size_t>("TestIotaMultiBatch",     "eitest_multi", N(2 * NUM_SSDS + 5, per / 2 - 3), idv);
}

// Final-chunk accounting: the header's `used` must reflect only in-range
// elements, never the full 4 MiB buffer. A one-element tail (most padding) and
// an exact-multiple tail (no padding) bracket the boundary.
void TestIotaPartialTail() {
    const size_t per = IotaElemsPerChunk<size_t>();
    auto idv = [](size_t g) { return (size_t) g; };
    auto N = [per](size_t chunks, size_t tail) { return (chunks - 1) * per + tail; };

    RunIotaCase<size_t>("TestIotaTailOneElement", "eitest_tail1", N(NUM_SSDS + 2, 1),   idv);
    RunIotaCase<size_t>("TestIotaTailExact",      "eitest_taile", N(NUM_SSDS + 2, per), idv);
}

// Non-size_t element type: a 16-byte record whose second word is derived from
// the value, catching truncation, misalignment, or wrong sizeof(T) handling.
void TestIotaRecordType() {
    const size_t per = IotaElemsPerChunk<Iota16>();
    auto mk = [](size_t g) { return Iota16(g); };
    const size_t n = (2 * NUM_SSDS + 3 - 1) * per + per / 4 + 5;  // multi-batch, partial tail
    RunIotaCase<Iota16>("TestIotaRecordType", "eitest_rec", n, mk);
}

// Determinism: although chunks are placed on random SSDs, the logical
// reconstruction and the per-index `used` accounting (a function of n alone)
// must be identical across runs.
void TestIotaDeterminism() {
    const size_t per = IotaElemsPerChunk<size_t>();
    const size_t n = (2 * NUM_SSDS + 4 - 1) * per + per / 2;  // multi-batch, partial tail
    auto idv = [](size_t g) { return (size_t) g; };
    std::cout << "TestIotaDeterminism (n=" << n << ", run twice)" << std::endl;

    std::vector<std::string> files_a, files_b;
    External_Sequence a = RunIota<size_t>("eitest_det_a", n, files_a);
    External_Sequence b = RunIota<size_t>("eitest_det_b", n, files_b);

    size_t ta = 0, tb = 0;
    bool a_ok = VerifyIotaData<size_t>(a, idv, ta);
    bool b_ok = VerifyIotaData<size_t>(b, idv, tb);
    Check(a_ok && ta == n, "TestIotaDeterminism: first run reconstructs iota");
    Check(b_ok && tb == n, "TestIotaDeterminism: second run reconstructs iota");

    parlay::sequence<chunk_header> ha = a.ordered_underlying_sequence;
    parlay::sequence<chunk_header> hb = b.ordered_underlying_sequence;
    auto by_index = [](const chunk_header &x, const chunk_header &y) { return x.index < y.index; };
    std::sort(ha.begin(), ha.end(), by_index);
    std::sort(hb.begin(), hb.end(), by_index);
    bool same = (ha.size() == hb.size());
    for (size_t i = 0; same && i < ha.size(); i++) {
        if (ha[i].index != hb[i].index || ha[i].used != hb[i].used) same = false;
    }
    Check(same, "TestIotaDeterminism: per-index chunk sizes identical across runs");

    RemoveFiles(files_a);
    RemoveFiles(files_b);
}

// Memory test: several full batches, confirming resident memory returns near
// baseline afterwards. A regression where the per-batch NUM_SSDS * 4 MiB output
// buffers leak would retain memory scaling with the batch count.
void TestIotaMemoryBounded() {
    const size_t per = IotaElemsPerChunk<size_t>();
    const size_t num_chunks = 4 * NUM_SSDS;  // four full batches
    const size_t n = num_chunks * per;
    auto idv = [](size_t g) { return (size_t) g; };
    std::cout << "TestIotaMemoryBounded (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batches)" << std::endl;

    const size_t rss_before = CurrentRssKb();
    std::vector<std::string> files;
    External_Sequence out = RunIota<size_t>("eitest_mem", n, files);
    const size_t rss_after = CurrentRssKb();
    const size_t peak = PeakRssKb();

    CheckIotaInvariants<size_t>(out, n, OutputNames("eitest_mem"), "TestIotaMemoryBounded");
    size_t total = 0;
    bool data_ok = VerifyIotaData<size_t>(out, idv, total);
    Check(data_ok && total == n, "TestIotaMemoryBounded: output still matches iota");

    const size_t retained_kb = rss_after > rss_before ? rss_after - rss_before : 0;
    const size_t leak_threshold_kb = 256 * 1024;  // 256 MiB
    std::cout << "    rss before=" << rss_before / 1024 << " MiB, after="
              << rss_after / 1024 << " MiB, retained=" << retained_kb / 1024
              << " MiB, peak=" << peak / 1024 << " MiB" << std::endl;
    Check(retained_kb < leak_threshold_kb,
          "TestIotaMemoryBounded: resident memory returns near baseline (no per-batch leak)");

    RemoveFiles(files);
}

// ===========================================================================
//  Correctness tests for ExternalMap (Plaidlay/ExternalMap.h)
//
//  ExternalMap reads an input External_Sequence, applies `f` to every element,
//  and writes the results back out. It is element-for-element, so each input
//  chunk maps to one output block when sizeof(R) <= sizeof(T), and fans out into
//  several output blocks when sizeof(R) > sizeof(T). Sorting the output by
//  `index` reproduces a stable global order; indices are re-densified to
//  0..M-1 after the fan-out.
// ===========================================================================

// Structural invariants for any successful map: whole-element byte counts that
// never exceed a block, output filenames drawn only from the provided set,
// indices forming exactly {0, ..., M-1}, and a total element count equal to the
// input's (map is element-for-element).
template<typename R>
void CheckMapInvariants(const External_Sequence &out, size_t total_elems,
                        const std::vector<std::string> &out_names,
                        const std::string &name) {
    const parlay::sequence<chunk_header> &hs = out.ordered_underlying_sequence;
    std::unordered_set<std::string> allowed(out_names.begin(), out_names.end());

    bool ok = true;
    size_t sum_elems = 0;
    std::vector<size_t> indices;
    indices.reserve(hs.size());
    for (const chunk_header &h : hs) {
        if (h.used % sizeof(R) != 0) ok = false;        // whole elements only
        if (h.used > kBufferBytes) ok = false;           // never exceeds a block
        if (allowed.find(h.filename) == allowed.end()) ok = false;  // valid sink
        sum_elems += h.used / sizeof(R);
        indices.push_back(h.index);
    }
    std::sort(indices.begin(), indices.end());
    for (size_t i = 0; i < indices.size(); i++) {
        if (indices[i] != i) ok = false;  // a permutation of 0..M-1, no dup/gap
    }
    if (sum_elems != total_elems) ok = false;
    Check(ok, name + ": output header invariants (count/index/size/filename)");
}

// Build `num_chunks` chunks of T (numbered sequentially via make(global_index)),
// run ExternalMap<T,R> with `f`, read the result back, and compare against the
// in-memory reference f(make(0)), f(make(1)), ... in order. `len(chunk_index)`
// sets each chunk's length.
template<typename T, typename R, typename MakeFn, typename LenFn, typename MapFn>
void RunMapCase(const std::string &name, const std::string &prefix,
                size_t num_chunks, MakeFn make, LenFn len, MapFn f) {
    std::cout << name << " (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batch(es))" << std::endl;

    std::vector<std::vector<T>> chunks;
    std::vector<R> reference;
    size_t counter = 0;
    for (size_t c = 0; c < num_chunks; c++) {
        const size_t n = len(c);
        std::vector<T> chunk;
        chunk.reserve(n);
        for (size_t k = 0; k < n; k++) {
            const T v = make(counter++);
            chunk.push_back(v);
            reference.push_back(f(v));
        }
        chunks.push_back(std::move(chunk));
    }

    std::vector<std::string> files;
    External_Sequence in = BuildInput<T>(prefix, chunks, files);
    std::vector<std::string> out_names = MakeOutputNames(prefix);
    files.insert(files.end(), out_names.begin(), out_names.end());

    External_Sequence out = ExternalMap<T, R>(in, f, out_names);
    std::vector<R> got = ReadOutput<R>(out);

    CheckMapInvariants<R>(out, reference.size(), OutputNames(prefix), name);
    Check(got.size() == reference.size(), name + ": element count matches reference");
    Check(got == reference, name + ": values match reference in stable index order");
    RemoveFiles(files);
}

// --- Map tests -------------------------------------------------------------

// Batch-boundary coverage (same-type map, FANOUT == 1): single chunk, under-full
// batch, exact batch, one past a batch, and a multi-batch partial tail.
void TestMapBatchBoundaries() {
    auto make = [](size_t i) { return (size_t) i; };
    auto f = [](size_t x) { return x * 2 + 1; };
    RunMapCase<size_t, size_t>("TestMapSingleChunk",  "emtest_one",   1,                make, VariedLen, f);
    RunMapCase<size_t, size_t>("TestMapUnderFull",    "emtest_under", NUM_SSDS - 1,     make, VariedLen, f);
    RunMapCase<size_t, size_t>("TestMapExactBatch",   "emtest_exact", NUM_SSDS,         make, VariedLen, f);
    RunMapCase<size_t, size_t>("TestMapOverBatch",    "emtest_over",  NUM_SSDS + 1,     make, VariedLen, f);
    RunMapCase<size_t, size_t>("TestMapMultiBatch",   "emtest_multi", 2 * NUM_SSDS + 5, make, VariedLen, f);
}

// Many tiny chunks: stresses per-chunk header bookkeeping and the stable sort.
void TestMapManyTinyChunks() {
    auto make = [](size_t i) { return (size_t) i; };
    auto f = [](size_t x) { return x ^ 0xABCDull; };
    RunMapCase<size_t, size_t>("TestMapManyTinyChunks", "emtest_tiny", 4 * NUM_SSDS + 9,
                               make, TinyLen, f);
}

// Shrinking type change (size_t -> int, sizeof(R) < sizeof(T), FANOUT == 1):
// confirms type-changing output is written and read back correctly.
void TestMapShrinkType() {
    auto make = [](size_t i) { return (size_t) i; };
    auto f = [](size_t x) { return (int) (x % 100000); };
    RunMapCase<size_t, int>("TestMapShrinkType", "emtest_shrink", 2 * NUM_SSDS + 3,
                            make, VariedLen, f);
}

// 16-byte record map (FANOUT == 1): the whole record is compared, catching
// truncation or misalignment in the read/map/write path.
void TestMapRecordType() {
    auto make = [](size_t i) { return Record{(uint64_t) i, (uint64_t) (i * 1000003ull + 7)}; };
    auto f = [](Record r) { return Record{r.key + 1, r.tag ^ 0xFFFFull}; };
    RunMapCase<Record, Record>("TestMapRecordType", "emtest_rec", 2 * NUM_SSDS + 3,
                               make, VariedLen, f);
}

// Growing type change with fan-out (uint8_t -> uint64_t, FANOUT == 8). One large
// chunk (> out_cap = 4 MiB / 8 = 512 Ki elements) must split into several output
// blocks, exercising the fan-out split and the index re-densification. The
// output chunk count must exceed the input chunk count.
void TestMapGrowFanout() {
    const std::string name = "TestMapGrowFanout";
    const size_t out_cap = (4u << 20) / sizeof(uint64_t);   // 512 Ki
    const size_t big = out_cap + out_cap / 2 + 17;          // ~1.5 output blocks
    std::cout << name << " (1 chunk of " << big << " uint8 -> uint64)" << std::endl;

    auto make = [](size_t i) { return (uint8_t) (i * 31 + 7); };
    auto f = [](uint8_t v) { return (uint64_t) v * 2654435761ull + 12345ull; };

    std::vector<std::vector<uint8_t>> chunks(1);
    std::vector<uint64_t> reference;
    reference.reserve(big);
    for (size_t k = 0; k < big; k++) {
        const uint8_t v = make(k);
        chunks[0].push_back(v);
        reference.push_back(f(v));
    }

    std::vector<std::string> files;
    External_Sequence in = BuildInput<uint8_t>("emtest_grow", chunks, files);
    std::vector<std::string> out_names = MakeOutputNames("emtest_grow");
    files.insert(files.end(), out_names.begin(), out_names.end());

    External_Sequence out = ExternalMap<uint8_t, uint64_t>(in, f, out_names);
    std::vector<uint64_t> got = ReadOutput<uint64_t>(out);

    CheckMapInvariants<uint64_t>(out, reference.size(), OutputNames("emtest_grow"), name);
    Check(out.ordered_underlying_sequence.size() > chunks.size(),
          name + ": one input chunk fanned out into multiple output blocks");
    Check(got.size() == reference.size(), name + ": element count matches reference");
    Check(got == reference, name + ": values match reference in stable index order");
    RemoveFiles(files);
}

// Memory test: several full batches, confirming resident memory returns near
// baseline. A regression where the per-batch output buffers leak would retain
// memory scaling with the batch count. Verified by streaming (one block at a
// time) to avoid the test itself holding a huge reference.
void TestMapMemoryBounded() {
    const size_t per = (4u << 20) / sizeof(size_t);  // full chunk = 512 Ki elements
    const size_t num_chunks = 4 * NUM_SSDS;          // four full batches
    auto make = [](size_t i) { return (size_t) i; };
    auto f = [](size_t x) { return x * 3 + 1; };
    std::cout << "TestMapMemoryBounded (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batches)" << std::endl;

    std::vector<std::vector<size_t>> chunks(num_chunks);
    size_t counter = 0;
    for (size_t c = 0; c < num_chunks; c++) {
        chunks[c].reserve(per);
        for (size_t k = 0; k < per; k++) chunks[c].push_back(make(counter++));
    }

    std::vector<std::string> files;
    External_Sequence in = BuildInput<size_t>("emtest_mem", chunks, files);
    std::vector<std::string> out_names = MakeOutputNames("emtest_mem");
    files.insert(files.end(), out_names.begin(), out_names.end());
    chunks.clear();
    chunks.shrink_to_fit();

    const size_t rss_before = CurrentRssKb();
    External_Sequence out = ExternalMap<size_t, size_t>(in, f, out_names);
    const size_t rss_after = CurrentRssKb();
    const size_t peak = PeakRssKb();

    CheckMapInvariants<size_t>(out, num_chunks * per, OutputNames("emtest_mem"),
                               "TestMapMemoryBounded");

    // Stream the output in index order and check each element equals f(make(g)).
    parlay::sequence<chunk_header> hs = out.ordered_underlying_sequence;
    std::sort(hs.begin(), hs.end(),
              [](const chunk_header &a, const chunk_header &b) { return a.index < b.index; });
    size_t g = 0;
    bool data_ok = true;
    for (const chunk_header &h : hs) {
        const size_t n = h.used / sizeof(size_t);
        if (n == 0) continue;
        std::vector<size_t> buf(n);
        int fd = open(h.filename.c_str(), O_RDONLY);
        CHECK(fd >= 0) << "could not open output file " << h.filename;
        ssize_t r = pread(fd, buf.data(), h.used, (off_t) h.begin_address);
        CHECK(r == (ssize_t) h.used) << "short read of output " << h.filename;
        close(fd);
        for (size_t k = 0; k < n; k++) {
            if (buf[k] != f(make(g))) data_ok = false;
            g++;
        }
    }
    Check(data_ok && g == num_chunks * per, "TestMapMemoryBounded: output still matches map");

    const size_t retained_kb = rss_after > rss_before ? rss_after - rss_before : 0;
    const size_t leak_threshold_kb = 256 * 1024;  // 256 MiB
    std::cout << "    rss before=" << rss_before / 1024 << " MiB, after="
              << rss_after / 1024 << " MiB, retained=" << retained_kb / 1024
              << " MiB, peak=" << peak / 1024 << " MiB" << std::endl;
    Check(retained_kb < leak_threshold_kb,
          "TestMapMemoryBounded: resident memory returns near baseline (no per-batch leak)");

    RemoveFiles(files);
}

// ===========================================================================
//  Correctness tests for ExternalReduce (Plaidlay/ExternalReduce.h)
//
//  ExternalReduce streams every element of an External_Sequence through an
//  associative `op` (wrapped in a parlay::monoid) and returns the single folded
//  value. T is the on-disk element type; R the accumulator type (may differ).
// ===========================================================================

// Build `num_chunks` chunks of T, run ExternalReduce<T,R> with op/identity, and
// compare against an in-memory fold of the same elements. `op` must be
// associative (the tests use commutative ops too, so build order is irrelevant).
template<typename T, typename R, typename MakeFn, typename LenFn, typename Op>
void RunReduceCase(const std::string &name, const std::string &prefix,
                   size_t num_chunks, MakeFn make, LenFn len, Op op, R identity) {
    std::cout << name << " (" << num_chunks << " chunks, "
              << (num_chunks + NUM_SSDS - 1) / NUM_SSDS << " batch(es))" << std::endl;

    std::vector<std::vector<T>> chunks;
    R reference = identity;
    size_t counter = 0;
    for (size_t c = 0; c < num_chunks; c++) {
        const size_t n = len(c);
        std::vector<T> chunk;
        chunk.reserve(n);
        for (size_t k = 0; k < n; k++) {
            const T v = make(counter++);
            chunk.push_back(v);
            reference = op(reference, v);
        }
        chunks.push_back(std::move(chunk));
    }

    std::vector<std::string> files;
    External_Sequence in = BuildInput<T>(prefix, chunks, files);
    R got = ExternalReduce<T, R>(in, op, identity);
    Check(got == reference, name + ": reduced value matches in-memory fold");
    RemoveFiles(files);
}

// --- Reduce tests ----------------------------------------------------------

// Sum over size_t across several batches with varied (non-block-aligned) chunk
// lengths, including the partial-tail batch.
void TestReduceSum() {
    auto make = [](size_t i) { return (size_t) i; };
    auto add = [](size_t a, size_t b) { return a + b; };
    RunReduceCase<size_t, size_t>("TestReduceSum", "ertest_sum", 2 * NUM_SSDS + 5,
                                  make, VariedLen, add, (size_t) 0);
}

// Max over size_t: a different associative op to confirm identity handling.
void TestReduceMax() {
    auto make = [](size_t i) { return (size_t) ((i * 2654435761ull) & 0xFFFFFFFFull); };
    auto mx = [](size_t a, size_t b) { return std::max(a, b); };
    RunReduceCase<size_t, size_t>("TestReduceMax", "ertest_max", NUM_SSDS + 7,
                                  make, VariedLen, mx, (size_t) 0);
}

// Type-changing reduce: sum 32-bit int elements into a 64-bit accumulator. The
// op accepts (size_t, int) for the streaming fold and (size_t, size_t) for the
// final fold over per-worker partials.
void TestReduceTypeChange() {
    auto make = [](size_t i) { return (int) (i % 50000); };
    auto add = [](size_t a, auto b) { return a + (size_t) b; };
    RunReduceCase<int, size_t>("TestReduceTypeChange", "ertest_typ", 2 * NUM_SSDS + 4,
                               make, VariedLen, add, (size_t) 0);
}

// All-empty chunks must reduce to the identity (exercises the zero-length chunk
// path in the reader and the size == 0 loop in ExternalReduce).
void TestReduceEmpty() {
    auto make = [](size_t i) { return (size_t) i; };
    auto add = [](size_t a, size_t b) { return a + b; };
    RunReduceCase<size_t, size_t>("TestReduceEmpty", "ertest_empty", NUM_SSDS + 3,
                                  make, [](size_t) { return (size_t) 0; }, add, (size_t) 0);
}

}  // namespace

int main() {
    std::cout << "=== ExternalFilter tests ===" << std::endl;

    TestBatchBoundaries();
    TestPredicateShapes();
    TestManyTinyChunks();
    TestRecordType();
    TestDeterminism();
    TestMemoryBounded();

    std::cout << "=== ExternalIota tests ===" << std::endl;

    TestIotaBatchBoundaries();
    TestIotaPartialTail();
    TestIotaRecordType();
    TestIotaDeterminism();
    TestIotaMemoryBounded();

    std::cout << "=== ExternalMap tests ===" << std::endl;

    TestMapBatchBoundaries();
    TestMapManyTinyChunks();
    TestMapShrinkType();
    TestMapRecordType();
    TestMapGrowFanout();
    TestMapMemoryBounded();

    std::cout << "=== ExternalReduce tests ===" << std::endl;

    TestReduceSum();
    TestReduceMax();
    TestReduceTypeChange();
    TestReduceEmpty();

    std::cout << "============================" << std::endl;
    if (g_failures == 0) {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }
    std::cout << g_failures << " check(s) failed." << std::endl;
    return 1;
}

#ifdef ENABLE_EXTERNAL_SEQ_DEMO
// Original randPerm/map/filter/scan demo, kept for reference. Not run by main().
void legacyDemo() {
    externalSeq<size_t> nums = externalSeqOps::randPerm<size_t>("nums", 24);
    std::cout << externalSeqOps::reduce<>(nums, add, (size_t)0) << std::endl;
    externalSeq<size_t> halved = externalSeqOps::map<size_t, size_t>(nums, "halved", [](size_t x) { return x / 2; });
    std::cout << externalSeqOps::reduce<>(halved, add, (size_t)0) << std::endl;
    externalSeq<size_t> modTen = externalSeqOps::filter<>(nums, "modTen", [](size_t a) {return a % 10 == 0;});
    std::cout << externalSeqOps::reduce<>(modTen, add, (size_t)0) << std::endl;
}

void mapThroughput() {
    parlay::internal::timer timer("Map");
    timer.next("Start prep");
    externalSeq<size_t> nums = externalSeqOps::randPerm<size_t>("nums", 24);
    timer.next("Start map");
    auto result = externalSeqOps::reduce<>(nums, add, (size_t)0);
    double time = timer.next_time();
    double throughput = GetThroughput(nums.files.to_vector(), time);
    std::cout << "throughput is " << throughput << " GB per sec" <<  std::endl;
}
#endif  // ENABLE_EXTERNAL_SEQ_DEMO
