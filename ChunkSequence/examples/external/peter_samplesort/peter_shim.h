// Plain-typed shim over Peter's external sample sort (peter_samplesort.h).
//
// Peter's SampleSort<T> and its scatter_gather / unordered_file_reader /
// ordered_file_writer / type_allocator headers ship their *own* copies of
// configs.h and utils/file_utils.h in this directory.  Those clash by include
// guard (SORTING_CONFIGS_H / SORTING_FILE_UTILS_H) with the main repo's
// configs.h (which additionally defines CHUNK_SIZE) and utils/file_utils.h
// (which additionally declares RaiseFdLimit / InitIoUringWithRetry), so Peter's
// sort cannot be included in the same translation unit as the chunk_seq code.
//
// This header exposes Peter's sort through *plain* types only (no Peter and no
// chunk types), so the comparison driver can call it without pulling Peter's
// headers into its own TU.  peter_shim.cpp is the sole TU that includes Peter's
// headers; it is compiled with this directory first on the include path so
// Peter's own configs.h/utils resolve, and it links against the main repo's
// file_utils.o (identical implementation — see CLAUDE.md / the byte-identical
// utils/file_utils.cpp) for FindFiles/GetFileName/GetSSDList/Read/etc.
//
// The on-disk input is generated here (not by the chunk-seq tabulate) in the
// raw per-drive layout Peter's FindFiles expects: marker-free files whose size
// is an exact O_DIRECT multiple, holding keys key_at(i)=parlay::hash64(i) for
// i in [0,n) — the identical multiset the chunk-seq side sorts, so the two
// out-of-core outputs (and the in-memory baseline) must agree exactly.

#ifndef PETER_SAMPLESORT_SHIM_H
#define PETER_SAMPLESORT_SHIM_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace peter_shim {

// Build an n-key input for Peter's sort under `prefix`, spread across the SSDs
// in Peter's per-drive file layout.  Keys are key_at(i)=parlay::hash64(i) for
// i in [0,n) — the same deterministic multiset the chunk-seq side sorts.
// Requires n % 512 == 0 (512 uint64 == one 4096-byte O_DIRECT block), so every
// file is O_DIRECT-aligned and marker-free (true_size == file_size), which is
// what Peter's Sort() assumes when it calls GetFileInfo(files, eof_marker=false).
// Returns the wall-clock build time in seconds.
double BuildInput(const std::string& prefix, std::size_t n);

// Sort the files found under `in_prefix` (via Peter's FindFiles) with
// SampleSort<uint64_t>, writing the sorted buckets under `out_prefix`.  Fills
// `result_files` with the result file names in globally-sorted (bucket) order
// and `result_true_sizes` with each file's logical byte count.  Returns the
// wall-clock sort time in seconds.
double Sort(const std::string& in_prefix, const std::string& out_prefix,
            std::vector<std::string>& result_files,
            std::vector<std::size_t>& result_true_sizes);

// Read the sorted output back into DRAM (result_true_sizes[i] bytes from
// result_files[i], concatenated in order) for the element-wise cross-check.
// Only called under the RAM budget.
std::vector<uint64_t> ReadBackSorted(const std::vector<std::string>& result_files,
                                     const std::vector<std::size_t>& result_true_sizes);

// Remove every file this shim / Peter's sort leaves on the drives for the given
// prefixes: the `in_prefix` input, the `out_prefix` output, and Peter's
// hard-coded "spfx_" intermediate buckets.
void Cleanup(const std::string& in_prefix, const std::string& out_prefix);

}  // namespace peter_shim

#endif  // PETER_SAMPLESORT_SHIM_H
