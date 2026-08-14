//
// io_backend.h -- pluggable storage backend for chunk_seq data.
//
// Every "file" this library creates lives under one of the GetSSDList() roots
// and is named by GetFileName(prefix, drive).  This header lets such a file be
// backed by a heap arena instead of a real O_DIRECT file, so the whole library
// (and its test suite) can run with no /mnt setup at all.
//
// chunk's {filename, begin_addr, used, index} metadata is unchanged: `filename`
// simply resolves to an arena rather than a path.
//
// ROUTING IS PER PATH, NOT PER PROCESS.  A path is heap-backed iff
//   (a) it sits under a GetSSDList() root, and
//   (b) its basename matches a prefix registered with SetPrefixBackend()
//       (or the default backend is Memory).
// Everything else falls through to the real syscall, which is what keeps the
// library/host boundary working for free: consolidate()'s output path,
// from_file()'s input path and chunk_csr::from_file are ordinary cwd//tmp files
// and stay ordinary files even in whole-process memory mode.
//
// Because routing is by path, a heap-backed and a disk-backed chunk_seq can
// coexist in one process, and a single chunk_seq may even hold chunks of both
// kinds -- each chunk routes independently by its own filename.
//
//   plaid::io::MemoryBacked scope("iota");   // this sequence in RAM ...
//   auto s = plaid::iota(n);                 // ... everything else on disk
//
// Whole-process mode is `PLAID_IN_MEMORY=1` in the environment, `--in_memory=1`
// on the command line, or SetDefaultBackend(Backend::Memory).
//
// Note: memory mode cannot reproduce I/O-error paths -- reads and writes to the
// arena never fail, and SYSCALL() only logs anyway.  Use the real backend when
// exercising error handling.
//

#ifndef PLAID_IO_BACKEND_H
#define PLAID_IO_BACKEND_H

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstddef>
#include <string>
#include <vector>

#include "configs.h"

namespace plaid {
namespace io {

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

enum class Backend { Disk, Memory };

// Backend for paths that match no registered prefix.  Defaults to Disk, or to
// Memory when PLAID_IN_MEMORY=1 is set in the environment.
void SetDefaultBackend(Backend b);
Backend GetDefaultBackend();

// True iff the default backend is Memory.  Cheap; safe to call in hot paths.
bool MemoryModeDefault();

// Route chunk_seqs whose filename prefix starts with `prefix` to `b`.  Longest
// registered prefix wins.  Matching mirrors the `name.rfind(p, 0) == 0` idiom
// bench_drives::clear_drives and every test's cleanup_prefix already use.
//
// These rules decide where *new* files are created; they do not move existing
// ones.  A file that already lives in the arena stays heap-backed no matter
// what the rules later say, so a chunk_seq built inside a MemoryBacked scope
// remains readable after that scope ends (otherwise its reads would fall
// through to a path that does not exist on disk).
//
// Caveat: library-internal intermediates are only picked up when they are built
// as `parent_prefix + suffix` (sort_buckets.h's `prefix + "_" + wave`, cut.h's
// `<file>_cut`).  Ones with unrelated prefixes (sample_sort's "ss_bucket_",
// cut.h's "bimul_zero") need registering separately -- which is why the test
// suite uses whole-process mode rather than per-prefix scopes.
void SetPrefixBackend(const std::string& prefix, Backend b);
void ClearPrefixBackends();

// Which backend a given path resolves to.  Paths outside every SSD root are
// always Disk.
Backend BackendFor(const char* path);

// RAII scope.  An empty prefix means "everything", i.e. whole-process mode.
class MemoryBacked {
 public:
  explicit MemoryBacked(const std::string& prefix = "");
  ~MemoryBacked();
  MemoryBacked(const MemoryBacked&) = delete;
  MemoryBacked& operator=(const MemoryBacked&) = delete;

 private:
  std::string prefix_;
  Backend previous_;
  bool was_registered_;
};

// ---------------------------------------------------------------------------
// File descriptors
// ---------------------------------------------------------------------------

// Virtual descriptors are allocated from this base, so `fd >= kVirtualFdBase`
// identifies a heap-backed fd and everything below it is a real kernel fd that
// the wrappers hand straight to libc.
constexpr int kVirtualFdBase = 1 << 28;

inline bool IsVirtual(int fd) { return fd >= kVirtualFdBase; }

// All of these mirror the libc signature *and* the libc return convention:
// >= 0 on success, -1 with errno set on failure.
int Open(const char* path, int flags, mode_t mode = 0644);
int Close(int fd);
ssize_t Pread(int fd, void* buf, size_t count, off_t offset);
ssize_t Pwrite(int fd, const void* buf, size_t count, off_t offset);
ssize_t Read(int fd, void* buf, size_t count);
ssize_t Write(int fd, const void* buf, size_t count);
off64_t Lseek64(int fd, off64_t offset, int whence);
int Fallocate(int fd, int mode, off_t offset, off_t len);
int Ftruncate(int fd, off_t length);
int Fstat(int fd, struct stat* st);

// Path-based helpers.
int Stat(const char* path, struct stat* st);
int Unlink(const char* path);

// Directory listing.  REPLACES std::filesystem::directory_iterator for the SSD
// roots and is load-bearing, not cosmetic: the (dir, error_code) overload used
// by bench_drives::clear_drives, chunk_sa_common's sweep() and dc3_test yields
// an *empty range* when /mnt/ssdN does not exist, so in memory mode teardown
// would silently free nothing and the whole arena would leak as an OOM instead
// of a test failure.  Returns real entries, heap entries, or both.
struct DirEntry {
  std::string path;
  size_t size;
};
std::vector<DirEntry> ListDir(const std::string& dir);

// fsync the filesystem holding `dir`.  No-op for heap-backed roots.
int SyncDir(const std::string& dir);

// Delete every file across all SSD roots whose basename starts with `prefix`,
// on whichever backend holds it.  Returns the number removed.  This is the
// shared replacement for the ~10 copy-pasted cleanup_prefix() helpers in the
// tests, and unlike their fixed 0..num_drives enumeration it also catches
// suffixed intermediates.
size_t UnlinkPrefix(const std::string& prefix);

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

size_t LiveBytes();  // bytes currently held by the arena
size_t PeakBytes();  // high-water mark since process start

// Hard cap on arena size, so exceeding it CHECK-fails with a dump of the
// largest files rather than inviting the OOM killer.  Default is
// min(4 GiB, RAM/2); override with PLAID_MEM_LIMIT_BYTES.
size_t LimitBytes();
void SetLimitBytes(size_t limit);

// Log the k largest arena files (path + bytes).
void DumpLargest(size_t k);

// Drop every heap-backed file.  Only for tests that want a clean slate.
void ResetArena();

}  // namespace io
}  // namespace plaid

#endif  // PLAID_IO_BACKEND_H
