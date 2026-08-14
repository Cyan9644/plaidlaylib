//
// Created for the in-memory storage backend.  See io_backend.h.
//

#include "utils/io_backend.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

#include "utils/file_utils.h"
#include "utils/logger.h"

namespace plaid {
namespace io {
namespace {

// ---------------------------------------------------------------------------
// Accounting
// ---------------------------------------------------------------------------

std::atomic<size_t>& LiveBytesCounter() {
  static std::atomic<size_t> v{0};
  return v;
}
std::atomic<size_t>& PeakBytesCounter() {
  static std::atomic<size_t> v{0};
  return v;
}
std::atomic<size_t>& LimitBytesCell() {
  static std::atomic<size_t> v{[] {
    if (const char* e = std::getenv("PLAID_MEM_LIMIT_BYTES"))
      return (size_t)std::strtoull(e, nullptr, 10);
    const size_t ram =
        (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    return std::min<size_t>(4ULL << 30, ram / 2);
  }()};
  return v;
}

bool EnvFlag(const char* name, bool dflt) {
  const char* e = std::getenv(name);
  if (e == nullptr) return dflt;
  return !(e[0] == '0' && e[1] == '\0');
}

// Enforce O_DIRECT's alignment rules on heap-backed fds too.  RAM does not care,
// but enforcing it anyway means "passes in memory mode" implies "passes on
// /mnt": memory mode doubles as an O_DIRECT linter, which is the cheapest
// mitigation for the main hazard of a runtime-selectable backend.
bool StrictAlign() {
  static const bool v = EnvFlag("PLAID_MEM_STRICT_ALIGN", true);
  return v;
}

// Opening a heap-backed file that does not exist is always a bug -- unlike a
// real filesystem there is no external agent that could have created it.  On
// disk this is silently non-fatal (chunk_seq_reader.h:149 logs a -1, then
// io_uring reads into an untouched buffer and the caller compares garbage), so
// make it loud here instead.
bool OpenMissingFatal() {
  static const bool v = EnvFlag("PLAID_MEM_OPEN_MISSING_FATAL", true);
  return v;
}

void AccountAlloc(size_t bytes) {
  const size_t live = LiveBytesCounter().fetch_add(bytes) + bytes;
  size_t peak = PeakBytesCounter().load(std::memory_order_relaxed);
  while (live > peak &&
         !PeakBytesCounter().compare_exchange_weak(peak, live)) {
  }
  if (live > LimitBytesCell().load(std::memory_order_relaxed)) {
    DumpLargest(10);
    LOG(FATAL) << "plaid::io in-memory arena exceeded PLAID_MEM_LIMIT_BYTES ("
               << LimitBytesCell().load() << "): now " << live << " bytes. "
               << "Lower the test size or raise the limit.";
  }
}

void AccountFree(size_t bytes) { LiveBytesCounter().fetch_sub(bytes); }

// ---------------------------------------------------------------------------
// MemFile -- a growable byte arena addressable at arbitrary offsets
// ---------------------------------------------------------------------------
//
// Storage is a vector of fixed CHUNK_SIZE segments, and the shared_mutex guards
// ONLY the segment-pointer vector.  That is the whole point of the segmented
// design: growth reallocates the char* array but never the segments themselves,
// so a thread that obtained a segment pointer under the shared lock keeps a
// pointer into a block that never moves.  A flat std::vector<char> would
// realloc out from under an in-flight memcpy, and many writer threads pwrite
// disjoint offsets of the *same* file concurrently:
//
//   * UnorderedFileWriter routes by explicit (file_index, file_offset), so any
//     writer thread can target any file at any offset.
//   * BucketWriter stamps each request with bk.append_off under bk.lock before
//     advancing the cursor, so its in-flight requests own strictly disjoint,
//     monotonically increasing extents -- possibly in the same segment, and
//     possibly straddling a segment boundary.
//   * process_inplace gives each bucket to exactly one worker (atomic
//     next_bucket++) and its runs are maximal non-overlapping extents.
//
// So concurrent access is always to disjoint byte ranges; MemFile only has to
// make *that* safe, which it does.
class MemFile {
 public:
  // Deliberately much finer than CHUNK_SIZE, and paired with the zero-elision
  // in Pwrite below.  The library writes whole CHUNK_SIZE blocks even when a
  // chunk holds a handful of elements (the writer pads the tail with zeros and
  // pushes a full block, because O_DIRECT wants aligned lengths).  On disk that
  // padding is nearly free; in RAM, allocating CHUNK_SIZE for every partial
  // chunk makes the footprint track the *padded chunk count* instead of the
  // live data -- convex_hull's quickhull recursion emits one padded block per
  // (worker, bucket) at every level, so 1.6 MB of points ballooned past 15 GiB.
  // With 64 KiB granules an all-zero granule is simply never allocated, so a
  // barely-filled chunk costs 64 KiB instead of 4 MiB.
  static constexpr size_t kSegBytes = 64 << 10;
  static_assert(CHUNK_SIZE % kSegBytes == 0 || kSegBytes % CHUNK_SIZE == 0,
                "segment size must tile the chunk grid");
  static_assert(kSegBytes % O_DIRECT_MULTIPLE == 0,
                "segments must stay O_DIRECT-aligned");

  ~MemFile() { Clear(); }

  // Always copies `n` bytes, zero-filling holes and anything past the
  // high-water mark, and never allocates.  Readers in this library already
  // ignore short reads -- and ReadEntireFile() would spin forever on a 0
  // return -- while AlignUp(c.used) reads legitimately run past `used` and, for
  // a count_sort bucket's last chunk, past true_bytes as well.
  size_t Pread(void* dst, size_t n, size_t off) const {
    char* d = (char*)dst;
    for (size_t done = 0; done < n;) {
      const size_t abs = off + done;
      const size_t si = abs / kSegBytes, so = abs % kSegBytes;
      const size_t take = std::min(n - done, kSegBytes - so);
      const char* seg = SegForRead(si);
      if (seg == nullptr) {
        std::memset(d + done, 0, take);
      } else {
        std::memcpy(d + done, seg + so, take);
      }
      done += take;
    }
    return n;
  }

  size_t Pwrite(const void* src, size_t n, size_t off) {
    const char* s = (const char*)src;
    for (size_t done = 0; done < n;) {
      const size_t abs = off + done;
      const size_t si = abs / kSegBytes, so = abs % kSegBytes;
      const size_t take = std::min(n - done, kSegBytes - so);
      // An all-zero write into a granule that was never allocated is a no-op:
      // unallocated granules already read as zeros.  This is what keeps the
      // padding of partially-filled chunks free.  If the granule *is* backed we
      // must still copy, since the zeros may be overwriting live data.
      char* seg = SegForRead(si);
      if (seg == nullptr) {
        if (AllZero(s + done, take)) {
          done += take;
          continue;
        }
        seg = SegForWrite(si);
      }
      std::memcpy(seg + so, s + done, take);
      done += take;
    }
    Bump(off + n);
    return n;
  }

  size_t Size() const { return size_.load(std::memory_order_acquire); }

  // fallocate/ftruncate-grow.  Lazy on purpose: eagerly materializing every
  // segment would allocate a 1 GiB tabulate target before a single data byte is
  // written, roughly doubling peak footprint.
  void GrowTo(size_t n) { Bump(n); }

  void Truncate(size_t n) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    const size_t keep = (n + kSegBytes - 1) / kSegBytes;
    for (size_t i = keep; i < segs_.size(); i++) {
      if (segs_[i] != nullptr) {
        std::free(segs_[i]);
        segs_[i] = nullptr;
        AccountFree(kSegBytes);
      }
    }
    if (keep < segs_.size()) segs_.resize(keep);
    size_.store(n, std::memory_order_release);
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lk(mu_);
    for (char* p : segs_) {
      if (p != nullptr) {
        std::free(p);
        AccountFree(kSegBytes);
      }
    }
    segs_.clear();
    size_.store(0, std::memory_order_release);
  }

 private:
  // Word-at-a-time so real data bails out almost immediately; only genuine
  // zero padding pays the full scan, and there it buys us skipping both the
  // allocation and the copy.
  static bool AllZero(const char* p, size_t n) {
    const size_t head = std::min(n, (size_t)((-(uintptr_t)p) & 7));
    for (size_t i = 0; i < head; i++)
      if (p[i] != 0) return false;
    const auto* w = (const uint64_t*)(p + head);
    const size_t words = (n - head) / 8;
    for (size_t i = 0; i < words; i++)
      if (w[i] != 0) return false;
    for (size_t i = head + words * 8; i < n; i++)
      if (p[i] != 0) return false;
    return true;
  }

  void Bump(size_t high) {
    size_t cur = size_.load(std::memory_order_relaxed);
    while (cur < high &&
           !size_.compare_exchange_weak(cur, high, std::memory_order_release,
                                        std::memory_order_relaxed)) {
    }
  }

  char* SegForRead(size_t i) const {
    std::shared_lock<std::shared_mutex> lk(mu_);
    return i < segs_.size() ? segs_[i] : nullptr;
  }

  char* SegForWrite(size_t i) {
    {
      std::shared_lock<std::shared_mutex> lk(mu_);
      if (i < segs_.size() && segs_[i] != nullptr) return segs_[i];
    }
    std::unique_lock<std::shared_mutex> lk(mu_);
    if (i >= segs_.size()) segs_.resize(i + 1, nullptr);
    if (segs_[i] == nullptr) {
      // O_DIRECT_MEMORY_ALIGNMENT so callers that hand these bytes back to a
      // real O_DIRECT write (mixed-backend pipelines) stay aligned.
      char* p = (char*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, kSegBytes);
      CHECK(p != nullptr) << "plaid::io: out of memory allocating a "
                          << kSegBytes << "-byte arena segment";
      std::memset(p, 0, kSegBytes);  // sparse reads must observe zeros
      AccountAlloc(kSegBytes);
      segs_[i] = p;
    }
    return segs_[i];
  }

  mutable std::shared_mutex mu_;
  std::vector<char*> segs_;
  std::atomic<size_t> size_{0};
};

// ---------------------------------------------------------------------------
// Registry: filename -> MemFile
// ---------------------------------------------------------------------------

std::mutex& RegistryMutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, std::shared_ptr<MemFile>>& Registry() {
  static std::unordered_map<std::string, std::shared_ptr<MemFile>> r;
  return r;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

std::atomic<Backend>& DefaultBackendCell() {
  static std::atomic<Backend> v{EnvFlag("PLAID_IN_MEMORY", false)
                                    ? Backend::Memory
                                    : Backend::Disk};
  return v;
}

std::mutex& PrefixMutex() {
  static std::mutex m;
  return m;
}
std::vector<std::pair<std::string, Backend>>& PrefixTable() {
  static std::vector<std::pair<std::string, Backend>> t;
  return t;
}

// The SSD roots are write-once (PopulateSSDList CHECKs that ssd_list is empty),
// so caching them on first use is safe.  Resolved lazily rather than at static
// init because ParseGlobalArguments must get to choose the drive set first.
const std::vector<std::string>& SsdRoots() {
  static const std::vector<std::string> roots = GetSSDList();
  return roots;
}

// A path is under an SSD root when the root is a prefix ending at a path
// boundary.  The '/' check matters: without it "/mnt/ssd1" also matches
// "/mnt/ssd10".
bool UnderSsdRoot(const char* path, std::string* basename) {
  for (const std::string& r : SsdRoots()) {
    if (std::strncmp(path, r.c_str(), r.size()) == 0 && path[r.size()] == '/') {
      *basename = path + r.size() + 1;
      return true;
    }
  }
  return false;
}

Backend BackendForBasename(const std::string& basename) {
  std::lock_guard<std::mutex> lk(PrefixMutex());
  const auto& table = PrefixTable();
  size_t best = std::string::npos;
  Backend result = DefaultBackendCell().load(std::memory_order_relaxed);
  for (size_t i = 0; i < table.size(); i++) {
    const std::string& key = table[i].first;
    if (basename.rfind(key, 0) != 0) continue;
    if (best == std::string::npos || key.size() > table[best].first.size()) {
      best = i;
      result = table[i].second;
    }
  }
  return result;
}

bool InRegistry(const char* path) {
  std::lock_guard<std::mutex> lk(RegistryMutex());
  return Registry().find(path) != Registry().end();
}

bool RoutedToMemory(const char* path, std::string* basename) {
  // Fast path: nothing is heap-backed unless some prefix or the default says so.
  const bool no_rules = [] {
    if (DefaultBackendCell().load(std::memory_order_relaxed) != Backend::Disk)
      return false;
    std::lock_guard<std::mutex> lk(PrefixMutex());
    return PrefixTable().empty();
  }();

  if (!UnderSsdRoot(path, basename)) return false;

  // An existing arena file is ALWAYS heap-backed, whatever the prefix table
  // currently says.  The registry, not the routing rules, is the truth about
  // where a file already lives -- otherwise a chunk_seq built inside a
  // `MemoryBacked` scope would become unreadable the moment that scope ended,
  // and reads would silently fall through to a nonexistent path on disk.
  // Routing rules only decide where *new* files get created.
  if (InRegistry(path)) return true;
  if (no_rules) return false;
  return BackendForBasename(*basename) == Backend::Memory;
}

// ---------------------------------------------------------------------------
// Virtual descriptor table
// ---------------------------------------------------------------------------

struct VFd {
  std::shared_ptr<MemFile> owner;  // keeps the arena alive past unlink()
  MemFile* file = nullptr;
  std::string path;
  int flags = 0;
  std::atomic<size_t> pos{0};
};

// A fixed slot array, published atomically: allocation and release take the
// mutex, but Pread/Pwrite resolve an fd with a single relaxed load and never
// touch a lock.  65536 is far past what this library needs -- readers open one
// fd per *distinct filename*, not per chunk.
constexpr size_t kMaxVirtualFds = 1 << 16;

std::array<std::atomic<VFd*>, kMaxVirtualFds>& FdSlots() {
  static std::array<std::atomic<VFd*>, kMaxVirtualFds> slots{};
  return slots;
}
std::mutex& FdMutex() {
  static std::mutex m;
  return m;
}
std::vector<int>& FreeFdSlots() {
  static std::vector<int> v;
  return v;
}
std::atomic<size_t>& NextFdSlot() {
  static std::atomic<size_t> v{0};
  return v;
}

VFd* Resolve(int fd) {
  const size_t slot = (size_t)(fd - kVirtualFdBase);
  if (slot >= kMaxVirtualFds) return nullptr;
  return FdSlots()[slot].load(std::memory_order_acquire);
}

int AllocFd(std::shared_ptr<MemFile> file, const std::string& path, int flags) {
  auto* v = new VFd();
  v->owner = std::move(file);
  v->file = v->owner.get();
  v->path = path;
  v->flags = flags;

  int slot;
  {
    std::lock_guard<std::mutex> lk(FdMutex());
    if (!FreeFdSlots().empty()) {
      slot = FreeFdSlots().back();
      FreeFdSlots().pop_back();
    } else {
      const size_t next = NextFdSlot().fetch_add(1);
      CHECK(next < kMaxVirtualFds)
          << "plaid::io: exhausted " << kMaxVirtualFds << " virtual fds";
      slot = (int)next;
    }
  }
  FdSlots()[slot].store(v, std::memory_order_release);
  return kVirtualFdBase + slot;
}

void CheckAlignment(const VFd* v, const void* buf, size_t count, off_t offset) {
  if (!StrictAlign()) return;
  if ((v->flags & O_DIRECT) == 0) return;
  CHECK(offset % (off_t)O_DIRECT_MULTIPLE == 0)
      << "plaid::io: O_DIRECT offset " << offset << " on " << v->path
      << " is not a multiple of " << O_DIRECT_MULTIPLE;
  CHECK(count % O_DIRECT_MULTIPLE == 0)
      << "plaid::io: O_DIRECT length " << count << " on " << v->path
      << " is not a multiple of " << O_DIRECT_MULTIPLE;
  CHECK((uintptr_t)buf % O_DIRECT_MEMORY_ALIGNMENT == 0)
      << "plaid::io: O_DIRECT buffer for " << v->path << " is not "
      << O_DIRECT_MEMORY_ALIGNMENT << "-byte aligned";
}

}  // namespace

// ---------------------------------------------------------------------------
// Backend selection (public)
// ---------------------------------------------------------------------------

void SetDefaultBackend(Backend b) {
  DefaultBackendCell().store(b, std::memory_order_relaxed);
}
Backend GetDefaultBackend() {
  return DefaultBackendCell().load(std::memory_order_relaxed);
}
bool MemoryModeDefault() {
  return DefaultBackendCell().load(std::memory_order_relaxed) ==
         Backend::Memory;
}

void SetPrefixBackend(const std::string& prefix, Backend b) {
  if (prefix.empty()) {
    SetDefaultBackend(b);
    return;
  }
  std::lock_guard<std::mutex> lk(PrefixMutex());
  for (auto& e : PrefixTable()) {
    if (e.first == prefix) {
      e.second = b;
      return;
    }
  }
  PrefixTable().emplace_back(prefix, b);
}

void ClearPrefixBackends() {
  std::lock_guard<std::mutex> lk(PrefixMutex());
  PrefixTable().clear();
}

Backend BackendFor(const char* path) {
  std::string basename;
  if (!UnderSsdRoot(path, &basename)) return Backend::Disk;
  return BackendForBasename(basename);
}

MemoryBacked::MemoryBacked(const std::string& prefix) : prefix_(prefix) {
  if (prefix_.empty()) {
    previous_ = GetDefaultBackend();
    was_registered_ = true;
    SetDefaultBackend(Backend::Memory);
    return;
  }
  std::lock_guard<std::mutex> lk(PrefixMutex());
  was_registered_ = false;
  previous_ = Backend::Disk;
  for (auto& e : PrefixTable()) {
    if (e.first == prefix_) {
      was_registered_ = true;
      previous_ = e.second;
      e.second = Backend::Memory;
      return;
    }
  }
  PrefixTable().emplace_back(prefix_, Backend::Memory);
}

MemoryBacked::~MemoryBacked() {
  if (prefix_.empty()) {
    SetDefaultBackend(previous_);
    return;
  }
  std::lock_guard<std::mutex> lk(PrefixMutex());
  auto& table = PrefixTable();
  for (size_t i = 0; i < table.size(); i++) {
    if (table[i].first != prefix_) continue;
    if (was_registered_) {
      table[i].second = previous_;
    } else {
      table.erase(table.begin() + i);
    }
    return;
  }
}

// ---------------------------------------------------------------------------
// File descriptors (public)
// ---------------------------------------------------------------------------

int Open(const char* path, int flags, mode_t mode) {
  std::string basename;
  if (!RoutedToMemory(path, &basename)) return ::open(path, flags, mode);

  std::shared_ptr<MemFile> file;
  {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    auto& reg = Registry();
    auto it = reg.find(path);
    if (it == reg.end()) {
      if ((flags & O_CREAT) == 0) {
        if (OpenMissingFatal()) {
          LOG(FATAL) << "plaid::io: open(\"" << path
                     << "\") without O_CREAT, but no such heap-backed file "
                        "exists.  On disk this silently yields a bad fd and "
                        "then a buffer of garbage; set "
                        "PLAID_MEM_OPEN_MISSING_FATAL=0 to allow it.";
        }
        errno = ENOENT;
        return -1;
      }
      file = std::make_shared<MemFile>();
      reg.emplace(path, file);
    } else {
      file = it->second;
    }
  }
  if ((flags & O_TRUNC) != 0) file->Clear();
  return AllocFd(std::move(file), path, flags);
}

int Close(int fd) {
  if (!IsVirtual(fd)) return ::close(fd);
  const size_t slot = (size_t)(fd - kVirtualFdBase);
  if (slot >= kMaxVirtualFds) {
    errno = EBADF;
    return -1;
  }
  VFd* v = FdSlots()[slot].exchange(nullptr, std::memory_order_acq_rel);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  delete v;
  {
    std::lock_guard<std::mutex> lk(FdMutex());
    FreeFdSlots().push_back((int)slot);
  }
  return 0;
}

ssize_t Pread(int fd, void* buf, size_t count, off_t offset) {
  if (!IsVirtual(fd)) return ::pread(fd, buf, count, offset);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  CheckAlignment(v, buf, count, offset);
  return (ssize_t)v->file->Pread(buf, count, (size_t)offset);
}

ssize_t Pwrite(int fd, const void* buf, size_t count, off_t offset) {
  if (!IsVirtual(fd)) return ::pwrite(fd, buf, count, offset);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  CheckAlignment(v, buf, count, offset);
  return (ssize_t)v->file->Pwrite(buf, count, (size_t)offset);
}

ssize_t Read(int fd, void* buf, size_t count) {
  if (!IsVirtual(fd)) return ::read(fd, buf, count);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  const size_t off = v->pos.fetch_add(count);
  CheckAlignment(v, buf, count, (off_t)off);
  return (ssize_t)v->file->Pread(buf, count, off);
}

ssize_t Write(int fd, const void* buf, size_t count) {
  if (!IsVirtual(fd)) return ::write(fd, buf, count);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  const size_t off = v->pos.fetch_add(count);
  CheckAlignment(v, buf, count, (off_t)off);
  return (ssize_t)v->file->Pwrite(buf, count, off);
}

off64_t Lseek64(int fd, off64_t offset, int whence) {
  if (!IsVirtual(fd)) return ::lseek64(fd, offset, whence);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  size_t base = 0;
  switch (whence) {
    case SEEK_SET: base = 0; break;
    case SEEK_CUR: base = v->pos.load(); break;
    case SEEK_END: base = v->file->Size(); break;
    default: errno = EINVAL; return -1;
  }
  const off64_t target = (off64_t)base + offset;
  if (target < 0) {
    errno = EINVAL;
    return -1;
  }
  v->pos.store((size_t)target);
  return target;
}

int Fallocate(int fd, int mode, off_t offset, off_t len) {
  if (!IsVirtual(fd)) return ::fallocate(fd, mode, offset, len);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  v->file->GrowTo((size_t)(offset + len));
  return 0;
}

int Ftruncate(int fd, off_t length) {
  if (!IsVirtual(fd)) return ::ftruncate(fd, length);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  if ((size_t)length >= v->file->Size()) {
    v->file->GrowTo((size_t)length);
  } else {
    v->file->Truncate((size_t)length);
  }
  return 0;
}

int Fstat(int fd, struct stat* st) {
  if (!IsVirtual(fd)) return ::fstat(fd, st);
  VFd* v = Resolve(fd);
  if (v == nullptr) {
    errno = EBADF;
    return -1;
  }
  std::memset(st, 0, sizeof(*st));
  st->st_mode = S_IFREG | 0644;
  st->st_size = (off_t)v->file->Size();
  st->st_blksize = O_DIRECT_MULTIPLE;
  st->st_blocks = (blkcnt_t)((v->file->Size() + 511) / 512);
  st->st_nlink = 1;
  return 0;
}

int Stat(const char* path, struct stat* st) {
  std::string basename;
  if (!RoutedToMemory(path, &basename)) return ::stat(path, st);
  std::shared_ptr<MemFile> file;
  {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    auto it = Registry().find(path);
    if (it == Registry().end()) {
      errno = ENOENT;
      return -1;
    }
    file = it->second;
  }
  std::memset(st, 0, sizeof(*st));
  st->st_mode = S_IFREG | 0644;
  st->st_size = (off_t)file->Size();
  st->st_blksize = O_DIRECT_MULTIPLE;
  st->st_blocks = (blkcnt_t)((file->Size() + 511) / 512);
  st->st_nlink = 1;
  return 0;
}

int Unlink(const char* path) {
  std::string basename;
  if (!RoutedToMemory(path, &basename)) return ::unlink(path);
  std::lock_guard<std::mutex> lk(RegistryMutex());
  auto it = Registry().find(path);
  if (it == Registry().end()) {
    errno = ENOENT;
    return -1;
  }
  // Erase immediately so no later Open() finds it; the arena itself dies with
  // the last shared_ptr, i.e. when the last still-open fd is closed.  (The
  // reader closes its fds only in its destructor, so a leaked fd pins memory
  // past the unlink -- POSIX semantics, but worth knowing when chasing
  // footprint.)
  Registry().erase(it);
  return 0;
}

std::vector<DirEntry> ListDir(const std::string& dir) {
  std::vector<DirEntry> out;

  // Real entries, if the directory exists at all.  Non-throwing overload, which
  // is also what the callers this replaces used.
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
    std::error_code sec;
    const auto sz = std::filesystem::file_size(e.path(), sec);
    out.push_back({e.path().string(), sec ? 0 : (size_t)sz});
  }

  // Heap entries.
  const std::string want = dir + "/";
  std::lock_guard<std::mutex> lk(RegistryMutex());
  for (const auto& kv : Registry()) {
    if (kv.first.rfind(want, 0) != 0) continue;
    if (kv.first.find('/', want.size()) != std::string::npos) continue;
    out.push_back({kv.first, kv.second->Size()});
  }
  return out;
}

int SyncDir(const std::string& dir) {
  // Nothing to flush for a heap-backed root, and on a dev box with no /mnt the
  // open() would just fail anyway.
  std::string basename;
  const std::string probe = dir + "/probe";
  if (RoutedToMemory(probe.c_str(), &basename)) return 0;

  int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (fd < 0) return -1;
  const int r = ::syncfs(fd);
  ::close(fd);
  return r;
}

size_t UnlinkPrefix(const std::string& prefix) {
  size_t removed = 0;
  for (const std::string& root : SsdRoots()) {
    for (const DirEntry& e : ListDir(root)) {
      const size_t slash = e.path.rfind('/');
      const std::string base =
          slash == std::string::npos ? e.path : e.path.substr(slash + 1);
      if (base.rfind(prefix, 0) != 0) continue;
      if (Unlink(e.path.c_str()) == 0) removed++;
    }
  }
  return removed;
}

// ---------------------------------------------------------------------------
// Accounting (public)
// ---------------------------------------------------------------------------

size_t LiveBytes() { return LiveBytesCounter().load(); }
size_t PeakBytes() { return PeakBytesCounter().load(); }
size_t LimitBytes() { return LimitBytesCell().load(); }
void SetLimitBytes(size_t limit) { LimitBytesCell().store(limit); }

void DumpLargest(size_t k) {
  std::vector<std::pair<size_t, std::string>> v;
  {
    std::lock_guard<std::mutex> lk(RegistryMutex());
    v.reserve(Registry().size());
    for (const auto& kv : Registry())
      v.emplace_back(kv.second->Size(), kv.first);
  }
  std::sort(v.begin(), v.end(), std::greater<>());
  LOG(ERROR) << "plaid::io arena: " << v.size() << " files, "
             << LiveBytes() / (1 << 20) << " MiB live, "
             << PeakBytes() / (1 << 20) << " MiB peak.  Largest:";
  for (size_t i = 0; i < std::min(k, v.size()); i++)
    LOG(ERROR) << "  " << v[i].second << "  " << v[i].first / (1 << 20)
               << " MiB";
}

namespace {
// PLAID_MEM_STATS=1 prints the arena high-water mark at exit, so one run tells
// you whether a size override is needed at all.
struct StatsAtExit {
  ~StatsAtExit() {
    if (!EnvFlag("PLAID_MEM_STATS", false)) return;
    LOG(INFO) << "plaid::io arena peak " << PeakBytes() / (1 << 20)
              << " MiB, still live at exit " << LiveBytes() / (1 << 20)
              << " MiB";
  }
};
const StatsAtExit stats_at_exit;
}  // namespace

void ResetArena() {
  std::lock_guard<std::mutex> lk(RegistryMutex());
  Registry().clear();
}

}  // namespace io
}  // namespace plaid
