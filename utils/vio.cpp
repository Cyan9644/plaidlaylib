#include "utils/vio.h"

#include <sys/sysinfo.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "utils/file_utils.h"

namespace plaid {
namespace vio {

const char kMemPathPrefix[] = "mem:";
static constexpr size_t kMemPathPrefixLen = sizeof(kMemPathPrefix) - 1;

bool is_mem_path(const char* path) {
  return path != nullptr &&
         std::strncmp(path, kMemPathPrefix, kMemPathPrefixLen) == 0;
}

// Defined below with the other public entry points.  MemFile::donate needs it:
// the knob has to gate both halves of zero-copy, or PLAID_MEMORY_ZEROCOPY=0
// would still hand write buffers to the store.
bool zerocopy_enabled();

namespace {

// ---------------------------------------------------------------------------
// Block pool
//
// Memory-backed files are built out of fixed-size blocks carved from much
// larger slabs.  Allocating each block on its own would be the mistake
// this codebase already documents in three places (chunk_seq.h's to_vector,
// ChunkPartition, delayed.h's read pool): an allocation that size goes to mmap,
// and the kernel serializes all workers on mmap_sem.  Pooling means the steady
// state does no mapping at all.
//
// Slabs are never handed back to the OS -- same bargain the reader's buffer
// pool strikes -- so the cap below is enforced on slab growth, while
// resident_bytes() reports the blocks actually held by live files (which does
// return to zero, and is what the leak assertions check).
// ---------------------------------------------------------------------------

// Two block size classes, picked per file from its first write.
//
// kSmallBlock is the conservative default, and is deliberately much smaller
// than CHUNK_SIZE: a memory-backed file rounds up to whole blocks, and plenty
// of workloads keep many files holding only a few kilobytes each --
// group_by_index opens one per bucket, and a few hundred buckets are ordinary.
// At CHUNK_SIZE granularity those cost 4 MiB apiece (gigabytes in total) where
// on disk they occupy a few KiB, which would make memory mode fail on
// workloads disk mode handles comfortably and would wreck the like-for-like
// comparison the mode exists to provide.
//
// kLargeBlock is exactly CHUNK_SIZE, for the files the engine writes one whole
// chunk at a time.  Those waste nothing -- a chunk is always written as a full
// CHUNK_SIZE block, zero-padded when partial -- and in exchange a chunk's
// bytes land in ONE contiguous allocation, which is what lets the reader hand
// out a pointer into the store instead of copying it.  At kSmallBlock a 4 MiB
// chunk is 64 pieces at descending addresses, so no such pointer exists.
constexpr size_t kSmallBlock = 64 * 1024;
constexpr size_t kLargeBlock = CHUNK_SIZE;

// Enough blocks per slab that growth is rare, without making any single
// aligned_alloc absurd.
constexpr size_t kSmallBlocksPerSlab = 256;  // 16 MiB slabs
constexpr size_t kLargeBlocksPerSlab = 16;   // 16 * CHUNK_SIZE = 64 MiB slabs

std::atomic<size_t> g_file_bytes{0};
std::atomic<size_t> g_borrowed_bytes{0};
std::atomic<size_t> g_copied_bytes{0};
std::atomic<size_t> g_declined_bytes{0};
std::atomic<size_t> g_donated_bytes{0};

size_t MemoryCapBytes() {
  static const size_t cap = [] {
    if (const char* e = getenv("PLAID_MEMORY_CAP_BYTES"))
      return (size_t)std::stoull(e);
    const size_t phys =
        (size_t)sysconf(_SC_PHYS_PAGES) * (size_t)sysconf(_SC_PAGE_SIZE);
    return phys / 2;
  }();
  return cap;
}

class BlockPool {
 public:
  // Deliberately leaked, never destroyed.  A MemFile returns its blocks to
  // this pool from its destructor, and the registry that owns those MemFiles
  // is itself a static: if the pool were destroyed first, teardown would write
  // through a dead object and corrupt the heap on the way out.  Outliving the
  // process sidesteps the ordering entirely -- the same bargain the reader's
  // buffer pool already strikes in chunk_seq.h.
  static BlockPool& instance() {
    static BlockPool* p = new BlockPool();
    return *p;
  }

  // `who` names the file being grown: when the cap trips, that is the one piece
  // of information that points at the offending intermediate.
  char* alloc(size_t block_size, const char* who, bool zero) {
    SizeClass& sc = class_for(block_size);
    char* p = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (sc.free_.empty()) grow(sc, block_size, who);
      p = sc.free_.back();
      sc.free_.pop_back();
    }
    if (zero) std::memset(p, 0, block_size);
    g_file_bytes.fetch_add(block_size, std::memory_order_relaxed);
    return p;
  }

  // A buffer donated straight into a file never passes through the free lists
  // above, so it is charged and un-charged here to keep one budget for all
  // memory-backed storage.
  void note_adopted(size_t bytes, const char* who) {
    std::lock_guard<std::mutex> lk(mu_);
    CHECK_LE(pool_bytes_ + bytes, MemoryCapBytes())
        << "vio: in-memory storage would exceed PLAID_MEMORY_CAP_BYTES ("
        << MemoryCapBytes() << " bytes) adopting a write buffer for \'" << who
        << "\'; already holding " << pool_bytes_ << " bytes.";
    pool_bytes_ += bytes;
  }

  void note_released(size_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    pool_bytes_ -= bytes;
  }

  void release(size_t block_size, char* p) {
    if (p == nullptr) return;
    g_file_bytes.fetch_sub(block_size, std::memory_order_relaxed);
    SizeClass& sc = class_for(block_size);
    std::lock_guard<std::mutex> lk(mu_);
    sc.free_.push_back(p);
  }

 private:
  struct SizeClass {
    std::vector<char*> free_;
  };

  SizeClass& class_for(size_t block_size) {
    return block_size == kLargeBlock ? large_ : small_;
  }

  void grow(SizeClass& sc, size_t block_size, const char* who) {  // holds mu_
    const size_t n =
        (block_size == kLargeBlock) ? kLargeBlocksPerSlab : kSmallBlocksPerSlab;
    const size_t slab_bytes = block_size * n;
    CHECK_LE(pool_bytes_ + slab_bytes, MemoryCapBytes())
        << "vio: in-memory storage would exceed PLAID_MEMORY_CAP_BYTES ("
        << MemoryCapBytes() << " bytes) while growing \'" << who
        << "\'; already holding " << pool_bytes_
        << " bytes.  Raise the cap, release intermediates, or run this "
           "sequence on disk.  Note this cap covers file storage only -- the "
           "reader\'s buffer pool and process_inplace\'s per-worker staging "
           "buffers are additional.";
    char* slab = (char*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, slab_bytes);
    CHECK(slab != nullptr) << "vio: out of memory allocating a " << slab_bytes
                           << "-byte slab for \'" << who << "\'";
    pool_bytes_ += slab_bytes;
    slabs_.push_back(slab);
    for (size_t i = 0; i < n; i++) sc.free_.push_back(slab + i * block_size);
  }

  std::mutex mu_;
  SizeClass small_;
  SizeClass large_;
  std::vector<char*> slabs_;
  size_t pool_bytes_ = 0;
};

// ---------------------------------------------------------------------------
// MemFile -- a sparse, heap-backed file
//
// Blocks materialize on write; a block that was never written reads as zeros,
// exactly like a hole in a sparse file on disk.  Reads and writes copy the
// block pointer out under a shared lock and then memcpy outside it: a block,
// once allocated, never moves, so only the block *table* needs guarding.
// ---------------------------------------------------------------------------
class MemFile {
 public:
  explicit MemFile(std::string name) : name_(std::move(name)) {}

  ~MemFile() {
    for (Block& b : blocks_) release_block(b);
    for (Block& b : retired_) release_block(b);
  }

  // Descriptor accounting, used only by truncate() to decide whether anyone
  // could still be holding a pointer into this file's blocks.
  void ref_fd() { open_fds_.fetch_add(1, std::memory_order_relaxed); }
  void unref_fd() { open_fds_.fetch_sub(1, std::memory_order_relaxed); }

  ssize_t pread(void* buf, size_t count, off_t offset) {
    char* out = (char*)buf;
    size_t done = 0;
    // One acquisition for the whole request rather than one per block: the
    // lock guards only table growth, and a block never moves once allocated.
    std::shared_lock<std::shared_mutex> lk(mu_);
    const size_t bs = block_size_;
    while (done < count) {
      const size_t pos = (size_t)offset + done;
      const size_t bi = pos / bs;
      const size_t within = pos % bs;
      const size_t n = std::min(count - done, bs - within);
      char* blk = (bi < blocks_.size()) ? blocks_[bi].p : nullptr;
      if (blk == nullptr)
        std::memset(out + done, 0, n);  // an unwritten hole
      else
        std::memcpy(out + done, blk + within, n);
      done += n;
    }
    g_copied_bytes.fetch_add(count, std::memory_order_relaxed);
    return (ssize_t)count;
  }

  // A pointer straight into this file's storage, or nullptr when the range is
  // not one contiguous resident run.  See vio::borrow() for the contract; in
  // particular the bytes are the file itself, so callers must not write them.
  char* borrow(size_t count, off_t offset) {
    if (count == 0) return nullptr;
    std::shared_lock<std::shared_mutex> lk(mu_);
    const size_t bs = block_size_;
    const size_t pos = (size_t)offset;
    const size_t within = pos % bs;
    if (within + count > bs) return nullptr;  // straddles a block boundary
    const size_t bi = pos / bs;
    if (bi >= blocks_.size()) return nullptr;  // past the end
    char* blk = blocks_[bi].p;
    if (blk == nullptr) return nullptr;  // an unwritten hole: must read zeros
    return blk + within;
  }

  // Adopt `buf` as this file's block covering [offset, offset+count), so a
  // write that would have been a memcpy becomes a pointer assignment.  Returns
  // false -- having copied nothing -- unless the buffer is exactly one whole,
  // still-unwritten block of the large class, in which case the caller falls
  // back to an ordinary pwrite.  On success the buffer belongs to this file and
  // the caller must not free it (see plaid::DonatableFree).
  bool donate(void* buf, size_t count, off_t offset) {
    if (!zerocopy_enabled()) return false;  // PLAID_MEMORY_ZEROCOPY=0
    if (buf == nullptr || count == 0) return false;
    if ((uintptr_t)buf % O_DIRECT_MEMORY_ALIGNMENT != 0) return false;
    if (count != kLargeBlock || (size_t)offset % kLargeBlock != 0) return false;

    std::unique_lock<std::shared_mutex> lk(mu_);
    if (!block_size_fixed_) {
      block_size_ = kLargeBlock;
      block_size_fixed_ = true;
    } else if (block_size_ != kLargeBlock) {
      return false;
    }
    const size_t bi = (size_t)offset / kLargeBlock;
    if (bi < blocks_.size() && blocks_[bi].p != nullptr) return false;
    if (bi >= blocks_.size()) blocks_.resize(bi + 1);
    // Adopted buffers skip the slab pool, so they have to be charged against
    // the same cap or memory mode could sail straight past it.
    BlockPool::instance().note_adopted(kLargeBlock, name_.c_str());
    g_file_bytes.fetch_add(kLargeBlock, std::memory_order_relaxed);
    blocks_[bi] = Block{(char*)buf, true};
    g_donated_bytes.fetch_add(kLargeBlock, std::memory_order_relaxed);
    size_ = std::max(size_, (size_t)offset + count);
    return true;
  }

  ssize_t pwrite(const void* buf, size_t count, off_t offset) {
    if (count == 0) return 0;

    size_t bs;
    // Materialize everything this request needs in one exclusive pass, so the
    // copy below -- and every concurrent writer to blocks that already exist --
    // only ever contends on the shared side.
    {
      std::unique_lock<std::shared_mutex> lk(mu_);
      if (!block_size_fixed_) {
        // The engine writes a whole chunk at a chunk-aligned offset; that
        // shape, and only that shape, earns the large class.  Everything else
        // -- BucketWriter's sub-chunk writev appends, the cut seams, a
        // single-element push_back -- keeps the small one.
        block_size_ =
            (count >= kLargeBlock && (size_t)offset % kLargeBlock == 0)
                ? kLargeBlock
                : kSmallBlock;
        block_size_fixed_ = true;
      }
      bs = block_size_;
      const size_t first = (size_t)offset / bs;
      const size_t last = ((size_t)offset + count - 1) / bs;
      if (last >= blocks_.size()) blocks_.resize(last + 1);
      for (size_t bi = first; bi <= last; bi++) {
        if (blocks_[bi].p != nullptr) continue;
        // A write covering a whole block overwrites every byte of it, so the
        // zero-fill can be skipped -- the common case, since the engine writes
        // whole CHUNK_SIZE runs at aligned offsets.
        const size_t bstart = bi * bs;
        const bool covers = (size_t)offset <= bstart &&
                            bstart + bs <= (size_t)offset + count;
        blocks_[bi] =
            Block{BlockPool::instance().alloc(bs, name_.c_str(), !covers),
                  false};
      }
      size_ = std::max(size_, (size_t)offset + count);
    }

    const char* in = (const char*)buf;
    size_t done = 0;
    std::shared_lock<std::shared_mutex> lk(mu_);
    while (done < count) {
      const size_t pos = (size_t)offset + done;
      const size_t within = pos % bs;
      const size_t n = std::min(count - done, bs - within);
      std::memcpy(blocks_[pos / bs].p + within, in + done, n);
      done += n;
    }
    g_copied_bytes.fetch_add(count, std::memory_order_relaxed);
    return (ssize_t)count;
  }

  // Shrinking frees the blocks past the new end; growing is sparse, so it only
  // moves the recorded size and lets the blocks materialize on write.
  //
  // "Frees" is conditional.  While any descriptor is open on this file a reader
  // may be holding a pointer directly into one of these blocks, so they are
  // retired instead and returned when the file itself dies.  With no descriptor
  // open nobody can hold such a pointer and they go back at once -- the common
  // case, since open(O_TRUNC) on a reused prefix runs before its own descriptor
  // is installed.
  int truncate(off_t length) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    const size_t len = (size_t)length;
    const size_t keep = (len + block_size_ - 1) / block_size_;
    const bool may_be_borrowed = open_fds_.load(std::memory_order_relaxed) > 0;
    for (size_t i = keep; i < blocks_.size(); i++) {
      if (blocks_[i].p == nullptr) continue;
      if (may_be_borrowed) {
        retired_.push_back(blocks_[i]);
        blocks_[i] = Block{};
      } else {
        release_block(blocks_[i]);
      }
    }
    if (keep < blocks_.size()) blocks_.resize(keep);
    size_ = len;
    return 0;
  }

  // fallocate's job here is only to make the file at least this large.
  int reserve(off_t offset, off_t len) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    size_ = std::max(size_, (size_t)offset + (size_t)len);
    return 0;
  }

 private:
  // A donated block came from a writer's aligned_alloc rather than the slab
  // pool, so it goes back with free() and its bytes are un-charged by hand.
  struct Block {
    char* p = nullptr;
    bool donated = false;
  };

  void release_block(Block& b) {
    if (b.p == nullptr) return;
    if (b.donated) {
      g_file_bytes.fetch_sub(block_size_, std::memory_order_relaxed);
      BlockPool::instance().note_released(block_size_);
      std::free(b.p);
    } else {
      BlockPool::instance().release(block_size_, b.p);
    }
    b = Block{};
  }

  mutable std::shared_mutex mu_;
  std::vector<Block> blocks_;
  // Blocks dropped by truncate() while a descriptor was open; see truncate().
  std::vector<Block> retired_;
  size_t block_size_ = kSmallBlock;
  bool block_size_fixed_ = false;
  std::atomic<size_t> open_fds_{0};
  size_t size_ = 0;
  std::string name_;
};

// ---------------------------------------------------------------------------
// Registry and virtual descriptor table
// ---------------------------------------------------------------------------

// All leaked on purpose, for the reason given on BlockPool::instance(): these
// hold (or are held by) objects whose destructors touch each other, and static
// destruction order between them is not something this file can pin down.
std::mutex& registry_mu() {
  static std::mutex* m = new std::mutex();
  return *m;
}
std::unordered_map<std::string, std::shared_ptr<MemFile>>& registry() {
  static auto* r = new std::unordered_map<std::string, std::shared_ptr<MemFile>>();
  return *r;
}

std::mutex& fdtab_mu() {
  static std::mutex* m = new std::mutex();
  return *m;
}
std::vector<std::shared_ptr<MemFile>>& fdtab() {
  static auto* t = new std::vector<std::shared_ptr<MemFile>>();
  return *t;
}
std::vector<int>& fdfree() {
  static auto* f = new std::vector<int>();
  return *f;
}

int install_fd(std::shared_ptr<MemFile> f) {
  std::lock_guard<std::mutex> lk(fdtab_mu());
  int slot;
  if (!fdfree().empty()) {
    slot = fdfree().back();
    fdfree().pop_back();
    fdtab()[slot] = std::move(f);
  } else {
    slot = (int)fdtab().size();
    fdtab().push_back(std::move(f));
  }
  return slot | kVirtualFdBit;
}

std::shared_ptr<MemFile> lookup_fd(int fd) {
  const int slot = fd & ~kVirtualFdBit;
  std::lock_guard<std::mutex> lk(fdtab_mu());
  CHECK_GE(slot, 0);
  CHECK_LT((size_t)slot, fdtab().size()) << "vio: bad virtual fd " << fd;
  std::shared_ptr<MemFile> f = fdtab()[slot];
  CHECK(f != nullptr) << "vio: use of closed virtual fd " << fd;
  return f;
}

}  // namespace

size_t resident_bytes() {
  return g_file_bytes.load(std::memory_order_relaxed);
}

std::atomic<int>& zerocopy_slot() {
  static std::atomic<int>* s = new std::atomic<int>(-1);  // -1 = unresolved
  return *s;
}

bool zerocopy_enabled() {
  std::atomic<int>& slot = zerocopy_slot();
  int v = slot.load(std::memory_order_relaxed);
  if (v < 0) {
    const char* e = getenv("PLAID_MEMORY_ZEROCOPY");
    v = (e != nullptr && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    slot.store(v, std::memory_order_relaxed);
  }
  return v != 0;
}

void set_zerocopy_enabled(bool on) {
  zerocopy_slot().store(on ? 1 : 0, std::memory_order_relaxed);
}

namespace {

bool verify_borrow_enabled() {
  static const bool on = [] {
    const char* e = getenv("PLAID_VERIFY_BORROW");
    return e != nullptr && !(e[0] == '0' && e[1] == '\0');
  }();
  return on;
}

uint64_t fold(const char* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a over 8-byte words, then tail
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    h = (h ^ w) * 1099511628211ULL;
  }
  for (; i < n; i++) h = (h ^ (unsigned char)p[i]) * 1099511628211ULL;
  return h;
}

struct Loan {
  size_t len;
  uint64_t hash;
  size_t refs;
};

std::mutex& loans_mu() {
  static std::mutex* m = new std::mutex();
  return *m;
}
std::unordered_map<const void*, Loan>& loans() {
  static auto* m = new std::unordered_map<const void*, Loan>();
  return *m;
}

}  // namespace

void release_borrow(const void* p) {
  if (!verify_borrow_enabled() || p == nullptr) return;
  std::lock_guard<std::mutex> lk(loans_mu());
  auto it = loans().find(p);
  if (it == loans().end()) return;  // never lent, or already checked
  if (--it->second.refs > 0) return;
  const uint64_t now = fold((const char*)p, it->second.len);
  CHECK_EQ(now, it->second.hash)
      << "vio: a borrowed buffer was modified while on loan (" << it->second.len
      << " bytes at " << p
      << ").  A borrowed buffer IS the memory-backed file, so this corrupted "
         "the source sequence.  Set allow_borrow=false on the reader whose "
         "consumer writes into the buffers it is handed.";
  loans().erase(it);
}

char* borrow(int fd, size_t len, off_t off) {
  if (!zerocopy_enabled()) return nullptr;
  if (!is_virtual_fd(fd)) return nullptr;  // a real file: nothing to borrow
  char* p = lookup_fd(fd)->borrow(len, off);
  if (p != nullptr)
    g_borrowed_bytes.fetch_add(len, std::memory_order_relaxed);
  else
    g_declined_bytes.fetch_add(len, std::memory_order_relaxed);
  if (p != nullptr && verify_borrow_enabled()) {
    std::lock_guard<std::mutex> lk(loans_mu());
    auto it = loans().find(p);
    if (it == loans().end())
      loans().emplace(p, Loan{len, fold(p, len), 1});
    else
      it->second.refs++;
  }
  return p;
}

size_t borrowed_bytes() {
  return g_borrowed_bytes.load(std::memory_order_relaxed);
}
size_t declined_bytes() {
  return g_declined_bytes.load(std::memory_order_relaxed);
}
size_t donated_bytes() {
  return g_donated_bytes.load(std::memory_order_relaxed);
}
size_t copied_bytes() {
  return g_copied_bytes.load(std::memory_order_relaxed);
}

void release_prefix(const std::string& prefix) {
  std::lock_guard<std::mutex> lk(registry_mu());
  for (auto it = registry().begin(); it != registry().end();) {
    if (it->first.find(prefix) != std::string::npos)
      it = registry().erase(it);
    else
      ++it;
  }
}

// ---------------------------------------------------------------------------
// POSIX shims
// ---------------------------------------------------------------------------

int open(const char* path, int flags, mode_t mode) {
  if (!is_mem_path(path)) return ::open(path, flags, mode);

  std::shared_ptr<MemFile> f;
  {
    std::lock_guard<std::mutex> lk(registry_mu());
    auto it = registry().find(path);
    if (it == registry().end()) {
      // Without O_CREAT this must fail exactly as a missing file would.  On
      // disk a stale header or a mistyped prefix fails loudly at open(); if a
      // memory-backed file were conjured up instead, the same bug would
      // silently read zeros and only surface as a wrong answer much later.
      if ((flags & O_CREAT) == 0) {
        errno = ENOENT;
        return -1;
      }
      f = std::make_shared<MemFile>(path);
      registry().emplace(path, f);
    } else {
      f = it->second;
    }
  }
  // The writer opens O_CREAT without O_TRUNC and leans on producers to clear
  // the file separately, so honouring O_TRUNC here is what keeps a reused
  // prefix from inheriting the previous run's blocks.
  if (flags & O_TRUNC) f->truncate(0);
  f->ref_fd();
  return install_fd(std::move(f));
}

int close(int fd) {
  if (!is_virtual_fd(fd)) return ::close(fd);
  const int slot = fd & ~kVirtualFdBit;
  std::lock_guard<std::mutex> lk(fdtab_mu());
  CHECK_LT((size_t)slot, fdtab().size()) << "vio: close of bad virtual fd " << fd;
  if (fdtab()[slot] != nullptr) fdtab()[slot]->unref_fd();
  fdtab()[slot].reset();
  fdfree().push_back(slot);
  return 0;
}

ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
  if (!is_virtual_fd(fd)) return ::pread(fd, buf, count, offset);
  return lookup_fd(fd)->pread(buf, count, offset);
}

ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset) {
  if (!is_virtual_fd(fd)) return ::pwrite(fd, buf, count, offset);
  return lookup_fd(fd)->pwrite(buf, count, offset);
}

int fallocate(int fd, int mode, off_t offset, off_t len) {
  if (!is_virtual_fd(fd)) return ::fallocate(fd, mode, offset, len);
  return lookup_fd(fd)->reserve(offset, len);
}

int ftruncate(int fd, off_t length) {
  if (!is_virtual_fd(fd)) return ::ftruncate(fd, length);
  return lookup_fd(fd)->truncate(length);
}

int unlink(const char* path) {
  if (!is_mem_path(path)) return ::unlink(path);
  std::lock_guard<std::mutex> lk(registry_mu());
  auto it = registry().find(path);
  if (it == registry().end()) {
    errno = ENOENT;
    return -1;
  }
  // POSIX semantics: drop the name now, but any descriptor still open keeps
  // its shared_ptr, so in-flight readers are unaffected and the storage goes
  // back to the pool on the last close.
  registry().erase(it);
  return 0;
}

// ---------------------------------------------------------------------------
// Ring
// ---------------------------------------------------------------------------

Ring::~Ring() { queue_exit(); }

int Ring::queue_init(unsigned entries, unsigned flags) {
  CHECK(!real_inited_) << "vio::Ring: queue_init called twice";
  entries_ = entries;
  flags_ = flags;
  return 0;
}

void Ring::queue_exit() {
  if (real_inited_) {
    io_uring_queue_exit(&real_ring_);
    real_inited_ = false;
  }
  pending_.clear();
  for (Cqe* c : done_) delete c;
  done_.clear();
  for (Cqe* c : cqe_pool_) delete c;
  cqe_pool_.clear();
}

int Ring::ensure_real() {
  if (real_inited_) return 0;
  int r = InitIoUringWithRetry(entries_, &real_ring_, flags_);
  if (r < 0) return r;
  real_inited_ = true;
  return 0;
}

Sqe* Ring::get_sqe() {
  pending_.emplace_back();
  return &pending_.back();
}

Cqe* Ring::acquire_cqe() {
  if (cqe_pool_.empty()) return new Cqe();
  Cqe* c = cqe_pool_.back();
  cqe_pool_.pop_back();
  *c = Cqe();
  return c;
}

void Ring::release_cqe(Cqe* c) { cqe_pool_.push_back(c); }

int Ring::submit() {
  if (pending_.empty()) return 0;

  size_t real_ops = 0;
  for (const Sqe& s : pending_)
    if (!is_virtual_fd(s.fd)) real_ops++;

  if (real_ops > 0) {
    int rc = ensure_real();
    if (rc < 0) {
      pending_.clear();
      return rc;
    }
  }

  int total = 0;
  size_t staged = 0;
  for (const Sqe& s : pending_) {
    // A memory-backed request is serviced right here and its completion queued
    // for the caller's next reap.  No call site depends on completion order --
    // each matches on user_data or reaps a fixed count -- so a synthetic
    // completion can be interleaved with the kernel's freely.
    if (is_virtual_fd(s.fd)) {
      Cqe* c = acquire_cqe();
      c->data = s.data;
      c->real = nullptr;
      switch (s.op) {
        case Sqe::Op::kRead:
          c->res = (int)pread(s.fd, s.buf, s.len, s.off);
          break;
        case Sqe::Op::kWrite:
          // A donatable write whose buffer happens to be exactly one whole,
          // unwritten block becomes a pointer assignment; anything else falls
          // back to the copy.  The completion tells the writer which happened,
          // since only in the first case must it not free the buffer.
          if (s.donate && lookup_fd(s.fd)->donate(s.buf, s.len, s.off)) {
            c->donated = true;
            c->res = (int)s.len;
          } else {
            c->res = (int)pwrite(s.fd, s.buf, s.len, s.off);
          }
          break;
        case Sqe::Op::kWritev: {
          // Gathered now, not lazily: BucketWriter recycles the request (and
          // with it the iovec array) as soon as the completion is reaped.
          off_t off = s.off;
          ssize_t got = 0;
          for (unsigned i = 0; i < s.nvec; i++) {
            got += pwrite(s.fd, s.iov[i].iov_base, s.iov[i].iov_len, off);
            off += (off_t)s.iov[i].iov_len;
          }
          c->res = (int)got;  // total bytes -- BucketWriter checks this
          break;
        }
      }
      done_.push_back(c);
      total++;
      continue;
    }

    // The real SQ holds `entries_` slots; a batch larger than that has to be
    // flushed part-way through, exactly as liburing itself would require.
    struct io_uring_sqe* sq = io_uring_get_sqe(&real_ring_);
    while (sq == nullptr) {
      int n = io_uring_submit(&real_ring_);
      if (n < 0) {
        pending_.clear();
        return n;
      }
      total += n;
      staged = 0;
      sq = io_uring_get_sqe(&real_ring_);
    }

    switch (s.op) {
      case Sqe::Op::kRead:
        io_uring_prep_read(sq, s.fd, s.buf, s.len, s.off);
        break;
      case Sqe::Op::kWrite:
        io_uring_prep_write(sq, s.fd, s.buf, s.len, s.off);
        break;
      case Sqe::Op::kWritev:
        io_uring_prep_writev(sq, s.fd, s.iov, s.nvec, s.off);
        break;
    }
    io_uring_sqe_set_data(sq, s.data);
    staged++;
  }

  if (staged > 0) {
    int n = io_uring_submit(&real_ring_);
    if (n < 0) {
      pending_.clear();
      return n;
    }
    total += n;
  }

  pending_.clear();
  return total;
}

int Ring::peek_cqe(Cqe** out) {
  if (!done_.empty()) {
    *out = done_.front();
    done_.pop_front();
    return 0;
  }
  if (!real_inited_) return -EAGAIN;

  struct io_uring_cqe* c;
  int rc = io_uring_peek_cqe(&real_ring_, &c);
  if (rc != 0) return rc;

  Cqe* w = acquire_cqe();
  w->res = c->res;
  w->data = io_uring_cqe_get_data(c);
  w->real = c;
  *out = w;
  return 0;
}

int Ring::wait_cqe(Cqe** out) {
  if (!done_.empty()) {
    *out = done_.front();
    done_.pop_front();
    return 0;
  }
  // Blocking on a ring that was never created can only ever hang, so make the
  // lazy-init bug that would cause it fail loudly instead.
  CHECK(real_inited_) << "vio::Ring::wait_cqe: no completion is pending and no "
                         "kernel ring exists -- this would block forever";

  struct io_uring_cqe* c;
  int rc = io_uring_wait_cqe(&real_ring_, &c);
  if (rc != 0) return rc;

  Cqe* w = acquire_cqe();
  w->res = c->res;
  w->data = io_uring_cqe_get_data(c);
  w->real = c;
  *out = w;
  return 0;
}

void Ring::cqe_seen(Cqe* cqe) {
  if (cqe == nullptr) return;
  if (cqe->real != nullptr) io_uring_cqe_seen(&real_ring_, cqe->real);
  release_cqe(cqe);
}

}  // namespace vio

namespace {
std::atomic<int>& default_storage_slot() {
  static std::atomic<int> s{-1};  // -1 = not yet resolved from the environment
  return s;
}
}  // namespace

storage default_storage() {
  std::atomic<int>& slot = default_storage_slot();
  int v = slot.load(std::memory_order_relaxed);
  if (v < 0) {
    v = (int)storage::disk;
    if (const char* e = getenv("PLAID_STORAGE")) {
      if (std::strcmp(e, "memory") == 0) {
        v = (int)storage::memory;
      } else if (std::strcmp(e, "disk") != 0) {
        LOG(FATAL) << "PLAID_STORAGE must be \"memory\" or \"disk\", got \""
                   << e << "\"";
      }
    }
    slot.store(v, std::memory_order_relaxed);
  }
  return (storage)v;
}

void set_default_storage(storage s) {
  default_storage_slot().store((int)s, std::memory_order_relaxed);
}

}  // namespace plaid
