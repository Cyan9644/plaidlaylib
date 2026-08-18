// vio.h -- the virtual I/O layer: one indirection point between the library's
// primitives and the operating system.
//
// Every read, write, open and io_uring submission in ChunkSequence/Primitives/
// goes through the shims declared here rather than calling libc or liburing
// directly.  Each shim inspects its file descriptor (or path) and either
// forwards to the real syscall -- the ordinary out-of-core path, unchanged --
// or services the request out of a heap-backed store.
//
// The point is that the choice is made per file, at runtime, so a single
// process can hold both kinds of chunk_seq at once.  Everything above this
// layer (buffer pools, thread counts, ring depths, chunk layout, O_DIRECT
// alignment, drive assignment) is deliberately identical in both modes: that
// is what makes a memory-mode run and a disk-mode run comparable, and lets a
// benchmark separate the cost of the I/O from the cost of the out-of-core
// restructuring.
//
// This header is as ignorant of the data model as file_utils.h is; anything
// that knows about chunks lives in ChunkSequence/Primitives/chunk_seq.h.

#ifndef PLAID_VIO_H
#define PLAID_VIO_H

#include <fcntl.h>
#include <liburing.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace plaid {

// Where a chunk_seq's bytes live.  Threaded through the producers that create
// files; see chunk_seq::mode(), which derives it back out of a filename.
enum class storage { disk, memory };

// The mode a root creator (tabulate, iota, from_file, ...) uses when the caller
// does not name one.  Defaults to disk, overridden by the PLAID_STORAGE
// environment variable ("memory"/"disk") or the --storage= flag that
// ParseGlobalArguments accepts.  Everything derived from a sequence inherits
// that sequence's mode instead of consulting this.
storage default_storage();
void set_default_storage(storage s);

namespace vio {

// ---------------------------------------------------------------------------
// Virtual file descriptors
//
// A descriptor handed out for a memory-backed file carries bit 30.  It must
// stay *positive*: SYSCALL() logs on any negative return, three call sites
// CHECK(fd >= 0) outright (secondary_primitives.h, delayed.h), and
// BucketWriter uses -1 as its "closed" sentinel.  Bit 30 is comfortably above
// any attainable RLIMIT_NOFILE, so it can never collide with a real fd.
// ---------------------------------------------------------------------------
constexpr int kVirtualFdBit = 0x40000000;

inline bool is_virtual_fd(int fd) { return fd > 0 && (fd & kVirtualFdBit) != 0; }

// The sentinel that marks a path as memory-backed.  It is a *prefix* so that
// callers which build a name by concatenation (e.g. filename + "_cut_start" in
// sequential_cut_no_compression) inherit the mode for free.
extern const char kMemPathPrefix[];  // "mem:"

bool is_mem_path(const char* path);
inline bool is_mem_path(const std::string& p) { return is_mem_path(p.c_str()); }

// ---------------------------------------------------------------------------
// POSIX shims.  Signatures mirror the libc originals so a call site changes by
// nothing more than its qualification.
// ---------------------------------------------------------------------------
int open(const char* path, int flags, mode_t mode = 0644);
int close(int fd);
ssize_t pread(int fd, void* buf, size_t count, off_t offset);
ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset);
int fallocate(int fd, int mode, off_t offset, off_t len);
int ftruncate(int fd, off_t length);
int unlink(const char* path);
inline int unlink(const std::string& p) { return unlink(p.c_str()); }

// ---------------------------------------------------------------------------
// io_uring shims
//
// A Ring wraps a real io_uring that is created *lazily*, on the first submit
// that carries a real descriptor.  A ring whose traffic is entirely
// memory-backed therefore never enters the kernel at all -- which matters, as
// ring setup is expensive enough that the library has a whole class
// (PersistentChunkSequenceReader) built to amortize it.
//
// Submission is deferred: get_sqe/prep_*/sqe_set_data record the request, and
// submit() then either memcpys it (virtual) or replays it into the real ring
// (real).  Completions are drained from a software queue first, then the
// kernel's.  No call site depends on completion order -- each either matches on
// user_data or reaps a fixed count -- so interleaving the two sources is safe.
// ---------------------------------------------------------------------------

// A completion.  `res` keeps its liburing name so existing `cqe->res` checks
// compile unchanged.  `real == nullptr` marks a synthetic completion, which is
// how cqe_seen knows whether it must advance the kernel's CQ head.
struct Cqe {
  int res = 0;
  void* data = nullptr;
  struct io_uring_cqe* real = nullptr;
  // Set when the memory-backed store adopted this write's buffer instead of
  // copying it.  The writer then releases the buffer without freeing it: those
  // bytes are the file now.  Always false for a real descriptor.
  bool donated = false;
};

// A recorded, not-yet-submitted request.
struct Sqe {
  enum class Op { kRead, kWrite, kWritev };
  Op op = Op::kRead;
  int fd = -1;
  void* buf = nullptr;
  const struct iovec* iov = nullptr;
  unsigned nvec = 0;
  unsigned len = 0;
  off_t off = 0;
  void* data = nullptr;
  // Offers this write's buffer to the store; see sqe_set_donate.
  bool donate = false;
};

class Ring {
 public:
  Ring() = default;
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;
  ~Ring();

  // Records (entries, flags) for the eventual lazy io_uring_queue_init.  Always
  // returns 0: a genuine init failure can only surface later, from submit().
  int queue_init(unsigned entries, unsigned flags);
  void queue_exit();

  // Never returns nullptr -- it grows its own pool instead.  Emulating SQ
  // exhaustion would deadlock UnorderedFileWriter's writer loop, which spins on
  // this call with no intervening submit().  Nothing is lost by growing: every
  // ring site already bounds its own in-flight count independently.
  Sqe* get_sqe();

  int submit();
  int wait_cqe(Cqe** out);
  int peek_cqe(Cqe** out);
  void cqe_seen(Cqe* cqe);

 private:
  Cqe* acquire_cqe();
  void release_cqe(Cqe* c);
  int ensure_real();

  struct io_uring real_ring_;
  bool real_inited_ = false;
  unsigned entries_ = 0;
  unsigned flags_ = 0;

  // deque, not vector: get_sqe hands out pointers that must survive the growth
  // caused by a later get_sqe in the same batch.
  std::deque<Sqe> pending_;
  std::deque<Cqe*> done_;
  std::vector<Cqe*> cqe_pool_;
};

// Free-function spellings, so a call site differs from the liburing original
// only by qualification.
inline int queue_init(unsigned entries, Ring* r, unsigned flags) {
  return r->queue_init(entries, flags);
}
inline void queue_exit(Ring* r) { r->queue_exit(); }
inline Sqe* get_sqe(Ring* r) { return r->get_sqe(); }
inline int submit(Ring* r) { return r->submit(); }
inline int wait_cqe(Ring* r, Cqe** out) { return r->wait_cqe(out); }
inline int peek_cqe(Ring* r, Cqe** out) { return r->peek_cqe(out); }
inline void cqe_seen(Ring* r, Cqe* c) { r->cqe_seen(c); }
inline void* cqe_get_data(Cqe* c) { return c->data; }

inline void prep_read(Sqe* s, int fd, void* buf, unsigned len, off_t off) {
  s->op = Sqe::Op::kRead;
  s->fd = fd;
  s->buf = buf;
  s->len = len;
  s->off = off;
}
inline void prep_write(Sqe* s, int fd, const void* buf, unsigned len,
                       off_t off) {
  s->op = Sqe::Op::kWrite;
  s->fd = fd;
  s->buf = const_cast<void*>(buf);
  s->len = len;
  s->off = off;
}
inline void prep_writev(Sqe* s, int fd, const struct iovec* iov, unsigned nvec,
                        off_t off) {
  s->op = Sqe::Op::kWritev;
  s->fd = fd;
  s->iov = iov;
  s->nvec = nvec;
  s->off = off;
}
inline void sqe_set_data(Sqe* s, void* data) { s->data = data; }

// Offer this write's buffer to a memory-backed store, which may take it as its
// own block rather than copying into one -- the write-side mirror of borrow().
// Only meaningful for a caller that owns the buffer outright and would free it
// once the write completes; the completion reports whether it was taken.
inline void sqe_set_donate(Sqe* s) { s->donate = true; }

// ---------------------------------------------------------------------------
// Memory-backed storage: accounting and lifetime
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Zero-copy reads
//
// A memory-backed chunk already sits in DRAM, so filling a caller's buffer from
// it is a pure memcpy that the disk path does not pay: O_DIRECT has the drive
// DMA straight into the reader's buffer, so nothing copies there at all.  On a
// 30-drive array that memcpy is enough to make memory mode the *slower* of the
// two on bandwidth-bound work.
//
// borrow() removes it.  It returns a pointer into the store itself when the
// requested range is one contiguous resident run, and nullptr otherwise -- a
// real descriptor, an unwritten hole, a span crossing a block boundary, or
// zero-copy disabled.  A nullptr is never an error: the caller falls back to
// its ordinary buffer-and-read path.
//
// The returned bytes stay valid while `fd` is open (truncate retires rather
// than releases blocks while any descriptor is open, for exactly this reason).
// Callers MUST treat them as read-only: they are the file, not a copy of it.
char* borrow(int fd, size_t len, off_t off);

// Hand a borrowed pointer back.  A no-op in ordinary runs -- a borrow owns
// nothing, so there is nothing to return -- but under PLAID_VERIFY_BORROW=1 it
// is where the buffer is checked for having been modified while out on loan.
// The four pack sites that compact in place are opted out of borrowing, but
// nothing in the type system stops a fifth consumer from writing through one,
// and the corruption would land silently in the source file.  Run the suite
// once with the flag on after touching any read path.
void release_borrow(const void* p);

// Off via PLAID_MEMORY_ZEROCOPY=0, which restores the copying path so a run can
// be compared against results taken before zero-copy existed.  The setter lets
// a test exercise both paths in one process.
bool zerocopy_enabled();
void set_zerocopy_enabled(bool on);

// Read-side accounting.
//
// borrowed_bytes/declined_bytes are the two outcomes of a borrow() request
// against a memory-backed descriptor, so their ratio is the zero-copy hit rate
// of the read path alone -- which is what tests assert, so that a silent fall
// back to copying cannot pass unnoticed.  A request against a real descriptor
// counts as neither: there was never anything to borrow.
//
// copied_bytes is broader: every byte memcpy'd by a memory-backed pread or
// pwrite,
// including callers that must copy no matter what (materialize/to_vector build
// a caller-owned parlay::sequence) and the paths deliberately left copying
// (process_inplace's multi-chunk runs, the cut seams).  Useful context, not a
// pass/fail signal.
size_t borrowed_bytes();
size_t declined_bytes();

// Write-side mirror of borrowed_bytes: bytes the store adopted outright
// (sqe_set_donate) instead of copying into a block of its own.
size_t donated_bytes();
size_t copied_bytes();

// Total bytes currently held by memory-backed files.  Tests assert this
// returns to zero after cleanup, which is how an unreleased intermediate gets
// caught.  Note it does NOT cover the reader's process-wide buffer pool (never
// freed by design) or process_inplace's per-worker staging buffers.
size_t resident_bytes();

// Drop every memory-backed file whose name carries `prefix`, mirroring what an
// example's cleanup_prefix loop does for real files.
void release_prefix(const std::string& prefix);

}  // namespace vio
}  // namespace plaid

// The primitive headers interleave global-scope code with `namespace plaid`
// blocks, so an unqualified `vio::` has to resolve in both.  Inside plaid the
// nested namespace is found directly; this alias covers the global regions.
namespace vio = plaid::vio;

#endif  // PLAID_VIO_H
