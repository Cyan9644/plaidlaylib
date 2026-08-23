// chunk_seq.h -- the chunk_seq data model and the I/O substrate every eager
// primitive is built on.
//
// Contents, in dependency order:
//   SimpleQueue                 bounded blocking queue for the I/O pipelines
//   UnorderedFileWriter<T>      the standardized async writer
//   chunk / chunk_seq           the data model + tabulate/iota/from_file/
//                               to_chunk_seq/consolidate/size
//   ChunkSequenceReader<T>      the standardized async reader
//   PersistentChunkSequenceReader<T>
//   ChunkEmitter / ExternalTransform / RemoveWorker    the unified engine
//   DensePack / DensePackStream the dense-output packer
//   NReader / NRemoveWorker     co-indexed lockstep read of N chunk_seqs
//   BucketWriter                per-worker staging -> sequential writev scatter
//
// A chunk_seq is a logical sequence stored out-of-core: chunks are packed at
// CHUNK_SIZE-aligned offsets (a multiple of O_DIRECT_MULTIPLE) so every read is
// O_DIRECT-aligned with no padding logic, and are spread across the SSD_COUNT
// drives (balls-in-bins) to saturate all drives in parallel.  Index-ordered
// invariant: chunks[i].index == i; every primitive returning a chunk_seq
// preserves it so callers can index by position.

#ifndef CHUNK_SEQ_H
#define CHUNK_SEQ_H

#include <fcntl.h>
#include <liburing.h>
#include <parlay/alloc.h>
#include <parlay/parallel.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "configs.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/utilities.h"
#include "utils/file_utils.h"

// ============================================================================
// SimpleQueue -- bounded blocking queue
//
// (was utils/simple_queue.h)
// ============================================================================

//
// Created by peter on 5/13/24.
//

enum class QueueCode { FINISH = 0, TIMEOUT = 1, SUCCESS = 2 };

template <typename T>
class SimpleQueue {
 public:
  explicit SimpleQueue(size_t size_limit = 0) : size_limit(size_limit) {}

  void Push(T data) {
    std::unique_lock<std::mutex> lock(mutex);
    while (size_limit && queue.size() >= size_limit) {
      writer_cond.wait(lock);
    }
    queue.push(data);
    reader_cond.notify_one();
  }

  std::pair<T, QueueCode> Poll(T default_result = nullptr,
                               int64_t timeout = -1) {
    std::unique_lock<std::mutex> lock(mutex);
    while (queue.empty()) {
      if (!open) {
        return {default_result, QueueCode::FINISH};
      }
      if (timeout == -1) {
        reader_cond.wait(lock);
      } else if (timeout == 0) {
        return {default_result, QueueCode::TIMEOUT};
      } else {
        auto result = reader_cond.wait_for(
            lock, std::chrono::duration(std::chrono::microseconds(timeout)));
        if (result == std::cv_status::timeout) {
          return {default_result, QueueCode::TIMEOUT};
        }
      }
    }
    T ret = queue.front();
    queue.pop();
    if (size_limit) {
      writer_cond.notify_one();
    }
    return {ret, QueueCode::SUCCESS};
  }

  void Close() {
    std::lock_guard<std::mutex> lock(mutex);
    open = false;
    reader_cond.notify_all();
  }

  // Re-open a queue that was previously Close()d so it can be reused.
  // Caller must ensure the queue is drained before calling.
  void Reopen() {
    std::lock_guard<std::mutex> lock(mutex);
    CHECK(queue.empty()) << "Reopen called with non-empty queue";
    open = true;
  }

  bool IsEmptyUnsafe() { return queue.empty(); }

  void SetSizeLimit(size_t new_limit) { size_limit = new_limit; }

  void Log(const std::string& message = "Queue size: ") {
    mutex.lock();
    size_t size = queue.size();
    mutex.unlock();
    LOG(INFO) << message << size;
  }

 private:
  std::queue<T> queue;
  std::mutex mutex;
  size_t size_limit;
  std::condition_variable reader_cond, writer_cond;
  bool open = true;
};

// ============================================================================
// UnorderedFileWriter<T> -- the standardized writer
//
// (was utils/unordered_file_writer.h)
// ============================================================================

//
// Created by peter on 3/2/24.
//

struct UnorderedWriterConfig {
  size_t io_uring_size = IO_URING_BUFFER_SIZE;
  size_t queue_size = 1000;
  size_t num_threads = 1;
  // Needed if only a file prefix is provided.
  // If the list of file names is supplied, this is ignored.
  size_t num_files = SSD_COUNT;
  bool allow_expand = false;

  UnorderedWriterConfig() = default;

  UnorderedWriterConfig(size_t io_uring_size, size_t queue_size,
                        size_t num_threads, size_t num_files)
      : io_uring_size(io_uring_size),
        queue_size(queue_size),
        num_threads(num_threads),
        num_files(num_files) {}
};

template <typename T>
class UnorderedFileWriter {
 private:
  struct WriteRequest;
  struct OpenedFile;

 public:
  UnorderedFileWriter() = default;

  explicit UnorderedFileWriter(
      const std::string& prefix,
      const UnorderedWriterConfig& config = UnorderedWriterConfig()) {
    Start(prefix, config);
  }

  void Start(const std::string& prefix,
             const UnorderedWriterConfig& config = UnorderedWriterConfig()) {
    file_name_prefix = prefix;
    std::vector<std::string> file_names;
    for (size_t i = 0; i < config.num_files; i++) {
      file_names.push_back(GetFileName(prefix, i));
    }
    Start(file_names, config);
  }

  void Start(const std::vector<std::string>& file_names,
             const UnorderedWriterConfig& config) {
    wait_queue.SetSizeLimit(config.queue_size);
    num_files = file_names.size();
    allow_expand = config.allow_expand;
    CHECK(num_files > 0);
    if (num_files < config.num_threads) {
      [[unlikely]] LOG(WARNING) << "Writing to " << num_files << " files with "
                                << config.num_threads << " threads. "
                                << "Some threads will not get a file.";
    }
    for (size_t i = 0; i < num_files; i++) {
      auto file = new OpenedFile(file_names[i]);
      global_files.push_back(file);
    }
    CHECK(config.num_threads > 0);
    for (size_t t = 0; t < config.num_threads; t++) {
      std::vector<OpenedFile*> file_list;
      for (size_t file_index = t; file_index < num_files;
           file_index += config.num_threads) {
        file_list.push_back(global_files[file_index]);
      }
      if (file_list.empty()) {
        continue;
      }
      worker_threads.push_back(std::make_unique<std::thread>(
          RunFileWriterWorker, this, file_list, config.io_uring_size));
    }
  }

  ~UnorderedFileWriter() { Wait(); }

  void Push(std::shared_ptr<T> data, size_t size, size_t file_index = -1,
            size_t file_offset = -1) {
    // FIXME: need to align writes to 512 byte blocks, otherwise we won't be
    // able to use O_DIRECT
    //   short term solution is to force multiples of 512 and throw an error
    //   otherwise; alternatively, use ftruncate to change the size of the file
    //   long term solution is to store the size of the last section in the end
    //   of the file (i.e. last 8 bytes)
    CHECK(size * sizeof(T) % O_DIRECT_MULTIPLE == 0)
        << "Size (in bytes) must be aligned to the size of a page in O_DIRECT "
           "mode. "
        << "Actual size: " << size * sizeof(T);
    CHECK((size_t)data.get() % O_DIRECT_MULTIPLE == 0)
        << "Buffers used by the UnorderedFileWriter must be aligned.";
    auto request =
        new WriteRequest(std::move(data), size, file_index, file_offset);
    wait_queue.Push(request);
  }

  WriteRequest* Poll() {
    static WriteRequest empty_request({nullptr}, 0);
    auto [result, _] = wait_queue.Poll(&empty_request);
    return result;
  }

  void Close() { wait_queue.Close(); }

  bool cleanup_done = false;

  /**
   * Block until file intermediate_writer finishes and return the number of
   * files
   * @return
   */
  size_t Wait() {
    if (cleanup_done) {
      return num_files;
    }
    cleanup_done = true;
    Close();
    for (auto& thread : worker_threads) {
      if (thread->joinable()) {
        thread->join();
      }
    }
    for (auto& f : global_files) {
      delete f;
    }
    return num_files;
  }

  bool AllowExpand() { return allow_expand; }

  void ExpandFiles(size_t new_num_files) {
    CHECK(this->num_files < new_num_files);
    while (this->global_files.size() < new_num_files) {
      global_files.push_back(
          new OpenedFile(GetFileName(file_name_prefix, global_files.size())));
    }
    num_files = global_files.size();
  }

 private:
  bool allow_expand = false;
  size_t num_files = 0;
  std::vector<std::unique_ptr<std::thread>> worker_threads;
  SimpleQueue<WriteRequest*> wait_queue;
  std::vector<OpenedFile*> global_files;
  std::string file_name_prefix;

  struct WriteRequest {
    std::shared_ptr<T> data;
    size_t size;
    size_t file_index = -1;
    size_t file_offset = -1;
    OpenedFile* file = nullptr;

    WriteRequest()
        : data(nullptr),
          size(0),
          file(nullptr),
          file_index(-1),
          file_offset(-1) {}

    WriteRequest(std::shared_ptr<T> data, size_t size)
        : data(std::move(data)), size(size) {}

    WriteRequest(std::shared_ptr<T> data, size_t size, size_t file_index,
                 size_t file_offset)
        : data(std::move(data)),
          size(size),
          file_index(file_index),
          file_offset(file_offset) {}
  };

  struct OpenedFile {
    int fd;
    size_t bytes_issued = 0;
    size_t bytes_written = 0;

    explicit OpenedFile(const std::string& name) {
      fd = open(name.c_str(), O_DIRECT | O_WRONLY | O_CREAT, 0644);
      SYSCALL(fd);
    }

    ~OpenedFile() { SYSCALL(close(fd)); }
  };

  enum Phase { NORMAL = 0, WAITING_FOR_COMPLETION = 1, ALL_DONE = 2 };

  static void RunFileWriterWorker(UnorderedFileWriter* writer,
                                  const std::vector<OpenedFile*> files,
                                  const size_t io_uring_size) {
    CHECK(!files.empty());
    struct io_uring ring;
    SYSCALL(
        InitIoUringWithRetry(io_uring_size, &ring, IORING_SETUP_SINGLE_ISSUER));

    size_t current_file = 0;
    size_t outstanding_request = 0;
    size_t sqe_unavailable_count = 0;
    size_t max_outstanding_requests = io_uring_size * 2;

    // FIXME: do we need to acquire a mutex for the second check?
    Phase phase = NORMAL;
    while (phase != ALL_DONE) {
      // reap io_uring result
      while (outstanding_request > 0) {
        struct io_uring_cqe* cqe;
        int wait_result = io_uring_peek_cqe(&ring, &cqe);
        if (wait_result != 0) {
          if (outstanding_request >= max_outstanding_requests ||
              phase == WAITING_FOR_COMPLETION) {
            wait_result = io_uring_wait_cqe(&ring, &cqe);
          }
        }
        if (wait_result == 0) {
          SYSCALL(cqe->res);
          auto* request = (WriteRequest*)io_uring_cqe_get_data(cqe);
          auto* file = request->file;
          file->bytes_written += request->size * sizeof(T);

          io_uring_cqe_seen(&ring, cqe);
          outstanding_request--;
          delete request;
        } else {
          break;
        }
      }
      if (phase >= WAITING_FOR_COMPLETION) {
        if (outstanding_request == 0) {
          phase = ALL_DONE;
        }
        continue;
      }
      bool submit_write = false;
      size_t requests_in_queue = 0;
      while (outstanding_request < max_outstanding_requests &&
             requests_in_queue < io_uring_size) {
        auto* request = writer->Poll();
        if (request->size == 0) {
          phase = WAITING_FOR_COMPLETION;
          break;
        }
        // submit this IO request to ring
        struct io_uring_sqe* sqe;
        while (true) {
          sqe = io_uring_get_sqe(&ring);
          if (sqe == nullptr) {
            [[unlikely]];
            sqe_unavailable_count++;
          } else {
            break;
          }
        }
        OpenedFile* file;
        if (request->file_index != (size_t)-1) {
          if (request->file_index >= writer->num_files) {
            [[unlikely]]
            if (writer->allow_expand) {
              writer->ExpandFiles(request->file_index + 1);
            } else {
              CHECK(request->file_index < writer->num_files);
            }
          }
          file = writer->global_files[request->file_index];
        } else {
          file = files[current_file];
          current_file = (current_file + 1) % files.size();
        }
        request->file = file;

        size_t num_bytes = request->size * sizeof(T);

        size_t offset = file->bytes_issued;
        if (request->file_offset != (size_t)-1) {
          offset = request->file_offset;
        }
        io_uring_prep_write(sqe, file->fd, request->data.get(), num_bytes,
                            offset);
        file->bytes_issued += num_bytes;
        io_uring_sqe_set_data(sqe, request);
        submit_write = true;
        outstanding_request++;
        requests_in_queue++;
      }
      if (submit_write) {
        SYSCALL(io_uring_submit(&ring));
      }
      requests_in_queue = 0;
    }
    if (sqe_unavailable_count > 0) {
      [[unlikely]];
      LOG(WARNING) << "io_uring sqe entires were unavailable "
                   << sqe_unavailable_count << " times. "
                   << "Consider expanding the ring buffer.";
    }
    io_uring_queue_exit(&ring);
  }
};

// ============================================================================
// chunk / chunk_seq -- the data model
//
// (was ChunkSequence/Primitives/chunk_seq.h)
// ============================================================================

// CHUNK_SIZE lives in configs.h (override with -DCHUNK_SIZE_BYTES=<n>).
constexpr size_t ELEMS_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);

struct chunk {
  std::string filename;  // the file that this chunk lives in
  size_t begin_addr;     // where in the file we should begin the read
  size_t used;   // how much of the prefix consists of data for this chunk
  size_t index;  // index of this chunk in the chunk_seq
};

// for now let's not use generics, assume we care about 64 bit integers or
// something
struct chunk_seq {
  // this vector is ordered by index
  std::vector<chunk> chunks;

  // Read all chunks in index order and write them contiguously to output_path.
  // Intended for local-filesystem output; reads from SSDs use O_DIRECT but the
  // output file is opened with ordinary (buffered) I/O so no alignment is
  // needed on the write side.
  void consolidate(const std::string& output_path) const {
    int out_fd = open(output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    // Fatal: every write below would go to fd -1 and the caller would read
    // back whatever happened to be at output_path already (a stale file from
    // an earlier run), which is worse than no output at all.
    CHECK(out_fd >= 0) << "consolidate: open(" << output_path
                       << ") failed: " << std::strerror(errno);

    // Process chunks in logical index order regardless of vector ordering.
    std::vector<const chunk*> ordered;
    ordered.reserve(chunks.size());
    for (const auto& c : chunks) ordered.push_back(&c);
    std::sort(
        ordered.begin(), ordered.end(),
        [](const chunk* a, const chunk* b) { return a->index < b->index; });

    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "consolidate: buffer allocation failed";

    // Cache fds so files referenced by multiple chunks are opened only once.
    std::map<std::string, int> fd_cache;
    for (const chunk* c : ordered) {
      if (c->used == 0) continue;
      auto [it, inserted] = fd_cache.emplace(c->filename, -1);
      if (inserted) {
        it->second = open(c->filename.c_str(), O_DIRECT | O_RDONLY);
        CHECK(it->second >= 0) << "consolidate: open(" << c->filename
                               << ") failed: " << std::strerror(errno)
                               << "; soft RLIMIT_NOFILE=" << SoftFdLimit();
      }
      const ssize_t got =
          pread(it->second, buf, AlignUp(c->used), (off_t)c->begin_addr);
      CHECK(got >= (ssize_t)c->used)
          << "consolidate: pread(" << c->filename << ", chunk " << c->index
          << ") returned " << got << " of " << c->used
          << " bytes: " << std::strerror(errno);
      const ssize_t put = write(out_fd, buf, c->used);
      CHECK(put == (ssize_t)c->used)
          << "consolidate: write(" << output_path << ") returned " << put
          << " of " << c->used << " bytes: " << std::strerror(errno);
    }

    free(buf);
    for (auto& [name, fd] : fd_cache) close(fd);
    close(out_fd);
  }
  const size_t headers_size() { return this->chunks.size(); }

  // Read the whole sequence into an in-DRAM std::vector<T> in index order.
  // Convenience for tests / small out-of-core results; assumes the sequence
  // fits in memory.  Chunks are read in parallel: element counts are prefix-
  // summed to give each chunk a fixed output slice, so workers scatter into
  // disjoint ranges of the result with no synchronization.
  //
  // Two things are hoisted OUT of the per-chunk parallel loop, because doing
  // them per chunk serializes everything and starves the drives (observed as a
  // ~5 GB/s, ~5% util, ~1.5% CPU phase — every worker blocked in the kernel):
  //   * one O_DIRECT-aligned scratch buffer PER WORKER, reused across its
  //     chunks, instead of aligned_alloc/free per chunk.  A 4 MB alloc goes
  //     through mmap/munmap, which take the process-wide mmap_sem *write* lock,
  //     so per-chunk alloc/free fully serializes the workers.
  //   * one shared read-only fd PER DISTINCT FILE, opened once up front
  //   (O_DIRECT
  //     reads carry an explicit offset, so a single fd is safe to share),
  //     instead of open()/close() per chunk.
  template <typename T = uint64_t>
  std::vector<T> to_vector() const {
    // Process chunks in logical index order regardless of vector ordering.
    // TODO: This is not necessary because we have the indexing invariant but
    // probably is fine anyways
    std::vector<const chunk*> ordered;
    ordered.reserve(chunks.size());
    for (const auto& c : chunks) ordered.push_back(&c);
    std::sort(
        ordered.begin(), ordered.end(),
        [](const chunk* a, const chunk* b) { return a->index < b->index; });

    // Prefix-sum element counts to place each chunk at a fixed output offset.
    std::vector<size_t> offset(ordered.size() + 1, 0);
    for (size_t i = 0; i < ordered.size(); i++) {
      CHECK(ordered[i]->used % sizeof(T) == 0)
          << "to_vector: chunk byte size not a multiple of sizeof(T)";
      offset[i + 1] = offset[i] + ordered[i]->used / sizeof(T);
    }

    std::vector<T> out(offset.back());
    if (ordered.empty()) return out;

    // Open each distinct file once, shared read-only across all workers.
    std::map<std::string, int> fds;
    for (const chunk* c : ordered)
      if (c->used && fds.find(c->filename) == fds.end()) {
        int fd = open(c->filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        fds[c->filename] = fd;
      }

    // One reusable aligned buffer per worker (lazily allocated on first use).
    const size_t W = std::max<size_t>(1, parlay::num_workers());
    std::vector<T*> wbuf(W, nullptr);

    parlay::parallel_for(
        0, ordered.size(),
        [&](size_t i) {
          const chunk* c = ordered[i];
          if (c->used == 0) return;
          const size_t w = parlay::worker_id();
          if (wbuf[w] == nullptr) {
            wbuf[w] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
            CHECK(wbuf[w] != nullptr) << "to_vector: buffer allocation failed";
          }
          // O_DIRECT needs an aligned buffer and an aligned read length; the
          // trailing padding beyond c->used is read but not copied out.
          SYSCALL(pread(fds.at(c->filename), wbuf[w], AlignUp(c->used),
                        (off_t)c->begin_addr));
          memcpy(out.data() + offset[i], wbuf[w], c->used);
        },
        /*granularity=*/1);

    for (T* b : wbuf)
      if (b) free(b);
    for (auto& [name, fd] : fds) close(fd);

    return out;
  }

  // ── scalar element access ────────────────────────────────────────────────
  // operator[] / push_back operate on a chunk_seq already materialized on
  // disk, one element at a time.  They rely on the index-ordered invariant
  // (chunks[k].index == k) so element i lives in chunks[i / ept] at byte
  // offset begin_addr + (i % ept) * sizeof(T), where ept = CHUNK_SIZE /
  // sizeof(T).

  // Read the single element at logical index i.  Reads one O_DIRECT-aligned
  // block (an 8-byte element never straddles a 4096 boundary since
  // O_DIRECT_MULTIPLE is a multiple of sizeof(T)).
  template <typename T = uint64_t>
  T operator[](size_t i) const {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
                  "sizeof(T) must divide CHUNK_SIZE");
    const size_t ept = CHUNK_SIZE / sizeof(T);
    const size_t ci = i / ept;
    const size_t off = (i % ept) * sizeof(T);  // byte offset within the chunk
    CHECK(ci < chunks.size()) << "operator[]: index " << i << " out of range";
    const chunk& c = chunks[ci];
    CHECK(off < c.used) << "operator[]: index " << i << " past end of chunk "
                        << ci;

    const size_t byte = c.begin_addr + off;
    const size_t block = AlignDown(byte);  // O_DIRECT-aligned start

    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
    CHECK(buf != nullptr) << "operator[]: buffer allocation failed";
    int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
    SYSCALL(fd);
    SYSCALL(pread(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
    T value;
    memcpy(&value, (char*)buf + (byte - block), sizeof(T));
    close(fd);
    free(buf);
    return value;
  }

  // Append one element to the end of the sequence, updating both the on-disk
  // data and the in-memory chunk_seq.  Requires a non-empty seq (there is no
  // filename to derive a first chunk from otherwise).
  template <typename T = uint64_t>
  void push_back(T value) {
    static_assert(CHUNK_SIZE % sizeof(T) == 0,
                  "sizeof(T) must divide CHUNK_SIZE");
    CHECK(!chunks.empty()) << "push_back: cannot push onto an empty chunk_seq";

    chunk& last = chunks.back();
    if (last.used < CHUNK_SIZE) {
      // Fast path: the last chunk has room.  Read-modify-write the single
      // aligned block holding the append position (the file was already
      // allocated to the slot's full CHUNK_SIZE).
      const size_t byte = last.begin_addr + last.used;
      const size_t block = AlignDown(byte);

      void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
      CHECK(buf != nullptr) << "push_back: buffer allocation failed";
      int fd = open(last.filename.c_str(), O_DIRECT | O_RDWR);
      SYSCALL(fd);
      SYSCALL(pread(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
      memcpy((char*)buf + (byte - block), &value, sizeof(T));
      SYSCALL(pwrite(fd, buf, O_DIRECT_MULTIPLE, (off_t)block));
      close(fd);
      free(buf);
      last.used += sizeof(T);
      return;
    }

    // Last chunk is full: allocate a new chunk.  Reuse the drive file already
    // present in the seq that holds the fewest chunks (balls-in-bins balance
    // without needing a prefix or all SSDs), and place it at the next slot.
    std::map<std::string, size_t> counts;
    for (const chunk& c : chunks) counts[c.filename]++;
    const std::string* target = nullptr;
    size_t best = SIZE_MAX;
    for (const auto& [name, cnt] : counts)
      if (cnt < best) {
        best = cnt;
        target = &name;
      }
    const std::string filename = *target;
    const size_t slot = best;  // dense per-file slot index
    const size_t begin_addr = slot * CHUNK_SIZE;

    // Grow the file to cover the new slot (matches tabulate's allocation).
    int fd = open(filename.c_str(), O_DIRECT | O_RDWR);
    SYSCALL(fd);
    const size_t file_size = (slot + 1) * CHUNK_SIZE;
    if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
      SYSCALL(ftruncate(fd, (off_t)file_size));

    void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, O_DIRECT_MULTIPLE);
    CHECK(buf != nullptr) << "push_back: buffer allocation failed";
    memset(buf, 0, O_DIRECT_MULTIPLE);
    memcpy(buf, &value, sizeof(T));
    SYSCALL(pwrite(fd, buf, O_DIRECT_MULTIPLE, (off_t)begin_addr));
    close(fd);
    free(buf);

    chunks.push_back({filename, begin_addr, sizeof(T), chunks.size()});
  }
};

namespace plaid {

/**
 * Build a chunk_seq by applying f to every index in [0, n).
 *
 * Output files are named result_prefix + drive_index on each SSD drive.
 * Each chunk holds a contiguous slice [start, start+count) of indices and is
 * randomly assigned to a drive for load balancing.  Within a drive file the
 * assigned chunks are packed at offsets 0, CHUNK_SIZE, 2*CHUNK_SIZE, …, so
 * every begin_addr is O_DIRECT-aligned.
 *
 * Writes go through UnorderedFileWriter (io_uring) into pre-fallocated files.
 * A queue of 64 in-flight buffers (256 MB) caps DRAM usage.
 */
template <typename T = uint64_t, typename F>
chunk_seq tabulate(size_t n, const std::string& result_prefix, F f,
                   size_t io_threads = 0) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_chunks = (n + ept - 1) / ept;
  const size_t num_drives = GetSSDList().size();
  // Number of io_uring writer threads (one ring each).  Defaults to one per
  // drive; a smaller count is used by callers running many tabulates
  // concurrently (e.g. sample_sort's per-bucket base cases) so the aggregate
  // ring count stays bounded rather than num_callers * num_drives.
  const size_t wthreads =
      (io_threads == 0) ? num_drives
                        : std::max<size_t>(1, std::min(io_threads, num_drives));

  // Randomly assign each chunk to a drive for balanced SSD utilization.
  std::vector<size_t> drive_of(num_chunks);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
  }

  // Group chunk indices by drive; insertion order gives each chunk its slot
  // index (position) within that drive's file.
  std::vector<std::vector<size_t>> drive_chunks(num_drives);
  for (size_t i = 0; i < num_chunks; i++)
    drive_chunks[drive_of[i]].push_back(i);

  // Precompute each chunk's slot so the parallel write loop can look it up
  // without touching drive_chunks again (avoids false sharing on the inner
  // vectors while many parlay workers are running).
  std::vector<size_t> slot_of(num_chunks);
  for (size_t d = 0; d < num_drives; d++)
    for (size_t s = 0; s < drive_chunks[d].size(); s++)
      slot_of[drive_chunks[d][s]] = s;

  // Build filenames and pre-allocate each drive file to its exact final
  // size so io_uring can write to arbitrary slot offsets immediately.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        const size_t file_size = drive_chunks[d].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        // fallocate guarantees contiguous allocation; fall back to ftruncate
        // on filesystems that don't support it (e.g. tmpfs).
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
          SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  // Build the output chunk descriptors (all metadata is known before any
  // data is written).
  std::vector<chunk> chunks(num_chunks);
  for (size_t i = 0; i < num_chunks; i++) {
    const size_t start = i * ept;
    const size_t count = std::min(ept, n - start);
    chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                 count * sizeof(T), i};
  }

  // One io_uring writer thread per drive; queue_size limits in-flight
  // buffers to 64 * 4 MB = 256 MB of DRAM at any moment.
  UnorderedWriterConfig wcfg;
  wcfg.num_threads = wthreads;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<T> writer;
  writer.Start(filenames, wcfg);

  // Stream chunks to the writer: a fixed pool of generator threads each walks
  // a strided slice of chunk indices, generating ONE chunk buffer at a time
  // and handing it off before producing the next.  Peak DRAM is therefore
  // bounded by (num_gen_threads + queue_size) * CHUNK_SIZE regardless of how
  // large n is — we never materialize all chunks at once.  SimpleQueue::Push
  // blocks when the writer queue is full, so generators throttle to write
  // throughput instead of racing ahead and exhausting memory.
  const size_t num_gen_threads =
      std::min((size_t)parlay::num_workers(), num_chunks);
  parlay::parallel_for(
      0, num_gen_threads,
      [&](size_t t) {
        for (size_t i = t; i < num_chunks; i += num_gen_threads) {
          const size_t start = i * ept;
          const size_t count = std::min(ept, n - start);

          T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
          CHECK(buf != nullptr) << "tabulate: buffer allocation failed";
          for (size_t j = 0; j < count; j++) buf[j] = f(start + j);
          if (count < ept) memset(buf + count, 0, (ept - count) * sizeof(T));

          // Hand ownership of this buffer to the writer (freed once its write
          // completes) and move on to the next chunk in this thread's slice.
          writer.Push(std::shared_ptr<T>(buf, free), CHUNK_SIZE / sizeof(T),
                      drive_of[i], slot_of[i] * CHUNK_SIZE);
        }
      },
      /*granularity=*/1);

  writer.Wait();
  return {chunks};
}

chunk_seq iota(size_t n) {
  return tabulate<uint64_t>(n, "iota", [](size_t i) { return (uint64_t)i; });
}

/**
 * Read a single local file into an out-of-core chunk_seq.  This is the inverse
 * of chunk_seq::consolidate(): consolidate() streams a chunk_seq's chunks in
 * index order into one contiguous file; from_file() slices one contiguous file
 * back into CHUNK_SIZE chunks spread balls-in-bins across the SSD drives.
 *
 * The input is treated as a flat array of T (its byte length must be a whole
 * number of elements).  The read side uses ordinary buffered I/O (no O_DIRECT):
 * like consolidate's output file it needs no alignment, and pread with an
 * explicit offset is safe to issue from many threads on one shared fd, so a
 * pool of generator threads reads strided chunk slices in parallel while the
 * io_uring writer pool (one ring per drive) lands them on the SSDs.  Peak DRAM
 * stays bounded by (num_gen_threads + queue_size) * CHUNK_SIZE regardless of
 * how large the input is, so the file may exceed DRAM.
 */
template <typename T = uint64_t>
chunk_seq from_file(const std::string& input_path,
                    const std::string& result_prefix = "chunkseq") {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");

  int in_fd = open(input_path.c_str(), O_RDONLY);
  SYSCALL(in_fd);
  struct stat st;
  SYSCALL(fstat(in_fd, &st));
  const size_t nbytes = (size_t)st.st_size;
  CHECK(nbytes % sizeof(T) == 0)
      << "from_file: file size " << nbytes << " not a multiple of sizeof(T)";
  const size_t n = nbytes / sizeof(T);

  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_chunks = (n + ept - 1) / ept;
  if (num_chunks == 0) {
    close(in_fd);
    return {};
  }
  const size_t num_drives = GetSSDList().size();

  // Balls-in-bins drive assignment + per-drive slot index, exactly as tabulate.
  std::vector<size_t> drive_of(num_chunks);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
  }
  std::vector<std::vector<size_t>> drive_chunks(num_drives);
  for (size_t i = 0; i < num_chunks; i++)
    drive_chunks[drive_of[i]].push_back(i);
  std::vector<size_t> slot_of(num_chunks);
  for (size_t d = 0; d < num_drives; d++)
    for (size_t s = 0; s < drive_chunks[d].size(); s++)
      slot_of[drive_chunks[d][s]] = s;

  // Create + size each destination drive file so io_uring can write any slot.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        const size_t file_size = drive_chunks[d].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
          SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  // Output descriptors: fresh file + slot offset; last chunk may be partial.
  std::vector<chunk> chunks(num_chunks);
  for (size_t i = 0; i < num_chunks; i++) {
    const size_t start = i * ept;
    const size_t count = std::min(ept, n - start);
    chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                 count * sizeof(T), i};
  }

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<T> writer;
  writer.Start(filenames, wcfg);

  // Generator pool: each thread walks a strided slice of chunk indices,
  // preads that chunk's bytes from the input file (explicit offset, so the
  // shared fd is safe across threads), zero-pads a partial tail, and hands the
  // buffer to the writer.  Push blocks when the writer queue is full, so the
  // readers throttle to write throughput.
  const size_t num_gen_threads =
      std::min((size_t)parlay::num_workers(), num_chunks);
  parlay::parallel_for(
      0, num_gen_threads,
      [&](size_t t) {
        for (size_t i = t; i < num_chunks; i += num_gen_threads) {
          const size_t start = i * ept;
          const size_t count = std::min(ept, n - start);
          const size_t bytes = count * sizeof(T);

          T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
          CHECK(buf != nullptr) << "from_file: buffer allocation failed";
          size_t got = 0;
          while (got < bytes) {
            ssize_t r = pread(in_fd, (char*)buf + got, bytes - got,
                              (off_t)(start * sizeof(T) + got));
            SYSCALL(r);
            CHECK(r > 0) << "from_file: unexpected EOF reading " << input_path;
            got += (size_t)r;
          }
          if (bytes < CHUNK_SIZE)
            memset((char*)buf + bytes, 0, CHUNK_SIZE - bytes);

          writer.Push(std::shared_ptr<T>(buf, free), CHUNK_SIZE / sizeof(T),
                      drive_of[i], slot_of[i] * CHUNK_SIZE);
        }
      },
      /*granularity=*/1);

  writer.Wait();
  close(in_fd);
  return {chunks};
}

/**
 * Convert an in-DRAM parlay::sequence (or any random-access range exposing
 * value_type, size() and operator[]) into an out-of-core chunk_seq, preserving
 * index order.  This is the inverse of materialize() / to_vector().
 *
 * Implemented on top of tabulate, so it reuses the same parallel io_uring
 * writer pipeline (one thread per drive, bounded in-flight DRAM).  The whole
 * input already lives in DRAM, so the per-element generator is a cheap indexed
 * read and the cost is dominated by the parallel SSD writes.
 */
template <typename Range>
chunk_seq to_chunk_seq(const Range& seq,
                       const std::string& result_prefix = "chunkseq",
                       size_t io_threads = 0) {
  using T = typename Range::value_type;
  return tabulate<T>(
      seq.size(), result_prefix, [&seq](size_t i) { return seq[i]; },
      io_threads);
}

/**
 * Sequential tabulate: the same output as tabulate(), built with blocking
 * O_DIRECT pwrites on the calling thread instead of the io_uring writer pool.
 *
 * Meant to be called from *inside* a parlay::parallel_for, where the caller
 * already supplies the parallelism (e.g. one call per bucket in
 * random_shuffle / sample_sort).  The eager tabulate would there spin up an
 * UnorderedFileWriter -- one io_uring ring per drive -- and a nested
 * parallel_for per call, so B concurrent calls mean B * num_drives rings
 * (RLIMIT_MEMLOCK pressure) and B nested parallel regions, all to write a
 * bucket small enough to fit in DRAM.  This version allocates one CHUNK_SIZE
 * bounce buffer, opens each destination drive file once, and writes its slots
 * back to back.
 */
template <typename T = uint64_t, typename F>
chunk_seq sequential_tabulate(size_t n, const std::string& result_prefix, F f) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE for O_DIRECT alignment");
  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_chunks = (n + ept - 1) / ept;
  if (num_chunks == 0) return {};
  const size_t num_drives = GetSSDList().size();

  // Balls-in-bins drive assignment, then group by drive so each drive file is
  // opened, sized, and filled exactly once.
  std::vector<size_t> drive_of(num_chunks);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
  }
  std::vector<std::vector<size_t>> drive_chunks(num_drives);
  for (size_t i = 0; i < num_chunks; i++)
    drive_chunks[drive_of[i]].push_back(i);

  std::vector<chunk> chunks(num_chunks);
  T* buf = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
  CHECK(buf != nullptr) << "sequential_tabulate: buffer allocation failed";

  for (size_t d = 0; d < num_drives; d++) {
    const std::vector<size_t>& mine = drive_chunks[d];
    if (mine.empty()) continue;

    const std::string filename = GetFileName(result_prefix, d);
    int fd =
        open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    SYSCALL(fd);
    const size_t file_size = mine.size() * CHUNK_SIZE;
    if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
      SYSCALL(ftruncate(fd, (off_t)file_size));

    // Slot s of this drive's file holds chunk mine[s], so every begin_addr is
    // CHUNK_SIZE-aligned exactly as in tabulate.
    for (size_t s = 0; s < mine.size(); s++) {
      const size_t i = mine[s];
      const size_t start = i * ept;
      const size_t count = std::min(ept, n - start);
      for (size_t j = 0; j < count; j++) buf[j] = f(start + j);
      if (count < ept) memset(buf + count, 0, (ept - count) * sizeof(T));
      SYSCALL(pwrite(fd, buf, CHUNK_SIZE, (off_t)(s * CHUNK_SIZE)));
      chunks[i] = {filename, s * CHUNK_SIZE, count * sizeof(T), i};
    }
    close(fd);
  }

  free(buf);
  return {chunks};
}

/**
 * Convert an in-DRAM range to a chunk_seq without the io_uring writer pool: the
 * to_chunk_seq counterpart of sequential_tabulate, and the inverse of
 * sequential_materialize.  Use inside a parallel_for; use to_chunk_seq when the
 * call is the only thing running.
 */
template <typename Range>
chunk_seq sequential_to_chunk_seq(
    const Range& seq, const std::string& result_prefix = "chunkseq") {
  using T = typename Range::value_type;
  return sequential_tabulate<T>(seq.size(), result_prefix,
                                [&seq](size_t i) { return seq[i]; });
}

// Total number of *elements* (not chunks) in the sequence.  O(1): every chunk
// but the last is full.  (Single-element access lives on the struct itself:
// chunk_seq::operator[] and chunk_seq::push_back.)
template <typename T = uint64_t>
size_t size(const chunk_seq& seq) {
  static_assert(CHUNK_SIZE % sizeof(T) == 0,
                "sizeof(T) must divide CHUNK_SIZE");
  if (seq.chunks.empty()) return 0;
  const size_t ept = CHUNK_SIZE / sizeof(T);
  return (seq.chunks.size() - 1) * ept + seq.chunks.back().used / sizeof(T);
}

// Materialize an index-ordered header list into a *fresh, independent* on-disk
// chunk_seq.  Callers that build chunk headers by hand (e.g. cut) may hand us
// headers that reference another sequence's files -- interior chunks shared by
// reference, seam chunks in "<file>_cut" scratch files.  Rather than alias
// those files, this copies every chunk's bytes into new files (named
// result_prefix + drive_index) spread balls-in-bins across the drives and
// packed at CHUNK_SIZE slots, so the returned sequence owns all its data and
// outlives the input. The per-chunk `used` and `index` are preserved (the head
// chunk may be partial, so this does NOT re-densify and does NOT verify the
// every-chunk-but-last-full invariant).  Reads and writes are O_DIRECT.
inline chunk_seq from_chunks(const parlay::sequence<chunk>& headers,
                             const std::string& result_prefix = "cut_out") {
  const size_t num_chunks = headers.size();
  if (num_chunks == 0) return {};
  const size_t num_drives = GetSSDList().size();

  // Assign each output chunk to a drive (balls-in-bins) and its slot within
  // that drive's file, exactly like tabulate.
  std::vector<size_t> drive_of(num_chunks);
  {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, num_drives - 1);
    for (size_t i = 0; i < num_chunks; i++) drive_of[i] = dist(rng);
  }
  std::vector<std::vector<size_t>> drive_chunks(num_drives);
  for (size_t i = 0; i < num_chunks; i++)
    drive_chunks[drive_of[i]].push_back(i);
  std::vector<size_t> slot_of(num_chunks);
  for (size_t d = 0; d < num_drives; d++)
    for (size_t s = 0; s < drive_chunks[d].size(); s++)
      slot_of[drive_chunks[d][s]] = s;

  // Create + size each destination drive file so O_DIRECT writes can land at
  // any slot offset immediately.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        const size_t file_size = drive_chunks[d].size() * CHUNK_SIZE;
        if (file_size == 0) return;
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        if (fallocate(fd, 0, 0, (off_t)file_size) != 0)
          SYSCALL(ftruncate(fd, (off_t)file_size));
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  // New descriptors: fresh file + slot offset, preserved `used` and `index`.
  std::vector<chunk> chunks(num_chunks);
  for (size_t i = 0; i < num_chunks; i++)
    chunks[i] = {filenames[drive_of[i]], slot_of[i] * CHUNK_SIZE,
                 headers[i].used, headers[i].index};

  // Copy each source chunk's bytes into its fresh slot, in parallel.  Read
  // AlignUp(used) into a CHUNK_SIZE buffer (zero-pad the tail so the on-disk
  // block is deterministic) and write a full O_DIRECT-aligned CHUNK_SIZE block.
  parlay::parallel_for(
      0, num_chunks,
      [&](size_t i) {
        const chunk& src = headers[i];
        void* buf = aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
        CHECK(buf != nullptr) << "from_chunks: buffer allocation failed";
        int rfd = open(src.filename.c_str(), O_RDONLY | O_DIRECT);
        SYSCALL(rfd);
        SYSCALL(pread(rfd, buf, AlignUp(src.used), (off_t)src.begin_addr));
        close(rfd);
        if (src.used < CHUNK_SIZE)
          memset((char*)buf + src.used, 0, CHUNK_SIZE - src.used);
        int wfd = open(filenames[drive_of[i]].c_str(), O_WRONLY | O_DIRECT);
        SYSCALL(wfd);
        SYSCALL(pwrite(wfd, buf, CHUNK_SIZE, (off_t)(slot_of[i] * CHUNK_SIZE)));
        close(wfd);
        free(buf);
      },
      /*granularity=*/1);

  return {chunks};
}

}  // namespace plaid

// ============================================================================
// ChunkSequenceReader<T> -- the standardized reader
//
// (was ChunkSequence/Primitives/chunk_seq_reader.h)
// ============================================================================

/**
 * Read a chunk_seq chunk-by-chunk using io_uring.
 *
 * Unlike UnorderedFileReader (which reads whole files), this reader submits
 * one io_uring read per chunk, reading exactly chunk.used bytes at
 * chunk.begin_addr within chunk.filename.  Chunks whose filename appears
 * multiple times share a cached file descriptor within each worker thread.
 *
 * Poll() returns (ptr, n_elements, chunk.index) tuples in completion order.
 * Callers must call allocator.Free(ptr) when done with a buffer.
 *
 * @tparam T Element type stored in each chunk.
 */
template <typename T>
class ChunkSequenceReader {
 public:
  // ptr to data buffer, number of T elements in the buffer, chunk index
  using BufferData = std::tuple<T*, size_t, size_t>;

  // Pool of CHUNK_SIZE buffers reused across reads.  The pool is a
  // PROCESS-WIDE, per-element-type singleton (all Allocator instances of a
  // given T share it), not per-reader: primitives like
  // ChunkFilter/ChunkPartition and the deep quickhull recursion create many
  // short-lived readers, and a per-reader pool would re-allocate (and fault in)
  // 50*CHUNK_SIZE = 200 MB on every one of them. The shared pool grows to peak
  // concurrent demand once and is then reused; its backing is intentionally
  // never freed (process-lifetime, reclaimed at exit) so buffers still in
  // flight in one reader are never pulled out from under it when another reader
  // is destroyed.  Alloc/Free are the same lock-guarded free-list ops as
  // before, just over the shared state.
  struct Allocator {
    static constexpr size_t BUFFER_SIZE = CHUNK_SIZE;
    static constexpr size_t INITIAL_COUNT = 50;
    static constexpr size_t ALLOC_BATCH = 50;
    static constexpr size_t ALLOC_THRESHOLD = 10;

    // Shared pool state (Meyers singletons: constructed on first use, correct
    // init order, one instance per T across the whole process).
    static std::vector<T*>& free_list() {
      static std::vector<T*> v;
      return v;
    }
    static std::mutex& free_list_lock() {
      static std::mutex m;
      return m;
    }
    static std::mutex& alloc_lock() {
      static std::mutex m;
      return m;
    }

    Allocator() {
      // Prime the pool once; the threshold guard makes later readers no-ops.
      AllocateMore(INITIAL_COUNT);
    }

    // No destructor: the shared backing outlives every reader and is reclaimed
    // by the OS at process exit (see the class comment above).

    void AllocateMore(size_t n) {
      std::lock_guard<std::mutex> lg(alloc_lock());
      {
        std::lock_guard<std::mutex> fl(free_list_lock());
        if (free_list().size() > ALLOC_THRESHOLD) return;
      }
      T* base =
          (T*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, BUFFER_SIZE * n);
      CHECK(base != nullptr) << "ChunkSequenceReader: out of memory";
      std::lock_guard<std::mutex> fl(free_list_lock());
      for (size_t i = 0; i < n; i++)
        free_list().push_back((T*)((intptr_t)base + i * BUFFER_SIZE));
    }

    T* Alloc() {
      while (true) {
        std::unique_lock<std::mutex> l(free_list_lock());
        if (!free_list().empty()) {
          T* p = free_list().back();
          free_list().pop_back();
          return p;
        }
        l.unlock();
        AllocateMore(ALLOC_BATCH);
      }
    }

    void Free(T* p) {
      std::lock_guard<std::mutex> l(free_list_lock());
      free_list().push_back(p);
    }
  };

  Allocator allocator;

  ChunkSequenceReader() = default;

  ~ChunkSequenceReader() {
    is_open = false;
    Wait();
    // Workers have all joined; no further reads can reference these fds.
    for (auto& [name, fd] : shared_fds) close(fd);
    shared_fds.clear();
  }

  void PrepChunks(const chunk_seq& seq) { chunks = seq.chunks; }

  /**
   * @param num_threads   Number of reader threads (each gets chunks
   * round-robin).
   * @param queue_depth   io_uring queue depth per thread.
   * @param max_requests  Max in-flight reads per thread.
   * @param buf_queue_sz  Max entries in the output buffer queue
   * (back-pressure).
   */
  void Start(size_t num_threads = 2, size_t queue_depth = 32,
             size_t max_requests = 16, size_t buf_queue_sz = 512) {
    CHECK(num_threads > 0);
    buffer_queue.SetSizeLimit(buf_queue_sz);
    active_threads = (int)num_threads;

    // Open one shared read-only fd per distinct file, once.  io_uring reads
    // pass an explicit offset (no shared file position), so a single fd is
    // safe to share across all worker threads — and this keeps the open-fd
    // count at O(distinct files) instead of O(num_threads * distinct files),
    // which is what caused EMFILE under highly-parallel recursive sorts.
    for (const chunk& c : chunks) {
      if (shared_fds.find(c.filename) == shared_fds.end()) {
        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        shared_fds[c.filename] = fd;
      }
    }

    for (size_t t = 0; t < num_threads; t++) {
      std::vector<chunk> work;
      for (size_t i = t; i < chunks.size(); i += num_threads)
        work.push_back(chunks[i]);
      worker_threads.push_back(std::make_unique<std::thread>(
          Worker, this, std::move(work), queue_depth, max_requests));
    }
  }

  /**
   * Get the next completed chunk.  Blocks until one is available or the
   * reader is exhausted.  Returns (nullptr, 0, 0) when done.
   */
  BufferData Poll() {
    static BufferData nil{nullptr, 0, 0};
    return buffer_queue.Poll(nil).first;
  }

  void Close() { buffer_queue.Close(); }

  void Wait() {
    for (auto& t : worker_threads)
      if (t->joinable()) t->join();
  }

 private:
  bool is_open = true;
  std::atomic<int> active_threads = 0;
  std::vector<chunk> chunks;
  std::vector<std::unique_ptr<std::thread>> worker_threads;
  SimpleQueue<BufferData> buffer_queue;
  // One read-only fd per distinct file, opened in Start(), shared across all
  // workers, closed in the destructor.  Not mutated after Start().
  std::map<std::string, int> shared_fds;

  struct ReadRequest {
    T* data;
    size_t chunk_index;
    size_t used_bytes;  // actual data bytes in this chunk (may be < CHUNK_SIZE)
  };

  static void Worker(ChunkSequenceReader* self, std::vector<chunk> work,
                     size_t queue_depth, size_t max_requests) {
    struct io_uring ring;
    SYSCALL(
        InitIoUringWithRetry(queue_depth, &ring, IORING_SETUP_SINGLE_ISSUER));

    // fds are opened once in Start() and shared read-only across workers;
    // the map is not mutated after Start(), so lookups here need no lock.
    auto get_fd = [&](const std::string& name) -> int {
      return self->shared_fds.at(name);
    };

    auto* pool = (ReadRequest*)malloc(max_requests * sizeof(ReadRequest));
    std::vector<ReadRequest*> free_pool;
    free_pool.reserve(max_requests);
    for (size_t i = 0; i < max_requests; i++) free_pool.push_back(pool + i);

    std::deque<chunk> pending(work.begin(), work.end());
    size_t outstanding = 0;
    size_t completed = 0;
    const size_t total = work.size();

    while ((completed < total) && self->is_open) {
      // Non-blocking reap of completed reads.
      while (outstanding > 0) {
        struct io_uring_cqe* cqe;
        if (io_uring_peek_cqe(&ring, &cqe) != 0) break;
        SYSCALL(cqe->res);
        auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
        self->buffer_queue.Push(
            {req->data, req->used_bytes / sizeof(T), req->chunk_index});
        free_pool.push_back(req);
        outstanding--;
        completed++;
        io_uring_cqe_seen(&ring, cqe);
      }

      // Submit new reads while we have capacity and pending chunks.
      bool submitted = false;
      while (!free_pool.empty() && !pending.empty() &&
             outstanding < max_requests) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        if (sqe == nullptr) break;

        const chunk c =
            pending.front();  // copy before pop to avoid dangling ref
        pending.pop_front();

        auto* req = free_pool.back();
        free_pool.pop_back();
        req->data = self->allocator.Alloc();
        req->chunk_index = c.index;
        req->used_bytes = c.used;

        // O_DIRECT requires the read size to be page-aligned.
        size_t read_size = AlignUp(c.used);
        io_uring_prep_read(sqe, get_fd(c.filename), req->data, read_size,
                           c.begin_addr);
        io_uring_sqe_set_data(sqe, req);
        outstanding++;
        submitted = true;
      }

      if (submitted) SYSCALL(io_uring_submit(&ring));

      // If the ring is full and there's nothing more to submit, wait
      // for at least one completion before looping.
      if (outstanding > 0 && (pending.empty() || free_pool.empty()) &&
          !submitted) {
        struct io_uring_cqe* cqe;
        SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        SYSCALL(cqe->res);
        auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
        self->buffer_queue.Push(
            {req->data, req->used_bytes / sizeof(T), req->chunk_index});
        free_pool.push_back(req);
        outstanding--;
        completed++;
        io_uring_cqe_seen(&ring, cqe);
      }
    }

    io_uring_queue_exit(&ring);
    free(pool);
    // Shared fds are closed once in ~ChunkSequenceReader, not per worker.

    self->active_threads--;
    if (self->active_threads == 0) self->Close();
  }
};

/**
 * Long-lived counterpart to ChunkSequenceReader<T>: opens fds and spawns
 * worker threads (each initializing its io_uring ring) ONCE, then replays
 * the SAME round-robin chunk list across many "rounds" instead of tearing
 * everything down after a single pass.
 *
 * Built for fixed-point-iteration callers (e.g. external_bellman_ford_fast,
 * ChunkSequence/examples/chunk_bellman_ford.h) that re-read an
 * unchanged chunk_seq many times: on WSL2, io_uring ring teardown frees its
 * RLIMIT_MEMLOCK charge asynchronously (see InitIoUringWithRetry in
 * utils/file_utils.h), so tearing down and rebuilding rings every round can
 * burn real wall-clock time retrying -ENOMEM. Keeping the rings (and worker
 * threads, and open fds) alive across rounds avoids that entirely.
 *
 * Does NOT support a different chunk list per round -- every StartRound()
 * replays the exact list fixed at Start(). A caller whose read plan changes
 * between calls must use ChunkSequenceReader instead. At most one round may
 * be in flight at a time (StartRound() must not be called again until the
 * previous round's TotalReads() completions have all been drained via
 * Poll()); callers driving this synchronously (one round fully finishes
 * before the next starts) satisfy this trivially.
 *
 * @tparam T Element type stored in each chunk.
 */
template <typename T>
class PersistentChunkSequenceReader {
 public:
  using BufferData = typename ChunkSequenceReader<T>::BufferData;

  // Same process-wide, per-T singleton pool as
  // ChunkSequenceReader<T>::Allocator (see that class's comment) -- already
  // persistent across reader instances, so no new persistence design is needed
  // for buffers specifically.
  typename ChunkSequenceReader<T>::Allocator allocator;

  PersistentChunkSequenceReader() = default;
  PersistentChunkSequenceReader(const PersistentChunkSequenceReader&) = delete;
  PersistentChunkSequenceReader& operator=(
      const PersistentChunkSequenceReader&) = delete;

  ~PersistentChunkSequenceReader() { Shutdown(); }

  /**
   * Called ONCE. Opens one shared fd per distinct file, fixes each
   * thread's round-robin chunk sub-list, and spawns num_threads persistent
   * worker threads -- each inits its io_uring ring here, then blocks
   * waiting for the first StartRound().
   */
  void Start(const chunk_seq& seq, size_t num_threads = 10,
             size_t queue_depth = 32, size_t max_requests = 16,
             size_t buf_queue_sz = 128) {
    CHECK(num_threads > 0);
    buffer_queue.SetSizeLimit(buf_queue_sz);
    total_reads = seq.chunks.size();

    // Same fd-sharing scheme as ChunkSequenceReader::Start: one read-only
    // fd per distinct file, shared read-only across all worker threads.
    for (const chunk& c : seq.chunks) {
      if (shared_fds.find(c.filename) == shared_fds.end()) {
        int fd = open(c.filename.c_str(), O_DIRECT | O_RDONLY);
        SYSCALL(fd);
        shared_fds[c.filename] = fd;
      }
    }

    for (size_t t = 0; t < num_threads; t++) {
      std::vector<chunk> work;
      for (size_t i = t; i < seq.chunks.size(); i += num_threads)
        work.push_back(seq.chunks[i]);
      worker_threads.push_back(std::make_unique<std::thread>(
          Worker, this, std::move(work), queue_depth, max_requests));
    }
  }

  // Total reads issued by one full round (sum of every thread's fixed work
  // list); callers poll exactly this many BufferData per round.
  size_t TotalReads() const { return total_reads; }

  // Begin a new round: bump the round generation and wake every worker to
  // replay its fixed chunk list. O(1), no I/O, no thread/ring churn.
  void StartRound() {
    {
      std::lock_guard<std::mutex> l(mu);
      generation++;
    }
    cv.notify_all();
  }

  /**
   * Get the next completed chunk of the current round. Blocks until one is
   * available or Shutdown() has been called. Returns (nullptr, 0, 0) once
   * shut down.
   */
  BufferData Poll() {
    static BufferData nil{nullptr, 0, 0};
    return buffer_queue.Poll(nil).first;
  }

  // Joins all worker threads (each exits its wait loop, tears down its
  // ring, and returns) and closes shared fds. Idempotent; also invoked by
  // the destructor.
  void Shutdown() {
    {
      std::lock_guard<std::mutex> l(mu);
      if (shutting_down) return;
      shutting_down = true;
    }
    cv.notify_all();
    for (auto& t : worker_threads)
      if (t->joinable()) t->join();
    for (auto& [name, fd] : shared_fds) close(fd);
    shared_fds.clear();
  }

 private:
  std::mutex mu;
  std::condition_variable cv;
  uint64_t generation = 0;
  bool shutting_down = false;

  std::vector<std::unique_ptr<std::thread>> worker_threads;
  // Never Close()d/Reopen()d between rounds -- a round boundary is purely
  // "the caller has drained TotalReads() completions", not a queue state
  // transition. Only Shutdown() ever closes it (via thread exit, mirroring
  // ChunkSequenceReader's active_threads-reaches-zero -> Close() pattern
  // is unnecessary here since callers stop polling once they've drained
  // TotalReads() themselves).
  SimpleQueue<BufferData> buffer_queue;
  // One read-only fd per distinct file, opened once in Start(), shared
  // across all workers and all rounds, closed once in Shutdown().
  std::map<std::string, int> shared_fds;
  size_t total_reads = 0;

  struct ReadRequest {
    T* data;
    size_t chunk_index;
    size_t used_bytes;  // actual data bytes in this chunk (may be < CHUNK_SIZE)
  };

  static void Worker(PersistentChunkSequenceReader* self,
                     std::vector<chunk> work, size_t queue_depth,
                     size_t max_requests) {
    struct io_uring ring;
    SYSCALL(
        InitIoUringWithRetry(queue_depth, &ring, IORING_SETUP_SINGLE_ISSUER));

    // fds are opened once in Start() and shared read-only across workers
    // and rounds; the map is not mutated after Start(), so lookups here
    // need no lock.
    auto get_fd = [&](const std::string& name) -> int {
      return self->shared_fds.at(name);
    };

    auto* pool = (ReadRequest*)malloc(max_requests * sizeof(ReadRequest));
    CHECK(pool != nullptr)
        << "PersistentChunkSequenceReader: allocation failed";

    uint64_t local_gen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> l(self->mu);
        self->cv.wait(l, [&] {
          return self->generation > local_gen || self->shutting_down;
        });
        if (self->shutting_down) break;
        local_gen = self->generation;
      }

      // One full pass over this thread's fixed work list -- same
      // submit/reap logic as ChunkSequenceReader::Worker, replayed
      // fresh every round against the same, already-open ring.
      std::vector<ReadRequest*> free_pool;
      free_pool.reserve(max_requests);
      for (size_t i = 0; i < max_requests; i++) free_pool.push_back(pool + i);

      std::deque<chunk> pending(work.begin(), work.end());
      size_t outstanding = 0;
      size_t completed = 0;
      const size_t total = work.size();

      while (completed < total) {
        // Non-blocking reap of completed reads.
        while (outstanding > 0) {
          struct io_uring_cqe* cqe;
          if (io_uring_peek_cqe(&ring, &cqe) != 0) break;
          SYSCALL(cqe->res);
          auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
          self->buffer_queue.Push(
              {req->data, req->used_bytes / sizeof(T), req->chunk_index});
          free_pool.push_back(req);
          outstanding--;
          completed++;
          io_uring_cqe_seen(&ring, cqe);
        }

        // Submit new reads while we have capacity and pending chunks.
        bool submitted = false;
        while (!free_pool.empty() && !pending.empty() &&
               outstanding < max_requests) {
          struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
          if (sqe == nullptr) break;

          const chunk c =
              pending.front();  // copy before pop to avoid dangling ref
          pending.pop_front();

          auto* req = free_pool.back();
          free_pool.pop_back();
          req->data = self->allocator.Alloc();
          req->chunk_index = c.index;
          req->used_bytes = c.used;

          // O_DIRECT requires the read size to be page-aligned.
          size_t read_size = AlignUp(c.used);
          io_uring_prep_read(sqe, get_fd(c.filename), req->data, read_size,
                             c.begin_addr);
          io_uring_sqe_set_data(sqe, req);
          outstanding++;
          submitted = true;
        }

        if (submitted) SYSCALL(io_uring_submit(&ring));

        // If the ring is full and there's nothing more to submit,
        // wait for at least one completion before looping.
        if (outstanding > 0 && (pending.empty() || free_pool.empty()) &&
            !submitted) {
          struct io_uring_cqe* cqe;
          SYSCALL(io_uring_wait_cqe(&ring, &cqe));
          SYSCALL(cqe->res);
          auto* req = (ReadRequest*)io_uring_cqe_get_data(cqe);
          self->buffer_queue.Push(
              {req->data, req->used_bytes / sizeof(T), req->chunk_index});
          free_pool.push_back(req);
          outstanding--;
          completed++;
          io_uring_cqe_seen(&ring, cqe);
        }
      }
      // Round done; loop back to wait for the next StartRound().
    }

    io_uring_queue_exit(&ring);
    free(pool);
    // Shared fds are closed once in Shutdown(), not per worker.
  }
};

// ============================================================================
// The unified engine -- ChunkEmitter / ExternalTransform / RemoveWorker
//
// (was ChunkSequence/Primitives/external_engine.h)
// ============================================================================

namespace plaid {

/**
 * The unified out-of-core transform engine shared by the eager streaming
 * primitives (ChunkMap, ChunkScan pass 2, …).
 *
 * Two building blocks live here:
 *   - ChunkEmitter<R>    : allocate output blocks and hand them to the writer,
 *                          recording a chunk descriptor for each.
 *   - ExternalTransform  : drive read -> body -> emit -> write for the
 *                          "one-or-more emit(s) per input chunk" family.
 *   - RemoveWorker     : drive read -> per-worker fold for the scalar family
 *                          (ChunkReduce, ChunkFindIf, ChunkScan pass 1).
 *
 * All three route through the single standardized reader (ChunkSequenceReader)
 * and writer (UnorderedFileWriter), so a chunk_seq produced by the engine is
 * indistinguishable from one produced by tabulate/dense-pack.
 *
 * Ownership rule for ExternalTransform: the engine owns each input buffer and
 * frees it back to the reader pool after `body` returns.  A body therefore
 * copies whatever it needs into freshly emitted output blocks (emit.alloc());
 * it must not retain or free the input pointer.  (This drops the old in-place
 * T==R buffer reuse; the extra in-DRAM copy is negligible next to the SSD
 * read+write it accompanies.)
 */
template <typename R>
class ChunkEmitter {
 public:
  ChunkEmitter(const std::vector<std::string>& filenames,
               std::vector<std::atomic<size_t>>& file_offsets,
               std::atomic<size_t>& out_count, std::vector<chunk>& out_chunks,
               UnorderedFileWriter<R>& writer, size_t num_drives)
      : filenames_(filenames),
        file_offsets_(file_offsets),
        out_count_(out_count),
        out_chunks_(out_chunks),
        writer_(writer),
        num_drives_(num_drives) {
    static_assert(CHUNK_SIZE % sizeof(R) == 0,
                  "sizeof(R) must divide CHUNK_SIZE for O_DIRECT alignment");
  }

  // Number of R elements that fit in one output block.
  size_t out_cap() const { return CHUNK_SIZE / sizeof(R); }

  // Allocate a fresh, O_DIRECT-aligned CHUNK_SIZE output block.
  R* alloc() const {
    R* p = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(p != nullptr) << "ChunkEmitter: allocation failed";
    return p;
  }

  /**
   * Emit one output block holding `count` valid R elements.  The caller must
   * have zero-padded the tail out to CHUNK_SIZE.  `logical_index` fixes this
   * block's position in the output order; ExternalTransform sorts by it and
   * (when compacting) renumbers to a dense 0..k-1 to restore the
   * index-ordered invariant.  Drive placement is deterministic balls-in-bins
   * via parlay::hash64 of the emission slot.
   */
  void emit(std::shared_ptr<R> buf, size_t count, size_t logical_index) const {
    const size_t slot = out_count_.fetch_add(1);
    const size_t d = parlay::hash64(slot) % num_drives_;
    const size_t base = file_offsets_[d].fetch_add(CHUNK_SIZE);
    out_chunks_[slot] =
        chunk{filenames_[d], base, count * sizeof(R), logical_index};
    writer_.Push(std::move(buf), CHUNK_SIZE / sizeof(R), d, base);
  }

  // Convenience overload: adopt a raw block from alloc() (freed with free()).
  void emit(R* buf, size_t count, size_t logical_index) const {
    emit(std::shared_ptr<R>(buf, free), count, logical_index);
  }

 private:
  const std::vector<std::string>& filenames_;
  std::vector<std::atomic<size_t>>& file_offsets_;
  std::atomic<size_t>& out_count_;
  std::vector<chunk>& out_chunks_;
  UnorderedFileWriter<R>& writer_;
  size_t num_drives_;
};

size_t get_used_bytes(const chunk_seq& seq) {
  size_t n = 0;
  for (size_t j = 0; j < seq.chunks.size(); j++) {
    n += seq.chunks[j].used;
  }
  return n;
}

/**
 * Streaming transform: read every chunk of `seq`, hand each to `body`, write
 * whatever `body` emits, and return the resulting chunk_seq.
 *
 * body signature:
 *   void body(const T* in, size_t n, size_t index, const ChunkEmitter<R>& emit)
 * where `in`/`n` are the input chunk's data and element count and `index` is
 * its position in the input.  A body may emit any number of output blocks (up
 * to `max_out_per_input`); use `index * max_out_per_input + sub` as the
 * logical index of the sub-th emitted block to keep global order.
 *
 * @param max_out_per_input  Upper bound on emits per input chunk (sizes the
 *                           output descriptor table).
 * @param compact  Renumber output chunk indices to a dense 0..k-1 after
 *                  sorting (restores out.chunks[i].index == i).
 */
template <typename T, typename R = T, typename Body>
chunk_seq ExternalTransform(const chunk_seq& seq,
                            const std::string& result_prefix, Body body,
                            size_t max_out_per_input = 1, bool compact = true) {
  const size_t num_drives = GetSSDList().size();

  // Create/truncate one output file per drive.  The writer opens files with
  // O_CREAT but not O_TRUNC, so stale data from a prior run is cleared here.
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(5, 32, 16);

  // One io_uring writer thread per drive; queue_size caps in-flight buffers
  // to 64 * 4 MB = 256 MB of DRAM at any moment.
  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  // Output descriptors are filled by the emitter at slot = fetch_add order.
  std::vector<chunk> out_chunks(seq.chunks.size() * max_out_per_input);
  std::atomic<size_t> out_count{0};
  std::vector<std::atomic<size_t>> file_offsets(num_drives);
  for (auto& a : file_offsets) a.store(0, std::memory_order_relaxed);

  ChunkEmitter<R> emit(filenames, file_offsets, out_count, out_chunks, writer,
                       num_drives);

  parlay::parallel_for(
      0, parlay::num_workers(),
      [&](size_t) {
        while (true) {
          auto [ptr, n, idx] = reader.Poll();
          if (ptr == nullptr) break;
          body((const T*)ptr, n, idx, emit);
          reader.allocator.Free(ptr);
        }
      },
      /*granularity=*/1);

  writer.Wait();

  // The reader delivers chunks out of order, so restore global order by the
  // logical index the body assigned, then densify if requested.
  out_chunks.resize(out_count.load());
  std::sort(out_chunks.begin(), out_chunks.end(),
            [](const chunk& a, const chunk& b) { return a.index < b.index; });
  if (compact)
    for (size_t k = 0; k < out_chunks.size(); k++) out_chunks[k].index = k;

  return {out_chunks};
}

/**
 * Scalar-fold driver: read every chunk of `seq` and return one accumulator per
 * parlay worker.  `worker(reader)` polls the shared reader to exhaustion
 * (Poll() returns nullptr when done) and returns its local accumulator; the
 * caller combines the per-worker results (e.g. parlay::reduce).  No writer is
 * started.  Workers may also just scatter into a shared array by chunk index
 * (as ChunkScan's pass 1 does) and return a placeholder.
 */
template <typename T, typename WorkerFn>
auto RemoveWorker(const chunk_seq& seq, size_t reader_threads, WorkerFn worker)
    -> parlay::sequence<
        std::invoke_result_t<WorkerFn, ChunkSequenceReader<T>&>> {
  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  // More IO threads than default keep the SSDs saturated; CPU is not the limit.
  reader.Start(reader_threads, 32, 8);
  return parlay::tabulate(
      parlay::num_workers(), [&](size_t) { return worker(reader); },
      /*granularity=*/1);
}

}  // namespace plaid

// ============================================================================
// Dense packing -- DensePack / DensePackStream
//
// (was ChunkSequence/Primitives/dense_pack.h)
// ============================================================================

namespace plaid {

// Virtual chunks processed per batch.  Each batch holds at most
// DENSE_PACK_BATCH_SIZE * CHUNK_SIZE bytes of producer output in memory
// simultaneously (512 MB at the default of 128).
static constexpr size_t DENSE_PACK_BATCH_SIZE = 128;

// One virtual chunk's contribution: `count` R elements at `data`, in logical
// order.  The storage is owned by the producer's Batch (see DensePack).
template <typename R>
struct DensePackRun {
  const R* data;
  size_t count;
};

/**
 * Shared dense-packing driver behind ChunkFilter and ChunkFlatTabulate.
 *
 * Both primitives produce, per virtual chunk, a variable-length run of
 * survivors in logical order and must pack those runs into a tightly packed
 * chunk_seq: every output chunk but the last holds exactly CHUNK_SIZE/sizeof(R)
 * elements.  The only difference between them is how a batch of runs is
 * produced (reading + predicate-compaction vs. calling a block function), so
 * that is the single point of variation `produce`.
 *
 * `produce(base, batch_n)` returns a movable Batch that:
 *   - keeps its run storage alive for the batch's lifetime, and
 *   - exposes `size()` (== batch_n) and `run(b) -> DensePackRun<R>`.
 * `run(b)` is queried after the Batch has settled in DensePack's local, so
 * data() pointers stay valid even for producers whose storage uses a
 * small-buffer optimization (e.g. parlay::sequence).
 *
 * Algorithm per batch (identical to the two originals it replaces):
 *   1. produce the batch's runs (in index order).
 *   2. prefix-sum run counts, accounting for the carry left by the prior batch.
 *   3. parallel scatter each run into pre-zeroed output buffers
 * (non-overlapping ranges by the prefix sums, so no races).
 *   4. push full output chunks; carry the trailing (< epct) survivors forward.
 * A final partial chunk is flushed after the loop.
 *
 * Output files grow via CHUNK_SIZE-aligned offsets (not pre-fallocated, as the
 * output size is unknown up front) and preserve the index-ordered invariant.
 *
 * @tparam R  Output element type.
 */
template <typename R, typename ProduceBatch>
chunk_seq DensePack(size_t num_virtual, const std::string& result_prefix,
                    ProduceBatch produce) {
  if (num_virtual == 0) return {};

  const size_t epct = CHUNK_SIZE / sizeof(R);  // elements per output chunk
  const size_t num_drives = GetSSDList().size();

  // Create/truncate one output file per drive so prior-run data is cleared
  // (the writer opens with O_CREAT but not O_TRUNC).
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  // Per-drive slot counter: next free CHUNK_SIZE-aligned slot in each file.
  // Advanced serially in the (sequential) push loop; no atomics needed.
  std::vector<size_t> next_slot(num_drives, 0);
  std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

  // Carry: survivors from the current batch that don't yet fill a full output
  // chunk.  Invariant: carry.size() < epct at all times between batches.
  std::vector<R> carry;
  carry.reserve(epct);

  std::vector<chunk> out_chunks;
  size_t out_idx = 0;

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  for (size_t base = 0; base < num_virtual; base += DENSE_PACK_BATCH_SIZE) {
    const size_t batch_n = std::min(DENSE_PACK_BATCH_SIZE, num_virtual - base);

    // 1. Produce this batch's runs (index order).  `batch` owns their
    //    storage until it is destroyed at the end of this iteration.
    auto batch = produce(base, batch_n);

    // 2. Prefix sums: offset[b] is the absolute position in the virtual
    //    output stream of run b's first element; the carry occupies
    //    [0, carry.size()).
    std::vector<size_t> offset(batch_n + 1);
    offset[0] = carry.size();
    for (size_t b = 0; b < batch_n; b++)
      offset[b + 1] = offset[b] + batch.run(b).count;
    const size_t total = offset[batch_n];
    const size_t num_out = total / epct;
    const size_t new_carry_cnt = total % epct;

    // 3. Allocate output buffers: num_out full chunks + 1 overflow for carry.
    //    We do NOT zero the whole buffer: every buffer's [0, total) is fully
    //    covered by the carry memcpy + prefix-sum-tiled scatter below, and the
    //    overflow buffer is only read back up to new_carry_cnt.  The only bytes
    //    that can reach disk unwritten are the tail past the epct packed
    //    elements on a full-chunk O_DIRECT write when sizeof(R) does not divide
    //    CHUNK_SIZE, so zero just that (a no-op when packed_bytes ==
    //    CHUNK_SIZE, i.e. for every current element type).
    const size_t packed_bytes = epct * sizeof(R);
    const size_t num_alloc = num_out + (new_carry_cnt > 0 ? 1 : 0);
    std::vector<R*> obuf(num_alloc, nullptr);
    for (size_t k = 0; k < num_alloc; k++) {
      obuf[k] = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(obuf[k] != nullptr) << "DensePack: buffer allocation failed";
      if (packed_bytes < CHUNK_SIZE)
        memset((char*)obuf[k] + packed_bytes, 0, CHUNK_SIZE - packed_bytes);
    }
    if (!carry.empty() && num_alloc > 0)
      memcpy(obuf[0], carry.data(), carry.size() * sizeof(R));

    // 4. Parallel scatter — non-overlapping by prefix sums, so no races.
    parlay::parallel_for(
        0, batch_n,
        [&](size_t b) {
          const DensePackRun<R> r = batch.run(b);
          if (r.count == 0) return;
          const R* src = r.data;
          size_t pos = offset[b], rem = r.count, src_o = 0;
          while (rem > 0) {
            const size_t k = pos / epct;
            const size_t off = pos % epct;
            const size_t can = std::min(rem, epct - off);
            memcpy(obuf[k] + off, src + src_o, can * sizeof(R));
            pos += can;
            src_o += can;
            rem -= can;
          }
        },
        /*granularity=*/1);

    // 5. Push full output chunks with balls-in-bins drive assignment.
    for (size_t k = 0; k < num_out; k++) {
      const size_t d = drive_dist(rng);
      const size_t slot = next_slot[d]++;
      writer.Push(std::shared_ptr<R>(obuf[k], free), CHUNK_SIZE / sizeof(R), d,
                  slot * CHUNK_SIZE);
      out_chunks.push_back(
          {filenames[d], slot * CHUNK_SIZE, CHUNK_SIZE, out_idx++});
    }

    // 6. Update the carry from the overflow buffer (or clear it).
    carry.resize(new_carry_cnt);
    if (new_carry_cnt > 0) {
      memcpy(carry.data(), obuf[num_out], new_carry_cnt * sizeof(R));
      free(obuf[num_out]);
    }
  }

  // Flush the final partial chunk (if any survivors remain in the carry).
  if (!carry.empty()) {
    R* buf = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buf != nullptr) << "DensePack: final chunk allocation failed";
    memset(buf, 0, CHUNK_SIZE);
    memcpy(buf, carry.data(), carry.size() * sizeof(R));
    const size_t d = drive_dist(rng);
    const size_t slot = next_slot[d]++;
    writer.Push(std::shared_ptr<R>(buf, free), CHUNK_SIZE / sizeof(R), d,
                slot * CHUNK_SIZE);
    out_chunks.push_back(
        {filenames[d], slot * CHUNK_SIZE, carry.size() * sizeof(R), out_idx++});
  }

  writer.Wait();
  return {out_chunks};
}

/**
 * Streaming reader-backed dense-packing driver: the read-overlapped sibling of
 * DensePack, used by ChunkFlatMap (→ KMP / Rabin-Karp) and ChunkFilter.
 *
 * Where DensePack processes the input in 128-chunk windows with a hard barrier
 * per window (read-all → compute-all → pack) off a fresh reader each window,
 * this driver streams: ONE persistent reader over the whole input, a dispatcher
 * that releases each input chunk to a worker the instant its read (and, for
 * halo>0, its right neighbor's read) lands, and an index-ordered packer that
 * threads the dense-packing carry sequentially while later chunks are still
 * being read + computed.  So reads overlap compute continuously and the per-
 * window reader churn / synchronous seam read are gone.
 *
 * Parallelism model is the same as DensePack — between chunks, each chunk's
 * `body` run sequentially on one worker — only the *scope* changes from a
 * per-window barrier to the whole sequence (mirrors delayed's for_each_chunk).
 *
 * `body(buf, n, gpos, halo_buf, halo_n) -> parlay::sequence<R>` returns chunk
 * i's output run in logical order.  For halo>0, halo_buf/halo_n is a read-only
 * view of chunk i+1's first min(halo, count) elements (the forward halo); at
 * the last chunk halo_n==0, halo_buf==nullptr.  The body must report only
 * outputs whose logical start falls in this chunk, using the halo purely as
 * lookahead.
 *
 * @tparam T     Input element type (matches the chunk_seq).
 * @tparam R     Output element type (sizeof(R) is not restricted here; the ≤8B
 *               on-disk cap is a delayed-layer constraint, not a packing one).
 * @tparam Body  Callable (const T*, size_t, uint64_t, const T*, size_t)
 *               -> parlay::sequence<R>.
 */
template <typename T, typename R, typename Body>
chunk_seq DensePackStream(const chunk_seq& seq,
                          const std::string& result_prefix, size_t halo,
                          Body body) {
  const size_t n_in = seq.chunks.size();
  if (n_in == 0) return {};

  const size_t epct = CHUNK_SIZE / sizeof(R);  // output elements per chunk
  const size_t num_drives = GetSSDList().size();

  // Global element index of each input chunk's first element (for gpos).
  std::vector<uint64_t> pos_of(n_in + 1);
  pos_of[0] = 0;
  for (size_t i = 0; i < n_in; i++)
    pos_of[i + 1] = pos_of[i] + seq.chunks[i].used / sizeof(T);

  // Create/truncate one output file per drive (writer opens O_CREAT, not
  // O_TRUNC).
  std::vector<std::string> filenames(num_drives);
  parlay::parallel_for(
      0, num_drives,
      [&](size_t d) {
        filenames[d] = GetFileName(result_prefix, d);
        int fd = open(filenames[d].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        SYSCALL(fd);
        SYSCALL(close(fd));
      },
      /*granularity=*/1);

  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<R> writer;
  writer.Start(filenames, wcfg);

  // One persistent reader over the whole sequence.  Deeper than the old
  // per-window Start(5,32,16): 10 threads x 32 in-flight ≈ 320 outstanding
  // reads (~10/drive at 30 drives) keeps the drives fed.  Queue sizes bound
  // live input buffers; all three are tunable against the trace.
  ChunkSequenceReader<T> reader;
  reader.PrepChunks(seq);
  reader.Start(/*threads=*/10, /*queue_depth=*/32, /*max_requests=*/32,
               /*buf_queue_sz=*/128);

  // Per-chunk streaming state.
  std::vector<T*> inbuf(n_in, nullptr);  // dispatcher-set input buffers
  std::vector<parlay::sequence<R>> results(n_in);  // worker-set output runs
  std::unique_ptr<std::atomic<uint8_t>[]> computed(
      new std::atomic<uint8_t>[n_in]());
  std::unique_ptr<std::atomic<int>[]> in_rc(new std::atomic<int>[n_in]);
  for (size_t j = 0; j < n_in; j++)
    // Consumers of inbuf[j]: chunk j (own) + chunk j-1 (as its halo, if any).
    in_rc[j].store(1 + ((halo > 0 && j >= 1) ? 1 : 0),
                   std::memory_order_relaxed);

  auto drop_input = [&](size_t j) {
    if (in_rc[j].fetch_sub(1, std::memory_order_acq_rel) == 1)
      reader.allocator.Free(inbuf[j]);
  };

  SimpleQueue<size_t> ready;                  // compute-ready chunk ids
  ready.SetSizeLimit(DENSE_PACK_BATCH_SIZE);  // back-pressures the reader

  std::mutex pmtx;  // guards packer wait/notify
  std::condition_variable pcv;

  // Dispatcher: assemble out-of-order completions; release a chunk the moment
  // it (and, for halo>0, its right neighbor) has landed.  Single-threaded, so
  // present[]/pushed[] need no atomics; the ready queue gives workers the
  // happens-before on inbuf[].
  std::thread dispatcher([&] {
    std::vector<char> present(n_in, 0), pushed(n_in, 0);
    auto try_push = [&](size_t i) {
      if (pushed[i] || !present[i]) return;
      const bool halo_ready = (halo == 0) || (i + 1 == n_in) || present[i + 1];
      if (!halo_ready) return;
      pushed[i] = 1;
      ready.Push(i);
    };
    for (size_t done = 0; done < n_in; done++) {
      auto [buf, n, cidx] = reader.Poll();
      (void)n;
      CHECK(buf != nullptr) << "DensePackStream: short read";
      inbuf[cidx] = buf;
      present[cidx] = 1;
      try_push(cidx);                                 // cidx may now be ready
      if (halo > 0 && cidx >= 1) try_push(cidx - 1);  // cidx is (cidx-1)'s halo
    }
    ready.Close();
  });

  // Index-ordered packer: consume results[next] in order, threading the carry
  // through a single partially-filled output chunk `cur`.  Runs concurrently
  // with ongoing reads + compute of later chunks; packing is O(output) and
  // never the bottleneck, so the carry stays strictly sequential.
  std::vector<chunk> out_chunks;
  size_t out_idx = 0;
  std::vector<size_t> next_slot(num_drives, 0);
  std::mt19937_64 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> drive_dist(0, num_drives - 1);

  std::thread packer([&] {
    // A full chunk is memcpy'd end-to-end before it is pushed, so we skip the
    // per-chunk zeroing and only zero-pad the tail bytes that actually reach
    // disk: the packed-elements tail on a full O_DIRECT write (a no-op when
    // sizeof(R) divides CHUNK_SIZE) and the trailing partial chunk's remainder.
    const size_t packed_bytes = epct * sizeof(R);
    R* cur = nullptr;
    size_t cur_n = 0;
    auto new_cur = [&] {
      cur = (R*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
      CHECK(cur != nullptr)
          << "DensePackStream: output buffer allocation failed";
      cur_n = 0;
    };
    new_cur();
    for (size_t next = 0; next < n_in; next++) {
      {
        std::unique_lock<std::mutex> lk(pmtx);
        pcv.wait(lk, [&] {
          return computed[next].load(std::memory_order_acquire) != 0;
        });
      }
      const R* src = results[next].data();
      size_t rem = results[next].size();
      while (rem > 0) {
        const size_t can = std::min(rem, epct - cur_n);
        memcpy(cur + cur_n, src, can * sizeof(R));
        cur_n += can;
        src += can;
        rem -= can;
        if (cur_n == epct) {  // full output chunk
          if (packed_bytes < CHUNK_SIZE)
            memset((char*)cur + packed_bytes, 0, CHUNK_SIZE - packed_bytes);
          const size_t d = drive_dist(rng);
          const size_t slot = next_slot[d]++;
          writer.Push(std::shared_ptr<R>(cur, free), epct, d,
                      slot * CHUNK_SIZE);
          out_chunks.push_back(
              {filenames[d], slot * CHUNK_SIZE, CHUNK_SIZE, out_idx++});
          new_cur();
        }
      }
      results[next] = parlay::sequence<R>();  // release run storage early
    }
    if (cur_n > 0) {  // final partial chunk
      memset((char*)cur + cur_n * sizeof(R), 0, CHUNK_SIZE - cur_n * sizeof(R));
      const size_t d = drive_dist(rng);
      const size_t slot = next_slot[d]++;
      writer.Push(std::shared_ptr<R>(cur, free), epct, d, slot * CHUNK_SIZE);
      out_chunks.push_back(
          {filenames[d], slot * CHUNK_SIZE, cur_n * sizeof(R), out_idx++});
    } else {
      free(cur);
    }
  });

  // Workers: build + compute each ready chunk, publish its run, release inputs.
  parlay::parallel_for(
      0, parlay::num_workers(),
      [&](size_t) {
        while (true) {
          auto [i, code] = ready.Poll((size_t)0);
          if (code == QueueCode::FINISH) break;
          const size_t n = seq.chunks[i].used / sizeof(T);
          const T* hbuf = nullptr;
          size_t hn = 0;
          if (halo > 0 && i + 1 < n_in) {
            hbuf = inbuf[i + 1];
            hn = std::min(halo, seq.chunks[i + 1].used / sizeof(T));
          }
          results[i] = body(inbuf[i], n, pos_of[i], hbuf, hn);
          computed[i].store(1, std::memory_order_release);
          {
            std::lock_guard<std::mutex> lk(pmtx);
          }  // pair with packer's wait
          pcv.notify_one();
          drop_input(i);                                    // own input buffer
          if (halo > 0 && i + 1 < n_in) drop_input(i + 1);  // halo buffer
        }
      },
      /*granularity=*/1);

  dispatcher.join();
  packer.join();
  writer.Wait();
  return {out_chunks};
}

}  // namespace plaid

// ============================================================================
// NReader / NRemoveWorker -- co-indexed N-way read
//
// (was ChunkSequence/Primitives/n_reader.h)
// ============================================================================

namespace plaid {

/**
 * NReader<T> — read N chunk sequences in lockstep and hand a worker the N
 * co-indexed buffers together from a single Poll().
 *
 * Motivation (see count_sort.h): some primitives need to walk two (or
 * more) chunk_seqs that share an index space at the same time — e.g. a values
 * sequence and a same-length bucket-id sequence — pairing element k of chunk i
 * in one with element k of chunk i in the other.  A single ChunkSequenceReader
 * only streams one sequence, and running two independent readers gives back
 * chunks in each reader's own io_uring completion order, so the two streams are
 * not aligned.  NReader owns one ChunkSequenceReader per input sequence and a
 * thin matching layer that re-pairs their output by chunk.index before handing
 * it out, so Poll() always returns a full set of N co-indexed buffers.
 *
 * Requirements on the inputs:
 *   - All N sequences must share the same index set (chunks[i].index == i for
 *     each, the library-wide index-ordered invariant), i.e. the same number of
 *     chunks.  A chunk index that is missing from any one sequence will never
 *     complete a match and is silently dropped.
 *   - This version uses a single element type T for every sequence (mirroring
 *     RemoveWorker<T>).  Sequences with different element types would need a
 *     variadic/heterogeneous variant; two same-typed sequences is the case the
 *     current callers need.
 *
 * Ownership: Poll() returns raw buffers owned by the per-sequence reader pools.
 * The worker must return them with Free(match) (or FreeSlot) exactly once, the
 * same contract as ChunkSequenceReader::allocator.Free.
 *
 * Note on buffering: if one reader races ahead of the others, its unmatched
 * buffers accumulate in the matcher (the reader pools grow to cover them).  For
 * co-generated sequences read at similar rates this stays small; it is bounded
 * only by how far the fastest reader can get ahead of the slowest.
 */
template <typename T>
class NReader {
 public:
  // One matched position across all N sequences.
  struct Match {
    std::vector<T*> ptrs;       // size N; ptrs[s] = seq s's buffer at index
    std::vector<size_t> sizes;  // size N; element count in ptrs[s]
    size_t index = 0;           // shared chunk index
    // Empty ptrs is the exhausted sentinel returned by Poll().
    bool valid() const { return !ptrs.empty(); }
  };

  NReader() = default;

  ~NReader() {
    // Pumps hold references into readers_, so they must be joined before the
    // readers (and this object) are torn down.
    for (auto& t : pumps_)
      if (t.joinable()) t.join();
  }

  NReader(const NReader&) = delete;
  NReader& operator=(const NReader&) = delete;

  // Register the input sequences (in the order Match::ptrs will use).
  void Prep(const std::vector<const chunk_seq*>& seqs) {
    CHECK(!seqs.empty()) << "NReader: need at least one sequence";
    n_ = seqs.size();
    readers_.resize(n_);
    for (size_t s = 0; s < n_; s++) {
      readers_[s] = std::make_unique<ChunkSequenceReader<T>>();
      readers_[s]->PrepChunks(*seqs[s]);
    }
  }

  /**
   * Start all N underlying readers and the matching pumps.  Reader tuning
   * mirrors RemoveWorker; each underlying reader gets `reader_threads` IO
   * threads.
   */
  void Start(size_t reader_threads = 10, size_t queue_depth = 32,
             size_t max_requests = 8, size_t match_queue_sz = 512) {
    CHECK(!readers_.empty()) << "NReader: Prep() before Start()";
    match_queue_.SetSizeLimit(match_queue_sz);
    active_pumps_ = (int)n_;
    for (size_t s = 0; s < n_; s++)
      readers_[s]->Start(reader_threads, queue_depth, max_requests);
    // Pumps started after readers so the readers are fully live first.
    for (size_t s = 0; s < n_; s++) pumps_.emplace_back(Pump, this, s);
  }

  /**
   * Return the next fully matched set of N co-indexed buffers.  Blocks until
   * one is ready.  A Match with empty ptrs (valid() == false) means every
   * sequence has been exhausted.  Safe to call from multiple workers.
   */
  Match Poll() {
    Match nil;  // empty ptrs => done
    return match_queue_.Poll(nil).first;
  }

  // Return seq s's buffer from a match to its owning reader's pool.
  void FreeSlot(size_t s, T* ptr) { readers_[s]->allocator.Free(ptr); }

  // Return every buffer in a match to its owning pool.
  void Free(const Match& m) {
    for (size_t s = 0; s < m.ptrs.size(); s++)
      if (m.ptrs[s] != nullptr) readers_[s]->allocator.Free(m.ptrs[s]);
  }

  size_t num_sequences() const { return n_; }

 private:
  // In-progress match awaiting the remaining sequences' buffers.
  struct Partial {
    std::vector<T*> ptrs;
    std::vector<size_t> sizes;
    size_t filled = 0;
  };

  // One pump per sequence: drain reader s and deposit each buffer into the
  // matcher keyed by chunk index; the buffer that completes a match publishes
  // the assembled Match to match_queue_.
  static void Pump(NReader* self, size_t s) {
    while (true) {
      auto [ptr, size, index] = self->readers_[s]->Poll();
      if (ptr == nullptr) break;

      Match ready;  // filled while holding the lock, published after
      {
        std::lock_guard<std::mutex> lk(self->match_mutex_);
        Partial& p = self->partial_[index];
        if (p.ptrs.empty()) {
          p.ptrs.assign(self->n_, nullptr);
          p.sizes.assign(self->n_, 0);
        }
        CHECK(p.ptrs[s] == nullptr) << "NReader: duplicate chunk index "
                                    << index << " in sequence " << s;
        p.ptrs[s] = (T*)ptr;
        p.sizes[s] = size;
        if (++p.filled == self->n_) {
          ready.ptrs = std::move(p.ptrs);
          ready.sizes = std::move(p.sizes);
          ready.index = index;
          self->partial_.erase(index);
        }
      }
      if (ready.valid()) self->match_queue_.Push(std::move(ready));
    }

    // Last pump to finish closes the output so pollers unblock.
    if (--self->active_pumps_ == 0) self->match_queue_.Close();
  }

  size_t n_ = 0;
  std::vector<std::unique_ptr<ChunkSequenceReader<T>>> readers_;
  std::vector<std::thread> pumps_;
  std::atomic<int> active_pumps_{0};

  std::mutex match_mutex_;
  std::unordered_map<size_t, Partial> partial_;
  SimpleQueue<Match> match_queue_;
};

/**
 * N-sequence analogue of RemoveWorker: read `seqs` in lockstep and return one
 * accumulator per parlay worker.  `worker(nreader)` polls the shared NReader to
 * exhaustion (Poll().valid() == false when done) and returns its local
 * accumulator; the caller combines the per-worker results.  Each worker must
 * Free() the matches it consumes.
 *
 * Example (paired values + bucket-ids walk):
 *   auto locals = NRemoveWorker<uint64_t>({&values, &ids}, 10,
 *       [&](NReader<uint64_t>& r) {
 *           LocalAcc acc;
 *           while (true) {
 *               auto m = r.Poll();
 *               if (!m.valid()) break;
 *               uint64_t* val = m.ptrs[0]; uint64_t* id = m.ptrs[1];
 *               for (size_t k = 0; k < m.sizes[0]; k++) acc.add(id[k], val[k]);
 *               r.Free(m);
 *           }
 *           return acc;
 *       });
 */
template <typename T, typename WorkerFn>
auto NRemoveWorker(const std::vector<const chunk_seq*>& seqs,
                   size_t reader_threads, WorkerFn worker)
    -> parlay::sequence<std::invoke_result_t<WorkerFn, NReader<T>&>> {
  NReader<T> reader;
  reader.Prep(seqs);
  reader.Start(reader_threads, 32, 8);
  return parlay::tabulate(
      parlay::num_workers(), [&](size_t) { return worker(reader); },
      /*granularity=*/1);
}

}  // namespace plaid

// ============================================================================
// BucketWriter -- the count-sort scatter
//
// (was ChunkSequence/Primitives/bucketed_file_writer.h)
// ============================================================================

// Bucketed file writer: decouples a scatter pass's per-worker buffer size from
// its write size, so a k-way partition need not fill a full CHUNK_SIZE buffer
// per (worker, bucket) before anything can be flushed.
//
// Extracted from direct_samplesort.h (itself a port of Peter's
// OrderedFileWriter / type_allocator.h), so both the hand-rolled direct sample
// sort and a primitives-based partitioner can share one writer implementation.
// See direct_samplesort.h's header comment for the rationale: a worker's
// scatter buffer here is one SAMPLE_SORT_BUCKET_SIZE (4 KiB) block, and the
// writer itself rebuilds large sequential writes out of them with iovecs.

namespace plaid {

constexpr size_t kFlushThresholdBytes = 1UL
                                        << 20;  // Initialize("spfx_", n, 1<<20)
constexpr size_t kWriterRingDepth = 128;  // OrderedFileWriter::RunIOThread
constexpr size_t kRequestsPerBucket =
    10;  // Initialize: pool = 10 * num_buckets

// ── scatter-buffer allocator (Peter's utils/type_allocator.h)
// ───────────────── parlay's block allocator with an alignment override, so a
// scatter buffer can go straight into an O_DIRECT iovec.  Per-worker free
// lists: handing a filled buffer to the writer and taking a fresh one costs no
// lock.
template <size_t Size>
struct AllocatorData {
  char data[Size];
};

template <typename T, size_t Align>
class AlignedTypeAllocator {
  static parlay::internal::block_allocator& allocator() {
    return parlay::internal::get_block_allocator<sizeof(T), Align>();
  }

 public:
  static T* alloc() { return static_cast<T*>(allocator().alloc()); }
  static void free(T* p) { allocator().free(static_cast<void*>(p)); }
  static void finish() { allocator().clear(); }
};

using BucketData = AllocatorData<SAMPLE_SORT_BUCKET_SIZE>;
using bucket_allocator = AlignedTypeAllocator<BucketData, O_DIRECT_MULTIPLE>;

// ── bucketed writer  (Peter's OrderedFileWriter)
// ────────────────────────────── One append-only O_DIRECT file per bucket.
// Write() takes ownership of a scatter buffer and appends it to the bucket's
// pending writev; once the request holds >= kFlushThresholdBytes (or
// IO_VECTOR_SIZE iovecs) it is queued to an io_uring thread.  Requests are
// stamped with their file offset when created, so a bucket file is a contiguous
// log in request-creation order even though the writes complete out of order —
// which is what lets the finished file be carved into chunks.
//
// A buffer whose length is not a multiple of O_DIRECT_MULTIPLE (only ever a
// worker's last, partial buffer for a bucket) cannot go in an O_DIRECT iovec,
// so it is parked; ReapResult() concatenates a bucket's parked buffers into one
// aligned tail and appends it as the final iovec (his
// GatherMisalignedPointers).
template <typename T>
class BucketWriter {
 public:
  struct Result {
    std::string filename;
    size_t true_bytes = 0;  // bytes of live data
    size_t file_bytes = 0;  // bytes on disk (true_bytes rounded up)
  };

  // disk_span spreads each bucket's data across `disk_span` independent
  // per-drive files instead of one (bucket b shard s -> GetFileName(prefix,
  // b*disk_span+s), so with disk_span == drive count, shard s of every
  // bucket lands on drive s -- reading/writing one bucket then touches every
  // drive instead of a single one).  Default 1 reproduces the original
  // one-file-per-bucket layout exactly.
  BucketWriter(const std::string& prefix, size_t num_buckets,
               size_t disk_span = 1)
      : num_buckets_(num_buckets),
        disk_span_(disk_span),
        buckets_(num_buckets * disk_span),
        results_(num_buckets * disk_span) {
    // One accumulating request is permanently held per (bucket,shard), so the
    // pool must exceed num_buckets*disk_span for a flush to make progress;
    // the surplus caps how many requests can be in flight (and hence DRAM).
    requests_.resize(num_buckets * disk_span * kRequestsPerBucket);
    for (Request& r : requests_) free_requests_.Push(&r);

    for (size_t b = 0; b < num_buckets; b++) {
      for (size_t s = 0; s < disk_span; s++) {
        const size_t i = b * disk_span_ + s;
        Bucket& bk = buckets_[i];
        results_[i].filename = GetFileName(prefix, i);
        bk.fd = open(results_[i].filename.c_str(),
                     O_WRONLY | O_DIRECT | O_CREAT | O_TRUNC, 0644);
        // Fatal, not logged: with fd == -1 every Write routed to this bucket
        // is silently discarded and the caller gets a short/empty bucket back
        // as if nothing went wrong.  One fd per (bucket, shard) means a
        // few-thousand-bucket count_sort/group_by_index needs a matching
        // RLIMIT_NOFILE -- ParseGlobalArguments lifts the soft limit to the
        // hard limit, so reaching here means the HARD limit is too low.
        CHECK(bk.fd >= 0) << "BucketWriter: open(" << results_[i].filename
                          << ") failed for bucket " << b << " of "
                          << num_buckets << " (shard " << s << " of "
                          << disk_span_ << "): " << std::strerror(errno)
                          << "; soft RLIMIT_NOFILE=" << SoftFdLimit();
        bk.cur = NewRequest(bk.fd, 0);
      }
    }
  }

  ~BucketWriter() {
    for (Bucket& bk : buckets_)
      if (bk.fd >= 0) close(bk.fd);
  }

  // Drains `pending_` until it is closed.  Run on kWriterIoThreads parlay
  // workers alongside the scatter workers.
  void RunIoThread() {
    struct io_uring ring;
    SYSCALL(InitIoUringWithRetry(kWriterRingDepth, &ring,
                                 IORING_SETUP_SINGLE_ISSUER));
    size_t in_ring = 0;
    bool more = true;

    while (more || in_ring > 0) {
      bool submitted = false;
      while (more && in_ring < kWriterRingDepth) {
        // Block for work only when there is nothing in flight to reap.
        auto [r, code] =
            pending_.Poll(nullptr, (in_ring == 0 && !submitted) ? -1 : 0);
        if (r == nullptr) {
          if (code == QueueCode::FINISH) more = false;
          break;
        }
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        CHECK(sqe != nullptr) << "BucketWriter: writer ring out of sqes";
        io_uring_prep_writev(sqe, r->fd, r->iov, r->n, r->offset);
        io_uring_sqe_set_data(sqe, r);
        in_ring++;
        submitted = true;
      }
      if (submitted) SYSCALL(io_uring_submit(&ring));

      bool must_reap =
          in_ring > 0 && (in_ring >= kWriterRingDepth || !more || !submitted);
      while (in_ring > 0) {
        struct io_uring_cqe* cqe;
        if (must_reap) {
          SYSCALL(io_uring_wait_cqe(&ring, &cqe));
        } else if (io_uring_peek_cqe(&ring, &cqe) != 0) {
          break;
        }
        SYSCALL(cqe->res);
        Request* r = (Request*)io_uring_cqe_get_data(cqe);
        CHECK((size_t)cqe->res == r->bytes)
            << "BucketWriter: short write (" << cqe->res << " of " << r->bytes
            << ")";
        io_uring_cqe_seen(&ring, cqe);
        in_ring--;
        must_reap = false;
        Recycle(r);
      }
    }
    io_uring_queue_exit(&ring);
  }

  // Takes ownership of `buf` (a bucket_allocator block); `count` is its live
  // element prefix.  `shard` selects which of bucket b's disk_span files this
  // block goes to -- the caller picks it (e.g. round-robin per (worker,bucket)
  // staging slot), the writer just routes by (b,shard).  Default 0 keeps the
  // original 3-argument call sites (disk_span==1 callers, e.g.
  // direct_samplesort.h) working unchanged.
  void Write(size_t b, T* buf, size_t count, size_t shard = 0) {
    Bucket& bk = buckets_[b * disk_span_ + shard];
    const size_t bytes = count * sizeof(T);
    if (bytes % O_DIRECT_MULTIPLE != 0) {
      std::lock_guard<std::mutex> l(bk.lock);
      bk.parked.emplace_back(buf, bytes);
      return;
    }
    std::unique_lock<std::mutex> l(bk.lock);
    Request* r = bk.cur;
    r->Add((char*)buf, bytes);
    if (r->bytes >= kFlushThresholdBytes || r->n == IO_VECTOR_SIZE) {
      // The next request appends where this one ends, so the file stays a
      // contiguous log even though the writes complete out of order.
      bk.append_off += r->bytes;
      bk.cur = NewRequest(bk.fd, bk.append_off);
      l.unlock();
      pending_.Push(r);
    }
  }

  // Flush every (bucket,shard)'s partial request + parked buffers, close the
  // pending queue (which ends the I/O threads), and report each shard's
  // on-disk extent (results_[b*disk_span+s]).  Not concurrent with Write();
  // the caller joins the I/O threads.
  std::vector<Result> ReapResult() {
    parlay::parallel_for(
        0, num_buckets_ * disk_span_,
        [&](size_t b) {
          Bucket& bk = buckets_[b];
          Request* r = bk.cur;

          size_t parked_bytes = 0;
          for (auto& [p, sz] : bk.parked) parked_bytes += sz;

          const size_t aligned_bytes = bk.append_off + r->bytes;
          results_[b].true_bytes = aligned_bytes + parked_bytes;
          results_[b].file_bytes = aligned_bytes + AlignUp(parked_bytes);

          if (parked_bytes > 0) {
            // Concatenate the parked tails into one aligned buffer and append
            // it as the request's final iovec.  Its trailing padding is zeroed
            // so the block written past the live data is deterministic.
            const size_t tail_bytes = AlignUp(parked_bytes);
            char* tail = (char*)std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT,
                                                   tail_bytes);
            CHECK(tail != nullptr) << "BucketWriter: tail alloc failed";
            size_t off = 0;
            for (auto& [p, sz] : bk.parked) {
              memcpy(tail + off, p, sz);
              off += sz;
            }
            memset(tail + parked_bytes, 0, tail_bytes - parked_bytes);
            for (auto& [p, sz] : bk.parked)
              bucket_allocator::free((BucketData*)p);
            CHECK(r->n < IO_VECTOR_SIZE)
                << "BucketWriter: no iovec left for tail";
            r->Add(tail, tail_bytes);
            r->owns_tail = true;  // aligned_alloc'd, not a scatter buffer
          }

          if (r->n > 0)
            pending_.Push(r);
          else
            free_requests_.Push(r);
          bk.cur = nullptr;
        },
        /*granularity=*/1);

    pending_.Close();
    return results_;
  }

  // Called after the I/O threads have joined: every write has landed.
  void CloseFiles() {
    for (Bucket& bk : buckets_) {
      if (bk.fd >= 0) SYSCALL(close(bk.fd));
      bk.fd = -1;
    }
  }

 private:
  struct Request {
    int fd = -1;
    size_t offset = 0;
    size_t bytes = 0;
    unsigned n = 0;
    bool owns_tail =
        false;  // last iovec is an aligned_alloc, not a scatter buffer
    struct iovec iov[IO_VECTOR_SIZE];

    void Add(char* p, size_t sz) {
      iov[n].iov_base = p;
      iov[n].iov_len = sz;
      n++;
      bytes += sz;
    }

    void Reset() {
      n = 0;
      bytes = 0;
      owns_tail = false;
    }
  };

  struct Bucket {
    std::mutex lock;
    int fd = -1;
    size_t append_off = 0;   // bytes already assigned to flushed requests
    Request* cur = nullptr;  // accumulating request; cur->offset == append_off
    std::vector<std::pair<T*, size_t>> parked;
  };

  // Blocks until a request is free; callers may hold a bucket lock (the io
  // threads never take one, so this cannot deadlock).
  Request* NewRequest(int fd, size_t offset) {
    Request* r = free_requests_.Poll(nullptr).first;
    CHECK(r != nullptr) << "BucketWriter: request pool closed";
    r->fd = fd;
    r->offset = offset;
    return r;
  }

  void Recycle(Request* r) {
    const size_t n_bufs = r->n - (r->owns_tail ? 1 : 0);
    for (size_t i = 0; i < n_bufs; i++)
      bucket_allocator::free((BucketData*)r->iov[i].iov_base);
    if (r->owns_tail) free(r->iov[r->n - 1].iov_base);
    r->Reset();
    free_requests_.Push(r);
  }

  const size_t num_buckets_;
  const size_t disk_span_;
  std::vector<Bucket> buckets_;
  std::vector<Result> results_;
  std::vector<Request> requests_;
  SimpleQueue<Request*> free_requests_;
  SimpleQueue<Request*> pending_;
};

}  // namespace plaid

#endif  // CHUNK_SEQ_H
