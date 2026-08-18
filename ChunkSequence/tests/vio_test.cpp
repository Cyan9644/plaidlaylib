#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "configs.h"
#include "utils/vio.h"

// Correctness test for the memory-backed storage under utils/vio.h, exercised
// directly rather than through any primitive: sparse-hole reads, writes that
// straddle block boundaries, O_TRUNC and ftruncate, POSIX unlink-while-open
// semantics, concurrent multi-threaded access to one file, and the accounting
// that resident_bytes() reports.
//
// This is the layer every primitive sits on, so a bug here would surface far
// away and much later; testing it in isolation keeps that from happening.

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << (ok ? "  OK   " : "  FAIL ") << what << "\n";
  if (!ok) failures++;
}


int open_mem(const char* p, int flags) { return vio::open(p, flags, 0644); }

// ── sparse reads: a block that was never written must read as zeros ─────────
void test_holes() {
  const char* p = "mem:/vio_test/holes";
  int fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);
  check(fd > 0, "holes: open returns a positive virtual fd");
  check(vio::is_virtual_fd(fd), "holes: fd is tagged virtual");

  // Write one block far out; everything before it is a hole.
  std::vector<char> one(4096, 'x');
  check(vio::pwrite(fd, one.data(), one.size(), 12u << 20) == (ssize_t)one.size(),
        "holes: pwrite past the end succeeds");

  std::vector<char> back(8192, 'q');
  check(vio::pread(fd, back.data(), back.size(), 0) == (ssize_t)back.size(),
        "holes: pread of an unwritten region succeeds");
  bool all_zero = true;
  for (char c : back) all_zero &= (c == 0);
  check(all_zero, "holes: unwritten region reads as zeros");

  std::vector<char> at(4096, 0);
  vio::pread(fd, at.data(), at.size(), 12u << 20);
  bool all_x = true;
  for (char c : at) all_x &= (c == 'x');
  check(all_x, "holes: written region reads back intact");

  vio::close(fd);
  vio::unlink(p);
}

// ── boundary splitting: reads/writes spanning several CHUNK_SIZE blocks ─────
void test_spanning() {
  const char* p = "mem:/vio_test/span";
  int fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);

  // Straddle a block boundary with an unaligned offset and a multi-block
  // length -- the shape process_inplace produces when it coalesces a run of
  // contiguous chunks into a single request.
  const size_t len = (size_t)(2.5 * CHUNK_SIZE);
  const off_t off = CHUNK_SIZE / 2 + 777;
  std::vector<char> src(len);
  std::mt19937_64 rng(7);
  for (size_t i = 0; i < len; i++) src[i] = (char)(rng() & 0xff);

  check(vio::pwrite(fd, src.data(), len, off) == (ssize_t)len,
        "span: multi-block unaligned pwrite");
  std::vector<char> dst(len, 0);
  check(vio::pread(fd, dst.data(), len, off) == (ssize_t)len,
        "span: multi-block unaligned pread");
  check(std::memcmp(src.data(), dst.data(), len) == 0,
        "span: contents survive a boundary-straddling round trip");

  // The byte just before the written range was never written.
  char before = 'z';
  vio::pread(fd, &before, 1, off - 1);
  check(before == 0, "span: the byte before the range is still a hole");

  vio::close(fd);
  vio::unlink(p);
}

// ── truncation, including the O_TRUNC path the writers depend on ────────────
void test_truncate() {
  const char* p = "mem:/vio_test/trunc";
  int fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);
  std::vector<char> blk(CHUNK_SIZE, 'a');
  vio::pwrite(fd, blk.data(), blk.size(), 0);
  vio::pwrite(fd, blk.data(), blk.size(), CHUNK_SIZE);
  const size_t two_blocks = vio::resident_bytes();
  check(two_blocks >= 2 * CHUNK_SIZE, "truncate: two blocks are resident");

  check(vio::ftruncate(fd, 0) == 0, "truncate: ftruncate(0) succeeds");
  check(vio::resident_bytes() < two_blocks,
        "truncate: ftruncate released the blocks");
  std::vector<char> back(64, 'b');
  vio::pread(fd, back.data(), back.size(), 0);
  bool zeroed = true;
  for (char c : back) zeroed &= (c == 0);
  check(zeroed, "truncate: truncated content reads as zeros");
  vio::close(fd);

  // Re-opening with O_TRUNC must clear a file that already has content --
  // UnorderedFileWriter opens O_CREAT without O_TRUNC and relies on the
  // producer having cleared the file separately.
  fd = open_mem(p, O_RDWR | O_CREAT);
  vio::pwrite(fd, blk.data(), blk.size(), 0);
  vio::close(fd);
  fd = open_mem(p, O_WRONLY | O_CREAT | O_TRUNC);
  std::vector<char> after(64, 'c');
  vio::pread(fd, after.data(), after.size(), 0);
  bool cleared = true;
  for (char c : after) cleared &= (c == 0);
  check(cleared, "truncate: O_TRUNC on reopen clears prior content");
  vio::close(fd);
  vio::unlink(p);
}

// ── open/unlink semantics ───────────────────────────────────────────────────
void test_open_semantics() {
  const char* missing = "mem:/vio_test/does_not_exist";
  int fd = open_mem(missing, O_RDONLY);
  check(fd == -1 && errno == ENOENT,
        "open: a missing memory path without O_CREAT fails ENOENT");

  // Unlink must not pull storage out from under an open descriptor.
  const char* p = "mem:/vio_test/unlinked";
  fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);
  const char* msg = "still here";
  vio::pwrite(fd, msg, 10, 0);
  check(vio::unlink(p) == 0, "unlink: succeeds on an open file");
  check(open_mem(p, O_RDONLY) == -1,
        "unlink: the name is gone immediately");
  char back[10] = {0};
  vio::pread(fd, back, 10, 0);
  check(std::memcmp(back, msg, 10) == 0,
        "unlink: an already-open fd still reads its data");
  vio::close(fd);
}

// ── many threads over one file, which is how every reader drives it ─────────
void test_concurrent() {
  const char* p = "mem:/vio_test/concurrent";
  int fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);

  const size_t nthreads = 16;
  const size_t per = 4;  // blocks each thread owns
  std::vector<std::thread> ts;
  std::atomic<int> bad{0};
  for (size_t t = 0; t < nthreads; t++) {
    ts.emplace_back([&, t] {
      std::vector<char> buf(CHUNK_SIZE);
      for (size_t i = 0; i < per; i++) {
        const size_t blk = t * per + i;
        std::memset(buf.data(), (int)(blk & 0x7f), buf.size());
        if (vio::pwrite(fd, buf.data(), buf.size(),
                        (off_t)(blk * CHUNK_SIZE)) != (ssize_t)buf.size())
          bad++;
      }
      std::vector<char> rd(CHUNK_SIZE);
      for (size_t i = 0; i < per; i++) {
        const size_t blk = t * per + i;
        if (vio::pread(fd, rd.data(), rd.size(), (off_t)(blk * CHUNK_SIZE)) !=
            (ssize_t)rd.size())
          bad++;
        for (char c : rd)
          if (c != (char)(blk & 0x7f)) {
            bad++;
            break;
          }
      }
    });
  }
  for (auto& th : ts) th.join();
  check(bad.load() == 0, "concurrent: 16 threads read back their own blocks");

  vio::close(fd);
  vio::unlink(p);
}

// ── accounting: storage must actually come back ─────────────────────────────
void test_accounting() {
  const size_t before = vio::resident_bytes();
  const char* p = "mem:/vio_test/accounting";
  int fd = open_mem(p, O_RDWR | O_CREAT | O_TRUNC);
  std::vector<char> blk(CHUNK_SIZE, 'k');
  for (int i = 0; i < 8; i++)
    vio::pwrite(fd, blk.data(), blk.size(), (off_t)i * CHUNK_SIZE);
  check(vio::resident_bytes() >= before + 8 * CHUNK_SIZE,
        "accounting: resident_bytes grew with the writes");
  vio::close(fd);
  vio::unlink(p);
  check(vio::resident_bytes() == before,
        "accounting: resident_bytes returns to its starting value");
}

}  // namespace

int main() {
  std::cout << "==================== vio (memory-backed storage) "
               "====================\n";
  test_holes();
  test_spanning();
  test_truncate();
  test_open_semantics();
  test_concurrent();
  test_accounting();

  std::cout << (failures == 0 ? "ALL PASS" : "SOME FAILED") << "\n";
  return failures == 0 ? 0 : 1;
}
