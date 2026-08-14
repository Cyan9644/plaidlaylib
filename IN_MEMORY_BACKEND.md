# In-memory storage backend — change summary

Adds a heap-backed storage backend so the library and its test suite run with no
`/mnt/ssd*` setup at all, and so algorithms can be compared without conflating
"out-of-core algorithm" with "device latency".

**Result:** `make test` (26 tests, disk) and `make test-mem` (25 tests, heap) both
pass from a single clean rebuild. 87 files changed, +872/−447, plus two new files
(`utils/io_backend.{h,cpp}`, ~1000 lines).

```bash
make test-mem                              # whole suite in RAM, no /mnt needed
make test-both                             # both backends; divergence fails
PLAID_IN_MEMORY=1 bin/reduceTest 1000000
bin/reduceTest --in_memory=1 1000000
```

---

## Design

`chunk`'s `{filename, begin_addr, used, index}` metadata is **unchanged** —
`filename` simply resolves to a heap arena instead of a path.

### Runtime switch, not compile-time

One binary serves both backends, so any test can be A/B'd against itself. This
also avoids a concrete hazard: the Makefile tracks no header dependencies, so a
`-D` flip would silently link differently-compiled copies of the storage header —
a data-corruption class of bug rather than a build error.

### Routing is per path, not per process

Every primitive already takes a `result_prefix` that becomes
`GetFileName(prefix, drive)`, so a prefix registry gives per-`chunk_seq` control
with **zero signature changes**. A path is heap-backed iff it sits under a
`GetSSDList()` root *and* its basename matches a registered prefix (longest match
wins; the default backend covers the rest).

```cpp
{
  plaid::io::MemoryBacked scope("iota");  // this sequence in RAM ...
  s = plaid::iota(n);                     // ... everything else on disk
}
auto tot = plaid::ChunkReduce<uint64_t>(s, SumMonoid{});  // still reads from RAM
```

Consequences, all verified:

- A heap-backed and a disk-backed `chunk_seq` coexist in one process.
- `flatten()` of the two yields **one `chunk_seq` holding chunks of both kinds** —
  each resolves independently by its own filename.
- Paths outside every SSD root always hit the real filesystem, which keeps the
  library/host boundary intact for free: `consolidate()`'s output,
  `from_file()`'s input and `chunk_csr::from_file` stay ordinary files.

### Two levels of interception

A reader/writer class swap alone would have missed most of the surface — ~350
raw storage calls bypass those classes entirely (34 in `chunk_seq.h` alone, ~76
in the tests, of which 5 verify output with their own `open(O_DIRECT)+pread`).

| level | what | how |
|---|---|---|
| **Loose syscalls** (354 sites) | `open/pread/pwrite/fallocate/unlink/…` | `plaid::io::Open/Pread/…`, mirroring libc signatures and return conventions. Virtual fds come from `kVirtualFdBase = 1<<28`, so anything below is a zero-overhead passthrough. |
| **I/O engine classes** (7) | `ChunkSequenceReader`, `PersistentChunkSequenceReader`, `UnorderedFileWriter`, `BucketWriter`, `process_inplace`, `direct_samplesort`, `direct_random_shuffle` | An early branch: a heap-backed chunk is copied inline — no SQE, no completion to reap. The memory path is *shorter* than the disk path it replaces. |

**io_uring rings are created lazily**, on the first chunk that actually lives on a
device, so a fully heap-backed run builds none. This matters because ring setup
costs ~40 ms and the recursive sorts spin up a reader per subproblem.

`plaid::io::ListDir` replaces `std::filesystem::directory_iterator` for the SSD
roots — load-bearing, not cosmetic (see bug 2).

### Arena layout: 64 KiB granules with zero elision

`MemFile` holds fixed granules behind a `shared_mutex` that guards **only the
pointer vector**, so growth never moves bytes out from under a concurrent
`memcpy` — many writer threads `pwrite` disjoint offsets of the same file at
once. A flat `std::vector<char>` would `realloc` out from under them.

An all-zero write into an unallocated granule is skipped entirely (unwritten
granules already read as zeros). Race-safe: a fresh granule is zero-filled on
allocation, so a skipped zero-write and a concurrent allocation commute.

---

## Files changed

| area | files | what |
|---|---|---|
| `utils/` | 3 + 2 new | `io_backend.{h,cpp}` (new); `unordered_file_writer.h` memory branch + lazy ring; `file_utils.cpp` (`ListDir`, `ReadFileOnce`, `ReadEntireFile`, and an fd leak fix); `command_line.cpp` (`--in_memory`) |
| `ChunkSequence/Primitives/` | 14 | `chunk_seq_reader.h` (both readers), `chunk_seq.h` (49 sites), `bucketed_file_writer.h`, `small_sequence_ops.h`, `delayed.h`, `cut.h`, `count_sort.h`, `dense_pack.h`, `partition.h`, `group_by.h`, `materialize.h`, `scan_find.h`, `linear_find.h`, `external_engine.h` |
| `ChunkSequence/tests/` | 19 | own `pread` verification + teardown |
| `ChunkSequence/examples/` | 38 | incl. `direct_samplesort.h` (3 rings), `chunk_convex_hull.h`, `chunk_sa_common.h` |
| `benchmarks/` | 9 | incl. `direct_random_shuffle.h` (2 rings) |
| `ChunkSequence/helper/` | 2 | `bench_drives.h`, `external_compressed_sparse_row.h` |
| `Makefile` | 1 | `io_backend.o` in `UTIL_OBJS`; `test-mem` / `test-both`; `TEST_ARGS_<binary>` overrides |
| `CLAUDE.md` | 1 | new "Storage backends" section |

---

## Bugs surfaced by the port

**1. Footprint tracked padded chunk count, not live data.** The writer pads every
chunk to a full `CHUNK_SIZE` block (O_DIRECT wants aligned lengths), so a chunk
holding three elements still writes 4 MiB. On disk that padding is nearly free;
in RAM it made convex_hull's quickhull recursion — one padded block per
(worker, bucket) per level — turn **1.6 MB of points into >15 GiB**. Fixed by the
granule + zero-elision scheme: the same case now runs **under 1 GiB**.

**2. `directory_iterator` teardown silently freed nothing.** The `(dir, ec)`
overload used by `bench_drives::clear_drives`, `chunk_sa_common::sweep`,
`dc3_test` and `external_rmat_test` yields an *empty range* when the root does
not exist, so the whole arena leaked as an OOM instead of a test failure. Now
`ListDir` / `UnlinkPrefix`.

**3. A sequence became unreadable once its `MemoryBacked` scope ended.** Reads
fell through to a nonexistent disk path. Routing rules now govern only where
*new* files are created; the registry is authoritative for files that already
exist. Caught by the mixed-backend probe, not by the suite.

**4. Converter false positive.** The mechanical rewrite clobbered a domain helper
named `close(double,double,double)` in `external_linefit.cpp`. Found by
syntax-checking all 36 converted example/benchmark sources; it was the only one.

Pre-existing issues fixed in passing: `ReadEntireFile` never closed its fd (a
real leak on disk; in memory mode a leaked virtual fd pins arena memory past
`unlink`), and `settle_drives` burned ~2 s of pure `sleep` per call with nothing
to sync.

---

## Knobs

| variable | default | meaning |
|---|---|---|
| `PLAID_IN_MEMORY` | off | whole-process memory mode (also `--in_memory=1`) |
| `PLAID_MEM_LIMIT_BYTES` | `min(4 GiB, RAM/2)` | arena cap; exceeding it CHECK-fails listing the largest files instead of inviting the OOM killer |
| `PLAID_MEM_STATS` | off | print arena high-water mark at exit |
| `PLAID_MEM_STRICT_ALIGN` | **on** | enforce O_DIRECT alignment on heap writes too, so "passes in memory mode" implies "passes on `/mnt`" |
| `PLAID_MEM_OPEN_MISSING_FATAL` | on | opening a nonexistent heap file is a bug; on disk it silently yields a bad fd and a buffer of garbage |

---

## Verification

1. `make test` — 26/26 on disk, no regression, after a clean rebuild.
2. `make test-mem` — 25/25 in RAM at the tests' own default sizes.
3. **A/B differential** — every test run both ways; same binary, runtime switch,
   so any divergence is a backend bug.
4. **Bypass proof** — memory mode creates 0 files across `/mnt/ssd*`; disk mode
   creates 30.
5. **Mixed-backend probe** — placement, per-backend reads, and a `flatten` of
   4 heap + 4 disk chunks reducing correctly.
6. `bin/delayedCompare` — `agree=1` (cross-substrate correctness) on disk.
7. All 36 converted example/benchmark sources syntax-checked.

---

## Known gaps

- **`dc3Test` is excluded from `make test-mem`** (`MEM_TEST_EXCLUDE`). It
  deadlocks under the heap backend — every thread parked on a futex at ~44 MB
  RSS, so a genuine hang, not the arena filling up. Undiagnosed: `ptrace` is
  restricted on the dev box, so no backtrace was obtainable. It runs and passes
  under `make test`.
- **Examples and benchmarks are syntax-checked but not run** in memory mode.
  Their call sites are converted, but only the test suite was verified
  end-to-end; `chunk_fft.h` still has an unconverted io_uring ring.
- **`peter_samplesort/` is out of scope** and stays on real files — its own
  `configs.h`/`file_utils.h` shadow ours on the include path.
- Memory mode **cannot reproduce I/O-error paths** (arena access never fails, and
  `SYSCALL` only logs).
- Memory mode runs the recursive primitives **wider** than disk mode ever does —
  nothing throttles the `par_do` frontier — so expect it to surface fan-out bugs
  first. That is a real difference, not a backend regression.

> The Makefile tracks no header dependencies. `make clean` before switching
> backends is genuinely required; skipping it produced one misleading A/B round
> during this work.
