# ChunkSequence — Parlay-style parallel primitives for multi-SSD out-of-core data

Research library implementing Parlay-style parallel primitives (map, reduce,
filter, scan, tabulate, …) for data stored across many SSDs.  Data is too large
for DRAM; all I/O goes through `io_uring` with `O_DIRECT`.

This repo is the standalone home of the **ChunkSequence** work.  It was extracted
from [ParAlg/PLAID](https://github.com/ParAlg/PLAID) (the original multi-SSD
project) and pulls PLAID's shared I/O utilities back in as a **git submodule**
(`plaid/`) rather than copying them, so they never drift.  Everything under
`ChunkSequence/` is original to this repo.

## Building

Uses **Make**.  Requires `g++` (C++17), `cmake`, `git`, `curl`, and system
`liburing` (Nix `shell.nix` provides the last).

```bash
# 0. Fetch the PLAID submodule (utils/ + sequence_algorithms/reduce.h).
git submodule update --init            # or clone with --recursive

# 1. First-time setup: fetch parlaylib + stb and build abseil from source.
make deps

# 2. Build all binaries (outputs to bin/).
make all

# Individual targets
make bin/chunkSeqMain
make bin/bwCompare
make bin/bwDelayed
make bin/raytracer bin/pathTracer bin/primes

# Run the ChunkSequence correctness tests (permTest, mapTest, reduceTest,
# filterTest, scanTest, combinedTest, delayedTest, flatTabulateTest —
# combinedTest chains the eager primitives and delayedTest covers the fused
# delayed pipeline; both sweep edge-case sizes).  Builds them if needed, runs
# each, and exits non-zero if any fails.
make test
make test TEST_ARGS=8000000   # override the per-test element count

# Cleanup
make clean       # remove object files and binaries
make distclean   # also remove deps/ and bin/
```

**Key compiler flags**: `-std=c++17 -O2`.  Link flags: `-luring -lpthread` plus
all abseil static libs.  Include roots (order matters): `.` (this repo) →
`plaid` (the submodule) → `deps/parlaylib` → `deps/abseil-cpp/install/include` →
`deps/stb`.  Because `-I.` precedes `-Iplaid`, this repo's local `configs.h` and
`sequence_algorithms/map.h` shadow the submodule's copies (see *Repository
layout*).

Nix environments are detected automatically; liburing include/lib paths are
picked up from `NIX_CFLAGS_COMPILE` / `NIX_LDFLAGS`.

> **Note**: the Makefile tracks no header dependencies, so editing a header
> (e.g. anything under `ChunkSequence/`) will *not* trigger a rebuild of a binary
> whose `.cpp` is unchanged.  Force it with `rm -f bin/<target>` (or `make clean`)
> before rebuilding.

## Repository layout

```
CLAUDE.md, Makefile, shell.nix, .envrc, .gitignore, .gitmodules
configs.h                   LOCAL copy of PLAID's configs.h — the machine knobs
                            (SSD_COUNT, SSD_ROOT, O_DIRECT_MULTIPLE, …) live here
                            so editing them for your box does NOT dirty the submodule.
sequence_algorithms/
  map.h                     LOCAL fork-patched copy: fixes a heap-corruption bug in
                            the in-place path (returns the reader-pool buffer via
                            allocator.Free instead of free()).  bwCompare needs it.
plaid/                      git submodule -> github.com/ParAlg/PLAID (pinned SHA).
                            Provides the shared library, byte-identical to upstream:
    utils/                    file_info.h, file_utils.{h,cpp}, logger.{h,cpp},
                              simple_queue.h, unordered_file_writer.h,
                              unordered_file_reader.h, type_allocator.h,
                              command_line.{h,cpp}
    sequence_algorithms/      reduce.h (map.h/filter.h are shadowed locally)
ChunkSequence/              the library — the reason this repo exists
  chunk_seq.h                 chunk / chunk_seq structs, tabulate, perm
  chunk_seq_reader.h          async chunk-level reader (io_uring)
  chunk_map.h                 ChunkMap    – chunk-level map
  chunk_reduce.h              ChunkReduce – chunk-level reduce
  chunk_filter.h              ChunkFilter – chunk-level filter (tightly packed output)
  chunk_scan.h                ChunkScan   – chunk-level exclusive prefix scan
  chunk_flat_tabulate.h       flat-tabulate helper
  chunk_delayed.h             delayed (fused) map/reduce/scan/filter/tabulate
  tests/                      correctness tests (→ permTest/…/flatTabulateTest)
  bench/                      scaling benchmarks (bw_compare.cpp → bwCompare;
                              delayed_compare.cpp → bwDelayed; driver + plotters)
  examples/                   raytracer.cpp, path_tracer.cpp, primes.cpp, make_png.py
deps/                       fetched by `make deps` (parlaylib, abseil, stb); gitignored
```

**Why the two local overrides** (`configs.h`, `sequence_algorithms/map.h`): both
are byte-identical to upstream today *except* the `map.h` bug fix.  Keeping them
local means (a) machine config is editable without touching the submodule, and
(b) the fixed `map.h` wins via include order.  Ideal end state: upstream the
`map.h`/`filter.h` fixes to ParAlg/PLAID, then delete the local
`sequence_algorithms/` and pull it from the submodule too.

## Machine setup

The library assumes `SSD_COUNT` (default 30) mount points named per `SSD_ROOT`
(default `/mnt/ssd%lu`), i.e. `/mnt/ssd0 … /mnt/ssd29`.  Edit these two constants
in `configs.h` for your box, plus `O_DIRECT_MULTIPLE` (alignment) if needed, then
rebuild.  On a dev box you can point all mounts at a shared tmpfs, but then the
"SSDs" share one RAM-backed device — keep benchmark sizes small (the bench docs
call out the tmpfs ceiling) and you'll only ever see the in-DRAM-overhead regime,
never the RAM-vs-storage cliff the scaling sweeps are designed to cross.

---

## Shared utilities (from the PLAID submodule)

### `FileInfo`  (`plaid/utils/file_info.h`)

Describes one file in a multi-file dataset: `file_name`, `file_index`,
`true_size` (data bytes), `file_size` (bytes on disk, O_DIRECT-rounded),
`before_size` (cumulative offset of preceding files).

### `UnorderedFileWriter<T>`  (`plaid/utils/unordered_file_writer.h`)

Writes data buffers to a set of output files asynchronously (io_uring).  This is
the writer every ChunkSequence primitive uses.

```cpp
UnorderedWriterConfig cfg;
cfg.num_files = files.size();
cfg.num_threads = 5;
UnorderedFileWriter<uint64_t> writer("output_prefix", cfg);
writer.Push(shared_ptr, n_elements, file_index, byte_offset);
writer.Wait();   // flush and close
```

`Push()` takes a `shared_ptr` so buffers can be freed once the write completes.

### `sequence_algorithms/` (file-level Map/Reduce/Filter)

The whole-file-granularity primitives PLAID ships.  Used here only as the
**baseline `bwCompare` measures ChunkSequence against**; `map.h` is the local
patched copy, `reduce.h` comes from the submodule.  Not part of the ChunkSequence
API.

---

## ChunkSequence

An alternative data layout and set of primitives that address data at **chunk**
granularity instead of whole-file granularity.

The chunk primitives live in the `ChunkSequenceOps` namespace (declared in
`chunk_seq.h`): `tabulate`, `perm`, `ChunkMap` (`chunk_map.h`), `ChunkReduce`
(`chunk_reduce.h`), `ChunkFilter` (`chunk_filter.h`), and `ChunkScan`
(`chunk_scan.h`).  Call them qualified, e.g. `ChunkSequenceOps::ChunkMap(...)`.
These are all **eager** (every op reads from and writes back to SSD).  A
**delayed/fused** variant lives in the nested `ChunkSequenceOps::delayed`
namespace (`chunk_delayed.h`) — see *Delayed (fused) sequences* below.

### `chunk` / `chunk_seq`  (`chunk_seq.h`)

```cpp
const size_t CHUNK_SIZE      = 4 << 20;                          // 4 MB
const size_t ELEMS_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);   // 524,288

struct chunk {
    std::string filename;   // file containing this chunk
    size_t begin_addr;      // byte offset of the chunk within that file
    size_t used;            // bytes of actual data (≤ CHUNK_SIZE)
    size_t index;           // position of this chunk in the chunk_seq
};

struct chunk_seq { std::vector<chunk> chunks; };
```

**Convention**: a `chunk_seq` corresponds to exactly **one file per drive** (i.e.
`SSD_COUNT` files total, one on each SSD mount point).  All chunks point into
those files at offsets that are multiples of `CHUNK_SIZE`, which is itself a
multiple of `O_DIRECT_MULTIPLE`, so every read is naturally aligned for O_DIRECT
io_uring without extra padding logic.  On creating a sequence via tabulate, the
chunks are randomly assigned to the drives, balls-in-bins style.  This saturates
all drives in parallel.

**Index-ordered invariant**: `chunk_seq.chunks` is always stored in index order,
i.e. `chunks[i].index == i`.  `tabulate` establishes this, and every primitive
that returns a `chunk_seq` (`ChunkMap`, `ChunkFilter`, …) must preserve it so
callers can index by position without a lookup.
The `chunk_seq_reader` opens each per-drive file once per worker thread (via its
fd cache) and issues individual `CHUNK_SIZE`-aligned reads for each chunk,
letting io_uring pipeline them across drives.

### `ChunkSequenceReader<T>`  (`chunk_seq_reader.h`)

```cpp
ChunkSequenceReader<uint64_t> reader;
reader.PrepChunks(seq);
reader.Start(/*num_threads=*/5, /*queue_depth=*/32, /*max_requests=*/16);

// consumer loop:
auto [ptr, n, chunk_index] = reader.Poll();  // (nullptr,0,0) when done
// …process ptr[0..n)…
reader.allocator.Free(ptr);
```

- Issues one io_uring read per chunk at `chunk.begin_addr` for
  `AlignUp(chunk.used)` bytes; reports only `chunk.used / sizeof(T)` elements.
- Caches open file descriptors per worker thread so files referenced by multiple
  chunks are opened once.
- `BufferData` is a 3-tuple `(ptr, n_elements, chunk_index)`.
- Has its own inline `Allocator` (aligned buffer pool); free buffers with
  `reader.allocator.Free(ptr)`.

### `ChunkSequenceOps::tabulate`  (`chunk_seq.h`)

```cpp
template <typename T = uint64_t, typename F>
chunk_seq ChunkSequenceOps::tabulate(size_t n, const std::string& result_prefix, F f);
```

Fills position `i` with `f(i)` for `i` in `[0, n)`, writing `ELEMS_PER_CHUNK`
elements per chunk.  Randomly assigns each chunk to one of the `GetSSDList().size()`
drives (uniform, `mt19937_64`) so writes are balanced.  Pre-fallocates one file
per drive at `GetFileName(result_prefix, drive)`, writes via `UnorderedFileWriter`.
`begin_addr` for slot `k` of a drive is `k * CHUNK_SIZE`, always O_DIRECT-aligned.
The last chunk is zero-padded; its `used` field holds the true byte count.

### `ChunkSequenceOps::perm`  (`chunk_seq.h`)

`chunk_seq ChunkSequenceOps::perm(size_t n);` — thin wrapper:
`tabulate(n, "perm", [](size_t i) { return (uint64_t)i; })`.

### Passing callables to eager primitives

`tabulate`, `ChunkMap`, and `ChunkFilter` accept any callable (`F`) — pass
lambdas directly, no `std::function` wrapper needed or wanted (wrapping defeats
inlining):

```cpp
// Good — lambda passed directly; compiler inlines the call
ChunkSequenceOps::ChunkMap<uint64_t>(seq, "out", [](uint64_t x) { return x + 1; });
```

**Type annotation rules** — because `F` is an unconstrained template parameter,
the compiler cannot infer `T` or `R` from the lambda.  Supply them explicitly:

| situation | what to write |
|---|---|
| Same-type map (`T == R`) | `ChunkMap<uint64_t>(seq, "out", f)` |
| Type-changing map | `ChunkMap<uint64_t, uint32_t>(seq, "out", f)` — omitting `R` silently truncates |
| `tabulate` with default `uint64_t` | `tabulate(n, "p", f)` |
| `tabulate` with non-default type | `tabulate<float>(n, "p", f)` — a lambda returning `float` does **not** set `T` |
| `ChunkFilter` | `ChunkFilter<uint64_t>(seq, "out", pred)` — `T` always explicit |

The `delayed` primitives follow the same pattern.  `ChunkReduce` and `ChunkScan`
are unaffected; they use a templated `Monoid` type and have no callable parameter.

### `ChunkMap`  (`chunk_map.h`)

```cpp
template <typename T, typename R = T, typename F>
chunk_seq ChunkSequenceOps::ChunkMap(const chunk_seq& seq, const std::string& result_prefix, F f);
```

Maps `f` over every element with the same one-file-per-drive layout as
`tabulate` (balls-in-bins drive assignment, `CHUNK_SIZE`-aligned slots,
pre-fallocated files).  Returns an index-ordered `chunk_seq`.  In-place
optimization applies when `T == R` (the reader buffer is transformed and handed
straight to the writer).  The output layout is computed fully up front
(independent of read-completion order).

### `ChunkReduce`  (`chunk_reduce.h`)

```cpp
template <typename T, typename R = T, typename Monoid>
R ChunkSequenceOps::ChunkReduce(const chunk_seq& seq, Monoid monoid);
```

Same monoid protocol as the file-level Reduce.  Pure streaming read, no writes.

### `ChunkFilter`  (`chunk_filter.h`)

```cpp
template <typename T, typename F>
chunk_seq ChunkSequenceOps::ChunkFilter(const chunk_seq& seq, const std::string& result_prefix, F pred);
```

Filters elements by `pred`, writing survivors as a **tightly packed** chunk_seq.
Output size is unknown upfront, so output files grow via `pwrite` at
`CHUNK_SIZE`-aligned offsets rather than being pre-fallocated.  All output chunks
except the final one have `used == CHUNK_SIZE` (dense packing).

Element order is preserved.  The input is processed in **index-contiguous**
batches of `FILTER_BATCH_SIZE = 128` chunks: batch `k` reads exactly the slice
`seq.chunks[k*128 : (k+1)*128)` with its own reader, sorts by `chunk_index`,
filters+compacts in parallel, prefix-sums survivor counts, scatters into
pre-allocated output buffers, flushes full chunks, and carries leftover survivors
(<`ELEMS_PER_CHUNK`) to the next batch — that cross-batch carry is what yields
dense packing.  Returns an index-ordered `chunk_seq`; the final chunk's `used`
holds the true survivor byte count.

### `ChunkScan`  (`chunk_scan.h`)

```cpp
template <typename T, typename R = T, typename Monoid>
std::pair<chunk_seq, R> ChunkSequenceOps::ChunkScan(const chunk_seq& seq,
                                                    const std::string& result_prefix,
                                                    Monoid monoid);
```

**Exclusive** prefix scan: `out[i] = monoid(in[0], …, in[i-1])`, with
`out[0] = monoid.identity`.  Returns `{result_seq, total}` where `total` is the
grand reduction (parlay scan convention).

Out-of-core, two-level (block) scan.  One accumulator per chunk fits in DRAM:
1. **Pass 1**: reduce each chunk in parallel into `chunk_sums[chunk_index]`.
2. **Block prefix** (sequential, `O(c)`): exclusive prefix over `chunk_sums` →
   `offset[i]` seeds chunk `i`; the running accumulator after the last chunk is
   the returned `total`.
3. **Pass 2**: re-read each chunk, run a sequential exclusive scan seeded with
   `offset[chunk_index]`, write with the same one-file-per-drive layout as
   `ChunkMap`/`tabulate`.

Returns an index-ordered `chunk_seq` plus the total.  In-place optimization
applies when `T == R`.

### Delayed (fused) sequences  (`chunk_delayed.h`)

A port of parlaylib's *block-iterable-delayed* (BID) design to the out-of-core
setting, in the nested `ChunkSequenceOps::delayed` namespace.  The eager
primitives round-trip every intermediate through the SSDs; a delayed sequence
**fuses** an operation chain so intermediates never touch disk.  `map` is lazy,
`reduce`/`force` consume in a single read pass, and `scan` is *partially delayed*.

```cpp
namespace d = ChunkSequenceOps::delayed;
auto s  = d::delay(seq);                          // wrap an on-SSD chunk_seq (lazy)
auto t  = d::tabulate(n, [](size_t i){ ... });    // generated source, no files
auto m  = d::map(s, f);                           // lazy; composes with no I/O
uint64_t r       = d::reduce(m, monoid);          // forces: one read pass, zero writes
auto [sd, total] = d::scan(m, monoid);            // partially delayed (offsets, then lazy)
chunk_seq out    = d::force(m, "out_prefix");     // materialize to SSD (index-ordered)
chunk_seq flt    = d::filter(m, "flt_prefix", pred);  // terminal, ChunkFilter-style packing
```

- `map` returns a delayed sequence of the same kind, composing element-wise with
  no temp buffer and no I/O — chain arbitrarily.
- `reduce` folds under a `ChunkReduce`-style monoid in one read pass.
- `scan` (exclusive) does one read pass for block offsets, then returns
  `{delayed_seq, total}` whose second pass stays lazy.
- `force` materializes to a real `chunk_seq` with the `ChunkMap` layout (fresh
  buffer per chunk — no in-place reuse, so scan chains stay correct).
- `filter` is a terminal modeled on `ChunkFilter` (index-contiguous 128-chunk
  batches, dense packing) reading through the fused iterator.
- For `reduce(map(map(seq,f),g),m)` the eager path moves 3 reads + 2 writes; the
  delayed path moves 1 read — quantified by `bench/delayed_compare.cpp` (→ `bwDelayed`).
- **Lifetime**: a delayed sequence (and any `scan` result derived from it) holds a
  pointer to its source `chunk_seq`; the source must outlive every terminal call.

---

## Benchmarks

### `bwCompare` — Chunk vs file-based Map/Reduce  (`ChunkSequence/bench/`)

Self-contained scaling harness comparing the ChunkSequence primitives against the
file-based `sequence_algorithms` ones across powers of two.  Only the operation
is timed — `bw_compare` generates the data first, outside the timed region.

```bash
make bin/bwCompare
ChunkSequence/bench/bench_chunk_vs_seq.sh [min_exp] [max_exp] [reps]   # defaults 20 26 1
# -> writes ChunkSequence/bench/results/chunk_bw.csv (the driver force-rebuilds bwCompare)
python3 ChunkSequence/bench/plot_chunk_bw.py     # needs matplotlib
# -> results/chunk_bw.png : Map vs ChunkMap and Reduce vs ChunkReduce (log-log)
```

Each run needs ~`32·n` bytes on the SSD mounts.  On a shared-tmpfs dev box keep
`max_exp` small; the driver clears `/mnt/ssd*/` between sizes.

### `bwDelayed` — in-memory vs chunk-eager vs chunk-delayed

`ChunkSequence/bench/delayed_compare.cpp` times the same pipeline three ways at
one size — in-memory `parlay::delayed` (the DRAM "speed-of-light" line),
chunk-eager (round-trips every intermediate through SSD), and chunk-delayed
(fused) — plus a raw-read ceiling (`ChunkReduce`), and checks all substrates agree
(exits non-zero on mismatch).  Pipelines: `map|reduce`, `map|map|reduce`, and the
write-terminated `force(map|map)`.

```bash
make bin/bwDelayed
./bin/bwDelayed <n>            # one size: per-substrate rows + CSV line + agree=1
ChunkSequence/bench/bench_delayed_scale.sh [min_exp] [max_exp] [reps]   # defaults 24 32 1
# -> ChunkSequence/bench/results/delayed_scale.csv
python3 ChunkSequence/bench/plot_delayed_scale.py     # needs matplotlib
```

The in-memory baseline is skipped once the input exceeds a RAM budget (default ½
physical RAM; override with env `DELAYED_INMEM_BUDGET_BYTES`).  Past that cliff
its CSV columns blank out and the plotted line stops — that boundary is the point
of the sweep, so it needs storage ≫ RAM (never visible on a tmpfs dev box).

### `bwChunkSize_<N>` — chunk-size sensitivity

`make bin/bwChunkSize_1048576` compiles `chunk_size_compare.cpp` with
`-DCHUNK_SIZE_BYTES=1048576`; `bench/bench_chunk_size.sh` sweeps 256 KB … 16 MB.

---

## Configuration constants  (`configs.h`)

| constant | default | meaning |
|---|---|---|
| `SSD_COUNT` | 30 | number of SSD mount points |
| `SSD_ROOT` | `/mnt/ssd%lu` | mount-path printf template |
| `READER_READ_SIZE` | 512 KB | io_uring read granularity (override `-DREADER_READ_SIZE_BYTES=N`) |
| `O_DIRECT_MULTIPLE` | 4096 | alignment for O_DIRECT buffers and offsets |
| `IO_URING_BUFFER_SIZE` | 64 | default io_uring ring depth |
| `CHUNK_SIZE` | 4 MB | size of one chunk (in `chunk_seq.h`; override `-DCHUNK_SIZE_BYTES=N`) |
| `ELEMS_PER_CHUNK` | 524 288 | `CHUNK_SIZE / sizeof(uint64_t)` (in `chunk_seq.h`) |
| `MAIN_MEMORY_SIZE` | 400 GB | assumed DRAM budget |
