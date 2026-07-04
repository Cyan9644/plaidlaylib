# external-chunk-seq — Parlay-style parallel primitives for multi-SSD out-of-core data

Research library implementing Parlay-style parallel primitives (map, reduce,
filter, scan, flat-tabulate, find_if, …) for data stored across many SSDs.  Data
is too large for DRAM; all I/O goes through `io_uring` with `O_DIRECT`.

This folder is the **merge** of two predecessor projects, `chunk-sequence` and
`Parlay_Primitives_for_MultiSSD`, which implemented the same idea with different
scaffolding.  It keeps `chunk-sequence`'s clean `chunk_seq` data model, namespace,
tests, and delayed layer, refactors the eager primitives onto a single transform
engine (from `Parlay_Primitives_for_MultiSSD`'s `ExternalTransform`), and ports
that project's `find_if`.  Everything else from both sides (duplicate primitives,
the old file-based `externalSeq`, WIP experiments, benchmarks, examples, the
second Bazel build, the PLAID submodule) was dropped to keep the library small.

## Building

Uses **Make**.  Requires `g++` (C++17), `cmake`, `git`, and system `liburing`
(Nix `shell.nix` provides the last).  Unlike the predecessors there is **no git
submodule** — the shared I/O utilities are vendored under `utils/`.

```bash
# 1. First-time setup: fetch parlaylib + build abseil from source.
make deps

# 2. Build + run the correctness tests (outputs to bin/).
make test
make test TEST_ARGS=8000000   # override the per-test element count

# Cleanup
make clean       # remove object files and test binaries
make distclean   # also remove deps/ and bin/
```

**Key flags**: `-std=c++17 -O2`; link `-luring -lpthread` plus abseil static libs.
Include roots: `-I.` (this repo) → `deps/parlaylib` → `deps/abseil-cpp/install/include`.
Nix liburing paths are auto-detected from `NIX_CFLAGS_COMPILE`/`NIX_LDFLAGS`.

> The Makefile tracks no header dependencies, so editing a header will not
> rebuild a binary whose `.cpp` is unchanged — `rm -f bin/<target>` (or
> `make clean`) first.

## Machine setup

Assumes `SSD_COUNT` (default 30) mount points named per `SSD_ROOT` (default
`/mnt/ssd%lu`), i.e. `/mnt/ssd0 … /mnt/ssd29`.  Edit `configs.h` for your box.
On a dev box you can point all mounts at one tmpfs, but keep benchmark sizes
small (the "SSDs" then share one RAM-backed device).

## Layout

```
configs.h                     machine knobs (SSD_COUNT, SSD_ROOT, O_DIRECT_MULTIPLE, …)
Makefile  shell.nix  .envrc
utils/                        vendored shared I/O utilities
  logger.{h,cpp}  file_info.h  file_utils.{h,cpp}  simple_queue.h
  unordered_file_writer.h     UnorderedFileWriter<T> — the standardized writer
  command_line.{h,cpp}        ParseGlobalArguments (used by the tests)
ChunkSequence/
  chunk_seq.h                 chunk / chunk_seq structs, tabulate, perm, consolidate
  chunk_seq_reader.h          ChunkSequenceReader<T> — the standardized async reader
  external_engine.h           ChunkEmitter + ExternalTransform + RemoveWorker
  dense_pack.h                shared batch/carry/prefix/scatter packer
  chunk_map.h                 ChunkMap        (thin body on ExternalTransform)
  chunk_reduce.h              ChunkReduce     (fold on RemoveWorker)
  chunk_scan.h                ChunkScan       (pass1 RemoveWorker + pass2 ExternalTransform)
  chunk_filter.h              ChunkFilter     (thin producer on DensePack)
  chunk_flat_tabulate.h       ChunkFlatTabulate (thin producer on DensePack)
  chunk_find_if.h             ChunkFindIf     (fold on RemoveWorker)
  chunk_delayed.h             delayed (fused) map/reduce/scan/filter/tabulate — untouched
  tests/                      correctness tests (→ permTest … findIfTest)
deps/                         fetched by `make deps` (parlaylib, abseil); gitignored
```

## Data model

```cpp
constexpr size_t CHUNK_SIZE      = 4 << 20;                    // 4 MB
constexpr size_t ELEMS_PER_CHUNK = CHUNK_SIZE / sizeof(uint64_t);

struct chunk    { std::string filename; size_t begin_addr, used, index; };
struct chunk_seq { std::vector<chunk> chunks; };  // always index-ordered
```

A `chunk_seq` is a logical sequence stored out-of-core.  Chunks are packed at
`CHUNK_SIZE`-aligned offsets (a multiple of `O_DIRECT_MULTIPLE`), so every read
is O_DIRECT-aligned with no padding logic, and are spread across the `SSD_COUNT`
drives (balls-in-bins) to saturate all drives in parallel.  **Index-ordered
invariant**: `chunks[i].index == i`; every primitive that returns a `chunk_seq`
preserves it so callers can index by position.

## The unified engine  (`external_engine.h`)

All eager primitives share the standardized reader (`ChunkSequenceReader<T>`,
`chunk_seq_reader.h`) and writer (`UnorderedFileWriter<T>`, `utils/`), through
three building blocks in `namespace ChunkSequenceOps`:

- **`ChunkEmitter<R>`** — `alloc()` a CHUNK_SIZE block; `emit(buf, count, index)`
  assigns a drive via `parlay::hash64(slot) % num_drives`, bumps a per-drive
  atomic offset, records a `chunk`, and pushes to the writer.
- **`ExternalTransform<T,R>(seq, prefix, body, max_out_per_input, compact)`** —
  read every chunk, hand each to `body(in, n, index, emit)`, write what it emits,
  then sort emitted chunks by index and (if `compact`) renumber to a dense
  0..k-1.  Ownership rule: the engine frees each input buffer after `body`
  returns, so a body copies what it needs into fresh emitted blocks.
- **`RemoveWorker<T>(seq, reader_threads, worker)`** — the scalar-fold family:
  each parlay worker polls the reader to exhaustion and returns a local
  accumulator (no writer).

Primitive mapping:

| primitive | built on |
|---|---|
| `ChunkMap`          | `ExternalTransform` (FANOUT emits when `sizeof(R) > sizeof(T)`) |
| `ChunkReduce`       | `RemoveWorker` + `parlay::reduce` |
| `ChunkScan`         | pass 1 `RemoveWorker` → per-chunk sums; sequential block prefix; pass 2 `ExternalTransform` seeded per chunk. Returns `{seq, total}` |
| `ChunkFindIf`       | `RemoveWorker` (per-worker min matching index; `n` if none) |
| `ChunkFilter`       | `DensePack` (reader source + predicate compaction) |
| `ChunkFlatTabulate` | `DensePack` (generator source, `f(start,end) -> sequence<R>`) |
| `tabulate` / `perm` | own writer pipeline (`chunk_seq.h`) — no reader stage to unify |

### Dense packing  (`dense_pack.h`)

`ChunkFilter` and `ChunkFlatTabulate` need **dense** output (every chunk but the
last is full), which requires cross-chunk carry state that the one-block-per-input
emitter cannot express.  Both are therefore thin producers over one shared
`DensePack<R>` driver, which owns the 128-virtual-chunk batch loop, cross-batch
carry, prefix sums, parallel scatter, and writer.  A producer returns a movable
`Batch` exposing `size()` and `run(b) -> DensePackRun<R>`; `run(b)` is read after
the batch has settled, so survivor pointers stay valid even for producers whose
storage uses a small-buffer optimization (e.g. `parlay::sequence`).

## Delayed (fused) sequences  (`chunk_delayed.h`)

A port of parlaylib's block-iterable-delayed design (namespace
`ChunkSequenceOps::delayed`) that fuses an operation chain so intermediates never
touch disk.  Left **unchanged** from `chunk-sequence`; it already routes through
the standardized reader/writer and keeps its own tuned drive.

```cpp
namespace d = ChunkSequenceOps::delayed;
auto m           = d::map(d::delay(seq), f);        // lazy; composes with no I/O
uint64_t r       = d::reduce(m, monoid);            // one read pass, zero writes
chunk_seq out    = d::force(m, "out_prefix");       // materialize to SSD
```

## Notes / known trade-offs

- The engine does **not** reuse the reader buffer in place for same-type map/scan
  and does **not** pre-`fallocate` streaming output files (it grows them by
  explicit offset).  Both are small perf trade-offs accepted for a single, simple
  engine; the extra in-DRAM copy is negligible next to the SSD read+write.
- Drive placement is deterministic (`parlay::hash64`) in the engine and random
  (`mt19937_64`) in `tabulate`/`DensePack`; both are balls-in-bins balanced.

## Configuration constants  (`configs.h`)

| constant | default | meaning |
|---|---|---|
| `SSD_COUNT` | 30 | number of SSD mount points |
| `SSD_ROOT` | `/mnt/ssd%lu` | mount-path printf template |
| `O_DIRECT_MULTIPLE` | 4096 | alignment for O_DIRECT buffers and offsets |
| `CHUNK_SIZE` | 4 MB | size of one chunk (`chunk_seq.h`; override `-DCHUNK_SIZE_BYTES=N`) |
