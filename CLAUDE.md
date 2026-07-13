# external-chunk-seq — Parlay-style parallel primitives for multi-SSD out-of-core data

Research library implementing Parlay-style parallel primitives (map, reduce,
filter, scan, flat-tabulate, find_if, …) for data stored across many SSDs.  Data
is too large for DRAM; all I/O goes through `io_uring` with `O_DIRECT`.
The primary goal of the project/library is to demonstrate that multi-SSD programming can be made relatively ergonomic with carefully chosen abstractions, while maintaining parallelism to rival in memory parallel algorithm implementations via techniques such as delaying to reduce IO trips. Examples are free to make calls into the reader and writer but these should be temporary solutions to reveal what abstractions are later needed; the ultimate goal is a useable set of abstractions that avoid burdening the user with the drive setup itself.

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
# 1. First-time setup: fetch parlaylib (+ its upstream examples, used as the
#    examples' in-memory baselines) + build abseil from source.
make deps

# 2. Build + run the correctness tests (outputs to bin/).
make test
make test TEST_ARGS=8000000   # override the per-test element count

# Cleanup
make clean       # remove object files and test binaries
make distclean   # also remove deps/ and bin/
```

**Key flags**: `-std=c++17 -O2`; link `-luring -lpthread` plus abseil static libs.
Include roots: `-I.` (this repo) → `-Ideps` (upstream example headers as
`"parlaylib-examples/<name>.h"`) → `deps/parlaylib` → `deps/abseil-cpp/install/include`.
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
  chunk_seq.h                 chunk / chunk_seq structs, tabulate, iota, consolidate
  chunk_seq_reader.h          ChunkSequenceReader<T> — the standardized async reader
  external_engine.h           ChunkEmitter + ExternalTransform + RemoveWorker
  dense_pack.h                shared batch/carry/prefix/scatter packer
  chunk_map.h                 ChunkMap        (thin body on ExternalTransform)
  chunk_reduce.h              ChunkReduce     (fold on RemoveWorker)
  chunk_scan.h                ChunkScan       (pass1 RemoveWorker + pass2 ExternalTransform)
  chunk_filter.h              ChunkFilter     (thin producer on DensePack)
  chunk_flat_tabulate.h       ChunkFlatTabulate (thin producer on DensePack)
  chunk_find_if.h             ChunkFindIf     (fold on RemoveWorker)
  chunk_delayed.h             delayed (fused) recursive-node layer: delay/tabulate/map/scan/zip + reduce/force/filter
  tests/                      correctness tests (→ iotaTest … findIfTest)
  examples/                   demonstration programs (→ primesExample …); dual-purpose
    primes.cpp                out-of-core prime sieve on ChunkFlatTabulate
    chunk_kmp.h  kmp.cpp      out-of-core KMP search (ChunkKmp, a producer on DensePack)
    chunk_rabin_karp.h  rabin_karp.cpp  out-of-core Rabin-Karp search (same shape as KMP)
benchmarks/                   perf benchmarks + single-file Python runner/plotter
  delayed_compare.cpp         in-mem delayed vs chunk-eager vs chunk-delayed (sweep n)
  chunk_size_compare.cpp      eager vs delayed across CHUNK_SIZE (-DCHUNK_SIZE_BYTES)
  run_benches.py              runs the sweeps (incl. examples) + plots to timestamped results/
deps/                         fetched by `make deps` (parlaylib, parlaylib-examples, abseil); gitignored
results/                      timestamped benchmark output (PNG + CSV); gitignored
```

## Benchmarks

`make bench` builds the two benchmark binaries, runs both parameter sweeps, and
writes plots + raw CSVs to `results/<YYYYmmdd-HHMMSS>/`.  A single Python driver
(`benchmarks/run_benches.py`) orchestrates: it shells out to `make` to build
each binary (all compilation stays in the Makefile), runs the sweep, parses the
`CSV,` line each binary prints, and plots with matplotlib (provided by
`shell.nix`).  Both benchmarks carry a cross-substrate correctness check, so a
`agree=0` mismatch aborts `make bench` non-zero — it doubles as a differential
test.

- **delayed scale** (`bin/delayedCompare`): fixed chunk size, sweep `n`.
- **chunk size** (`bin/chunkSizeCompare_<bytes>`): fixed `n`, sweep `CHUNK_SIZE`.
  One binary is compiled per size via the `chunkSizeCompare_%` pattern rule,
  which passes `-DCHUNK_SIZE_BYTES=$*` (the stem = size in bytes).

`make bench` defaults are sized for a ~5 GiB tmpfs dev box; override via env or
the driver's flags, e.g. `make bench BENCH_CHUNK_SIZES="2097152 8388608"` or
`python3 benchmarks/run_benches.py --delayed --n-values "1M 8M 64M"`.  The driver
deletes the benchmarks' data files (`iota<drive>` + `bw_*`) between every sweep
point and after the run so nothing accumulates on the drives (`--no-clean` to
disable).  It also best-effort `fstrim`s the mounts once at startup
(`--fstrim-glob`, default `/mnt/ssd*`; a no-op on tmpfs, `--no-fstrim` to
disable) — once rather than between points, since fstrim can be slow on real SSDs.

`make bench-full` runs the same sweeps tuned for the benchmark machine (500 GiB
RAM, 30x 1TB SSDs): delayed scale over `2^30 … 2^39` elements (8 B each) and the
chunk-size test at `268435456` elements across `256KiB … 16MiB` chunks.  This is
multi-TB of I/O — intended for the real machine, not a tmpfs dev box.  (`fstrim`
does real work there and may need privileges — run under `sudo` or `--no-fstrim`.)

## Examples

`ChunkSequence/examples/` holds demonstration programs that use the primitives on
a real problem.  Each is **dual-purpose** like the benchmarks: run by hand it
prints human-readable output, and it always ends with a machine-readable `CSV,`
line the runner greps.  `make examples` builds them all (one per file, to
`bin/<name>Example` via the `%Example` pattern rule; order-only dep on
`deps/parlaylib-examples`).

Each example also times **the corresponding upstream parlaylib example**
(`deps/parlaylib-examples/`, fetched by `make deps`) in DRAM as an in-memory
baseline.  The fetch **patches three upstream bugs** (see the sed commands and
comments in the Makefile rule: an `int` loop index that segfaults KMP past
2^31 chars, a missing KMP state reset after a match that reads past the
pattern, and Rabin-Karp comparing the last window against the powers-scan
total `x^n` instead of the text-hash total, dropping a match at position n−m);
all three were confirmed and the fixes verified against brute force with
exact-position property tests.  **Checkouts that fetched
`deps/parlaylib-examples` before the patches existed must
`rm -rf deps/parlaylib-examples && make deps` to re-fetch.**  The baseline is
gated by a RAM budget exactly like `delayed_compare`: half of
physical RAM, overridable via `EXAMPLE_INMEM_BUDGET_BYTES`; past the budget the
run is skipped and the CSV field left blank, so the plotted in-mem line stops
at the RAM cliff.  When the baseline does run, the binary cross-checks the
count **and the full contents** (the out-of-core output is read back and
compared element-wise) and exits non-zero on a mismatch — a differential test
in the spirit of the benchmarks' `agree`.  Unlike the substrate benchmarks,
the examples sweep does **not** abort on a problem (mismatch or crash): the
runner warns immediately, drops any point that produced no CSV line, keeps
sweeping, and repeats all warnings in the end-of-run summary (also persisted
to `warnings.txt` in the results dir, next to the fstrim outcome note).

- `primes.cpp` → `bin/primesExample [n] [out_path]`: out-of-core Eratosthenes
  sieve on `ChunkFlatTabulate`.  Prints `pi(n)`, output throughput, and the last
  few primes; consolidating the full list to a local file is opt-in via
  `out_path` (skipped at bench scale).  Baseline: upstream `primes(n)`
  (`primes.h`, the original of the local `in_mem_primes`, which stays so the
  out-of-core sieve is self-contained); ~10n-byte footprint.  Emits
  `CSV,n,time_s,inmem_time_s,count,throughput_gb_s`.
- `kmp.cpp` → `bin/kmpExample [n] [m]`: out-of-core KMP string search over an
  n-char synthetic text (pattern = the text's first m chars, m constant across
  the sweep).  The algorithm itself, `ChunkKmp` (`examples/chunk_kmp.h`, tested
  by `kmpTest`), is a `DensePack` producer: per-chunk sequential KMP with
  cross-chunk matches caught via batch-local overlap — chunk k+1's head is
  already in DRAM in the same batch; one small sync read per batch seam
  (requires pattern ≤ one chunk).  It lives in `examples/` rather than the
  library proper.  Baseline: upstream `knuth_morris_pratt.h` on the same text
  (~n-byte footprint).  Emits
  `CSV,n,m,build_s,search_s,inmem_search_s,count,throughput_gb_s`; the sweep
  plots `search_s` (text build excluded from both series).
- `rabin_karp.cpp` → `bin/rabin_karpExample [n] [m]`: out-of-core Rabin-Karp
  search, same driver shape and chunk structure as `kmp.cpp` (`ChunkRabinKarp`
  in `examples/chunk_rabin_karp.h`, tested by `rabinKarpTest`).  Within a chunk
  it uses a rolling polynomial hash mod the Mersenne prime 2^31−1 (Horner
  orientation, so no modular inverse; hash hits are double-checked) rather than
  parlaylib's prefix-hash scans, which out-of-core would write an 8x hash array
  to disk.  Baseline: upstream `rabin_karp.h`, which *does* materialize those
  prefix-hash scans in DRAM (~9n-byte footprint) — that contrast is the point
  of the comparison.  Emits the same CSV columns as kmp; the sweep plots
  `search_s`.

Examples are benchmarked by a **separate opt-in sweep**.  The sweep is
parameterized by **input size in bytes** (not the binary's element count `n`), so
heterogeneous examples move the same number of bytes at each point and their times
are directly comparable: each `EXAMPLES` entry carries `elem_bytes` (its primary
on-disk sequence's element size) and `input_seqs` (how many input sequences it
reads), and `size_to_n()` converts a target size to the binary's argv[1] as
`n = size / (elem_bytes*input_seqs)`, rounded down to a whole number of
`CHUNK_SIZE`-chunks (preserving the O_DIRECT chunk-aligned invariant).  So
`bigint_add` (two 8-byte-limb operands, 16 B/n) gets **half** the `n` of a single
8-byte sequence at the same size — its input is split across two operands.
`make bench-examples` sweeps dev-box (tmpfs) sizes (`128MiB … 1GiB`) and writes
`<name>_scale.{csv,png}` (both series per plot; x-axis is the uniform input size,
and the CSV carries an `input_bytes` column) into the same timestamped `results/`
dir as the other benchmarks; `make bench-examples-mid` goes up to 256 GiB and
`make bench-examples-full` up to 1 TiB (benchmark-machine sizes).  Add an example
by dropping a `.cpp` in `examples/` and appending one entry to the `EXAMPLES`
registry in `run_benches.py` (set its `elem_bytes`/`input_seqs`).  **Name-clash warning**: the upstream parlaylib example
headers define their symbols at global scope with no include guards (e.g.
`field` in `rabin_karp.h`, `primes(long)` in `primes.h`), so when a new example
pulls one in, check carefully for clashes against the chunk-side code (our
ports live in `ChunkSequenceOps::detail` for exactly this reason) and don't
include more than one upstream header per translation unit without verifying
they coexist.  The examples sweep is **not** part of `make bench` / `--all`.

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
| `tabulate` / `iota` | own writer pipeline (`chunk_seq.h`) — no reader stage to unify |

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
touch disk.  For `reduce(map(map(delay(seq),f),g),m)` the eager path moves
3n reads + 2n writes; the delayed path moves 1n reads and 0 writes.

```cpp
namespace d = ChunkSequenceOps::delayed;
auto  m       = d::map(d::delay(seq), f);          // lazy; composes with no I/O
uint64_t r    = d::reduce(m, monoid);              // one read pass, zero writes
auto [s, tot] = d::scan(m, monoid);                // partially delayed; {seq, total}
chunk_seq z   = d::force(d::zip(d::delay(a), d::delay(b), pad), "out");
```

### The node model

A delayed sequence is a **recursive tree of value-type nodes**, each templated
(no `std::function`) so the fused chain inlines.  Every node exposes:

- `length()` / `num_chunks()` / `chunk_len(i)` — sizing over the `ELEMS_PER_CHUNK`
  grid.
- `plan(i, planner)` — register the physical reads logical chunk `i` needs
  (leaves call `planner.need(src, chunks[i])`; internal nodes forward to children
  left-to-right).
- `build(i, resolver)` — construct the fused forward-iterator for chunk `i`,
  pulling each leaf's buffer from `resolver.next()` in the same order `plan`
  registered them (the positional match is the core invariant; `build` must visit
  children in `plan`'s order).

Node kinds and the combinators that build them:

| node | from | plan | build |
|---|---|---|---|
| `leaf_source<T>` | `delay(seq)` | one read `chunks[i]` | pointer into that buffer |
| `leaf_index<F>` | `tabulate(n,f)` | none (generated) | counting iterator over `f` |
| `map_node` | `map(d,g)` | forward to child | wrap child in `map_iter` |
| `scan_node` | `scan(d,m)` | forward to child | seed `scan_iter` with the chunk's offset |
| `zip_node` | `zip(a,b[,pad])` | union of both children | pad each child, `zip_iter` → `std::pair` |

Because `plan`/`build` just recurse, **zip composes arbitrarily**:
`zip(zip(A,B),C)` (N-ary via nesting), `zip(A, map(B))`, and
`zip(A, scan(map(zip(A,B),f)))` (the out-of-core carry-lookahead **big-integer
add** shape) all work.  `zip(a,b)` requires equal length; `zip(a,b,pad)` pads the
shorter side with a runtime fill value.  Padding lives in `zip_node`: it wraps
each child in a `pad_iter` to that child's real element count for the chunk, so a
shorter operand emits `pad` past its end and a child with no chunk at `i`
contributes no read.

### Drivers and terminals

Two drivers execute a tree; both plan each chunk with a `Planner` that **dedups
reads by source** — a `chunk_seq` appearing in several leaves of one chunk (e.g.
A,B in both `zip(A,B)` and a scan of it) is read once, not per occurrence:

- **`for_each_chunk`** (streaming; used by `reduce`, `scan` pass-1, `force`):
  **one** long-lived `ChunkSequenceReader` for the whole pass, a dispatcher
  thread that assembles chunks from the reader's out-of-order completions and
  releases each to a parlay worker the instant its reads land — reads and compute
  overlap continuously, no window barrier.  `body` must be chunk-disjoint /
  order-independent (true for reduce and force).  (An earlier windowed driver
  re-created a reader per 128-chunk window; that ~40 ms/window setup cost is why
  this path streams instead.)
- **`for_each_window`** (collect a `FILTER_BATCH_SIZE` window, then compute; used
  by `filter`): needed because filter's dense-packing carry threads sequentially
  in index order and cannot consume chunks out of order.

`scan` is partially delayed: pass 1 (a streaming read pass) computes per-chunk
offsets + total; pass 2 is a lazy `scan_node`.  `force` writes one file per drive
(balls-in-bins) and returns an index-ordered `chunk_seq`.

### Constraints

- **≤8-byte on-disk elements**: the chunk grid assumes 8-byte elements, so
  `force`/`filter` `static_assert` `sizeof(R) ≤ 8`.  `zip`'s `std::pair` elements
  are transient inside the fused pass; map them to a scalar before `force`.
- **Lifetime**: every source `chunk_seq` (both operands of a zip) must outlive
  every terminal call on a sequence derived from it.

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
| `CHUNK_SIZE` | 4 MB | size of one chunk (`configs.h`; override `-DCHUNK_SIZE_BYTES=N`) |
