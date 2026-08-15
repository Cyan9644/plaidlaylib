# plaidlaylib — Parlay-style parallel primitives for multi-SSD out-of-core data

Research library implementing Parlay-style parallel primitives (map, reduce,
filter, scan, flat-tabulate, find_if, …) for data stored across many SSDs.  Data
is too large for DRAM; all I/O goes through `io_uring` with `O_DIRECT`.
The primary goal of the project/library is to demonstrate that multi-SSD programming can be made relatively ergonomic with carefully chosen abstractions, while maintaining parallelism to rival in memory parallel algorithm implementations via techniques such as delaying to reduce IO trips. Examples are free to make calls into the reader and writer but these should be temporary solutions to reveal what abstractions are later needed; the ultimate goal is a useable set of abstractions that avoid burdening the user with the drive setup itself.

The library is deliberately small: **five headers**, six test binaries, eight
example binaries.  It reached that size through a whitelist cleanup that dropped
the accumulated experimental surface (parked research, superseded alternates,
head-to-head comparison drivers); `CLEANUP.md` records what was kept, what was
dropped and why, and where every surviving piece moved to.

## Building

Uses **Make**.  Requires `g++` (C++17), `cmake`, `git`, and system `liburing`
(Nix `shell.nix` provides the last).  There is **no git submodule** — the shared
I/O utilities are vendored under `utils/`.

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

Header dependencies **are** tracked (`-MMD -MP`, one `.d` per target under
`build/`), so editing a header rebuilds the binaries that include it — no
`rm -f bin/<target>` dance.  Note the link rules use `$< $(UTIL_OBJS)` rather
than `$^` on purpose: `-include`ing the `.d` files makes every header a
prerequisite, and `$^` would hand those headers to `g++` as source inputs.

## Machine setup

Assumes `SSD_COUNT` (default 30) mount points named per `SSD_ROOT` (default
`/mnt/ssd%lu`), i.e. `/mnt/ssd0 … /mnt/ssd29`.  Edit `configs.h` for your box.
On a dev box you can point all mounts at one tmpfs, but keep sizes small — the
"SSDs" then share one RAM-backed device, and a run that would be trivial on the
real machine can fill it and take the box down with it.  Two known cases:
`primitive_demosExample count_sort` needs `NUM_BUCKETS * CHUNK_SIZE` ≈ 16 GiB of
bucket files *regardless of n*, so it cannot run on a small tmpfs at any size.

## Layout

```
configs.h                     machine knobs (SSD_COUNT, SSD_ROOT, O_DIRECT_MULTIPLE, …)
Makefile  shell.nix  .envrc  CLAUDE.md  CLEANUP.md
utils/                        non-chunk-aware plumbing
  file_utils.{h,cpp}            paths, O_DIRECT alignment, fd/memlock limits,
                                SYSCALL/ASSERT + InitLogger, ParseGlobalArguments
  trace_marker.h                PLAID_TRACE phase markers (read by io_trace.py)
  bench_drives.h                shared benchmark/test drive helpers (clear_drives, …)
ChunkSequence/
  Primitives/                 the whole library — five headers, in dependency order
    chunk_seq.h                 data model + I/O substrate: SimpleQueue,
                                UnorderedFileWriter, chunk/chunk_seq (+ tabulate,
                                iota, from_file, to_chunk_seq, consolidate, size),
                                ChunkSequenceReader, ChunkEmitter/ExternalTransform/
                                RemoveWorker, DensePack/DensePackStream, NReader,
                                BucketWriter
    delayed.h                   the fused/lazy layer (delay/tabulate/map/scan/zip +
                                reduce/force/filter).  Depends only on chunk_seq.h
    primitives.h                the six core eager primitives: ChunkMap,
                                ChunkReduce, ChunkScan, ChunkFilter,
                                ChunkFlatTabulate, ChunkFlatMap
    secondary_primitives.h      the rest of the eager layer: segmented_reduce,
                                pack/pack_if/pack_value, histogram, find_if,
                                partition, flatten, materialize, cut, scan_find,
                                linear_find.  Includes primitives.h, so including
                                it gets the whole eager layer
    sort.h                      out-of-core sort/shuffle (process_inplace,
                                ChunkOperation/apply, count_sort, group_by,
                                sample, sample_sort, random_shuffle,
                                Permutation)
  tests/                      six binaries, each exiting 0 on PASS
    primitives_test.cpp         every case for chunk_seq.h/primitives.h/sort.h
    delayed_test.cpp            the delayed layer
    kmp_test.cpp  rabin_karp_test.cpp  bigint_add_test.cpp  convex_hull_test.cpp
  examples/                   seven demonstration programs + the primitive demos
    primes.cpp                  out-of-core prime sieve on ChunkFlatTabulate
    kmp.cpp  chunk_kmp.h        out-of-core KMP search
    rabin_karp.cpp  chunk_rabin_karp.h   out-of-core Rabin-Karp search
    bigint_add.cpp  chunk_bigint_add.h   out-of-core big-integer add (delayed-fused)
    linefit.cpp  chunk_linefit.h         fully-delayed least-squares fit
    convex_hull.cpp  chunk_convex_hull.h out-of-core upper convex hull
    samplesort.cpp              driver for plaid::sample_sort (Primitives/sort.h)
    primitive_demos.cpp         one binary, 11 per-primitive demos on argv[1]
    in_memory_baselines.h       DRAM references for linefit + sample sort
benchmarks/                   perf benchmarks + Python runner/plotter
  delayed_compare.cpp           in-mem delayed vs chunk-eager vs chunk-delayed
  chunk_size_compare.cpp        eager vs delayed across CHUNK_SIZE
  run_benches.py  summary_figure.py  io_trace.py  plot_style.py  clean_bench_data.py
benchresults/                 figure sources + historical plot output (kept in full)
deps/                         fetched by `make deps`; gitignored
results/                      timestamped benchmark output; gitignored
```

## Tests

`make test` builds and runs all six binaries, continuing past a failure and
exiting non-zero if any failed.  `TEST_ARGS` is forwarded to every binary
(`make test TEST_ARGS=8000000`); a case with no argument uses its own default.

`bin/primitivesTest` holds sixteen cases — iota, map, reduce, scan,
segmented_reduce, find_if, histogram, scalar, filter, flat_tabulate, flat_map,
partition, group_by, chunk_operation, combined, samplesort — each in its own
namespace with its original `main` renamed to `run()`, ordered cheap-first so a
substrate break surfaces before the expensive sorts.  `ParseGlobalArguments` is
called **once** by the dispatcher, not per case: it consumes the global flags and
populates the SSD list, so a second call would reset that list to the defaults
and discard any `--ssd=` selection.

## Examples

Each example is **dual-purpose**: run by hand it prints human-readable output,
and it always ends with a machine-readable `CSV,` line the runner greps.
`make examples` builds them all to `bin/<name>Example` via one pattern rule
(there are no per-binary override rules — every example lives in
`ChunkSequence/examples/<name>.cpp`).

Each example also times an **in-memory baseline** in DRAM: the corresponding
upstream parlaylib example (`deps/parlaylib-examples/`) where one exists, or
`in_memory_baselines.h` for linefit and sample sort.  The fetch **patches three
upstream bugs** (see the sed commands in the Makefile: an `int` loop index that
segfaults KMP past 2^31 chars, a missing KMP state reset after a match that reads
past the pattern, and Rabin-Karp comparing the last window against the powers-scan
total `x^n` instead of the text-hash total, dropping a match at position n−m); all
three were confirmed and the fixes verified against brute force with exact-position
property tests.  **Checkouts that fetched `deps/parlaylib-examples` before the
patches existed must `rm -rf deps/parlaylib-examples && make deps` to re-fetch.**
The baseline is gated by a RAM budget — half of physical RAM, overridable via
`EXAMPLE_INMEM_BUDGET_BYTES`; past the budget the run is skipped and the CSV field
left blank, so the plotted in-mem line stops at the RAM cliff.  When the baseline
does run, the binary cross-checks the count **and the full contents** (the
out-of-core output is read back and compared element-wise) and exits non-zero on a
mismatch — a differential test in the spirit of the benchmarks' `agree`.

- `primes.cpp` → `bin/primesExample [n] [out_path]`: out-of-core Eratosthenes
  sieve on `ChunkFlatTabulate`.  Prints `pi(n)`, output throughput, and the last
  few primes; consolidating the full list to a local file is opt-in via
  `out_path` (skipped at bench scale).  Emits
  `CSV,n,time_s,inmem_time_s,count,throughput_gb_s`.
- `kmp.cpp` → `bin/kmpExample [n] [m]`: out-of-core KMP string search over an
  n-char synthetic text (pattern = the text's first m chars, m constant across
  the sweep).  `ChunkKmp` (`examples/chunk_kmp.h`, tested by `kmpTest`) is a
  `DensePack` producer: per-chunk sequential KMP with cross-chunk matches caught
  via batch-local overlap — chunk k+1's head is already in DRAM in the same
  batch; one small sync read per batch seam (requires pattern ≤ one chunk).
  Emits `CSV,n,m,build_s,search_s,inmem_search_s,count,throughput_gb_s`; the
  sweep plots `search_s` (text build excluded from both series).
- `rabin_karp.cpp` → `bin/rabin_karpExample [n] [m]`: same driver shape and chunk
  structure as `kmp.cpp` (`ChunkRabinKarp` in `examples/chunk_rabin_karp.h`,
  tested by `rabinKarpTest`).  Within a chunk it uses a rolling polynomial hash
  mod the Mersenne prime 2^31−1 (Horner orientation, so no modular inverse; hash
  hits are double-checked) rather than parlaylib's prefix-hash scans, which
  out-of-core would write an 8x hash array to disk.  Baseline: the **same
  rolling-hash algorithm** run in DRAM, so this is a same-algorithm
  DRAM-vs-out-of-core comparison at a ~n-byte footprint — not parlaylib's ~9n
  prefix-hash variant.  Same CSV columns as kmp.
- `bigint_add.cpp` → `bin/bigint_addExample [n]`: out-of-core n-limb big-integer
  add (`examples/chunk_bigint_add.h`) as a fused delayed chain (zip → classify →
  carry-scan → add → force).  Baseline: parlaylib `bigint_reference::add`.
  Emits `CSV,n,build_s,add_s,inmem_add_s,result_limbs,gb_s`.
- `linefit.cpp` → `bin/linefitExample [n]`: fully-delayed least-squares fit
  (`examples/chunk_linefit.h`) — zip x and y into one delayed sequence and reduce,
  never materializing the zipped points.  Baseline: `in_memory_baselines.h`.
- `convex_hull.cpp` → `bin/convex_hullExample [n]`: out-of-core 2D **upper convex
  hull** via quickhull.  The point cloud is a `chunk_seq` of 32-byte `hpoint`s
  (two coords + original index + pad; 32 divides CHUNK_SIZE, so the fat element
  stays O_DIRECT-aligned — the eager engine is generic in element size, only the
  *delayed* layer caps at 8 B).  `UpperHull` (`examples/chunk_convex_hull.h`,
  tested by `convexHullTest`) recurses like parlaylib's `quickhull.h`, but each
  level's "farthest point from the dividing line" is a `ChunkReduce` (argmax
  monoid, tie-break by original index to match upstream's `maximum<pair>`), so the
  working set streams off the SSDs.  Each level's split is a **single**
  `ChunkPartition` pass routing each point to left/right/drop in one read — not
  two `ChunkFilter`s — and the base case's farthest-point pick ties by original
  index, so it is order-independent (works on the partition's completion-ordered
  output).  A **DRAM base case** finishes any sub-region below a byte budget
  (`CONVEX_HULL_DRAM_BUDGET_BYTES`, default `min(4 GiB, RAM/8)`) with an in-memory
  quickhull — the "shrink until it fits, then go in-memory" pattern, so only the
  top levels touch disk.  The budget is kept **small** on purpose: the streaming
  partition discards interior points at device speed, whereas in-memory quickhull
  on a huge region is slow (per-level sequence allocation over billions of
  points), so the out-of-core levels should do the bulk discarding and hand the
  base case only a small residual.  The in-mem baseline is additionally capped at
  n < 2^31 (upstream indexes with `int`).  Carries the points themselves (not
  indices into a global array) to avoid random per-element reads.  Emits
  `CSV,n,build_s,hull_s,inmem_hull_s,count,throughput_gb_s`.  Finds the upper hull
  only; the lower hull is symmetric and a full hull is upper ++ lower with shared
  endpoints dropped.
- `samplesort.cpp` → `bin/samplesortExample [n]`: driver for `plaid::sample_sort`
  (`Primitives/sort.h`) — oversample → `heap_tree` pivots → `group_by_index` →
  per-bucket DRAM sort → `flatten`.  Bucket count is chosen so each bucket fits in
  DRAM, so the per-bucket step is one in-memory pass rather than a recursion.
- `primitive_demos.cpp` → `bin/primitive_demosExample <primitive> [n ...]`: one
  binary holding the eleven per-primitive demos — `map`, `reduce`, `scan`,
  `tabulate`, `zip`, `filter`, `pack`, `count_sort`, `histogram_by_index`, `cut`,
  `random_shuffle`.  Each was its own binary before the cleanup and moved here
  verbatim inside its own namespace with `main()` renamed to `run()`; behaviour,
  cross-checks and CSV columns are unchanged.  **Known broken: `cut` segfaults**
  (a pre-existing break carried over from the parked `external_TODO` tree, not
  introduced by the cleanup — verified identical at the pre-cleanup commit).

**Name-clash warning**: the upstream parlaylib example headers define their
symbols at global scope with no include guards (e.g. `field` in `rabin_karp.h`,
`primes(long)` in `primes.h`), so when a new example pulls one in, check
carefully for clashes against the chunk-side code (our ports live in
`plaid::detail` for exactly this reason) and don't include more than one upstream
header per translation unit without verifying they coexist.

## Benchmarks

`make bench` builds the two benchmark binaries, runs both parameter sweeps, and
writes plots + raw CSVs to `results/<YYYYmmdd-HHMMSS>/`.  A single Python driver
(`benchmarks/run_benches.py`) orchestrates: it shells out to `make` to build each
binary (all compilation stays in the Makefile), runs the sweep, parses the `CSV,`
line each binary prints, and plots with matplotlib (provided by `shell.nix`).
Both benchmarks carry a cross-substrate correctness check, so an `agree=0`
mismatch aborts `make bench` non-zero — it doubles as a differential test.

- **delayed scale** (`bin/delayedCompare`): fixed chunk size, sweep `n`.
- **chunk size** (`bin/chunkSizeCompare_<bytes>`): fixed `n`, sweep `CHUNK_SIZE`.
  One binary is compiled per size via the `chunkSizeCompare_%` pattern rule,
  which passes `-DCHUNK_SIZE_BYTES=$*` (the stem = size in bytes).

`make bench` defaults are sized for a small dev box; override via env or the
driver's flags, e.g. `make bench BENCH_CHUNK_SIZES="2097152 8388608"`.  The driver
deletes the benchmarks' data files between every sweep point and after the run so
nothing accumulates on the drives (`--no-clean` to disable); the glob list is
derived from each `EXAMPLES` entry's `data_globs`, so **an entry with a wrong glob
silently leaks files** (this bit `count_sort`, whose `csrt_bucket_*` never matched
the real `csrt_bucket0`, stranding ~4 GB per run).  It also best-effort `fstrim`s
the mounts once at startup (`--fstrim-glob`, default `/mnt/ssd*`; a no-op on
tmpfs, `--no-fstrim` to disable).

`make bench-full` runs the same sweeps tuned for the benchmark machine (500 GiB
RAM, 30x 1TB SSDs): delayed scale over `2^30 … 2^39` elements (8 B each) and the
chunk-size test at `268435456` elements across `256KiB … 16MiB` chunks.  This is
multi-TB of I/O — intended for the real machine, not a tmpfs dev box.

Examples are benchmarked by a **separate opt-in sweep**, parameterized by **input
size in bytes** (not the binary's element count `n`), so heterogeneous examples
move the same number of bytes at each point and their times are directly
comparable: each `EXAMPLES` entry carries `elem_bytes` (its primary on-disk
sequence's element size) and `input_seqs` (how many input sequences it reads), and
`size_to_n()` converts a target size to the binary's argv[1] as
`n = size / (elem_bytes*input_seqs)`, rounded down to a whole number of
`CHUNK_SIZE`-chunks (preserving the O_DIRECT chunk-aligned invariant).  So
`bigint_add` (two 8-byte-limb operands, 16 B/n) gets **half** the `n` of a single
8-byte sequence at the same size.  Entries for the eleven primitive demos carry
`pre_argv` (the subcommand name), which `run_examples` places ahead of `n`.
`make bench-examples` sweeps dev-box sizes (`128MiB … 1GiB`); `bench-examples-mid`
goes to 256 GiB and `bench-examples-full` to 1 TiB.  The examples sweep does
**not** abort on a problem: the runner warns immediately, drops any point that
produced no CSV line, keeps sweeping, and repeats all warnings in the end-of-run
summary (also persisted to `warnings.txt`).  It is **not** part of
`make bench` / `--all`.

`make bench-summary` (`benchmarks/summary_figure.py`) draws the combined
relative-performance bar chart: 17 entries (10 primitives + 7 examples), each run
once at the largest n where its own in-mem baseline still fits DRAM.  Its
`SUMMARY_ENTRIES` asserts both alphabetical order and membership in
`run_benches.EXAMPLES`, so the two files must be edited together.

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

## The unified engine  (`Primitives/chunk_seq.h`)

All eager primitives share the standardized reader (`ChunkSequenceReader<T>`) and
writer (`UnorderedFileWriter<T>`) -- both now in `Primitives/chunk_seq.h` -- through
three building blocks in `namespace plaid`:

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
| `ChunkSegmentedReduce` | `RemoveWorker`, one pass; per-chunk segments classified fully-owned (direct write) vs boundary (chunk_idx-keyed, sequential O(n_chunks) merge) — same shape as `ChunkScan`'s pass-1 boundary handling, generalized from one running total to arbitrary contiguous segment bounds. Needed whenever the monoid isn't invertible (e.g. `min`), since `ChunkScan` prefix-differencing only recovers per-segment reduces for invertible monoids like sum. |
| `ChunkFindIf`       | `RemoveWorker` (per-worker min matching index; `n` if none) |
| `ChunkFilter`       | `DensePack` (reader source + predicate compaction) |
| `ChunkFlatTabulate` | `DensePack` (generator source, `f(start,end) -> sequence<R>`) |
| `ChunkPack` / `pack_if` / `pack_value` | `DensePack` (boolean-array / chunk-parallel-flag / predicate gate; `Primitives/pack.h`) |
| `ChunkFlatMap`      | `DensePackStream` (`flatten∘map`, `f(data,n,start,halo,halo_n)`; optional forward **halo** for boundary-crossing maps; `Primitives/flat_map.h`) |
| `ChunkHistogramByIndex` / `ChunkHistogramByKey` | `RemoveWorker` (per-worker bucket-count fold; `Primitives/histogram_by_index.h`) |
| `ChunkPartition`    | own single-reader + single-writer pass (`Primitives/primitives.h`); k-way split with a `PARTITION_DROP` sentinel |
| `NReader` / `NRemoveWorker` | own N-way co-indexed reader (`Primitives/chunk_seq.h`); lockstep read of N parallel `chunk_seq`s (e.g. values + bucket-ids for count-sort) |
| `tabulate` / `iota` | own writer pipeline (`Primitives/chunk_seq.h`) — no reader stage to unify |

(`ChunkSegmentedReduce` was also exposed per-vertex on CSR graphs, as
`chunk_csr::segmented_reduce_over_edges` using `degree_scan` for the segment
bounds — one streaming pass reducing every vertex's in-edge range at once.  The
CSR/graph family was dropped in the cleanup; see `CLEANUP.md` for the commit it
is recoverable from.)

`Primitives/sort.h` holds the out-of-core sort/shuffle substrate the
sample sort is built on.  It carries **only what is reachable** from
`sample_sort`, `apply`, `group_by_*`, `count_sort_by_key`, `random_shuffle_method`
and `Permutation` — the experimental variants that accumulated around it
(`count_sort_serial`, the `(seq, ids)` `count_sort` overload,
`group_by_index_partition_small`, `primitive_quicksort`, and a ~165-line
commented-out parallel `count_sort`) were removed in the cleanup; see
`CLEANUP.md`.  `count_sort` distributes elements
into per-bucket external sequences through the `BucketWriter` scatter
(per-worker staging → sequential `writev`); a bucket
list is then finished by `process_inplace` (on-disk
read/compute/write pipeline coalescing contiguous chunk runs, in place over
each sequence's own chunks, driven by an arbitrary `Processor` lambda) /
`process_inplace_budgeted` (the same, but DRAM-budget-checked and
wave-batched, keeping the in-place, original-layout write-back — so the caller
doesn't have to pre-size every
sequence to fit DRAM the way `count_sort`'s bucket count already does for
`sort_inplace` and `Permutation::Run`).  The `ChunkOperation`
enum (`Sort`/`Shuffle`) + `apply<Op>(seqs, ...)` is the named-operation
front door on top of `process_inplace_budgeted`, for callers who'd rather
select a named operation than hand-write a raw `Processor` lambda.  Supporting
pieces, all in `Primitives/primitives.h`: `cut` (slice / shift / guard-limb cuts,
used by bigint), `flatten` (concatenate a list of `chunk_seq`s by reindexing),
`materialize` (read a `chunk_seq` — or a delayed source — into a DRAM
`parlay::sequence`), the single-element probes `scan_find` / `linear_find`, and
in `sort.h`, `random_shuffle` (count-sort bucketing + per-bucket shuffle).

### Dense packing  (`Primitives/chunk_seq.h`)

`ChunkFilter` and `ChunkFlatTabulate` need **dense** output (every chunk but the
last is full), which requires cross-chunk carry state that the one-block-per-input
emitter cannot express.  Both are therefore thin producers over one shared
`DensePack<R>` driver, which owns the 128-virtual-chunk batch loop, cross-batch
carry, prefix sums, parallel scatter, and writer.  A producer returns a movable
`Batch` exposing `size()` and `run(b) -> DensePackRun<R>`; `run(b)` is read after
the batch has settled, so survivor pointers stay valid even for producers whose
storage uses a small-buffer optimization (e.g. `parlay::sequence`).

**No full-buffer zeroing** (relies on `sizeof(R) | CHUNK_SIZE`): output buffers
are *not* memset to zero before use.  Each batch fully overwrites every buffer's
`[0, total)` — the carry prefix by memcpy, the rest by the prefix-sum-tiled
scatter — so full chunks (pushed at a full `CHUNK_SIZE`) and the overflow buffer
(read back only up to the carry count, then freed) never expose uninitialized
memory.  The only bytes that could reach disk unwritten are the tail past the
`epct = CHUNK_SIZE/sizeof(R)` packed elements on a full O_DIRECT write, so only
that tail is zeroed — **and it is empty exactly when `sizeof(R)` divides
`CHUNK_SIZE`** (the same divisibility that keeps chunks O_DIRECT-aligned; true for
all current element types — 8 B, 32 B `hpoint`), making the zero-fill a no-op in
the common case.  The trailing partial chunk always zero-pads its remainder since
it too is written as a full `CHUNK_SIZE` block.  `DensePackStream` (the streaming
sibling) follows the same rule.  If a future `R` does *not* divide `CHUNK_SIZE`,
correctness is preserved by the per-buffer tail zero, but revisit this before
relying on it.

### k-way split  (`Primitives/primitives.h`)

`ChunkPartition` splits one `chunk_seq` into `k` output `chunk_seq`s in a **single
streaming read pass** (`key_fn(elem) -> bucket` or `PARTITION_DROP`) — the k-way
generalization of `ChunkFilter`, done with **one** long-lived reader and **one**
writer instead of k filter passes.  Each bucket is dense-except-last and returned
**separately** (do NOT concatenate — that buries a partial chunk mid-sequence and
breaks the delayed `zip` grid + `size`); ordering within a bucket is completion
order, not input order.  The scatter is **per-worker private** (each worker fills
its own pool-recycled assembly buffer per bucket, `writer.Push` outside all locks),
mirroring `ChunkReduce`'s fully-private folds so routing hits device read speed; a
final sequential tail-merge consolidates the per-worker partials to keep each bucket
dense-except-last.  A *shared*-per-bucket assembly (count_sort style) instead
serializes on 1–2 bucket locks for a few-bucket split and craters to ~1/10th
bandwidth (low CPU **and** low disk util together = lock convoy, not an I/O limit).

## Delayed (fused) sequences  (`Primitives/delayed.h`)

A port of parlaylib's block-iterable-delayed design (namespace
`plaid::delayed`) that fuses an operation chain so intermediates never
touch disk.  For `reduce(map(map(delay(seq),f),g),m)` the eager path moves
3n reads + 2n writes; the delayed path moves 1n reads and 0 writes.

```cpp
namespace d = plaid::delayed;
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
- **TODO (perf): stream the delayed `filter`.**  `delayed::filter`
  (`Primitives/delayed.h`) still uses the whole-window `for_each_window` driver
  (read-all → compute-all per `FILTER_BATCH_SIZE` window), so it waits for every
  read in a window before any compute — the pre-optimization behavior the eager
  `ChunkFilter` already shed when it moved onto `DensePackStream`.  Its per-chunk
  survivor scan is chunk-disjoint/order-independent (same shape as `reduce`/`force`
  on `for_each_chunk`); only the prefix+scatter+carry is sequential.  It could
  stream like `DensePackStream` does: order-independent compaction on
  `for_each_chunk`'s dispatcher (which, unlike `DensePackStream`, already handles
  the delayed layer's multi-read-per-chunk planning) publishing per-chunk
  `results`, plus a concurrent index-ordered packer thread threading the carry.
  Currently **no example** uses `delayed::filter` — only `delayed_test` exercises
  it (the `filter_compare` benchmark that also did was dropped in the cleanup), so
  there is no measured workload standing to gain today.

## Configuration constants  (`configs.h`)

| constant | default | meaning |
|---|---|---|
| `SSD_COUNT` | 30 | number of SSD mount points |
| `SSD_ROOT` | `/mnt/ssd%lu` | mount-path printf template |
| `O_DIRECT_MULTIPLE` | 4096 | alignment for O_DIRECT buffers and offsets |
| `CHUNK_SIZE` | 4 MB | size of one chunk (`configs.h`; override `-DCHUNK_SIZE_BYTES=N`) |
