# Whitelist cleanup

## Why

The repo had accumulated a large experimental surface around a small library:
parked research branches, superseded alternates kept "for reference", and
head-to-head comparison drivers, plus 30 separate primitive headers with 26
separate test binaries each carrying its own copy of the same pass/fail
scaffolding. Finding the library inside it had become the hard part.

This was a whitelist reset rather than an incremental tidy: keep the core
primitive library, the delayed layer, and seven examples (kmp, rabin_karp,
primes, bigint_add, linefit, convex_hull, sample sort); collapse the many small
files into a few large ones; drop everything else.

**Everything below is recoverable from commit `9c96e4a`**, the state immediately
before the first deletion:

```bash
git show 9c96e4a:ChunkSequence/examples/external_TODO/chunk_dc3.h
git checkout 9c96e4a -- ChunkSequence/helper/          # restore a whole subtree
```

Nothing was deleted that does not exist in that commit.

## Counts

| | before (`9c96e4a`) | after |
|---|---|---|
| tracked files | 464 | 268 |
| tracked files excluding `benchresults/` | 239 | 43 |
| `.h` / `.cpp` files | 200 | 32 |
| lines of `.h` / `.cpp` | 42,081 | 18,477 |
| library headers | 30 | 5 |
| test binaries | 26 | 6 |
| example binaries | 37 | 8 |

`benchresults/` was kept in full — every dated result directory,
`Existing_Figures/`, and `Old_Figures/`. It is the historical record of the runs
and accounts for the bulk of the remaining file count.

## What was kept

| | |
|---|---|
| `Primitives/chunk_seq.h` | data model + I/O substrate: `SimpleQueue`, `UnorderedFileWriter`, `chunk`/`chunk_seq` (+ `tabulate`, `iota`, `from_file`, `to_chunk_seq`, `consolidate`, `size`), `ChunkSequenceReader`, `ChunkEmitter`/`ExternalTransform`/`RemoveWorker`, `DensePack`/`DensePackStream`, `NReader`, `BucketWriter` |
| `Primitives/delayed.h` | the fused/lazy layer — unchanged apart from one include. Depends only on `chunk_seq.h` |
| `Primitives/primitives.h` | the six core eager primitives: `ChunkMap`, `ChunkReduce`, `ChunkScan`, `ChunkFilter`, `ChunkFlatTabulate`, `ChunkFlatMap` |
| `Primitives/secondary_primitives.h` | the rest of the eager layer: `ChunkSegmentedReduce`, `pack`/`pack_if`/`pack_value`, histogram, `ChunkFindIf`, `ChunkPartition`, `flatten`, `materialize`, `cut`, `scan_find`, `linear_find`. Includes `primitives.h` |
| `Primitives/sort.h` | out-of-core sort/shuffle: `process_inplace`(+`_budgeted`), `sort_inplace`, `ChunkOperation`/`apply`, `count_sort`, `count_sort_by_key`, `fuse`, `group_by_index`, `group_by_key`, `sample`, `sample_sort`, `random_shuffle_method`, `Permutation`. Pruned to only what is reachable from a live caller |
| tests | `primitivesTest` (16 cases), `delayedTest`, `kmpTest`, `rabinKarpTest`, `bigintAddTest`, `convexHullTest` |
| examples | `primes`, `kmp`, `rabin_karp`, `bigint_add`, `linefit`, `convex_hull`, `samplesort`, plus `primitive_demos` (11 subcommands) |
| `utils/` | `file_utils.{h,cpp}`, `trace_marker.h`, `bench_drives.h` |
| benchmarks | `delayed_compare.cpp`, `chunk_size_compare.cpp`, `run_benches.py`, `summary_figure.py`, `io_trace.py`, `plot_style.py`, `clean_bench_data.py` |
| `benchresults/` | kept in full |

## File map: old path → new path

Every surviving piece that moved. Bodies moved **verbatim** — comments included —
so a `git log -S` for any symbol still finds its history.

### Into `Primitives/chunk_seq.h`
| was | |
|---|---|
| `utils/simple_queue.h` | `SimpleQueue` |
| `utils/unordered_file_writer.h` | `UnorderedFileWriter<T>` |
| `Primitives/chunk_seq.h` | `chunk` / `chunk_seq` + constructors |
| `Primitives/chunk_seq_reader.h` | `ChunkSequenceReader`, `PersistentChunkSequenceReader` |
| `Primitives/external_engine.h` | `ChunkEmitter`, `ExternalTransform`, `RemoveWorker` |
| `Primitives/dense_pack.h` | `DensePack`, `DensePackStream`, `DENSE_PACK_BATCH_SIZE` |
| `Primitives/n_reader.h` | `NReader`, `NRemoveWorker` |
| `Primitives/bucketed_file_writer.h` | `BucketWriter` |

### Into `Primitives/primitives.h` and `Primitives/secondary_primitives.h`
`Primitives/{map,reduce,scan,segmented_reduce,filter,flat_tabulate,flat_map,pack,histogram_by_index,partition,flatten,materialize,cut,scan_find,linear_find}.h`,
plus `Primitives/TODO/chunk_find_if.h` (promoted out of `TODO/` — `findIfTest`
already exercised it).

These then split by prominence: `primitives.h` keeps the six an out-of-core
program reaches for first (`ChunkMap`, `ChunkReduce`, `ChunkScan`,
`ChunkFilter`, `ChunkFlatTabulate`, `ChunkFlatMap`); everything else moved to
`secondary_primitives.h`, which includes `primitives.h` so one include still
gets the whole eager layer.  Each call site was repointed by which symbols it
actually uses, so the five examples that only need core primitives include only
`primitives.h`.

### Into `Primitives/sort.h`
`Primitives/{small_sequence_ops,operation,count_sort,group_by,sample,random_shuffle}.h`
and `plaid::sample_sort` lifted out of `examples/external/external_samplesort.h`
— then pruned to the reachable set (see "Second pass" below).

### Into `utils/file_utils.{h,cpp}`
`utils/file_info.h`, `utils/logger.{h,cpp}`, `utils/command_line.{h,cpp}`.

### Examples (flattened out of `examples/external/` and `examples/in_memory/`)
| was | now |
|---|---|
| `examples/external/{primes,kmp,rabin_karp,bigint_add,convex_hull}.cpp` | `examples/<same>.cpp` |
| `examples/external/external_linefit.cpp` / `.h` | `examples/linefit.cpp` / `examples/chunk_linefit.h` |
| `examples/external/external_samplesort.cpp` | `examples/samplesort.cpp` |
| `examples/external/chunk_{kmp,rabin_karp,bigint_add,convex_hull}.h` | `examples/<same>` |
| `examples/in_memory/{linefit,sample_sort}.h` | `examples/in_memory_baselines.h` |
| `examples/external/{map,reduce,scan,tabulate,zip,filter,pack,count_sort,histogram_by_index}.cpp` | `examples/primitive_demos.cpp` (`<name>` subcommand) |
| `examples/external_TODO/chunk_cut.cpp` | `examples/primitive_demos.cpp` (`cut` subcommand) |
| `examples/external/external_random_shuffle.cpp` | `examples/primitive_demos.cpp` (`random_shuffle` subcommand) |
| `ChunkSequence/helper/bench_drives.h` | `utils/bench_drives.h` |

### Tests
| was | now |
|---|---|
| `tests/{iota,map,reduce,scan,segmented_reduce,find_if,histogram,scalar,filter,flat_tabulate,flat_map,partition,group_by,chunk_operation,combined,samplesort_striped}_test.cpp` | `tests/primitives_test.cpp` (one namespace per case) |
| `tests/{delayed,kmp,rabin_karp,bigint_add,convex_hull}_test.cpp` | unchanged |

## What was dropped, and why

### Parked research — the largest group, and the most likely to be wanted back
All of it is intact at `9c96e4a`.

- **Suffix arrays**: `chunk_suffix_array.h` (prefix doubling), `chunk_dc3.h`
  (Kärkkäinen–Sanders skew) + `chunk_sa_common.h`, and their `suffix_array.cpp` /
  `dc3.cpp` drivers and `dc3_test.cpp`.
- **The graph family**: `external_compressed_sparse_row.h` (`chunk_csr`), the
  RMAT generator `graph_utils/external_rmat.h` + `external_rmat_test.cpp`,
  `external_bellman_ford.h` + `bellman_ford.cpp`, `external_bfs.h` + `bfs.cpp`,
  `external_kcore.h`, `external_page_rank.h`, `csr_bfs.h`, and the in-memory
  graph baselines. This also removed the only caller of
  `ChunkSegmentedReduce`'s per-vertex CSR wrapper.
- **FFT**: `chunk_fft.h`, `fft.cpp`, `fft_transpose.cpp` (four-step FFT,
  transpose-free and on-disk-transpose variants).
- **Others**: `chunk_bigint_mul.h` + `bigint_mul.cpp` + `bigint_mul_test.cpp`
  (Karatsuba), `ExternalKthSmallest.h` + `kth_smallest.cpp`,
  `chunk_word_count.h` + `word_count.cpp` + `word_count_test.cpp`,
  `external_even_squares.h` + `even_squares.cpp`,
  `chunk_convex_hull_lazy_filter.h` + its driver,
  `bigint_add_eager.cpp` (the materialized-intermediate variant of bigint add).

### Superseded
- `direct_samplesort.h`, `orig_samplesort.h`, `orig_quicksort.h`,
  `external_samplesort_rewritten.h` — alternates to the kept `sample_sort`.
- `sample_sort_random`, `sample_sort_singledrive` — unused siblings inside
  `external_samplesort.h`; only a dropped benchmark driver called the former.
- `Primitives/sort_buckets.h` — already dead (its only call site was commented
  out); `process_inplace_budgeted` does the job.
- `Primitives/TODO/{chunk_count_sort_handwritten,group_by_index_handwritten}.h`
  — superseded by `count_sort.h` / `group_by.h`.
- `benchmarks/old_filter/old_chunk_delayed.h` — the pre-optimization delayed layer.

### Second pass: pruning `sort.h` to the reachable set

The first pass moved the sort/shuffle headers over wholesale. A follow-up
reachability audit — walking out from every live caller in `tests/`, `examples/`
and `benchmarks/` — found that a third of the merged file had no call site at
all, inside `sort.h` or out. Removed (547 lines, `sort.h` 2007 → 1460):

- **A ~165-line commented-out parallel `count_sort`** — an alternate two-level
  buffering implementation, already commented out in the original `count_sort.h`
  and carried over verbatim by the merge. It was never live code.
- `count_sort_serial` — the delayed-source serial variant. No callers; the live
  parallel `count_sort(const D&, ...)` superseded it.
- `count_sort(const chunk_seq& seq, const chunk_seq& ids, ...)` — the
  materialized-ids overload. Every live call site passes a delayed source and a
  `size_t num_buckets`, so they all bind the other overload.
- `group_by_index_partition_small` — a `ChunkPartition`-based alternative to
  `group_by_index` for small bucket counts. No callers.
- `primitive_quicksort` — the per-bucket DRAM base sorter. This one was pulled in
  from `external_TODO/` during the first pass **on the strength of an `#include`
  in `sample.h` rather than an actual call**; `sample_sort` finishes its buckets
  with `apply<ChunkOperation::Sort>` instead. It should not have come across.

What remains in `sort.h` is exactly the closure of `sample_sort`, `apply`,
`group_by_index`, `group_by_key`, `count_sort_by_key`, `fuse`,
`random_shuffle_method` and `Permutation`. Verified by rebuilding all 14 binaries
and re-running the suite after the cut — these are called by name, so any real
use would have been a hard compile error.

### Comparison scaffolding
- `examples/external_TODO/peter_samplesort/` — the vendored competitor sort with
  its own `configs.h`/`utils/`, plus the `peter_shim.o` isolation build and the
  `PETER_DIR` machinery in the Makefile that existed only to keep its include
  guards from clashing with ours.
- Every "vs" driver: `external_samplesort_vs_peter`, `direct_samplesort_vs_peter`,
  `samplesort_three_way`, `apply_sort_vs_samplesort`,
  `samplesort_vs_samplesort_random`, `random_shuffle_three_way`,
  `fitmem_sort`, `fitmem_kth_smallest`, `kth_smallest_delayed`,
  `direct_random_shuffle.h`, `external_quicksort.h`.
- One-off micro-benchmarks: `filter_compare.cpp`, `zip_depth_compare.cpp`,
  `work_exponent_compare.cpp` and their Python drivers, plus `csv_from_log.py`
  and `samplesort_phase_bench.py`.
- `scripts/` — eight `clean_*.sh` scripts, all targeting dropped drivers.

### Dead or broken
- `Primitives/merge.h` — never compiled; the body contains a literal `if()`.
- `Primitives/unique.h`, `Primitives/reverse.h` — unreferenced.
- `ChunkSequence/helper/heaptree.h` — a vendored `heap_tree` copy nothing included
  (the live code uses `parlay::internal::heap_tree`).
- `tests/tmp_csr_check.cpp` — a scratch file.
- `err.log`, tracked `build_local/*.o` — stray artifacts.
- `.gitmodules` — pointed at `chunk-sequence/PLAID`, a path that no longer exists.

## Consequences

Things that will surprise someone returning to this repo:

- **Test binary names changed.** There is no `bin/mapTest`, `bin/scanTest`, … —
  those sixteen cases are all inside `bin/primitivesTest`. `make test` is
  unaffected; muscle memory is.
- **Header edits now rebuild correctly.** `-MMD -MP` dependency tracking was
  added, so the old "the Makefile tracks no header dependencies, `rm -f
  bin/<target>` first" ritual is obsolete. With the library down to four headers
  this mattered much more than it used to. The link rules deliberately use
  `$< $(UTIL_OBJS)` rather than `$^`: `-include`ing the `.d` files makes every
  header a prerequisite, and `$^` would pass those headers to `g++` as source
  inputs.
- **The summary figure lost four bars**, 21 → 17. `bellman_ford`, `fft` and
  `kth_smallest` lost their examples outright; `cut` is excluded because its demo
  segfaults (see below).
- **`primitive_demosExample cut` is broken.** It segfaults on any input. This is
  **pre-existing**, not a cleanup regression — verified by building and running
  the original `chunk_cutExample` at `9c96e4a`, which crashes identically at the
  same point. It came from `external_TODO/`, i.e. parked, unmaintained code. The
  `cut` primitive itself lives on in `primitives.h`; only its demo driver is
  broken.
- **`primitive_demosExample count_sort` cannot run on a small tmpfs dev box.** It
  allocates `NUM_BUCKETS (4096) * CHUNK_SIZE (4 MiB)` ≈ 16 GiB of bucket files
  *regardless of `n`*, so it needs the real multi-SSD machine. Also pre-existing
  (the demo body is byte-identical to the original).
- **A file-leak bug was fixed in passing.** `run_benches.py`'s `count_sort` entry
  declared its bucket glob as `csrt_bucket_*`, but the files are named
  `csrt_bucket0`, so the trailing underscore meant no clean pass ever matched
  them — every run stranded ~4 GB. Now `csrt_bucket*`. Since
  `BENCH_FILE_GLOBS` is derived from each entry's `data_globs`, and
  `clean_bench_data.py` reuses it, a wrong glob anywhere silently leaks; worth
  checking when adding an entry.
- **`run_benches.py` gained `pre_argv`**, placed ahead of `n`, so the eleven
  primitive-demo entries can target the one shared binary with their subcommand
  as `argv[1]`.

## Verification performed

- `make test` — all 6 binaries pass (`primitivesTest` 16/16 cases, `delayedTest`
  148 assertions, the four example tests green).
- `make examples` — all 8 binaries build; the 7 examples were each run and each
  passed its in-DRAM differential cross-check (exit 0).
- 9 of the 11 `primitive_demos` subcommands run and emit their original `CSV,`
  columns; the two exceptions are the pre-existing `cut` and `count_sort` issues
  documented above.
- `make format-check` — clean.
- Dependency tracking confirmed: touching a header rebuilds its dependents.
- `run_benches.py` and `summary_figure.py` import cleanly; `summary_figure.py`'s
  ordering and registry-membership asserts pass against the trimmed registry.
