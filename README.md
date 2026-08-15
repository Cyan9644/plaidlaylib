# plaidlaylib

Parlay-style parallel primitives (map, reduce, filter, scan, flat-tabulate,
find_if, sort, …) for data too large for DRAM, stored across many SSDs. All I/O
goes through `io_uring` with `O_DIRECT`.

The hope of the project is to show that multi-SSD programming can be made
ergonomic with certain abstractions, without giving up the parallelism of an
in-memory implementation. When applicable we also partly by delay work to cut 
the number of I/Os.

## Quick start

```bash
make deps     # fetch parlaylib + upstream examples, build abseil (first time only)
make test     # build and run the six correctness binaries
make examples # build the eight example binaries into bin/
```

Needs `g++` (C++17), `cmake`, `git`, and system `liburing`. 
Assumes `SSD_COUNT` mount points named per `SSD_ROOT` — by default
`/mnt/ssd0 … /mnt/ssd29`. Edit `configs.h` for your machine.

## What is where

```
configs.h                machine knobs: SSD_COUNT, SSD_ROOT, CHUNK_SIZE, O_DIRECT_MULTIPLE
Makefile                 all compilation; header deps tracked via -MMD -MP
CLAUDE.md                the full reference
```

### The library — `ChunkSequence/Primitives/`, five headers

Strict dependency order; each includes only the one before it.

| header | what's in it |
|---|---|
| `chunk_seq.h` | data model + I/O substrate: `chunk`/`chunk_seq`, `tabulate`/`iota`/`from_file`/`to_chunk_seq`/`consolidate`, `ChunkSequenceReader`, `UnorderedFileWriter`, `SimpleQueue`, the `ExternalTransform`/`RemoveWorker`/`ChunkEmitter` engine, `DensePack`/`DensePackStream`, `NReader`, `BucketWriter` |
| `delayed.h` | the fused/lazy layer: `delay`/`tabulate`/`map`/`scan`/`zip` + the `reduce`/`force`/`filter` terminals. Chains compose with no intermediate I/O |
| `primitives.h` | the six core eager primitives: `ChunkMap`, `ChunkReduce`, `ChunkScan`, `ChunkFilter`, `ChunkFlatTabulate`, `ChunkFlatMap` |
| `secondary_primitives.h` | the rest of the eager layer: `ChunkSegmentedReduce`, `pack`/`pack_if`/`pack_value`, histograms, `ChunkFindIf`, `ChunkPartition`, `flatten`, `materialize`, `cut`, `scan_find`, `linear_find`. Includes `primitives.h`, so this one include gets the whole eager layer |
| `sort.h` | out-of-core sort/shuffle: `count_sort`, `group_by_index`/`group_by_key`, `process_inplace`(`_budgeted`), `ChunkOperation`/`apply`, `sample`, `sample_sort`, `random_shuffle_method`, `Permutation` |

### Tests — `ChunkSequence/tests/`, six binaries

`make test` runs all six, continues past a failure, and exits non-zero if any
failed. `make test TEST_ARGS=8000000` overrides the element count for every case.

| binary | covers |
|---|---|
| `primitivesTest` | 16 cases for `chunk_seq.h` / `primitives.h` / `secondary_primitives.h` / `sort.h`, cheap-first so a substrate break surfaces before the expensive sorts |
| `delayedTest` | the delayed layer |
| `kmpTest` `rabinKarpTest` `bigintAddTest` `convexHullTest` | the four examples that carry a correctness test |

### Examples — `ChunkSequence/examples/`
| | |
|---|---|
| `primes.cpp` | out-of-core sieve on `ChunkFlatTabulate` |
| `kmp.cpp` + `chunk_kmp.h` | KMP search; per-chunk automaton, cross-chunk matches via batch-local overlap |
| `rabin_karp.cpp` + `chunk_rabin_karp.h` | rolling-hash search mod 2^31−1 |
| `bigint_add.cpp` + `chunk_bigint_add.h` | n-limb add as one fused delayed chain (zip → classify → carry-scan → add → force) |
| `linefit.cpp` + `chunk_linefit.h` | fully-delayed least-squares fit; the zipped points never materialize |
| `convex_hull.cpp` + `chunk_convex_hull.h` | upper hull by quickhull; each level is one `ChunkPartition` pass, with a DRAM base case |
| `samplesort.cpp` | driver for `plaid::sample_sort` |
| `primitive_demos.cpp` | one binary, 11 per-primitive demos dispatched on `argv[1]`: `map`, `reduce`, `scan`, `tabulate`, `zip`, `filter`, `pack`, `count_sort`, `histogram_by_index`, `cut`, `random_shuffle` |

### Everything else

```
utils/          non-chunk-aware plumbing: paths, O_DIRECT alignment, fd/memlock
                limits, SYSCALL/ASSERT, ParseGlobalArguments, PLAID_TRACE markers
benchmarks/     three benchmark binaries + the Python sweep runners and plotters
benchresults/   figure sources and historical plot output
deps/           fetched by `make deps`; gitignored
results/        timestamped benchmark output; gitignored
