# Removing the "dense-except-last" chunk invariant — breakage report

## Context

A `chunk_seq` currently carries two invariants:

1. **index-ordered** — `chunks[i].index == i` (**keeping** this), and
2. **dense-except-last** — every chunk but the final one has `used == CHUNK_SIZE`.

The goal is to drop (2) so any chunk may be partially filled, accepting the load-imbalance
cost. Decisions taken: **scope is tolerate-only** (consumers must handle ragged input;
producers keep emitting dense output), and **zero-length chunks (`used == 0`) are legal**.

The load-bearing consequence of (2) — the thing every break below is an instance of — is
the identity

> global element index `g` ⟺ (physical chunk `g / epc`, offset `g % epc`)

and its dual, "logical chunk `i` covers global elements `[i*epc, (i+1)*epc)`". Removing
(2) means every `/ epc`, `% epc`, and `i * epc` that addresses **stored** data must become
a prefix sum plus a predecessor search.

---

## The most important finding: the invariant is already violated today

This is not a greenfield relaxation. Four mechanisms already emit chunk_seqs with short
**interior** chunks:

| producer | mechanism |
|---|---|
| [`flatten`](ChunkSequence/Primitives/flatten.h#L16-L43) | re-indexes and glues chunk lists, no re-densify. Callers: `external_samplesort.h:210,411,599`, `random_shuffle.h:156,248`, `unique.h:57`, `sort_buckets.h:110`, `chunk_csr::merge` |
| [`count_sort.h:586-595`](ChunkSequence/Primitives/count_sort.h#L586-L595) | with `disk_span > 1`, each shard contributes its own short trailing chunk mid-bucket. `fuse()` at 598-608 likewise |
| [`delayed::force`](ChunkSequence/Primitives/delayed.h#L1368-L1370) | stamps `used = chunk_len(i) * sizeof(R)`; for any `R` narrower than 8 B **every** chunk is short |
| [`cut`](ChunkSequence/Primitives/cut.h#L197-L205) / `from_chunks` / `push_back` | build a partial *first* chunk / preserve ragged by design / create a `used == sizeof(T)` chunk |

So several breaks below are **live bugs, not hypotheticals**. Three worth calling out:

- `ExternalKthSmallest.h:245` calls `size<T>(buckets[b])` on a `count_sort` bucket — wrong
  whenever `disk_span > 1`.
- `chunk_csr::merge` → ragged `edges` → `segmented_reduce_over_edges`, which computes
  `global_start = chunk_idx * ept`. Corrupts Bellman-Ford / graph results.
- `external_linefit.h:25-29` computes `n` **correctly** (sums `used`) but then feeds the
  same sequence to `delay<double>`, which computes length via the invariant — so it divides
  a sum taken over the wrong element count by the right `n`.

This reframes the change: it is less "relax a guarantee" than "make the library honest
about a shape it already produces."

---

## Part 1 — The delayed layer (the only structurally hard problem)

`delayed.h` assumes **every node in a fused tree shares one uniform logical chunk grid**,
so "chunk `i`" denotes the same element range in every node. That is what lets
`zip_node::build` hand chunk `i` to both children and pair them positionally. The initial
hypothesis (zip breaks) is correct, but the root cause is one level down and affects more
than zip.

### 1a. `leaf_source` — the root break
[delayed.h:256-285](ChunkSequence/Primitives/delayed.h#L256-L285), built by
[`delay()` at 567-574](ChunkSequence/Primitives/delayed.h#L567-L574):

```cpp
const size_t len = nc == 0 ? 0 : (nc - 1) * epc + seq.chunks[nc-1].used / sizeof(T);
size_t chunk_len(size_t i) const { const size_t base = i*epc;
                                   return base >= len ? 0 : std::min(epc, len - base); }
```

Three independent failures:

- **`len` overstates the length** — must be `Σ used/sizeof(T)`. Propagates to `zip`'s
  `max(lenA,lenB)`, `force`'s output size, `size(d)`, linefit's denominator.
- **`chunk_len(i)` returns `epc` for a short interior chunk**, but `plan(i)` reads only
  `chunks[i].used` bytes. The body then walks past the live data. Because reader buffers
  are a full `CHUNK_SIZE` ([chunk_seq_reader.h:53](ChunkSequence/Primitives/chunk_seq_reader.h#L53))
  this stays inside the allocation — so it is **recycled-pool garbage, silent and
  non-deterministic**, not a segfault.
- **`num_chunks()` decouples from `chunks.size()`** — once `len` is correct,
  `ceil(len/epc) < chunks.size()`, and `plan`'s `if (i < num_chunks())` guard means
  trailing physical chunks are **never read**. With today's inflated `len` the reverse
  happens: `src->chunks[i]` indexes a `std::vector` out of bounds.

Note the drivers already receive the ground truth and throw it away —
[delayed.h:696-697, 808-809, 1082-1083](ChunkSequence/Primitives/delayed.h#L808-L809) all
do `auto [buf, n, rid] = reader.Poll(); (void)n;` and use `d.chunk_len(ci)` instead.

### 1b. `cut_source` — worst offender
[delayed.h:317-386](ChunkSequence/Primitives/delayed.h#L317-L386). `segments()` at 349-351
is the global-index→physical-address map in pure grid arithmetic:

```cpp
const size_t g0 = start_index + i * epc;
s.phys_lo = g0 / epc;  s.offset_lo = g0 % epc;
const size_t avail_lo = src->chunks[s.phys_lo].used / sizeof(T) - s.offset_lo;
```

- Wrong chunk and wrong offset under any raggedness.
- `avail_lo` **underflows** (`size_t`) once `offset_lo` exceeds the chunk's real element
  count — giving a huge `avail_lo`, so `take_lo = cl`, `hi` is never planned, and
  `cut_iter` walks `cl` elements from a bogus offset **past the end of the buffer**.
- The **"at most two physical reads"** assumption (stated in the class comment at
  [309-316](ChunkSequence/Primitives/delayed.h#L309-L316)) fails: one logical chunk can now
  span arbitrarily many physical chunks. `Seg`'s fixed two-slot shape, `plan`'s two
  `p.need`s, `build`'s two `r.next()`s and `cut_iter`'s two-pointer walk all need variable
  fanout.
- `cut()`'s bounds `CHECK(end_index <= total)` at
  [592](ChunkSequence/Primitives/delayed.h#L592) uses an invariant-derived `total`, so it
  passes for out-of-range cuts.

Live consumers: `external_bellman_ford.h:137` and `external_compressed_sparse_row.h:77` —
per-vertex adjacency slices over a CSR edge array, with `sizeof(weighted_edge) == 32`, so
the two-chunk assumption is already load-bearing.

### 1c. `zip_node` — confirmed, with four distinct failures
[delayed.h:458-492](ChunkSequence/Primitives/delayed.h#L458-L492).

**`zip` does no alignment at all.** Line 479 is the assumption written down:

```cpp
const size_t eb = i * ELEMS_PER_CHUNK;   // never asks either child where its chunk i begins
const size_t rA = eb >= lenA ? 0 : std::min(n, lenA - eb);
```

The node interface (`length/num_chunks/chunk_len/plan/build`) has **no "start offset of
chunk i" accessor**, so there is nothing to align with.

1. **Misaligned pairing.** If A's chunks are `[EPC/2, EPC/2, EPC]` and B's are `[EPC, ...]`,
   `a.build(1)` yields A's elements from global `EPC/2` while zip believes chunk 1 starts at
   `EPC`. Every pair is off. Silent.
2. **`pad_iter::remaining` derived from `lenA`, not `a.chunk_len(i)`**
   ([481-484](ChunkSequence/Primitives/delayed.h#L481-L484) →
   [158](ChunkSequence/Primitives/delayed.h#L158)). `rA` can exceed the child's real chunk
   content, so `pad_iter` keeps dereferencing instead of switching to `pad` — stale pool
   bytes, and a **heap OOB read** when `rA*sizeof(T) > CHUNK_SIZE` (possible whenever
   `sizeof(T) > 8`, since `n` is capped at `ELEMS_PER_CHUNK`, not `epc`). If the child
   returned the `nullptr` dummy, this is a **null deref**.
3. **Chunk-count mismatch** ([469](ChunkSequence/Primitives/delayed.h#L469)) — zip grids on
   `max(lenA,lenB)`, ignoring children's counts, so a child's tail chunks are never planned.
4. The strict-length `CHECK` at [612-614](ChunkSequence/Primitives/delayed.h#L612-L614)
   compares invariant-derived lengths, giving **both** false positives (equal-length
   sequences fragmented differently) and false negatives (unequal lengths with compensating
   fragmentation).

### 1d. `leaf_index` (`tabulate`) — breaks when zipped
[delayed.h:404](ChunkSequence/Primitives/delayed.h#L404):
`make_counting(i * ELEMS_PER_CHUNK, f)`. It has no backing file, so nothing *forces* it
onto a grid — it simply chose the uniform one. Zipped against a ragged `leaf_source`,
`A[S_i + j]` gets paired with `f(i*EPC + j)`. Silently wrong, no diagnostic.

This is exactly what
[`Permutation::Run`](ChunkSequence/Primitives/random_shuffle.h#L232-L238) does to recover
global indices — `zip(delay(seq), tabulate(...))` — so it is correct only while `seq` is
dense.

### 1e. Zero-length chunks specifically
Since `used == 0` is to be legal, one more hard break:
[delayed.h:944-950](ChunkSequence/Primitives/delayed.h#L944-L950) in
`sequential_for_each_chunk` does `if (c.used == 0) continue;` — skipping the read but
leaving that pool slot holding **the previous call's data**, which `Resolver` then hands
out as real. Unreachable today; a silent-corruption break the moment empty chunks are
legal. Predecessor searches also need care: runs of equal prefix-sum values mean
`upper_bound` vs `lower_bound` is no longer interchangeable.

### 1f. Nodes needing no change
`map_node` (409-428) and `scan_node` (432-452) forward `chunk_len` verbatim.
`filter_node` (499-556) defines its own output grid and reads its child only through
`d.chunk_len(k)` — and its `locate()` predecessor search over an exact prefix sum
([514-517](ChunkSequence/Primitives/delayed.h#L514-L517)) is the pattern the leaves need.
`scan` is decomposition-agnostic: `per_chunk_reduce` keys by logical chunk index and
`segmented_reduce_generic` already prefix-sums `chunk_len`
([1228-1231](ChunkSequence/Primitives/delayed.h#L1228-L1231)). The `Planner` dedup keys on
`(source, chunk.index)` — invariant (1), not (2) — so it survives untouched.

### Two candidate designs — (A) recommended

**(A) Re-window ragged leaves onto a single uniform logical grid.** Keep the grid as the
delayed layer's contract; make the *leaf* absorb the raggedness. `cut_source` already does
90% of this. Generalize it: a per-source `used` prefix sum (snapshotted in a `shared_ptr`
at `delay()`/`cut()` time), binary search replacing every `/epc` and `%epc`, and an
**N-segment** iterator replacing `cut_iter`'s two-pointer walk. `leaf_source` becomes
`cut_source` with `start_index == 0`. `zip`, `tabulate`, `map`, `scan`, `filter`, both
drivers and the Planner all stay untouched.

**(B) Add `chunk_start(i)` to the node interface and align zip by element position.**
This is the natural reading — but it does not terminate. If A's and B's chunk boundaries
differ, there is *no* decomposition where both children's chunk `i` aligns; you must merge
the two boundary sets, at which point at least one child's logical chunk spans several of
its physical chunks. **Design B collapses into Design A at the point where data is
fetched**, having also broken zip against `tabulate`.

The tradeoff to weigh before choosing A: it implies **one logical grid for all element
sizes**. Today `leaf_source<T>` grids on `epc = CHUNK_SIZE/sizeof(T)` while
`zip`/`tabulate` grid on `ELEMS_PER_CHUNK` — a latent misalignment for any non-8-byte type
that is currently unreachable only because nobody zips one against a `tabulate`. Unifying
on `ELEMS_PER_CHUNK` fixes that, but for a 32-byte `weighted_edge` one logical chunk
becomes 16 MB across 4 physical reads. That is the accepted perf hit, but it is a
memory-footprint decision worth making deliberately.

### Other delayed-layer issues to fix while in here

- [`force` 1381-1387](ChunkSequence/Primitives/delayed.h#L1381-L1387) — writes `n` elements
  into a `CHUNK_SIZE` buffer where `n = d.chunk_len(ci)`. The
  `static_assert(sizeof(R) <= 8)` at 1325 guards the wide case, but **not** a chain rooted
  at `leaf_source<uint32_t>` (`epc = CHUNK_SIZE/4`) mapped to `uint64_t`:
  `n*sizeof(R) = 2*CHUNK_SIZE` → **heap overflow plus a `size_t` underflow in the `memset`
  length**. Latent hard break today.
- [`Resolver::next()` 240](ChunkSequence/Primitives/delayed.h#L240) and
  [`filter_node::build` 540](ChunkSequence/Primitives/delayed.h#L540) are unbounded. Safe
  today by construction; any plan/build divergence introduced by this refactor turns them
  into OOB reads or an infinite loop. Add bounds as part of the work.
- [`scan_node` 449](ChunkSequence/Primitives/delayed.h#L449) silently seeds `identity` past
  `offsets->size()`; make it a `CHECK`. Also snapshot the leaf prefix sum — `push_back`
  mutates `chunks.back().used` and can append, staleing the offsets.
- `filter`'s `static_assert(sizeof(R) <= 8)` at 1407 is spurious (it grids on
  `epct = CHUNK_SIZE/sizeof(R)` throughout); `force`'s can become a runtime `CHECK`.

---

## Part 2 — Eager layer, hard breaks

### 1. `ChunkSequenceOps::size` — fix this first
[chunk_seq.h:613-623](ChunkSequence/Primitives/chunk_seq.h#L613-L623). The doc comment *is*
the invariant ("O(1): every chunk but the last is full"). Becomes `Σ used/sizeof(T)`,
O(#chunks) — cheap (headers are in DRAM) but no longer O(1); cache it on `chunk_seq` if
that matters. Unblocks `sample.h:38`, `cut.h:139`, `chunk_bigint_add.h:76,77,142,143`,
`chunk_convex_hull.h:214,281`, `ExternalKthSmallest.h:245`, `scalar_test.cpp`.

### 2. `chunk_seq::operator[]`
[chunk_seq.h:169-171](ChunkSequence/Primitives/chunk_seq.h#L169-L171) — `ci = i/ept`,
`off = (i%ept)*sizeof(T)`. Needs prefix sum + `upper_bound`. On the critical path of
`ChunkBigIntAdd` (`chunk_bigint_add.h:83,84,114,197`). `push_back` stays correct (it only
inspects the last chunk) but is itself a producer of ragged sequences.

### 3. `ChunkSegmentedReduce`
[segmented_reduce.h:39](ChunkSequence/Primitives/segmented_reduce.h#L39) —
`global_start = chunk_idx * ept` drives the whole fully-owned/boundary classification.
Needs a prefix-sum array indexed by `chunk_idx`, computed before `RemoveWorker`. Also fixes
`chunk_csr::segmented_reduce_over_edges` → Bellman-Ford.

### 4. `ChunkFindIf`
[TODO/chunk_find_if.h:47](ChunkSequence/Primitives/TODO/chunk_find_if.h#L47) —
`idx * epct + j`. Its `n` at line 36 already sums `used`; only the index math is wrong. Its
header comment at 20-23 states the invariant explicitly.

### 5. `sample`
[sample.h:38](ChunkSequence/Primitives/sample.h#L38) — `total = size<T>(seq)` then
`uniform_int_distribution(0, total-1)`. Lines 40-41 already use `scan_size` correctly, so
fixing `size` fixes it — but note the failure mode is an over-reported `total` feeding
`scan_find` an out-of-range index, i.e. a read past the end.

### 6. `NReader` co-indexing, and its `count_sort` consumer
[n_reader.h:34-38](ChunkSequence/Primitives/n_reader.h#L34-L38) pairs by `chunk.index`
only. Pairing *element k of chunk i* additionally requires **identical per-chunk element
counts** — free today from "both dense, same length", not free after. The consumer at
[count_sort.h:93-99](ChunkSequence/Primitives/count_sort.h#L93-L99) takes
`n = match.sizes[0]` and indexes `val_ptr[k]` with it, never consulting `match.sizes[1]`.
Needs a CHECK at minimum.

### 7. The forward halo
[dense_pack.h:410-413](ChunkSequence/Primitives/dense_pack.h#L410-L413):
`hn = std::min(halo, seq.chunks[i+1].used / sizeof(T))` — sourced from **exactly one** next
chunk. A short chunk `i+1` silently yields `hn < halo` and the consumer misses
boundary-crossing events. [flat_map.h:34-37, 48](ChunkSequence/Primitives/flat_map.h#L34-L37)
`CHECK(halo < CHUNK_SIZE/sizeof(T))` guards physical capacity, not actual length. Affects
`ChunkFlatMap`, `chunk_kmp.h`, `chunk_rabin_karp.h`, and `external_rmat.h` (halo=1 dedup).
Needs multi-chunk halo assembly — and with empty chunks legal it must skip over them.
`external_rmat.h:188-193`'s `CHECK(ch.used > 0)` should **stay**.

### 8. `small_sequence_ops` run coalescing — a coupled pair
[L174-180](ChunkSequence/Primitives/small_sequence_ops.h#L174-L180)'s coalescing guard
includes `bs.chunks[ci-1].used == CHUNK_SIZE`, so many partial chunks collapse every run to
length 1 — one `open()` + one SQE per chunk. **Perf only.** But that guard is exactly what
makes the [L222-226](ChunkSequence/Primitives/small_sequence_ops.h#L222-L226) single-run
fast path safe ("buf's live bytes are already a gap-free prefix"). **Do not relax L178
without also deleting the L222 fast path** — the multi-run path below it already
compacts/expands correctly via `used`.

### 9. CSR access path
[external_compressed_sparse_row.h:67-104](ChunkSequence/helper/external_compressed_sparse_row.h#L67-L104).
`get_adjacent` / `edge_exist` → `sequential_cut_no_compression`, which walks headers
accumulating `used` ([cut.h:148-155](ChunkSequence/Primitives/cut.h#L148-L155)) and is
**already ragged-safe** — only its `size<T>` guard at
[cut.h:139](ChunkSequence/Primitives/cut.h#L139) breaks. `delay_get_adjacent` breaks via
`delayed::cut`; `segmented_reduce_over_edges` via #3.

---

## Part 3 — Already ragged-safe (no change needed)

Most of the library, worth stating explicitly:

- [`chunk_seq_reader.h`](ChunkSequence/Primitives/chunk_seq_reader.h) — entirely
  `used`-driven (`req->used_bytes = c.used`, `read_size = AlignUp(c.used)`).
  **Constraint to keep:** `used <= CHUNK_SIZE`, since the pool buffer is exactly one
  `CHUNK_SIZE` ([L53](ChunkSequence/Primitives/chunk_seq_reader.h#L53)).
- `to_vector`, `consolidate`, `from_chunks` — prefix-sum `used`; `from_chunks` is already
  documented as not re-densifying.
- `ExternalTransform` / `RemoveWorker` / `ChunkEmitter` — bodies get `(ptr, n, index)` with
  `n` from `used`; `emit` records `count * sizeof(R)`. So **`ChunkMap` is safe**.
- `ChunkReduce`, `ChunkScan` (keys by `chunk_idx` throughout), `ChunkHistogramByIndex`,
  `ChunkFlatTabulate` (grids a virtual index range, not stored chunks), `ChunkPartition`
  and `ChunkFilter` as consumers.
- `DensePackStream`'s [`pos_of`](ChunkSequence/Primitives/dense_pack.h#L252-L255) — the
  model to copy; `gpos` is already a true prefix sum.
- `pack.h:75-82`, `materialize.h:30-33`, `scan_find.h:19-33`, `linear_find.h:31-45`,
  `unique.h:34`, `random_shuffle.h:44-48`, `sort_buckets.h:28-30,64-73`,
  `external_samplesort.h:83,269,459`, `operation.h`, `bucketed_file_writer.h`.

Every `* ept` in `chunk_seq.h` other than `size`/`operator[]` (lines 277, 332, 361, 416,
457, 482, 549, 585) sits inside a **producer** defining its own dense output — correct
as-is. Likewise `dense_pack.h:132-165` and `delayed.h:1473-1496` are output-packing math,
and `small_sequence_ops.h:192,272` index CHUNK_SIZE *slots*, not elements.

`merge.h` and `reverse.h` are broken stubs (`merge.h` does not compile; `reverse.h:37`
reverses the physical chunk rather than `used/sizeof(T)` elements, already wrong today) —
they need writing against the new model, not fixing. `chunk_fft.h` maintains its own
uniform grid over data it generates itself and does not touch the delayed layer, so it is
safe as long as it keeps producing dense output.

---

## Part 4 — Tests and docs that encode the invariant

**Tests that will fail and need updating** (all assert dense-except-last on output — under
tolerate-only scope, most of these assertions remain *valid* since producers stay dense;
they just stop being library-wide guarantees):

`filter_test.cpp:150-162`, `flat_tabulate_test.cpp:36-46` (`check_packing`),
`partition_test.cpp:67-81`, `kmp_test.cpp:84-94`, `rabin_karp_test.cpp:85-95`,
`external_rmat_test.cpp:63-79` (`check_dense`), `iota_test.cpp:74`,
`scalar_test.cpp:43-94`, `segmented_reduce_test.cpp:88-109`,
`delayed_test.cpp:734` (uses `ci * ELEMS_PER_CHUNK` as a reference offset — the same
mistake [materialize.h:118-124](ChunkSequence/Primitives/materialize.h#L118-L124) documents
having already fixed once).

**The coverage gap is the real risk:** every test source is built by `iota`/`tabulate`, so
the suite constructs **no sequence with a short interior chunk** and cannot detect any break
above.

**Docs stating the invariant:** `CLAUDE.md` (Data model §, the `ChunkPartition` note, the
`dense_pack` note, the delayed-layer constraints), plus in-code prose at
`partition.h:37-48` and `169-173` (the "do NOT concatenate" paragraph — obsoleting it is
the change's main payoff), `chunk_seq.h:157-160,613-614,632-634`, `flat_map.h:34-37`,
`filter.h:17-19`, `pack.h:53-56`, `TODO/chunk_find_if.h:20-23`, `dense_pack.h:49,69-70`,
`n_reader.h:34-38`, `cut.h:92-96,112-114`, `small_sequence_ops.h:61-76,222-225`,
`external_rmat.h:188-191`, `delayed.h:50,309-316`, `chunk_bigint_add.h:130-135`,
`chunk_kmp.h:65-66`, `chunk_rabin_karp.h:97-98`, and several files under
`examples/external_TODO/`.

---

## Suggested implementation order

1. **`size<T>()`** → sum of `used`. Unblocks `sample`, `cut`, bigint, convex hull,
   kth-smallest.
2. **Promote a shared prefix-sum helper.**
   [`scan_find.h:19`](ChunkSequence/Primitives/scan_find.h#L19) `scan_size` already is one —
   use it in `operator[]`, `ChunkSegmentedReduce:39`, `ChunkFindIf:47`.
3. **Thread a global `start` offset through `ExternalTransform`'s body signature**
   ([external_engine.h:113](ChunkSequence/Primitives/external_engine.h#L113)) so no body
   ever recomputes `index * ept`. `DensePackStream:252-255` shows the shape.
4. **The delayed leaves** — `leaf_source`/`cut_source` merge, per-source prefix sum,
   N-segment iterator (design A). Without this, `unique`, `random_shuffle`, `count_sort`,
   Bellman-Ford, CSR and linefit stay broken.
5. **Multi-chunk halo** in `dense_pack.h:410-413`, skipping empty chunks.
6. **`NReader`** per-chunk count CHECK, and `count_sort.h:93-99`.
7. **Zero-length-chunk cleanups**: `delayed.h:947` stale buffer, predecessor-search
   `upper_bound`/`lower_bound` handling of equal-prefix runs.
8. **Bounds hardening** flagged in Part 1: `Resolver::next`, `filter_node::build` loop,
   `scan_node` offsets CHECK, `force`'s buffer-overflow guard.

**Explicitly out of scope** under tolerate-only: deleting `ChunkPartition`'s sequential
tail merge ([partition.h:169-204](ChunkSequence/Primitives/partition.h#L169-L204)),
dropping the `DensePack` carry machinery, and relaxing `small_sequence_ops.h:178`. Each is
a real simplification/speedup, but each makes a producer emit ragged output and is better
taken one at a time afterwards.

---

## Verification

- `make test` is a regression net only — it covers partial-*last*-chunk cases heavily but
  constructs no ragged interior, so it cannot catch any of the above.
- **The critical new coverage** is a test helper that builds a deliberately ragged
  `chunk_seq` — hand-written headers of varying `used` over an existing file (via
  `from_chunks`), including `used == 0` chunks and a run of them — then runs every
  primitive through it against a DRAM baseline. Easiest natural source: `count_sort` with
  `disk_span > 1`, which already produces one.
- Highest-value targeted cases: `delayed::zip` of two differently-ragged sources; `zip` of a
  ragged source against `tabulate` (the `Permutation::Run` shape); `delayed::cut` spanning
  >2 physical chunks; `ChunkSegmentedReduce` with bounds straddling short chunks;
  `ChunkFlatMap` with a halo over a short chunk; `chunk_seq::operator[]` across ragged
  chunks; `size<T>()` on a `flatten`ed bucket list.
- The three live bugs named at the top double as end-to-end checks: `external_linefit` on a
  ragged input, `ExternalKthSmallest` with `disk_span > 1`, and Bellman-Ford over a merged
  `chunk_csr` should all go from wrong to right.
