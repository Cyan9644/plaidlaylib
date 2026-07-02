# Delaying `filter` and `flatten` in the out-of-core (chunk) setting

Notes on why `map`/`scan` delay cleanly in [chunk_delayed.h](chunk_delayed.h) but
`filter` and `flatten` do not — both in general and for the specific "delayed"
construction the block-delayed paper describes — and which special cases *can*
still be fused.

Notation: `n` = input elements, `c` = number of chunks (`≈ n / ELEMS_PER_CHUNK`),
`σ` = selectivity (fraction of elements a predicate keeps, so `σn` survivors).

---

## Baseline: why `map` and `scan` delay easily

A delayed sequence is a *(source + per-chunk iterator factory)*. `map` and `scan`
delay trivially because **output chunk `i` is element-for-element aligned with input
chunk `i`**:

- the element count of output chunk `i` equals input chunk `i`'s count, which is
  known from chunk metadata (`chunk.used / sizeof(T)`) — no data read required;
- a "block" is just "input chunk `i`, transformed on the fly," so the consumer
  loops a known count and parallelizes across chunks;
- `scan` is *partially* delayed: one read pass computes the `O(c)` block offsets,
  then the second pass stays lazy. The offsets are cheap because each is a
  reduction of a chunk, again 1:1 with the input.

Everything below breaks because this 1:1, count-known-up-front alignment is lost.

---

## Why `filter` is hard

### In general

`filter`'s output structure is **data-dependent**: the number of survivors in a
chunk is unknown until you read the chunk and run the predicate. Consequently:

- the total output size is unknown up front;
- the global position of each chunk's survivors is unknown;
- **output-block boundaries no longer line up with input-chunk boundaries** —
  compaction shifts every survivor leftward by the count of everything dropped
  before it.

The positions you'd need to address the output by index are prefix-sums of the
per-chunk survivor **counts**, and *a survivor count cannot be obtained without
running the predicate over every element*. So "just computing the offsets" is
already a full filtering pass.

### The paper's delayed construction (flatten-style)

The block-delayed paper expresses filter as a `flatten` of per-block survivors:

```
filter(A, p)  ≡  flatten( map_over_blocks(A, λ block → compacted_survivors(block)) )
```

1. filter each block into a small buffer of its survivors (these per-block buffers
   sum to exactly the survivor count — "allocations precisely equal to the size of
   the surviving elements");
2. scan the per-block counts → global `offsets` + total `m`;
3. the *delayed* part is the flatten: a re-blocked BID whose fixed-size output
   block `j` is produced by an iterator seeded via `upper_bound(offsets, j·B)` and
   walking a `flatten_iterator` forward — so **one output block can span survivors
   from more than one input block**.

parlaylib implements `flatten` exactly this way (delayed). Its
`block_delayed::filter` happens to *force* the final flatten (`filter_map`
materializes a contiguous `sequence`), but the lazy machinery exists and filter
*could* ride on it.

### Why it's hard *out-of-core* specifically

In DRAM the per-block survivor buffers are cheap and fit in memory; the only thing
"delayed" is skipping the final re-blocking copy. Out-of-core the survivor *values*
(`σn`) are the bulky thing, and the whole premise is data > DRAM, so **they do not
fit in RAM**. That leaves two variants, both costly:

- **Materialize the survivors** — write each chunk's survivors to SSD. This is a
  real `σn` write (with O_DIRECT padding slack per region). The delayed re-blocking
  then reads those regions back with **unaligned, sub-`CHUNK_SIZE` reads** at output-
  block boundaries, and adjacent output blocks double-read the region they straddle.
  This is dominated by — and strictly messier than — just compacting densely at
  write time, which is what the eager [chunk_filter.h](chunk_filter.h) /
  `delayed::filter` terminal already do (one aligned write pass, immediately a clean
  `chunk_seq`).

- **Stay fully lazy** (store nothing) — *this is the tempting-but-bogus one.* The
  offsets still force a full counting pass: you read all of `A`, run the predicate
  on every element, and keep only the `O(c)` counts, **throwing the survivors away**.
  Then producing values re-reads the input and re-runs the predicate. So fully-lazy
  does the filtering work **twice** and reads the input **twice**, trading the `σn`
  write for a second full read + a second predicate evaluation. On bytes moved it
  only breaks even when `σ → 1` (almost everything survives, so the avoided write
  was nearly a whole pass anyway), and even then the second read is the misaligned
  one. You can also only *resume* filtering at an input-chunk boundary, not at an
  arbitrary survivor offset, so producing output block `j` means re-reading from the
  preceding chunk boundary and skip-filtering forward — extra read + discarded work
  per output-block boundary.

**Conclusion:** out-of-core you essentially always *store* the survivors. The
offset/re-blocking "delayed filter" is faithful to the paper but is dominated by the
dense eager filter, because delay cannot avoid materializing `σn` whenever the
consumer needs the result addressable by position.

---

## Why `flatten` doesn't seem straightforward (and is doubly hard)

### The data model blocks it first

A `chunk_seq` stores **flat POD** (`uint64_t`). There are no nested sequences to
flatten — the input type `flatten` needs (a sequence of sequences) does not exist in
our representation. To even have something to flatten you'd need an on-disk encoding
of variable-length runs, which the chunk model doesn't provide. So flatten is "not
applicable" before any structural argument.

### Even with nested sequences, the structure is hard

Suppose elements *were* `(offset, length)` descriptors of sub-sequences. Flatten then
has filter's data-dependent shape, with one mitigating difference: the sub-sequence
lengths may be available as **metadata** (so the offset scan need not read the bulk
data — unlike filter, whose counts require running the predicate). The paper's
delayed flatten uses that: scan the lengths → `offsets`, return a re-blocked BID
(`upper_bound` + `flatten_iterator`), copying no data. parlaylib implements this.

### Why it's hard out-of-core

The re-blocking that's nearly free in DRAM becomes the problem on SSD:

- an output block spans several input sub-sequences, which live at **different
  offsets in different files on different drives**, so materializing one output block
  is a *scatter-gather* of unaligned, sub-`CHUNK_SIZE` reads — alignment waste and
  read amplification, and it breaks the "one sequential `CHUNK_SIZE` read per chunk"
  model the reader is built around;
- if the sub-sequences are small and numerous you get tiny reads and go IOPS-bound
  (cf. the sub-256 KiB note in [configs.h](../configs.h)); if they're large it
  degenerates to a plain concatenation.

So flatten is blocked twice: by the flat-POD data model, and — even past that — by
re-blocked reads that are scattered and misaligned across drives.

---

## What *can* be delayed cheaply

**Principle.** Out-of-core, delay pays off only when it (a) avoids materializing a
bulky (`σn`- or `n`-sized) intermediate **and** (b) the consumer does not need that
intermediate addressable by global position. Operations that *fold/stream* — that
only need to **visit** each element once — sidestep the offset problem entirely.

- **`filter → reduce`** (and `filter → map → reduce`) — the clean win. reduce needs
  no offsets, no re-blocking, no second pass: read each chunk once, apply the
  upstream maps, test the predicate, fold survivors into a per-block local, then
  combine the per-block locals **in chunk-index order** (the reader is out of order,
  but locals are indexed by `chunk_index`, an `O(c)` ordered combine like
  `ChunkScan`'s block prefix; a commutative monoid needs no ordering). Single read,
  zero writes, nothing materialized — because reduce never addresses the output by
  position.

- **`filter → {count, sum, any associative aggregate}`** — same mechanism as reduce.

- **`map` fused *into* a filter's read pass** — already done by the existing
  `delayed::filter` terminal: preceding maps are applied while reading for the
  filter, so they cost no extra I/O. This is "delaying the map into the filter," and
  it is free.

- **`filter → filter → reduce`** — a chain of predicates folded in one pass (nested
  filtering, terminal reduce), still a single read.

- **`flatten → reduce`** *would* be the analogous cheap case (fold each sub-sequence,
  combine) — but it is blocked by the data model, not the structure: there are no
  nested sequences to fold.

**What does *not* delay cheaply** — `filter → scan`, `filter → force`
(materialize to a `chunk_seq`), `filter → zip`: all need the survivors addressable by
index, so they must materialize `σn` and gain little from delay. Same for the flatten
analogues.

---

## Summary

| pipeline | delayable out-of-core? | why |
|---|---|---|
| `map`, `scan` | yes (scan: partial) | output chunk `i` aligns 1:1 with input chunk `i`; counts/offsets known cheaply |
| `filter → reduce` / `→ map → reduce` / `→ count/sum` | **yes, fully fused** | reduce only visits each survivor once; no offsets, one read, no writes |
| `filter → scan` / `→ force` / `→ zip` | no | consumer needs `σn` survivors addressable by index → must materialize |
| `filter` as a re-blocked delayed BID (paper-style) | technically yes, not worth it | survivors don't fit in RAM → must write `σn` anyway; lazy re-blocking adds misaligned/duplicated reads; fully-lazy re-does the filter twice |
| `flatten` (any) | no | chunks hold flat POD (no nested sequences); even given them, re-blocked reads scatter across drives at unaligned offsets |

Bottom line: the only filter/flatten fusion that genuinely pays out-of-core is into a
**folding terminal** (`reduce` and friends), where no offsets and no second pass are
needed. Everything that needs the result laid out and addressable must materialize
it, and there the eager dense [chunk_filter.h](chunk_filter.h) layout already does
the cheapest correct thing.
