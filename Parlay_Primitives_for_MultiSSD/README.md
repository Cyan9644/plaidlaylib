This github is a platform for the CAAR REU studies on implementing Parlay primtitives for the MultiSSD setting.

Some ideas for primitives that could port well (or not so well)written by Alex,:

----------------------------------------------------------------------------
**PARLAY::FILTER**:

The basic block algorithm relies on the flatten operation and an existing implementation of filter.

This is the idea:
1. Break the input sequence into equal-sized blocks.
2. Filter each block in parallel and pack the resulting blocks into a contiguous area.
3. Flatten the contiguous area, resulting in a sequence of only the filtered elements.

Problems:
Flatten isn't implemented for MultiSSD and isn't so simple with the block algorithm given because it relies on merging with binary search.
This idea of a "contiguous area" is not clearly defined in this context. We don't want to just use a single SSD, so probably these are round-robined for parallel access.
However, if you were then going to flatten the full sequence, how exactly does this work when the individual sequences are spread across multiple SSDs?



There is also a cheating idea that immediately allocates memory the size of the input and assigns each thread to block iterable index i with n/#blocks work.
This is clearly bad, though, as the final size of the filtered array is unlikely to be close to as large as what we're attempting to filter.





Another parallel algorithm for filter:

1. Break the input into equal-sized blocks
2. Filter each block in parallel and calculate the toal number of elements that passed the filter in each block as well as the boolean predicate array that indicates whether element i satisfied or did not satisfy the predicate. Then write each block count to a small buffer in DRAM.
3. Perform parallel scan on the block counts buffer. This resulting prefix sum array gives us, for each index i, the starting position for which block i can begin writing its elements (count[i] - i of them)
4. Within each block, calculate another scan on the boolean predicate array -- this gives us the local offset for writing for each block. Crucially the prefix is increased iff the predicate evaluates to 0 at an index, which gives us the offset for the filtered array.
5. Allocate an array to hold the final elements of size block[last] + local_offset[block_last]
6.
```
auto filtered_index = Block_offsets[i] + local_offset; //perform writes to an auxiliary array
```

Problems:

We need auxiliary memory for the pass-count buffer and n additional memory for the boolean predicate array. This is probably not ideal when DRAM is so limited.
We need to allocate a large auxiliary array for the filter to write into UNLESS we use a delayed version. This would definitely need to be implemented with a BID because random-access would be very latent; stream-of-blocks has the issue that the parallel work is tied to individual blocks, which are likely to be imbalanced, and that accessing an indiviudal element requires block instantiation, removing our random access abilities. Stream of blocks also requires large block sizes to overcome the overhead of synchronization. 

----------------------------------------------------------------------------------
**Note on BIDs (Block of streams) vs Stream-of-blocks approach:**

In the stream-of-blocks approach, the stream is the outer loop; this means that all blocks must wait for each other to synchronize because the structure is inherently sequential. Parallelism here occurs in blocks themselves, not across blocks. In the context of a filter operation, the individual blocks may contain any number of remaining elements -- blocks read further down the line have no way of knowing their offset until it is computed for all blocks beneath them.

A BID, on the other hand, is defined by a length N, block size B, and a function that generates a stream for any block index. The iteration space is therefore predictably of size N/B even blocks. When you apply a filter to a BID, it doesn't evaluate it. Instead, it updates the streams with the new filter condition.


----------------------------------------------------------------------------------


This requires a MultiSSD version of the scan algorithm, as the boolean predicate array is exactly as long as the total number of elements (i.e. definitely does not fit in DRAM)






----------------------------------------------------------------------------
**PARLAY::SCAN && PARLAY::SCAN_INCLUSIVE**: 

ADDITIONAL OPTIMIZATION: We can cut out an additional write here by not actually computing prefix sums on the first pass. We just need the total sums for the offset buffer, and we can later overwrite the garbage data. This allows us to read data on the second pass, compute the prefix sums, and directly apply the offsets before writing back to disk. Because the scan inherently requires that 0-(n-1) index sums be computed for index n, we do double the typical work in DRAM but do not have to write to disk a second time.

The parallel scan algorithm to compute prefix sums typically works in three phases:

In the first phase, we take blocks of the data in parallel (already a good sign for the SSD setting) and compute individual prefix sums using some arbitrary method.
In phase two, calculate prefix sums over the final element of each block -- this gives us the offsets for the final blocks.
In the third phase, we can add the offset for each block to its existing prefix sums to get the overall sums.

When we say "arbitrary method," probably what we mean in DRAM is a Blelloch scan for work-efficiency, but there's something to be said for a sequential implementation. Similar to the Paralle Block-Delayed Sequences paper

The method for Scan described in Parallel Block Delayed Sequences is the one described above, where there are actually multiple layers of scanning on top of one another.

Complications for the MultiSSD setting:

The first phase is simple as long as we can account for striping across SSDs, and the data does clearly need to be striped in order to hide latency of I/O (need to saturate the PCIe lanes*). The individual prefix sum computation should probably use a Blelloch scan unless there's some reason not to.



In phase two, we do something similar, but we need to remember that the entire reason we use the MultiSSD setting is because not everything can fit in DRAM. Let's assume naively that we can fit at maximum one block in DRAM at a time. This implies that we need to write each block back to NVM as an intermediate step. In this case, we should probably reserve some space that's indexed by the block number (to avoid concurrency issues) to store the final prefix sum of each block (i.e. the last element of the block) to avoid issuing additional reads. If each block can fit in DRAM, the last prefix sum array absolutely will; this means that we can run Scan on them in memory potentially on another thread if the CPU overhead isn't too high. This will give us the block offsets, but unfortunately we'll need to read and write a second time to update each of the blocks on SSD.



Possible modifications and qualifications:

Blelloch scan is probably the best algorithm for the in-memory computation because it's efficient in terms of both work and depth. Depending on the number of threads, though, we might want to consider options that better match the I/O latency. 
Blelloch scan may also be an issue depending on the amount of DRAM we have -- it's work efficient (meaning it doesn't perform substantially more operations than the best sequential algorithm), but we still need to do ~2n operations through the up and down passes on the data. I don't think we really have a better option, though, unless we can do some NVIDIA CUB shennanigans (I have no idea what this is but apparently it's single pass, maybe not parallelizable though). It's possible that the parallleism from the scan process won't outweigh the synchronization overhead, in which case we would just use a sequential scan.

In order to hide the I/O latency, we probably need to be able to fit multiple blocks in DRAM -- this allows us to be reading into DRAM while performing computation to keep the CPU active.



*We probably can't just issue multiple reads to the same SSD because the bandwidth will saturate, and we also need this to be topology-aware with respect to the root complexes. Apparently there are some other issues with this as well, but they don't affect the algorithm itself.









-----------------------------------------------
**Parlay::flatten**

Flatten relies heavily on efficient scan, so if we can implement that algorithm, we can get a good chunk of this one done too.

Block flatten works like this:

1. We first need to perform a parallel prefix sum on the sublists.
Let's imagine we're flattening th sequence:

{[A, B, C], [D], [E, G, H, I]}

Then our prefix sum should be [0, 3, 4].

We also need to calculate the expected output size, which is just the sum of the sizes of the subsequences, in this case 8.


Next we partition the output array ~equally -- this means that if we have 2 threads, each of them in this case is responsible for 4 elements in the final output array.

Next comes the merge step, which is where the implementation might break down for the MultiSSD setting.


We need to merge the sorted offsets array with the sorted start index array for the blocks:

So merge [0, 3, 4] with [0, 4] -> [0 (offset 0), 0 (block start index 0), 3 (offset 1), 4 (offset 2), 4 (block start index 1)];

What exactly does this tell us?

Suppose that we're in the first thread which handles indices 0-3 in the final array.

Then we know that the start of the next block, block 1, is at 4 -- this means we need to handle everything below this.
So we copy the sequence 0 from 0-2 because we see that index 3 is where the next offset begins.
We then copy sequence 1 from 2-3 because after that is where the next block begins.




Possible notes for the MultiSSD setting:

We may not want to partition equally to overlap I/O and computation (otherwise every thread tries to do I/O at the same time)
If we can't rely on Parlay's work-stealing scheduler, we'll need to implement one ourselves


------------------------------------------------

Other stuff:

parlay::find_if: The basic parlay version uses a doubling sort to find the first index such that the value satisfies a given unary predicate. This is obviously very easy if the data is stored in blocks, but we have to assume that it's distributed across SSDs, which will incur random read costs. On the bright side, there is no writing involved so write amplification from random access is not an issue. CPU work is probably light.

parlay::delayed_map: This should be simple because MultiSSD already has a map function, but if the data is distributed across SSDs, call-time evaluation could incur multiple SSD read costs. Maybe at least a small cache?

parlay::delayed_map: This should be easy to implement but probably won't be great for the MultiSSD setting -- we already have map, which actually instantiates the transformed array. The issue here is that accessing each indivdual element could incur SSD read costs.


parlay::tabulate:: Should be fine.


parlay::delayed_tabulate: Should be fine.


parlay::iota: Should be easy -- this is effectively a map:
```
auto x = parlay::map(input, [&](long i){
return i;

});
```

parlay::pack && parlay::pack_index: This is essentially just a filter that acts on a boolean list instead of a predicate. If we can do filter, we can probably do pack.

parlay::set_union: This is not a primitive, but there are two interesting ways to think about this problem as an application.
If the data is unsorted, you can join the two sets in a sequence, flatten, and then remove duplicates.


if the data is sorted, there is an easy algorithm to do merge in DRAM:
find where the median of the first sequence (at spot i) would fit in the second sequence (at spot j)
and recur on the subproblems 
set_union(set1.cut(0, i), set2.cut(0,j);
set_union (set1.cut(i, n1), set2.cut(j, n2));
adjusting indices accordingly.

However, the binary search is awful for SSDs because it incurs many reads which are likely to be in distinct blocks.

parlay::group_by_key



parlay::histogram

parlay::lexicographical_compare

parlay::lower/upper bound: Good luck with this one



parlay::to_sequence
