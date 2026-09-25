Many algorithms in this repo have been cleaned, formatted, and error-checked by AI programming tools (e.g. Claude code).

See the original repository at: https://github.com/Cyan9644/Parlay_Primitives_for_MultiSSD.

The Parallel Library for Asynchronous I/O Device (PLAID) is a C++ library for computation on multicore machines with an array of auxiliary SSDs. PLAID provides a number of functional programming tools and primitives that can be composed to write general-purpose parallel algorithms at a high level. Performance for these algorithms is often competitive with in-memory counterparts, though the relative factor is dependent on how the program is constructed as well as the requirements of the method.

PLAID represents sequences using the plaid namespace. In particular, our equivalent of a vector sequence like std::vector or parlay::sequence is the chunk_seq datatype, which uses an underlying std::vector of chunk descriptors in logical order. These chunk descriptors, which can be specified to point to chunks of any size, have 4 fields: the corresponding block's logical index, backing file, byte offset within that file, and number of valid bytes. Chunks are randomly distributed to SSDs to ensure load balance and efficiency for arbitrary access patterns.

PLAID supports parallel block-delayed operations on disk sequences (https://dl.acm.org/doi/epdf/10.1145/3503221.3508434), avoiding intermediate writes that increase disk wear and increasing CPU utilization on passes. These exist in the plaid::delayed namespace and have the same names as their eager counterparts.

The library is designed for streaming-style algorithms that can easily overlap I/O and computation to avoid disk-latency bottlenecks, but supports some imperative access. PLAID significantly simplifies the design of multi-SSD algorithms, and we provide a number of examples for users to learn from in the examples directory. 

Particularly informative examples are line-fit, samplesort, and big-integer; these can be written almost trivially with a namespace change from their equivalents in ParlayLib, which we strive to support in all cases. Some examples require unique block logic such as chunk-wise iteration rather than element-wise (e.g., Rabin-Karp and KMP), but we avoid the use of a low-level reader/writer. 
