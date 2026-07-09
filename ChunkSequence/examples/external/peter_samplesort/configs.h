//
// Created by peter on 3/1/24.
//

#ifndef SORTING_CONFIGS_H
#define SORTING_CONFIGS_H

#include <cstddef>
#include <string>

constexpr size_t MAIN_MEMORY_SIZE = 400ULL * (1 << 30);

const std::string SSD_ROOT = "/mnt/ssd%lu";
// number of SSD directories in the file system
constexpr size_t SSD_COUNT= 30;
// number of SSDs that can be utilized in parallel; this is different from SSD_COUNT if RAID is used
constexpr size_t SSD_PARALLELISM = SSD_COUNT;

// Override at build time with -DREADER_READ_SIZE_BYTES=<n>; default 512 KiB.
// map_edges sweep on Twitter shows a U-shaped throughput curve with a
// minimum at 512 KiB; below 256 KiB the SSDs become IOPS-bound, above
// 1 MiB load balancing across the 12 shards starts to coarsen.
#ifndef READER_READ_SIZE_BYTES
#define READER_READ_SIZE_BYTES (512 * 1024)
#endif
constexpr size_t READER_READ_SIZE = READER_READ_SIZE_BYTES;

constexpr size_t SAMPLE_SORT_BUCKET_SIZE = 4 << 10;

constexpr size_t IO_VECTOR_SIZE = 1024;

// This is machine-dependent:
// On Google Cloud machines, keep this value.
// On baldr, aligned memory allocations are not required.
// On other systems, make this 512 and see if it works.
constexpr size_t O_DIRECT_MULTIPLE = 4096;
constexpr size_t O_DIRECT_MEMORY_ALIGNMENT = O_DIRECT_MULTIPLE;
// It should never be necessary to change this unless O_DIRECT_MULTIPLE is very large
constexpr size_t METADATA_SIZE = 2;
constexpr size_t IO_URING_BUFFER_SIZE = 64;

#endif //SORTING_CONFIGS_H