//I had an AI rewrite this file because it was unreadable

#include <algorithm>
#include <cmath>
#include <array>
#include <atomic>
#include <random>
#include <string>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/sequence.h>
#include "ExternalBoolean.h"


inline parlay::sequence<size_t> base_sieve(long m) {
    if (m < 2) return parlay::sequence<size_t>();
    std::vector<char> is_prime(m + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    for (long p = 2; p * p <= m; p++) {
        if (is_prime[p]) {
            for (long k = p * p; k <= m; k += p) is_prime[k] = 0;
        }
    }
    parlay::sequence<size_t> out;
    for (long i = 2; i <= m; i++) {
        if (is_prime[i]) out.push_back(i);
    }

    return out;
}


External_Sequence primes(size_t n, const std::vector<std::string> &new_filenames, const std::vector<std::string> &out_files) {

  if (n < 2) return External_Sequence(0);


  long sqrt_n = (long) std::sqrt((double) n);
  while ((size_t)(sqrt_n + 1) * (size_t)(sqrt_n + 1) <= n) sqrt_n++;
  while (sqrt_n > 0 && (size_t) sqrt_n * (size_t) sqrt_n > n) sqrt_n--;
  parlay::sequence<size_t> base_primes = base_sieve(sqrt_n);

  constexpr size_t buffer_size_bytes = 4 << 20;
  constexpr size_t flags_per_chunk = buffer_size_bytes / sizeof(bool);
  constexpr size_t buffer_size = buffer_size_bytes / sizeof(size_t);
  // The sieve below assumes each chunk starts on an even number, so flag index
  // parity matches integer parity (even integers, the non-primes, sit at even
  // indices). chunk_start = index * flags_per_chunk, so this holds iff the chunk
  // size is even. Guard it: a future tweak to buffer_size_bytes must keep it so.
  static_assert(flags_per_chunk % 2 == 0, "chunk size must be even for parity-skipping");
  size_t num_chunks = (n + 1 + flags_per_chunk - 1) / flags_per_chunk;

  External_Sequence seq(num_chunks);
  BoolCreate<bool>(n + 1, seq, new_filenames, true);

  auto& flag_headers = seq.ordered_underlying_sequence;
  UnorderedChunkReader<bool, buffer_size_bytes> reader;
  reader.PrepFiles(flag_headers);
  reader.Start();

  size_t expected_reads = (flag_headers.size() + NUM_SSDS - 1) / NUM_SSDS;

  External_Sequence res(flag_headers.size());
  parlay::sequence<chunk_header>* chunk_header_arr = &res.ordered_underlying_sequence;

  UnorderedChunkWriter<size_t> writer;
  UnorderedChunkWriterConfig wconfig;
  wconfig.num_threads = WRITER_THREADS;
  writer.Start(out_files, wconfig);

  std::vector<size_t*> buffer(NUM_SSDS);
  std::array<std::atomic<size_t>, NUM_SSDS> file_offsets{};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distrib(0, NUM_SSDS - 1);

  size_t write_count = 0;
  while (write_count < expected_reads) {

    for (int i = 0; i < NUM_SSDS; i++) {
        buffer[i] = (size_t*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, buffer_size_bytes);
    }

    std::atomic<int> counter(0);

    std::vector<unsigned int> random_holder(NUM_SSDS);
    std::atomic<bool> bad_flags[NUM_SSDS];
    std::vector<int> slot_for(NUM_SSDS, -1);

    for (int k = 0; k < NUM_SSDS; k++) {
        random_holder[k] = distrib(gen);
        bad_flags[k] = false;
    }

    parlay::parallel_for(0, NUM_SSDS, [&](size_t i){

    auto [ptr, size, _, index, which_chunk, filename] = reader.Poll();
    if (ptr == nullptr) {
        bad_flags[i] = true;
        return;
    }

    size_t chunk_start = index * flags_per_chunk;
    size_t chunk_end = chunk_start + size;

    for (size_t p : base_primes) {
        // First multiple of p in this chunk, never below p*p (smaller multiples
        // were already crossed off by smaller primes).
        size_t start = std::max<size_t>(p * p, ((chunk_start + p - 1) / p) * p);
        size_t step = p;
        if (p != 2) {
            // Even multiples of an odd prime are even numbers, already marked by
            // p == 2. Skip them: jump to the first odd multiple and stride 2*p.
            if ((start & 1) == 0) start += p;   // start is odd*p -> still a multiple
            step = 2 * p;
        }
        if (start >= chunk_end) continue;
        // Strength-reduce to a flag index so the inner loop is a plain stride.
        for (size_t k = start - chunk_start; k < size; k += step) {
            ptr[k] = false;
        }
    }
    if (chunk_start == 0) {
        if (size > 0) ptr[0] = false;
        if (size > 1) ptr[1] = false;
    }

    size_t* out = buffer[i];
    size_t produced = 0;
    // chunk_start is even (see static_assert), so even indices hold even integers,
    // which are all composite except 2. Emit 2 explicitly (first chunk only) and
    // then scan odd indices only, halving the loads over the flag buffer.
    if (chunk_start == 0 && size > 2 && ptr[2]) out[produced++] = 2;
    for (size_t k = 1; k < size; k += 2) {
        if (ptr[k]) out[produced++] = chunk_start + k;
    }
    reader.allocator.Free(ptr);

    chunk_header chunked;
    chunked.index = index;                       
    chunked.filename = out_files[random_holder[i]];
    chunked.used = produced * sizeof(size_t);
    chunked.begin_address = 0;                    
    int slot = counter.fetch_add(1);
    slot_for[i] = slot;
    (*chunk_header_arr)[write_count * NUM_SSDS + slot] = chunked;
        });

        for (int r = 0; r < NUM_SSDS; r++) {
            if (!(bad_flags[r])) {
                size_t base_offset = file_offsets[random_holder[r]].fetch_add(buffer_size_bytes);
                (*chunk_header_arr)[write_count * NUM_SSDS + slot_for[r]].begin_address = base_offset;
                writer.Push(std::shared_ptr<size_t>(buffer[r], free), buffer_size, random_holder[r], base_offset);
            } else {
                free(buffer[r]);  
            }
        }
        write_count++;
    }

    writer.Wait();

    std::sort(res.begin(), res.end(), [&](const chunk_header& i, const chunk_header& j){
        return i.index < j.index;
    });
    return res;
}

inline External_Sequence primes(size_t n, const std::string &prefix) {
    std::vector<std::string> new_filenames, out_files;
    new_filenames.reserve(NUM_SSDS);
    out_files.reserve(NUM_SSDS);
    for (int i = 0; i < NUM_SSDS; i++) {
        new_filenames.push_back(prefix + "_flag_" + std::to_string(i));
        out_files.push_back(prefix + "_prime_" + std::to_string(i));
    }
    return primes(n, new_filenames, out_files);
}