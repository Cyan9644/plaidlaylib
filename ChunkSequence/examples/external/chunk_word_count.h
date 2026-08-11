#ifndef CHUNK_WORD_COUNT_H
#define CHUNK_WORD_COUNT_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "parlay/primitives.h"

#include "ChunkSequence/Primitives/chunk_seq.h"
#include "ChunkSequence/Primitives/external_engine.h"

namespace ChunkSequenceOps {

/**
 * Out-of-core word count -- the streaming, hash-fold analogue of parlaylib's
 * `word_counts` (deps/parlaylib-examples/word_counts.h), which tokenizes a text
 * and groups identical words with `histogram_by_key` (a hash pass), then sorts
 * only the small distinct-pairs list by count.
 *
 * Design (see plan): word count needs a *sort* only when the DISTINCT vocabulary
 * does not fit in DRAM; for real text the vocabulary is tiny even for a multi-TB
 * corpus, so we group by a 64-bit hash of each word into a per-worker DRAM hash
 * table and merge -- ONE streaming read of the text, ZERO writes, no sort of the
 * token stream.  Built directly on the `RemoveWorker` fold driver
 * (external_engine.h), the same per-worker-accumulator shape as ChunkReduce /
 * ChunkFindIf.
 *
 * A word straddling a chunk boundary is handled without a second pass: each
 * worker counts only the tokens fully *interior* to a chunk and stashes the
 * chunk's leading/trailing partial-word fragments; the caller stitches
 * `tail[i] ++ head[i+1]` across every seam in one sequential O(num_chunks) merge.
 * A token may only straddle a single boundary unless it exceeds CHUNK_SIZE, which
 * is guarded (a word longer than one chunk is unsupported -- same spirit as KMP's
 * "pattern <= one chunk").
 *
 * The map keys on the hash and keeps one representative word string per hash for
 * display and as a collision guard (the string is compared on every hash hit,
 * exactly as a hash table would; a 64-bit collision among the small vocabulary is
 * astronomically unlikely and fires a CHECK).
 *
 * NOTE: the interior-token counting + seam stitch here is the reusable seed of a
 * future general `reduce_by_key`; kept inline in this example for now.
 */

// hash -> (count, one representative word).
using WordCounts = std::unordered_map<uint64_t, std::pair<uint64_t, std::string>>;

// FNV-1a over the word bytes; deterministic and identical on both the
// out-of-core and DRAM-baseline sides so counts can be compared by hash.
inline uint64_t HashWord(const char* p, size_t len) {
    uint64_t h = 1469598103934665603ULL;         // FNV offset basis
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)(unsigned char)p[i];
        h *= 1099511628211ULL;                    // FNV prime
    }
    return h;
}

namespace detail {

// Insert/bump one word into a map, keeping a representative and verifying it on
// every hit (the hash-table key compare a real hash map would do anyway).
inline void AddWord(WordCounts& m, const char* p, size_t len) {
    const uint64_t h = HashWord(p, len);
    auto [it, inserted] = m.try_emplace(h, 0, std::string());
    if (inserted) {
        it->second.second.assign(p, len);
    } else {
        CHECK(it->second.second.size() == len &&
              std::memcmp(it->second.second.data(), p, len) == 0)
            << "WordCount: 64-bit hash collision between distinct words";
    }
    it->second.first++;
}

inline void AddWord(WordCounts& m, const std::string& w) {
    if (!w.empty()) AddWord(m, w.data(), w.size());
}

}  // namespace detail

/**
 * Count occurrences of every `delim`-separated word in `text` (a chunk_seq of
 * char).  Returns the distinct (hash -> {count, representative}) map.
 *
 * @param text            the input text, one char per element.
 * @param delim           the token separator (default space, matching upstream).
 * @param reader_threads  io_uring reader threads feeding the fold.
 */
inline WordCounts WordCount(const chunk_seq& text, char delim = ' ',
                            size_t reader_threads = 10) {
    const size_t num_chunks = text.chunks.size();
    if (num_chunks == 0) return {};

    // Per-worker accumulator: a local map plus, for each chunk this worker
    // handled, that chunk's boundary fragments (to be stitched by the caller).
    struct Local {
        WordCounts counts;
        std::vector<std::tuple<size_t, std::string, std::string>> seams;  // {index, head, tail}
    };

    auto locals = RemoveWorker<char>(text, reader_threads,
        [&](ChunkSequenceReader<char>& reader) {
            Local L;
            while (true) {
                auto [ptr, n, idx] = reader.Poll();
                if (ptr == nullptr) break;
                const char* buf = ptr;

                // Locate the first and last delimiter to peel off the leading /
                // trailing partial-word fragments (which belong to seams, not
                // this chunk's interior).
                size_t fs = n;                        // first delimiter, n if none
                for (size_t i = 0; i < n; i++)
                    if (buf[i] == delim) { fs = i; break; }

                if (fs == n) {
                    // No delimiter: a token spans the whole chunk (word bigger
                    // than CHUNK_SIZE) -- unsupported.  An empty chunk is fine.
                    CHECK(n == 0) << "WordCount: token exceeds one chunk (word > CHUNK_SIZE)";
                    L.seams.emplace_back(idx, std::string(), std::string());
                    reader.allocator.Free(ptr);
                    continue;
                }

                size_t ls = fs;                       // last delimiter (>= fs)
                for (size_t i = n; i-- > 0;)
                    if (buf[i] == delim) { ls = i; break; }

                std::string head = (fs > 0) ? std::string(buf, buf + fs) : std::string();
                std::string tail = (ls + 1 < n) ? std::string(buf + ls + 1, buf + n)
                                                : std::string();

                // Interior tokens: everything between the first and last
                // delimiter is bounded by a delimiter on both sides.
                size_t i = fs;
                while (i <= ls) {
                    while (i <= ls && buf[i] == delim) i++;
                    if (i > ls) break;
                    const size_t s = i;
                    while (i <= ls && buf[i] != delim) i++;
                    detail::AddWord(L.counts, buf + s, i - s);
                }

                L.seams.emplace_back(idx, std::move(head), std::move(tail));
                reader.allocator.Free(ptr);
            }
            return L;
        });

    // Merge per-worker maps and gather the per-chunk boundary fragments.
    WordCounts merged;
    std::vector<std::string> head(num_chunks), tail(num_chunks);
    for (auto& L : locals) {
        for (auto& kv : L.counts) {
            auto [it, inserted] = merged.try_emplace(kv.first, 0, std::string());
            it->second.first += kv.second.first;
            if (inserted) it->second.second = std::move(kv.second.second);
        }
        for (auto& s : L.seams) {
            head[std::get<0>(s)] = std::move(std::get<1>(s));
            tail[std::get<0>(s)] = std::move(std::get<2>(s));
        }
    }

    // Stitch the seams.  head[0] is the text's first complete word and
    // tail[last] its last; every interior boundary joins tail[i] ++ head[i+1].
    // Each straddling word is counted exactly once and none is double counted
    // (an interior head/tail is consumed by exactly one boundary).
    detail::AddWord(merged, head[0]);
    detail::AddWord(merged, tail[num_chunks - 1]);
    for (size_t i = 0; i + 1 < num_chunks; i++)
        detail::AddWord(merged, tail[i] + head[i + 1]);

    return merged;
}

/**
 * The distinct (word, count) pairs sorted by descending count (ties by word),
 * i.e. the shape upstream `word_counts` returns.  The distinct set is small, so
 * this is a plain DRAM sort -- exactly upstream's final step.
 */
inline std::vector<std::pair<std::string, uint64_t>>
SortedByCount(const WordCounts& counts) {
    std::vector<std::pair<std::string, uint64_t>> pairs;
    pairs.reserve(counts.size());
    for (const auto& kv : counts)
        pairs.emplace_back(kv.second.second, kv.second.first);
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    return pairs;
}

}  // namespace ChunkSequenceOps

#endif  // CHUNK_WORD_COUNT_H
