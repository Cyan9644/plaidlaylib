#ifndef N_READER_H
#define N_READER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "parlay/primitives.h"
#include "absl/log/check.h"

#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/simple_queue.h"
#include "configs.h"

namespace ChunkSequenceOps {

/**
 * NReader<T> — read N chunk sequences in lockstep and hand a worker the N
 * co-indexed buffers together from a single Poll().
 *
 * Motivation (see chunk_count_sort.h): some primitives need to walk two (or
 * more) chunk_seqs that share an index space at the same time — e.g. a values
 * sequence and a same-length bucket-id sequence — pairing element k of chunk i
 * in one with element k of chunk i in the other.  A single ChunkSequenceReader
 * only streams one sequence, and running two independent readers gives back
 * chunks in each reader's own io_uring completion order, so the two streams are
 * not aligned.  NReader owns one ChunkSequenceReader per input sequence and a
 * thin matching layer that re-pairs their output by chunk.index before handing
 * it out, so Poll() always returns a full set of N co-indexed buffers.
 *
 * Requirements on the inputs:
 *   - All N sequences must share the same index set (chunks[i].index == i for
 *     each, the library-wide index-ordered invariant), i.e. the same number of
 *     chunks.  A chunk index that is missing from any one sequence will never
 *     complete a match and is silently dropped.
 *   - This version uses a single element type T for every sequence (mirroring
 *     RemoveWorker<T>).  Sequences with different element types would need a
 *     variadic/heterogeneous variant; two same-typed sequences is the case the
 *     current callers need.
 *
 * Ownership: Poll() returns raw buffers owned by the per-sequence reader pools.
 * The worker must return them with Free(match) (or FreeSlot) exactly once, the
 * same contract as ChunkSequenceReader::allocator.Free.
 *
 * Note on buffering: if one reader races ahead of the others, its unmatched
 * buffers accumulate in the matcher (the reader pools grow to cover them).  For
 * co-generated sequences read at similar rates this stays small; it is bounded
 * only by how far the fastest reader can get ahead of the slowest.
 */
template<typename T>
class NReader {
public:
    // One matched position across all N sequences.
    struct Match {
        std::vector<T*> ptrs;       // size N; ptrs[s] = seq s's buffer at index
        std::vector<size_t> sizes;  // size N; element count in ptrs[s]
        size_t index = 0;           // shared chunk index
        // Empty ptrs is the exhausted sentinel returned by Poll().
        bool valid() const { return !ptrs.empty(); }
    };

    NReader() = default;

    ~NReader() {
        // Pumps hold references into readers_, so they must be joined before the
        // readers (and this object) are torn down.
        for (auto& t : pumps_)
            if (t.joinable()) t.join();
    }

    NReader(const NReader&) = delete;
    NReader& operator=(const NReader&) = delete;

    // Register the input sequences (in the order Match::ptrs will use).
    void Prep(const std::vector<const chunk_seq*>& seqs) {
        CHECK(!seqs.empty()) << "NReader: need at least one sequence";
        n_ = seqs.size();
        readers_.resize(n_);
        for (size_t s = 0; s < n_; s++) {
            readers_[s] = std::make_unique<ChunkSequenceReader<T>>();
            readers_[s]->PrepChunks(*seqs[s]);
        }
    }

    /**
     * Start all N underlying readers and the matching pumps.  Reader tuning
     * mirrors RemoveWorker; each underlying reader gets `reader_threads` IO
     * threads.
     */
    void Start(size_t reader_threads = 10, size_t queue_depth = 32,
               size_t max_requests = 8, size_t match_queue_sz = 512) {
        CHECK(!readers_.empty()) << "NReader: Prep() before Start()";
        match_queue_.SetSizeLimit(match_queue_sz);
        active_pumps_ = (int)n_;
        for (size_t s = 0; s < n_; s++)
            readers_[s]->Start(reader_threads, queue_depth, max_requests);
        // Pumps started after readers so the readers are fully live first.
        for (size_t s = 0; s < n_; s++)
            pumps_.emplace_back(Pump, this, s);
    }

    /**
     * Return the next fully matched set of N co-indexed buffers.  Blocks until
     * one is ready.  A Match with empty ptrs (valid() == false) means every
     * sequence has been exhausted.  Safe to call from multiple workers.
     */
    Match Poll() {
        Match nil;  // empty ptrs => done
        return match_queue_.Poll(nil).first;
    }

    // Return seq s's buffer from a match to its owning reader's pool.
    void FreeSlot(size_t s, T* ptr) { readers_[s]->allocator.Free(ptr); }

    // Return every buffer in a match to its owning pool.
    void Free(const Match& m) {
        for (size_t s = 0; s < m.ptrs.size(); s++)
            if (m.ptrs[s] != nullptr) readers_[s]->allocator.Free(m.ptrs[s]);
    }

    size_t num_sequences() const { return n_; }

private:
    // In-progress match awaiting the remaining sequences' buffers.
    struct Partial {
        std::vector<T*> ptrs;
        std::vector<size_t> sizes;
        size_t filled = 0;
    };

    // One pump per sequence: drain reader s and deposit each buffer into the
    // matcher keyed by chunk index; the buffer that completes a match publishes
    // the assembled Match to match_queue_.
    static void Pump(NReader* self, size_t s) {
        while (true) {
            auto [ptr, size, index] = self->readers_[s]->Poll();
            if (ptr == nullptr) break;

            Match ready;  // filled while holding the lock, published after
            {
                std::lock_guard<std::mutex> lk(self->match_mutex_);
                Partial& p = self->partial_[index];
                if (p.ptrs.empty()) {
                    p.ptrs.assign(self->n_, nullptr);
                    p.sizes.assign(self->n_, 0);
                }
                CHECK(p.ptrs[s] == nullptr)
                    << "NReader: duplicate chunk index " << index
                    << " in sequence " << s;
                p.ptrs[s] = (T*)ptr;
                p.sizes[s] = size;
                if (++p.filled == self->n_) {
                    ready.ptrs = std::move(p.ptrs);
                    ready.sizes = std::move(p.sizes);
                    ready.index = index;
                    self->partial_.erase(index);
                }
            }
            if (ready.valid()) self->match_queue_.Push(std::move(ready));
        }

        // Last pump to finish closes the output so pollers unblock.
        if (--self->active_pumps_ == 0) self->match_queue_.Close();
    }

    size_t n_ = 0;
    std::vector<std::unique_ptr<ChunkSequenceReader<T>>> readers_;
    std::vector<std::thread> pumps_;
    std::atomic<int> active_pumps_{0};

    std::mutex match_mutex_;
    std::unordered_map<size_t, Partial> partial_;
    SimpleQueue<Match> match_queue_;
};

/**
 * N-sequence analogue of RemoveWorker: read `seqs` in lockstep and return one
 * accumulator per parlay worker.  `worker(nreader)` polls the shared NReader to
 * exhaustion (Poll().valid() == false when done) and returns its local
 * accumulator; the caller combines the per-worker results.  Each worker must
 * Free() the matches it consumes.
 *
 * Example (paired values + bucket-ids walk):
 *   auto locals = NRemoveWorker<uint64_t>({&values, &ids}, 10,
 *       [&](NReader<uint64_t>& r) {
 *           LocalAcc acc;
 *           while (true) {
 *               auto m = r.Poll();
 *               if (!m.valid()) break;
 *               uint64_t* val = m.ptrs[0]; uint64_t* id = m.ptrs[1];
 *               for (size_t k = 0; k < m.sizes[0]; k++) acc.add(id[k], val[k]);
 *               r.Free(m);
 *           }
 *           return acc;
 *       });
 */
template<typename T, typename WorkerFn>
auto NRemoveWorker(const std::vector<const chunk_seq*>& seqs,
                   size_t reader_threads, WorkerFn worker)
    -> parlay::sequence<std::invoke_result_t<WorkerFn, NReader<T>&>> {
    NReader<T> reader;
    reader.Prep(seqs);
    reader.Start(reader_threads, 32, 8);
    return parlay::tabulate(parlay::num_workers(),
                            [&](size_t) { return worker(reader); },
                            /*granularity=*/1);
}

} // namespace ChunkSequenceOps

#endif // N_READER_H
