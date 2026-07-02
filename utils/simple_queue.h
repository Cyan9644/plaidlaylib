//
// Created by peter on 5/13/24.
//

#ifndef SORTING_SIMPLE_QUEUE_H
#define SORTING_SIMPLE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include "absl/log/check.h"

enum class QueueCode {
    FINISH = 0,
    TIMEOUT = 1,
    SUCCESS = 2
};

template<typename T>
class SimpleQueue {
public:
    explicit SimpleQueue(size_t size_limit = 0) : size_limit(size_limit) {}

    void Push(T data) {
        std::unique_lock<std::mutex> lock(mutex);
        while (size_limit && queue.size() >= size_limit) {
            writer_cond.wait(lock);
        }
        queue.push(data);
        reader_cond.notify_one();
    }

    std::pair<T, QueueCode> Poll(T default_result = nullptr, int64_t timeout = -1) {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty()) {
            if (!open) {
                return {default_result, QueueCode::FINISH};
            }
            if (timeout == -1) {
                reader_cond.wait(lock);
            } else if (timeout == 0) {
                return {default_result, QueueCode::TIMEOUT};
            } else {
                auto result = reader_cond.wait_for(lock,
                                                   std::chrono::duration(std::chrono::microseconds(timeout)));
                if (result == std::cv_status::timeout) {
                    return {default_result, QueueCode::TIMEOUT};
                }
            }
        }
        T ret = queue.front();
        queue.pop();
        if (size_limit) {
            writer_cond.notify_one();
        }
        return {ret, QueueCode::SUCCESS};
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mutex);
        open = false;
        reader_cond.notify_all();
    }

    // Re-open a queue that was previously Close()d so it can be reused.
    // Caller must ensure the queue is drained before calling.
    void Reopen() {
        std::lock_guard<std::mutex> lock(mutex);
        CHECK(queue.empty()) << "Reopen called with non-empty queue";
        open = true;
    }

    bool IsEmptyUnsafe() {
        return queue.empty();
    }

    void SetSizeLimit(size_t new_limit) {
        size_limit = new_limit;
    }

    void Log(const std::string& message = "Queue size: ") {
        mutex.lock();
        size_t size = queue.size();
        mutex.unlock();
        LOG(INFO) << message << size;
    }

private:
    std::queue<T> queue;
    std::mutex mutex;
    size_t size_limit;
    std::condition_variable reader_cond, writer_cond;
    bool open = true;
};

#endif //SORTING_SIMPLE_QUEUE_H
