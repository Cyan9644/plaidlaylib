//
// Created by peter on 11/26/24.
//

#ifndef SORTING_TYPE_ALLOCATOR_H
#define SORTING_TYPE_ALLOCATOR_H

#include "parlay/alloc.h"

template <size_t size>
class AllocatorData {
    char data[size];
};

/**
 * A modified parlay::type_allocator that allows the user to specify an arbitrary alignment
 * 
 * @tparam T The type of object that needs to be allocated.
 * @tparam Align Alignment requirement (in bytes).
 */
template <typename T, size_t Align>
class AlignedTypeAllocator {

    static parlay::internal::block_allocator& get_allocator() {
        return parlay::internal::get_block_allocator<sizeof(T), Align>();
    }

public:

    // Allocate uninitialized storage appropriate for storing an object of type T
    static T* alloc() {
        void* buffer = get_allocator().alloc();
        assert(reinterpret_cast<uintptr_t>(buffer) % Align == 0);
        return static_cast<T*>(buffer);
    }

    // Free storage obtained by alloc()
    static void free(T* ptr) {
        assert(ptr != nullptr);
        assert(reinterpret_cast<uintptr_t>(ptr) % Align == 0);
        get_allocator().free(static_cast<void*>(ptr));
    }

    // Allocate storage for and then construct an object of type T using args...
    template<typename... Args>
    static T* create(Args... args) {
        static_assert(std::is_constructible_v<T, Args...>);
        return new (alloc()) T(std::forward<Args>(args)...);
    }

    // Destroy an object obtained by create(...) and deallocate its storage
    static void destroy(T* ptr) {
        assert(ptr != nullptr);
        ptr->~T();
        free(ptr);
    }

    // for backward compatibility -----------------------------------------------
    static constexpr inline size_t default_alloc_size = 0;
    static constexpr inline bool initialized = true;

    template <typename ... Args>
    static T* allocate(Args... args) { return create(std::forward<Args>(args)...); }
    static void retire(T* ptr) { destroy(ptr); }
    static void init(size_t, size_t) {};
    static void init() {};
    static void reserve(size_t n = default_alloc_size) { get_allocator().reserve(n); }
    static void finish() { get_allocator().clear(); }
    static size_t block_size () { return get_allocator().get_block_size(); }
    static size_t num_allocated_blocks() { return get_allocator().num_allocated_blocks(); }
    static size_t num_used_blocks() { return get_allocator().num_used_blocks(); }
    static size_t num_used_bytes() { return num_used_blocks() * block_size(); }
    static void print_stats() { get_allocator().print_stats(); }
};

#endif //SORTING_TYPE_ALLOCATOR_H
