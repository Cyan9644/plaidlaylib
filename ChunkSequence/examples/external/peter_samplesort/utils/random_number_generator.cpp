//
// Created by peter on 3/6/24.
//

#include "random_number_generator.h"
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "utils/unordered_file_writer.h"

#include <random>
#include <cstdlib>

template<typename T>
parlay::sequence<T> RandomSequence(size_t n, T max_num) {
    parlay::random_generator rng;
    auto limit = std::numeric_limits<T>();
    if (max_num == 0) {
        max_num = limit.max();
    }
    std::uniform_int_distribution<T> dist(limit.min(), max_num);
    parlay::sequence<T> sequence(n);
    parlay::parallel_for(0, n, [&](size_t i) {
        auto r = rng[i];
        sequence[i] = dist(r);
    });
    return sequence;
}

template parlay::sequence<int64_t> RandomSequence(size_t, int64_t limit);

template parlay::sequence<uint64_t> RandomSequence(size_t, uint64_t limit);

template parlay::sequence<int32_t> RandomSequence(size_t, int32_t limit);

template parlay::sequence<uint32_t> RandomSequence(size_t, uint32_t limit);

template parlay::sequence<int8_t> RandomSequence(size_t, int8_t limit);

template parlay::sequence<uint8_t> RandomSequence(size_t, uint8_t limit);

/** Zipf-like random distribution.
 *
 * https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
 * License: CC BY-SA 4.0
 *
 * "Rejection-inversion to generate variates from monotone discrete
 * distributions", Wolfgang Hörmann and Gerhard Derflinger
 * ACM TOMACS 6.3 (1996): 169-184
 */
template<class IntType = unsigned long, class RealType = double>
class zipf_distribution
{
public:
    typedef RealType input_type;
    typedef IntType result_type;

    static_assert(std::numeric_limits<IntType>::is_integer, "");
    static_assert(!std::numeric_limits<RealType>::is_integer, "");

    zipf_distribution(const IntType n=std::numeric_limits<IntType>::max(),
                      const RealType q=1.0)
            : n(n)
            , q(q)
            , H_x1(H(1.5) - 1.0)
            , H_n(H(n + 0.5))
            , dist(H_x1, H_n)
    {}

    IntType operator()(parlay::random_generator& rng)
    {
        while (true) {
            const RealType u = dist(rng);
            const RealType x = H_inv(u);
            const auto k = clamp<IntType>(std::round(x), 1, n);
            if (u >= H(k + 0.5) - h(k)) {
                return k;
            }
        }
    }

private:
    /** Clamp x to [min, max]. */
    template<typename T>
    static constexpr T clamp(const T x, const T min, const T max)
    {
        return std::max(min, std::min(max, x));
    }

    /** exp(x) - 1 / x */
    static double
    expxm1bx(const double x)
    {
        return (std::abs(x) > epsilon)
               ? std::expm1(x) / x
               : (1.0 + x/2.0 * (1.0 + x/3.0 * (1.0 + x/4.0)));
    }

    /** H(x) = log(x) if q == 1, (x^(1-q) - 1)/(1 - q) otherwise.
     * H(x) is an integral of h(x).
     *
     * Note the numerator is one less than in the paper order to work with all
     * positive q.
     */
    const RealType H(const RealType x)
    {
        const RealType log_x = std::log(x);
        return expxm1bx((1.0 - q) * log_x) * log_x;
    }

    /** log(1 + x) / x */
    static RealType
    log1pxbx(const RealType x)
    {
        return (std::abs(x) > epsilon)
               ? std::log1p(x) / x
               : 1.0 - x * ((1/2.0) - x * ((1/3.0) - x * (1/4.0)));
    }

    /** The inverse function of H(x) */
    const RealType H_inv(const RealType x)
    {
        const RealType t = std::max(-1.0, x * (1.0 - q));
        return std::exp(log1pxbx(t) * x);
    }

    /** That hat function h(x) = 1 / (x ^ q) */
    const RealType h(const RealType x)
    {
        return std::exp(-q * std::log(x));
    }

    static constexpr RealType epsilon = 1e-8;

    IntType                                  n;     ///< Number of elements
    RealType                                 q;     ///< Exponent
    RealType                                 H_x1;  ///< H(x_1)
    RealType                                 H_n;   ///< H(n)
    std::uniform_real_distribution<RealType> dist;  ///< [H(x_1), H(n)]
};

template<typename T>
parlay::sequence<T> GenerateZipfianDistribution(size_t n, double s, T limit) {
    parlay::random_generator random;
    zipf_distribution<T> distribution(limit, s);

    return parlay::tabulate(n, [&](size_t i) {
        auto rng = random[i];
        return (T) distribution(rng);
    });
}

template<typename T>
parlay::sequence<T> GenerateExponentialDistribution(size_t count, double lambda) {
    parlay::random_generator random;
    std::exponential_distribution<double> distribution(lambda);

    return parlay::tabulate(count, [&](size_t i) {
        auto rng = random[i];
        return (T) distribution(rng);
    });
}

template parlay::sequence<uint64_t> GenerateExponentialDistribution(size_t, double);
template parlay::sequence<int64_t> GenerateExponentialDistribution(size_t, double);
template parlay::sequence<uint32_t> GenerateExponentialDistribution(size_t, double);
template parlay::sequence<int32_t> GenerateExponentialDistribution(size_t, double);

template<typename T>
void WriteNumbers(const std::string &prefix, size_t n, const T *data) {
    UnorderedFileWriter<T> writer(prefix);
    constexpr auto WRITE_SIZE = 4 << 20;
    const auto step = WRITE_SIZE / sizeof(T);
    for (size_t i = 0; i < n; i += step) {
        T *buffer = (T *) std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, WRITE_SIZE);
        memcpy(buffer, data + i, WRITE_SIZE);
        std::shared_ptr<T> temp(buffer, free);
        writer.Push(temp, step);
    }
    writer.Close();
}

template<typename T>
void WriteNumbers(const std::string &prefix, size_t n, const std::function<parlay::sequence<T>(size_t)> &generator) {
    UnorderedFileWriter<T> writer(prefix);
    constexpr auto WRITE_SIZE = 4 << 20;
    const auto step = WRITE_SIZE / sizeof(T);
    parlay::parallel_for(0, n / step, [&](size_t i) {
        T *buffer = (T *) std::aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, WRITE_SIZE);
        auto seq = generator(step);
        memcpy(buffer, seq.data(), WRITE_SIZE);
        std::shared_ptr<T> temp(buffer, free);
        writer.Push(temp, step);
    }, 1);
}

template<typename T>
void GenerateZipfianRandomNumbers(const std::string &prefix, size_t n, double s) {
    WriteNumbers<T>(prefix, n, [&](size_t sz) {
        return GenerateZipfianDistribution<T>(sz, s, n);
    });
}

template<typename T>
void GenerateExponentialRandomNumbers(const std::string &prefix, size_t n, double s) {
    WriteNumbers<T>(prefix, n, [&](size_t sz) {
        return GenerateExponentialDistribution<T>(sz, s);
    });
}

template<typename T>
void GenerateUniformRandomNumbers(const std::string &prefix, size_t count, T limit_max) {
    const size_t GRANULARITY = 1 << 20;
    UnorderedFileWriter<T> writer(prefix);
    CHECK(count % GRANULARITY == 0) << "Can only generate numbers that are a multiple of " << GRANULARITY;
    std::atomic<int64_t> num_blocks = (int64_t) (count / GRANULARITY);
    parlay::parallel_for(0, parlay::num_workers(), [&](size_t _) {
        std::random_device device;
        std::mt19937 rng(device());
        auto limit = std::numeric_limits<T>();
        if (limit_max == 0) {
            limit_max = limit.max();
        }
        std::uniform_int_distribution<T> distribution(limit.min(), limit_max);
        while (true) {
            auto current = num_blocks--;
            if (current <= 0) {
                return;
            }
            std::shared_ptr<T> result((T *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, GRANULARITY * sizeof(T)), free);
            for (size_t i = 0; i < GRANULARITY; i++) {
                result.get()[i] = distribution(rng);
            }
            writer.Push(result, GRANULARITY);
        }
    }, 1);
    writer.Close();
}

template void GenerateUniformRandomNumbers<int64_t>(const std::string &prefix, size_t count, int64_t limit);

template void GenerateUniformRandomNumbers<uint64_t>(const std::string &prefix, size_t count, uint64_t limit);

template void GenerateUniformRandomNumbers<int32_t>(const std::string &prefix, size_t count, int32_t limit);

template void GenerateUniformRandomNumbers<uint32_t>(const std::string &prefix, size_t count, uint32_t limit);

template void GenerateZipfianRandomNumbers<size_t>(const std::string &prefix, size_t n, double s);

template void GenerateExponentialRandomNumbers<size_t>(const std::string &prefix, size_t n, double s);
