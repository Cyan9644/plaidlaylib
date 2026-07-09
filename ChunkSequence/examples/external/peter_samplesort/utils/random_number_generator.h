//
// Created by peter on 3/6/24.
//

#ifndef SORTING_RANDOM_NUMBER_GENERATOR_H
#define SORTING_RANDOM_NUMBER_GENERATOR_H

#include <cstdlib>
#include <string>

#include "parlay/sequence.h"

template<typename T>
parlay::sequence<T> RandomSequence(size_t n, T limit = std::numeric_limits<T>::max());

template <typename T>
void GenerateUniformRandomNumbers(const std::string &prefix, size_t count, T limit = std::numeric_limits<T>::max());

template<typename T>
parlay::sequence<T> GenerateZipfianDistribution(size_t n, double s, T limit = std::numeric_limits<T>::max());

template <typename T>
void GenerateZipfianRandomNumbers(const std::string &prefix, size_t count, double s);

template<typename T>
parlay::sequence<T> GenerateExponentialDistribution(size_t n, double s);

template<typename T>
void GenerateExponentialRandomNumbers(const std::string &prefix, size_t count, double s);

#endif //SORTING_RANDOM_NUMBER_GENERATOR_H
