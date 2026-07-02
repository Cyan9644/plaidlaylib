#pragma once

// okay this is just a temp file to get some code going
#ifndef PLAIDLAY_H
#define PLAIDLAY_H
#include <vector>
#include <iostream>

#include <cassert>


template <typename T>
class naiveSeq {
    private:
        std::vector<T> data;
        friend std::ostream& operator<<(std::ostream& os, const naiveSeq<T>& seq) {
            os << "[ ";
            for (const T& elem : seq.data) {
                os << elem << " ";
            }
            os << "]";
            return os;
        }
    public:
        naiveSeq() {}
        naiveSeq(std::vector<T> in) : data(in) {}
        naiveSeq(size_t n) : data(n) {}
        ~naiveSeq() {}

        auto begin() {return data.begin();}
        auto end() {return data.end();}

        auto begin() const {return data.cbegin();}
        auto end() const {return data.cend();}

        auto size() const {return data.size();}

        T& operator[](int i) { return data[i]; }
        const T& operator[](int i) const { return data[i]; }
};

namespace plaidlayNaive {
    // here f should map int -> T
    template<typename T, typename Func>
    naiveSeq<T> tabulate(int n, Func f) {
        std::vector<T> data;
        for (int i = 0; i < n; i++) {
            data.push_back(f(i));
        }
        return naiveSeq<T>(data);
    }
    // okay I will now write a map function, parameterized over the return type U
    // Func must map T -> U
    template <typename T, typename Func>
    auto map(const naiveSeq<T>& seq, Func f) {
        using U = decltype(f(*seq.begin()));
        std::vector<U> out;
        for (const T& elem : seq) {
            out.push_back(f(elem));
        }
        return naiveSeq<U>(out);
    }
    // here Func should map (U, T) -> U
    template <typename T, typename U, typename Func>
    auto reduce(const naiveSeq<T>& seq, Func f, U identity) {
        U out = identity;
        for (const T& elem : seq) {
            out = f(out, elem);
        }
        return out;
    }
    // filter, we take a seq and a boolean predicate
    // Func here maps T -> bool
    template <typename T, typename Func>
    naiveSeq<T> filter (const naiveSeq<T>& seq, Func f) {
        std::vector<T> out;
        for (const T& elem: seq) {
            if (f(elem)) {
                out.push_back(elem);
            }
        }
        return naiveSeq<T>(out);
    }
    // Func must map (T,T) -> t
    template <typename T, typename Func>
    std::pair<naiveSeq<T>, T> scan(const naiveSeq<T>& seq, Func f, T identity) {
        int n = seq.size();
        std::vector<T> out(n);
        out[0] = identity;
        for (int i = 1; i < n; i++) {
            out[i] = out[i-1] + seq[i-1];
        }
        return std::make_pair(std::move(naiveSeq<T>(out)), out[n-1] + seq[n-1]);
    }
    template <typename T>
    naiveSeq<T> flatten(const naiveSeq<naiveSeq<T>>& seq) {
        std::vector<T> res;
        for(const auto& out : seq){
            for(const auto& in : out){
                res.push_back(in);
            }
        }
        return naiveSeq<T>(res);
    }
    template <typename T>
    naiveSeq<T> cut(const naiveSeq<T> seq, int start, int end){
        assert(start >= 0 && end >= 0);
        assert(start <= end);
        std::vector<T> res;
        for(int i = start; i < end; i++){
            res.push_back(seq[i]);
        }
        return naiveSeq<T>(res);
    }
}

#endif


