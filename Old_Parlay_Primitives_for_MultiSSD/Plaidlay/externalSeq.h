#include "utils/file_info.h"
#include "utils/file_utils.h"
#include "utils/unordered_file_writer.h"
#include "sequence_algorithms/map.h"
#include "sequence_algorithms/filter.h"
#include "sequence_algorithms/reduce.h"
#include "scan.h"
#include <parlay/sequence.h>
#include <vector>


#define BUFFERSIZE 4096 //2 ** 12
// this is some seq, later possibly will implement delaying, that "owns" some
// number of files on drive with owns in quotes because the file system handles
// memory
//
// currently templated but really the type info doesn't come into play until we
// have some way to consume to an in memory variable
//
// should be able to map, reduce, filter, flatten, scan, and to convert to an in memory seq
// maybe printing would be nice for small seqs? and write out to a file for big ones?
template <typename T>
class externalSeq {
    private:
    public:
        // I'm making these public since this class is basically jsut a wrapper, behaves like a struct
        parlay::sequence<FileInfo> files;
        // the prefix that all of the file names start with
        std::string prefix;
        externalSeq() {}
        // FindFiles / GetFileInfo hand us a std::vector<FileInfo>, so accept that and
        // convert into the parlay::sequence we store internally.
        externalSeq(const std::vector<FileInfo>& files, std::string prefix) : files(files.begin(), files.end()), prefix(prefix) {}
        ~externalSeq() {}

        // TODO: at least for now, we can't support these. depends on how we do delaying I guess
        // if you wanted to you could probably implement these with a single small read or write
        // but I'm not sure how alignment factors into that
        //
        // auto begin() {return data.begin();}
        // auto end() {return data.end();}

        // auto begin() const {return data.cbegin();}
        // auto end() const {return data.cend();}

        // auto size() const {return data.size();}

        // T& operator[](int i) { return data[i]; }
        // const T& operator[](int i) const { return data[i]; }
};
namespace externalSeqOps {
    void nop(void *ptr) {}
    template<typename T>
    externalSeq<T> randPerm(const std::string &prefix, size_t power_of_two) {
        size_t n = 1UL << power_of_two;
        // TODO: this part right here is copied from sample-sort.cpp for now
        auto nums = (size_t *) aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, n * sizeof(size_t));
            {
                auto perm = parlay::random_permutation(n, parlay::random(std::random_device()()));
                parlay::parallel_for(0, n, [&](size_t i) {
                    nums[i] = perm[i];
                });
            }
            UnorderedWriterConfig config;
            config.num_threads = 2;
            config.io_uring_size = 64;
            UnorderedFileWriter<size_t> writer(prefix, config);
            size_t step = std::min(1UL << 20, n);
            for (size_t i = 0; i < n; i += step) {
                writer.Push(std::shared_ptr<size_t>(nums + i, nop), std::min(step, n - i));
            }
            writer.Close();
            writer.Wait();
            free(nums);
        auto files = FindFiles(prefix);
        GetFileInfo(files);
        return externalSeq<T>(files, prefix);
    }
    template <typename T, typename R = T, bool in_place = sizeof(T) == sizeof(R)>
    externalSeq<R> map(externalSeq<T> seq, const std::string& new_prefix, std::function<R(T)> f) {
    //always do out of place for now, I get a crash in place
        Map<T, R, false>(seq.files.to_vector(), new_prefix, f);
        auto new_files = FindFiles(new_prefix);
        GetFileInfo(new_files);
        return externalSeq<T>(new_files, new_prefix);
    }
    // here Func should map (U, T) -> U
    template <typename T, typename U=T, typename Func>
    auto reduce(const externalSeq<T>& seq, Func f, U identity) {
        parlay::monoid monoid(f, identity);
        return Reduce<T, U>(seq.files.to_vector(), monoid);
    }
    // filter, we take a seq and a boolean predicate
    // Func here maps T -> bool
    template <typename T, typename Func>
    externalSeq<T> filter (const externalSeq<T>& seq, const std::string& new_prefix, Func f) {
        Filter<T>(seq.files.to_vector(), new_prefix, f);
        auto new_files = FindFiles(new_prefix);
        GetFileInfo(new_files);
        return externalSeq<T>(new_files, new_prefix);
    }


    //test for naive file scan
    template <typename T>
    externalSeq<T> naiveScan(const externalSeq<T>& seq, const std::string& new_prefix){
        Scan<T>(seq.files.to_vector(), new_prefix);
        auto filer = FindFiles(new_prefix);
        GetFileInfo(filer);
        return externalSeq<T>(filer, new_prefix);

    

    }

      // Usually the part after the prefix is the index, but if it's empty, we just ignore it.
        // auto index_substring = path_str.substr(index + 1 + prefix.size());
        // if (!index_substring.empty()) {
        //     file_index = std::stol(index_substring);
        // }
        // result.emplace_back(path_str, file_index, 0, dir_entry.file_size());

    //this should currently only work for a single file
    template <typename T> 
    void check_in_mem(const externalSeq<T>& seq, const std::string& new_prefix){
        auto filer = FindFiles(new_prefix); //filer is now a std::vector of FileInfos
        auto file = filer[0];//get very first file
        int filedes = open(file.file_name.c_str(), O_RDWR, 0644);
        struct stat checker;
        if(fstat(filedes, &checker) != 0){
            printf("something went wrong when checking file length with fstat");
        }
        off_t size = checker.st_size;
        size_t buffer[BUFFERSIZE];
        if(read(filedes, buffer, BUFFERSIZE) < 0){
             printf("something went wrong when checking file length with fstat");
        }
        for(int i =0; i < BUFFERSIZE; i++){
            printf("%d ", buffer[i]);
        }


        close(filedes);
        



    }
    // template <typename T>
    // void write_random(const const externalSeq<T>& seq, const std::string& new_prefix){
        

    //     // auto iota = parlay::iota(0, 8192, [&](size_t i)){
    //             std::iota(numbers.begin(), numbers.end(), 10);
    //     // }
    // }
}
