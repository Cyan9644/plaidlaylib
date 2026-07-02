//
// Created by peter on 3/2/24.
//

#ifndef SORTING_FILE_UTILS_H
#define SORTING_FILE_UTILS_H

#include <vector>
#include <string>
#include "utils/file_info.h"
#include "configs.h"

std::vector<FileInfo> FindFiles(const std::string &prefix, bool parallel = false);

inline size_t GetFileSize(const std::string &file_name);

void GetFileInfo(std::vector<FileInfo> &info, bool eof_marker = false, bool compute_before_size = true);

void ComputeBeforeSize(std::vector<FileInfo> &files);

inline size_t AlignDown(size_t original, size_t alignment) {
    return original / alignment * alignment;
}

inline size_t AlignUp(size_t original, size_t alignment) {
    return (original + alignment - 1) / alignment * alignment;
}

/**
 * Ensure that a byte offset conforms to disk alignment requirements by rounding down.
 * For example, 4098 would become 4096 if O_DIRECT_MULTIPLE is set to 4096.
 *
 * @param original
 * @return
 */
constexpr size_t AlignDown(size_t original) {
    return original / O_DIRECT_MULTIPLE * O_DIRECT_MULTIPLE;
}
/**
 * Ensure that a byte offset conforms to disk alignment requirements by rounding up.
 * For example, 4098 would become 8192 if O_DIRECT_MULTIPLE is set to 4096.
 *
 * @param original
 * @return
 */
constexpr size_t AlignUp(size_t original) {
    return (original + O_DIRECT_MULTIPLE - 1) / O_DIRECT_MULTIPLE * O_DIRECT_MULTIPLE;
}

void Read(int fd, void* buffer, size_t read_size);
void Write(int fd, const void *buffer, size_t write_size);

void ReadFileOnce(const std::string &file_name, void* buffer, size_t offset);
void ReadFileOnce(const std::string &file_name, void *buffer, size_t start, size_t read_size);

void* ReadEntireFile(const std::string &file_name, size_t read_size);

void PopulateSSDList();
void PopulateSSDList(size_t count, bool random, bool verbose);
void PopulateSSDList(const std::vector<int> &ssd_numbers, bool verbose);
std::string GetFileName(const std::string &prefix, size_t file_number);
std::vector<std::string> GetSSDList();

void MakeFileEndMarker(unsigned char *buffer, size_t size, size_t real_size);

double GetThroughput(size_t size, double time);
double GetThroughput(const std::vector<FileInfo> &files, double time);

#endif //SORTING_FILE_UTILS_H
