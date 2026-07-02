//
// Created by peter on 3/21/24.
//

#ifndef SORTING_FILE_INFO_H
#define SORTING_FILE_INFO_H

#include <string>
#include <utility>

struct FileInfo {
    std::string file_name;
    size_t file_index = 0;
    size_t true_size = 0;
    size_t file_size = 0;
    // sum of the size of the files before this file (if this file is a part of a sequence of files)
    size_t before_size = 0;

    FileInfo() = default;

    FileInfo(std::string file_name, const FileInfo &original) :
        file_name(std::move(file_name)),
        file_index(original.file_index), true_size(original.true_size), file_size(original.file_size) {};

    FileInfo(std::string file_name, size_t file_index, size_t true_size, size_t file_size) :
        file_name(std::move(file_name)),
        file_index(file_index), true_size(true_size), file_size(file_size) {};

};


#endif //SORTING_FILE_INFO_H
