cc_library(
    name = "logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:initialize",
    ],
)

cc_library(
    name = "random_number_generator",
    srcs = ["random_number_generator.cpp"],
    hdrs = ["random_number_generator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":io_utils",
        "@parlaylib//parlay:parallel",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "io_utils",
    srcs = [],
    linkopts = ["-luring"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_info",
        ":ordered_file_writer",
        ":unordered_file_reader",
        ":unordered_file_writer",
    ],
)

cc_library(
    name = "file_info",
    srcs = ["file_info.h"],
    deps = [],
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cpp"],
    hdrs = ["file_utils.h"],
    deps = [
        ":file_info",
        ":logger",
        "//:config",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "unordered_file_writer",
    srcs = ["unordered_file_writer.h"],
    deps = [
        ":file_utils",
        ":logger",
        ":simple_queue",
    ],
)

cc_library(
    name = "unordered_file_reader",
    srcs = ["unordered_file_reader.h"],
    deps = [
        ":logger",
        ":simple_queue",
    ],
)

cc_library(
    name = "ordered_file_writer",
    srcs = ["ordered_file_writer.h"],
    deps = [
        ":aligned_type_allocator",
        ":file_info",
        ":logger",
        ":simple_queue",
        "//:config",
        "@parlaylib//parlay:primitives",
    ],
)

cc_library(
    name = "timer",
    srcs = ["timer.h"],
    visibility = ["//visibility:public"],
    deps = ["@parlaylib//parlay/internal:get_time"],
)

cc_library(
    name = "simple_queue",
    srcs = ["simple_queue.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "random_read",
    srcs = ["random_read.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        "@com_google_absl//absl/log",
        "@parlaylib//parlay:sequence",
    ],
)

cc_library(
    name = "command_line",
    srcs = ["command_line.cpp"],
    hdrs = ["command_line.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        "//:config",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "aligned_type_allocator",
    srcs = ["type_allocator.h"],
    visibility = ["//visibility:public"],
    deps = ["@parlaylib//parlay:alloc"],
)
