cc_binary(
    name = "sample_sort",
    srcs = ["sample-sort.cpp"],
    deps = [
        "//scatter_gather_algorithms:sample_sort",
        "//utils:command_line",
        "//utils:io_utils",
        "//utils:random_number_generator",
    ],
)

cc_binary(
    name = "permutation",
    srcs = ["permutation.cpp"],
    deps = [
        "//scatter_gather_algorithms:permutation",
        "//utils:command_line",
    ],
)

cc_binary(
    name = "sequence",
    srcs = ["sequence.cpp"],
    deps = [
        "//sequence_algorithms:filter",
        "//sequence_algorithms:map",
        "//sequence_algorithms:reduce",
        "//utils:command_line",
        "//utils:io_utils",
        "@com_google_absl//absl/log",
    ],
)

cc_binary(
    name = "speed_test",
    srcs = ["speed-test.cpp"],
    deps = [
        "//benchmarks:distribution_benchmarks",
        "//benchmarks:in_memory_benchmarks",
        "//benchmarks:io_benchmarks",
        "//utils:aligned_type_allocator",
        "//utils:command_line",
        "//utils:io_utils",
        "//utils:logger",
        "//utils:random_number_generator",
        "//utils:random_read",
    ],
)

cc_binary(
    name = "random_io_bench",
    srcs = ["random_io_bench.cpp"],
    deps = [
        "//:config",
        "//utils:command_line",
        "//utils:io_utils",
        "//utils:logger",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@parlaylib//parlay/internal:get_time",
    ],
)

cc_binary(
    name = "io_uring_test",
    srcs = ["io_uring_test.cpp"],
    linkopts = [
        "-luring",
    ],
    deps = [
        "//utils:io_utils",
        "//utils:logger",
        "@parlaylib//parlay/internal:get_time",
    ],
)

cc_library(
    name = "config",
    hdrs = ["configs.h"],
    visibility = ["//visibility:public"],
    deps = [],
)
