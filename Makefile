CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -fno-omit-frame-pointer
# -I. lets local headers resolve by repo-root-relative path
# ("ChunkSequence/…", "utils/…", "configs.h").  -Ideps lets the fetched
# upstream example headers resolve as "parlaylib-examples/<name>.h".
INCLUDES := -I. -Ideps -Ideps/parlaylib -Ideps/abseil-cpp/install/include
LDFLAGS  := -luring -lpthread

BINDIR := bin
OBJDIR := build
$(shell mkdir -p $(BINDIR) $(OBJDIR))

# Detect Nix environment and add liburing include/lib paths
ifdef NIX_CFLAGS_COMPILE
  INCLUDES += $(patsubst -isystem%,-I%,$(filter -isystem%,$(NIX_CFLAGS_COMPILE)))
endif
ifdef NIX_LDFLAGS
  LDFLAGS  += $(filter -L%,$(NIX_LDFLAGS))
endif

# If liburing.h is not on the default search path, find it in the Nix store.
ifeq ($(shell echo '\#include <liburing.h>' | $(CXX) -x c++ - -fsyntax-only 2>/dev/null; echo $$?),0)
else
  LIBURING_INC := $(firstword $(wildcard /nix/store/*-liburing-*-dev/include))
  LIBURING_LIB := $(firstword $(wildcard /nix/store/*-liburing-[0-9]*/lib))
  ifneq ($(LIBURING_INC),)
    INCLUDES += -I$(LIBURING_INC)
    LDFLAGS  += -L$(LIBURING_LIB)
  endif
endif

ABSL_LIBDIR := $(firstword $(wildcard deps/abseil-cpp/install/lib deps/abseil-cpp/install/lib64))
ABSL_LIBS   := $(shell find $(ABSL_LIBDIR) -name '*.a' 2>/dev/null | sort)

# Vendored shared utilities (utils/), compiled into this repo's $(OBJDIR).
UTIL_OBJS := $(OBJDIR)/logger.o $(OBJDIR)/command_line.o $(OBJDIR)/file_utils.o

# ChunkSequence correctness tests (each exits 0 on PASS, non-zero on FAIL).
TEST_BINARIES := $(BINDIR)/iotaTest $(BINDIR)/mapTest $(BINDIR)/reduceTest \
                 $(BINDIR)/filterTest $(BINDIR)/scanTest $(BINDIR)/combinedTest \
                 $(BINDIR)/delayedTest $(BINDIR)/flatTabulateTest \
                 $(BINDIR)/flatMapTest $(BINDIR)/findIfTest \
                 $(BINDIR)/histogramTest $(BINDIR)/kmpTest $(BINDIR)/rabinKarpTest \
                 $(BINDIR)/scalarTest $(BINDIR)/bigintAddTest $(BINDIR)/convexHullTest \
                 $(BINDIR)/partitionTest $(BINDIR)/segmentedReduceTest \
                 $(BINDIR)/dc3Test

# ChunkSequence examples (dual-purpose: demo + a machine-readable CSV line).
EXAMPLE_BINARIES := $(BINDIR)/primesExample $(BINDIR)/kmpExample \
                    $(BINDIR)/rabin_karpExample $(BINDIR)/kth_smallestExample \
                    $(BINDIR)/external_samplesortExample $(BINDIR)/external_linefitExample \
                    $(BINDIR)/fitmem_sortExample $(BINDIR)/fitmem_kth_smallestExample \
                    $(BINDIR)/bigint_addExample $(BINDIR)/bigint_add_eagerExample \
                    $(BINDIR)/chunk_cutExample \
                    $(BINDIR)/external_samplesort_vs_peterExample \
                    $(BINDIR)/direct_samplesort_vs_peterExample \
                    $(BINDIR)/samplesort_three_wayExample \
                    $(BINDIR)/external_random_shuffleExample \
                    $(BINDIR)/convex_hullExample \
                    $(BINDIR)/bellman_fordExample \

# Peter's external sample sort (the second contestant in the
# external_samplesort_vs_peter comparison) ships its own configs.h /
# utils/file_utils.h that clash by include guard with the main repo's, so it is
# isolated in one shim TU compiled with its own directory first on the include
# path.  Its shared utils (file_utils.cpp/logger/...) are byte-identical to the
# main repo's, so it links against $(UTIL_OBJS) — do NOT also compile Peter's
# utils/*.cpp (that would duplicate FindFiles/GetFileName/... symbols).
PETER_DIR := ChunkSequence/examples/external/peter_samplesort

LINK = $(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

.PHONY: all clean distclean deps test examples bench bench-full bench-examples bench-examples-full trace

all:
	$(MAKE) deps
	$(MAKE) $(TEST_BINARIES)

# ── tests ──────────────────────────────────────────────────────────────────────

# Build and run every correctness test in sequence.  Runs all of them even if
# one fails, then exits non-zero if any failed.  Pass extra args (e.g. a custom
# element count) via TEST_ARGS, e.g. `make test TEST_ARGS=8000000`.
test: $(TEST_BINARIES)
	@fail=0; \
	for t in $(TEST_BINARIES); do \
	  echo "==================== $$t $(TEST_ARGS) ===================="; \
	  $$t $(TEST_ARGS) || fail=1; \
	  echo; \
	done; \
	if [ $$fail -ne 0 ]; then echo "SOME TESTS FAILED"; exit 1; \
	else echo "ALL TESTS PASSED"; fi

# ── dependency fetching ────────────────────────────────────────────────────────

deps: deps/parlaylib deps/parlaylib-examples deps/abseil-cpp/install

deps/parlaylib:
	mkdir -p deps
	git clone https://github.com/ParAlg/parlaylib.git deps/parlaylib-full
	cd deps/parlaylib-full && git checkout 6b4a4cdbfeb3c481608a42db0230eb6ebb87bf8d
	mv deps/parlaylib-full/include deps/parlaylib
	rm -rf deps/parlaylib-full

# Upstream parlaylib example algorithms (knuth_morris_pratt.h, rabin_karp.h,
# primes.h, …), used as the in-memory comparison baselines by the examples.
# A separate clone of the same pinned commit: the deps/parlaylib rule above
# keeps only include/, and won't re-fire on checkouts that already have it.
#
# Three upstream bugs are patched after the fetch (each confirmed, and the
# fix verified, with an exact-position brute-force property test):
#  1. kmp: the search loop index is `int`, so any text over 2^31 chars
#     truncates it negative -> wild read -> SIGSEGV.
#  2. kmp: after a full match the automaton state is never reset, so the next
#     comparison reads pattern[m] (one past the end); if that garbage byte
#     matches the text the state runs away off both arrays.  Fix = take the
#     failure transition when reporting, the standard KMP step.
#  3. rabin_karp: the last window is compared against `total` (the powers-scan
#     total, x^n) instead of `sum` (the text-hash total), so a match at the
#     final position n-m is missed.
deps/parlaylib-examples:
	mkdir -p deps
	git clone https://github.com/ParAlg/parlaylib.git deps/parlaylib-examples-full
	cd deps/parlaylib-examples-full && git checkout 6b4a4cdbfeb3c481608a42db0230eb6ebb87bf8d
	mv deps/parlaylib-examples-full/examples deps/parlaylib-examples
	rm -rf deps/parlaylib-examples-full
	sed -i 's/for (int i=start;/for (long i=start;/' \
	    deps/parlaylib-examples/knuth_morris_pratt.h
	sed -i 's/if (tail == n-1) out.push_back(i - tail);/if (tail == n-1) { out.push_back(i - tail); tail = failure_p[tail]; }/' \
	    deps/parlaylib-examples/knuth_morris_pratt.h
	sed -i 's/total = total\] (long i)/total = sum] (long i)/' \
	    deps/parlaylib-examples/rabin_karp.h

deps/abseil-cpp/install:
	mkdir -p deps
	git clone --depth 1 --branch 20240722.1 \
	    https://github.com/abseil/abseil-cpp.git deps/abseil-cpp
	rm -rf deps/abseil-cpp/.git
	cd deps/abseil-cpp && cmake -S . -B build \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	    -DABSL_BUILD_TESTING=OFF \
	    -DABSL_ENABLE_INSTALL=ON \
	    -DBUILD_SHARED_LIBS=OFF \
	    -DABSL_PROPAGATE_CXX_STD=ON \
	    -DCMAKE_CXX_STANDARD=17 \
	    -DCMAKE_INSTALL_PREFIX=$(CURDIR)/deps/abseil-cpp/install && \
	    cmake --build build -j$$(nproc) && \
	    cmake --install build

# ── compilation rules ──────────────────────────────────────────────────────────

# Vendored utilities -> this repo's $(OBJDIR).
$(OBJDIR)/%.o: utils/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ── test binaries ──────────────────────────────────────────────────────────────

$(BINDIR)/iotaTest: ChunkSequence/tests/iota_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/mapTest: ChunkSequence/tests/map_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/reduceTest: ChunkSequence/tests/reduce_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/filterTest: ChunkSequence/tests/filter_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/scanTest: ChunkSequence/tests/scan_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/combinedTest: ChunkSequence/tests/combined_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/delayedTest: ChunkSequence/tests/delayed_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/flatTabulateTest: ChunkSequence/tests/flat_tabulate_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/flatMapTest: ChunkSequence/tests/flat_map_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/findIfTest: ChunkSequence/tests/find_if_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/histogramTest: ChunkSequence/tests/histogram_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/kmpTest: ChunkSequence/tests/kmp_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/rabinKarpTest: ChunkSequence/tests/rabin_karp_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/scalarTest: ChunkSequence/tests/scalar_test.cpp $(UTIL_OBJS)
	$(LINK)

# bigintAddTest includes an example header (examples/chunk_bigint_add.h); no
# order-only deps/parlaylib-examples prereq is needed (no upstream baseline).
$(BINDIR)/bigintAddTest: ChunkSequence/tests/bigint_add_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/partitionTest: ChunkSequence/tests/partition_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/segmentedReduceTest: ChunkSequence/tests/segmented_reduce_test.cpp $(UTIL_OBJS)
	$(LINK)

# dc3Test: out-of-core DC3 suffix array vs brute force + a streaming-vs-DRAM
# differential.  Header-only algorithm (chunk_dc3.h), no upstream baseline.
$(BINDIR)/dc3Test: ChunkSequence/tests/dc3_test.cpp $(UTIL_OBJS)
	$(LINK)

# convexHullTest includes upstream quickhull.h (its in-DRAM differential
# baseline), so it needs the order-only deps/parlaylib-examples prereq.
$(BINDIR)/convexHullTest: ChunkSequence/tests/convex_hull_test.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/tempMain: ChunkSequence/examples/temp_main.cpp $(UTIL_OBJS)
	$(LINK)

# ── examples ───────────────────────────────────────────────────────────────────

# Build every example.  Each example lives in ChunkSequence/examples/<name>.cpp
# and builds to bin/<name>Example via the generic pattern rule below.
examples: $(EXAMPLE_BINARIES)

# Order-only prereq: examples include upstream parlaylib example headers
# ("parlaylib-examples/…") as their in-memory baselines, and run_benches.py
# builds these targets directly (not via `make all`, which runs `deps` first).
$(BINDIR)/%Example: ChunkSequence/examples/%.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

# kth_smallest, external_samplesort, and external_linefit (the
# External-primitives examples) live one level deeper, in examples/external/,
# than the %Example pattern rule reaches, so they need explicit rules.  Same
# recipe.
$(BINDIR)/kth_smallestExample: ChunkSequence/examples/external/kth_smallest.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/external_samplesortExample: ChunkSequence/examples/external/external_samplesort.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/external_linefitExample: ChunkSequence/examples/external/external_linefit.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/external_random_shuffleExample: ChunkSequence/examples/external/external_random_shuffle.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

# fitmem variants: single-level (buckets fit in DRAM) sample sort / kth-smallest,
# same examples/external/ location and recipe as the fully external siblings.
$(BINDIR)/fitmem_sortExample: ChunkSequence/examples/external/fitmem_sort.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/fitmem_kth_smallestExample: ChunkSequence/examples/external/fitmem_kth_smallest.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

$(BINDIR)/chunk_cutExample: ChunkSequence/examples/external/chunk_cut.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

# bellman_ford: out-of-core external_bellman_ford vs the in-memory reference
# (examples/in_memory/graph/), same examples/external/ location as its
# siblings. Unlike them it needs an extra -I: bellman_ford.h's own
# `#include "helper/ligra_light.h"` is unprefixed (copied verbatim from
# deps/parlaylib-examples/bellman_ford.h, which resolves it via quoted-include
# fallback to its own containing directory), so compiling the local copy
# directly needs deps/parlaylib-examples on the search path explicitly.
$(BINDIR)/bellman_fordExample: ChunkSequence/examples/external/bellman_ford.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(CXX) $(CXXFLAGS) -Ideps/parlaylib-examples $(INCLUDES) $^ -o $@ \
	    $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

# external_samplesort_vs_peter: our sample sort vs Peter's, head-to-head.  The
# driver TU uses the main includes; Peter's sort is linked in via peter_shim.o,
# compiled separately below with $(PETER_DIR) first so its own configs.h/utils
# resolve (never the main repo's).
$(OBJDIR)/peter_shim.o: $(PETER_DIR)/peter_shim.cpp | deps/parlaylib deps/abseil-cpp/install
	$(CXX) $(CXXFLAGS) -I$(PETER_DIR) $(INCLUDES) -c $< -o $@

$(BINDIR)/external_samplesort_vs_peterExample: ChunkSequence/examples/external/external_samplesort_vs_peter.cpp $(UTIL_OBJS) $(OBJDIR)/peter_shim.o | deps/parlaylib-examples
	$(LINK)

# direct_samplesort_vs_peter: same head-to-head, but our contestant is the
# direct-I/O sort (direct_samplesort.h) rather than the primitives-based one, so
# the pair of sweeps isolates the substrate from the algorithm.  Same shim.
$(BINDIR)/direct_samplesort_vs_peterExample: ChunkSequence/examples/external/direct_samplesort_vs_peter.cpp $(UTIL_OBJS) $(OBJDIR)/peter_shim.o | deps/parlaylib-examples
	$(LINK)

# samplesort_three_way: all three out-of-core sorts (Peter's, our direct-I/O one,
# our primitives one) in one run on the same keys, with the run order rotated
# across rounds so no sort is always the one paying for the previous one's
# deleted files.  Same shim as the pairwise drivers.
$(BINDIR)/samplesort_three_wayExample: ChunkSequence/examples/external/samplesort_three_way.cpp $(UTIL_OBJS) $(OBJDIR)/peter_shim.o | deps/parlaylib-examples
	$(LINK)

# suffix_array: out-of-core prefix-doubling suffix array (built on the direct-I/O
# sample sort) vs upstream parlaylib suffix_array in DRAM.  Same examples/external/
# location and plain recipe as the other external siblings; the upstream baseline
# header resolves via -Ideps (in $(INCLUDES)).
$(BINDIR)/suffix_arrayExample: ChunkSequence/examples/external/suffix_array.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

# dc3: out-of-core DC3 / skew suffix array (streaming Kärkkäinen–Sanders on the
# direct-I/O sample sort) vs upstream parlaylib suffix_array in DRAM.  Same
# examples/external/ location and plain recipe as suffix_arrayExample.
$(BINDIR)/dc3Example: ChunkSequence/examples/external/dc3.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINK)

# ── benchmarks ─────────────────────────────────────────────────────────────────

# delayed_compare: one binary, swept over n at runtime.
$(BINDIR)/delayedCompare: benchmarks/delayed_compare.cpp $(UTIL_OBJS)
	$(LINK)

# chunk_size_compare: compiled once per CHUNK_SIZE via -DCHUNK_SIZE_BYTES=<stem>.
# e.g. `make bin/chunkSizeCompare_2097152` bakes in a 2 MiB chunk size.
$(BINDIR)/chunkSizeCompare_%: benchmarks/chunk_size_compare.cpp $(UTIL_OBJS)
	$(CXX) $(CXXFLAGS) -DCHUNK_SIZE_BYTES=$* $(INCLUDES) $^ -o $@ \
	    $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

# One-off exploratory benchmarks (old vs new delayed-filter windowing; zip
# chain depth scaling). Deliberately NOT part of TEST_BINARIES/EXAMPLE_BINARIES
# or the bench/bench-full recipes — build and run manually, same as bin/tempMain.
$(BINDIR)/filterCompare: benchmarks/filter_compare.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/zipDepthCompare: benchmarks/zip_depth_compare.cpp $(UTIL_OBJS)
	$(LINK)

# Run both benchmark sweeps and write timestamped images + CSVs under results/.
# The Python driver builds each binary via make, runs the sweep, and plots.
# Override the sweep via env, e.g. `make bench BENCH_CHUNK_SIZES="2097152 8388608"`.
# `bench` uses small dev-box (tmpfs) defaults.
bench:
	python3 benchmarks/run_benches.py --all --outdir results

# Full-scale sweep tuned for the benchmark machine (500 GiB RAM, 30x 1TB SSDs):
# delayed scale 2^30..2^39 elements (8 B each), chunk-size test at 2^28 elements.
# Multi-TB of I/O — intended for the real machine, not a tmpfs dev box.
bench-full:
	python3 benchmarks/run_benches.py --all --outdir results \
	    --n-values "2^30 2^31 2^32 2^33 2^34 2^35 2^36 2^37 2^38 2^39" \
	    --n 268435456 \
	    --chunk-sizes "256KiB 512KiB 1MiB 2MiB 4MiB 8MiB 16MiB"

# Opt-in examples sweep: time each example across a sweep of input SIZE (each
# binary's element count is derived per example so all examples move the same
# bytes; see size_to_n in run_benches.py).  Kept separate from `bench`/`bench-full`
# (examples are heterogeneous and some are expensive).  `bench-examples` uses
# small dev-box (tmpfs) defaults (128MiB .. 1GiB).
bench-examples:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,convex_hull" --outdir results

# Mid-scale examples sweep: input sizes up to 256 GiB.
bench-examples-mid:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,convex_hull" --outdir results \
	    --example-sizes "1GiB 4GiB 16GiB 64GiB 256GiB"

# Full-scale examples sweep tuned for the benchmark machine (500 GiB RAM, 30x 1TB
# SSDs): input sizes up to 1 TiB.  Multi-TB of I/O — not for a tmpfs dev box.
bench-examples-full:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,convex_hull" --outdir results \
	    --example-sizes "1GiB 4GiB 16GiB 64GiB 256GiB 1TiB"

# Single-run IO/CPU trace of one example (per-SSD read/write throughput + %util +
# CPU over time; build/op phases marked).  Meaningful on real block devices only.
#   make trace EXAMPLE=kmp SIZE=1GiB          (add ARGS='m ...' for extra positionals)
trace:
	python3 benchmarks/io_trace.py $(EXAMPLE) --size $(SIZE) $(if $(ARGS),-- $(ARGS),)

# ── cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(UTIL_OBJS) $(OBJDIR)/peter_shim.o $(TEST_BINARIES) $(EXAMPLE_BINARIES) \
	      $(BINDIR)/delayedCompare $(BINDIR)/chunkSizeCompare_*

distclean: clean
	rm -rf deps $(BINDIR) $(OBJDIR)
