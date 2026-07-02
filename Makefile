CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -fno-omit-frame-pointer
# -I. lets local headers resolve by repo-root-relative path
# ("ChunkSequence/…", "utils/…", "configs.h").
INCLUDES := -I. -Ideps/parlaylib -Ideps/abseil-cpp/install/include
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
TEST_BINARIES := $(BINDIR)/permTest $(BINDIR)/mapTest $(BINDIR)/reduceTest \
                 $(BINDIR)/filterTest $(BINDIR)/scanTest $(BINDIR)/combinedTest \
                 $(BINDIR)/delayedTest $(BINDIR)/flatTabulateTest $(BINDIR)/findIfTest \
                 $(BINDIR)/histogramTest

# ChunkSequence examples (dual-purpose: demo + a machine-readable CSV line).
EXAMPLE_BINARIES := $(BINDIR)/primesExample

LINK = $(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

.PHONY: all clean distclean deps test examples bench bench-full bench-examples bench-examples-full

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

deps: deps/parlaylib deps/abseil-cpp/install

deps/parlaylib:
	mkdir -p deps
	git clone https://github.com/ParAlg/parlaylib.git deps/parlaylib-full
	cd deps/parlaylib-full && git checkout 6b4a4cdbfeb3c481608a42db0230eb6ebb87bf8d
	mv deps/parlaylib-full/include deps/parlaylib
	rm -rf deps/parlaylib-full

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

$(BINDIR)/permTest: ChunkSequence/tests/perm_test.cpp $(UTIL_OBJS)
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

$(BINDIR)/findIfTest: ChunkSequence/tests/find_if_test.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/histogramTest: ChunkSequence/tests/histogram_test.cpp $(UTIL_OBJS)
	$(LINK)

# ── examples ───────────────────────────────────────────────────────────────────

# Build every example.  Each example lives in ChunkSequence/examples/<name>.cpp
# and builds to bin/<name>Example via the generic pattern rule below.
examples: $(EXAMPLE_BINARIES)

$(BINDIR)/%Example: ChunkSequence/examples/%.cpp $(UTIL_OBJS)
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

# Opt-in examples sweep: time each example across a sweep of n.  Kept separate
# from `bench`/`bench-full` (examples are heterogeneous and some are expensive).
# `bench-examples` uses small dev-box (tmpfs) defaults.
bench-examples:
	python3 benchmarks/run_benches.py --examples --outdir results

# Full-scale examples sweep tuned for the benchmark machine (500 GiB RAM, 30x 1TB
# SSDs): sieve range 2^32 .. 2^40.  Multi-TB of I/O — not for a tmpfs dev box.
bench-examples-full:
	python3 benchmarks/run_benches.py --examples --outdir results \
	    --example-n-values "2^32 2^34 2^36 2^38 2^40"

# ── cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(UTIL_OBJS) $(TEST_BINARIES) $(EXAMPLE_BINARIES) \
	      $(BINDIR)/delayedCompare $(BINDIR)/chunkSizeCompare_*

distclean: clean
	rm -rf deps $(BINDIR) $(OBJDIR)
