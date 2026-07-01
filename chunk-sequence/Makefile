CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -fno-omit-frame-pointer
# Include order matters: this repo (-I.) wins over the PLAID submodule (-Iplaid),
# so the local configs.h and sequence_algorithms/map.h shadow the submodule's.
INCLUDES := -I. -Iplaid -Ideps/parlaylib -Ideps/abseil-cpp/install/include -Ideps/stb
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

# The shared utility .cpp files live in the PLAID submodule.  Compile them into
# this repo's own $(OBJDIR) (never in-place) so the submodule tree stays clean.
UTIL_OBJS := $(OBJDIR)/logger.o $(OBJDIR)/command_line.o $(OBJDIR)/file_utils.o

BINARIES := $(BINDIR)/chunkSeqMain $(BINDIR)/permTest $(BINDIR)/mapTest \
            $(BINDIR)/reduceTest $(BINDIR)/filterTest $(BINDIR)/scanTest $(BINDIR)/combinedTest \
            $(BINDIR)/delayedTest $(BINDIR)/flatTabulateTest $(BINDIR)/bwCompare $(BINDIR)/bwDelayed \
            $(BINDIR)/raytracer $(BINDIR)/pathTracer $(BINDIR)/primes

# ChunkSequence correctness tests (each exits 0 on PASS, non-zero on FAIL).
TEST_BINARIES := $(BINDIR)/permTest $(BINDIR)/mapTest $(BINDIR)/reduceTest $(BINDIR)/filterTest $(BINDIR)/scanTest $(BINDIR)/combinedTest $(BINDIR)/delayedTest $(BINDIR)/flatTabulateTest

LINK = $(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

.PHONY: all clean distclean deps test

all:
	$(MAKE) deps
	$(MAKE) $(BINARIES)

# ── tests ──────────────────────────────────────────────────────────────────────

# Build and run every ChunkSequence correctness test in sequence.  Runs all of
# them even if one fails, then exits non-zero if any failed.  Pass extra args
# (e.g. a custom element count) via TEST_ARGS, e.g. `make test TEST_ARGS=8000000`.
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

deps: deps/parlaylib deps/abseil-cpp/install deps/stb/stb_image.h

deps/parlaylib:
	mkdir -p deps
	git clone https://github.com/ParAlg/parlaylib.git deps/parlaylib-full
	cd deps/parlaylib-full && git checkout 6b4a4cdbfeb3c481608a42db0230eb6ebb87bf8d
	mv deps/parlaylib-full/include deps/parlaylib
	rm -rf deps/parlaylib-full

deps/stb/stb_image.h:
	mkdir -p deps/stb
	curl -fsSL https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h \
	    -o deps/stb/stb_image.h

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

# Shared utilities from the PLAID submodule -> this repo's $(OBJDIR).
$(OBJDIR)/%.o: PLAID/utils/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ── binaries ───────────────────────────────────────────────────────────────────

$(BINDIR)/chunkSeqMain: ChunkSequence/chunk_seq_main.cpp $(UTIL_OBJS)
	$(LINK)

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

$(BINDIR)/bwCompare: ChunkSequence/bench/bw_compare.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/bwDelayed: ChunkSequence/bench/delayed_compare.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/raytracer: ChunkSequence/examples/raytracer.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/pathTracer: ChunkSequence/examples/path_tracer.cpp $(UTIL_OBJS)
	$(LINK)

$(BINDIR)/primes: ChunkSequence/examples/primes.cpp $(UTIL_OBJS)
	$(LINK)

# Chunk-size sensitivity benchmark: one binary per chunk size, compiled with
# -DCHUNK_SIZE_BYTES=N so the chunk layout is baked in at compile time.
# Usage: make bin/bwChunkSize_1048576   (for 1 MB chunks, etc.)
$(BINDIR)/bwChunkSize_%: ChunkSequence/bench/chunk_size_compare.cpp $(UTIL_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DCHUNK_SIZE_BYTES=$* $^ -o $@ $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

# ── cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(UTIL_OBJS) $(BINARIES)

distclean: clean
	rm -rf deps $(BINDIR) $(OBJDIR)
