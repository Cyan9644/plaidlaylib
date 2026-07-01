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
                 $(BINDIR)/delayedTest $(BINDIR)/flatTabulateTest $(BINDIR)/findIfTest

LINK = $(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS) -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

.PHONY: all clean distclean deps test

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

# ── cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(UTIL_OBJS) $(TEST_BINARIES)

distclean: clean
	rm -rf deps $(BINDIR) $(OBJDIR)
