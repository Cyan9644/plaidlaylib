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
  # Headers live in the -dev output, liburing.so in the runtime output.  Glob
  # for the .so itself rather than a lib/ directory: "*-liburing-[0-9]*/lib"
  # also matches the -dev output (whose lib/ holds only pkgconfig/), and it
  # sorts first, so a directory glob silently picks the one without the library.
  LIBURING_INC := $(firstword $(wildcard /nix/store/*-liburing-*-dev/include))
  LIBURING_LIB := $(patsubst %/,%,$(dir $(firstword \
                    $(wildcard /nix/store/*-liburing-*/lib/liburing.so))))
  ifneq ($(LIBURING_INC),)
    INCLUDES += -I$(LIBURING_INC)
  endif
  ifneq ($(LIBURING_LIB),)
    LDFLAGS  += -L$(LIBURING_LIB) -Wl,-rpath,$(LIBURING_LIB)
  endif
endif

ABSL_LIBDIR := $(firstword $(wildcard deps/abseil-cpp/install/lib deps/abseil-cpp/install/lib64))
ABSL_LIBS   := $(shell find $(ABSL_LIBDIR) -name '*.a' 2>/dev/null | sort)

# Vendored shared utilities (utils/), compiled into this repo's $(OBJDIR).
UTIL_OBJS := $(OBJDIR)/file_utils.o

# ChunkSequence correctness tests (each exits 0 on PASS, non-zero on FAIL).
# primitivesTest covers everything in Primitives/{chunk_seq,primitives,sort}.h;
# delayedTest covers Primitives/delayed.h; the rest are the four examples that
# carry a correctness test.
TEST_BINARIES := $(BINDIR)/primitivesTest $(BINDIR)/delayedTest \
                 $(BINDIR)/kmpTest $(BINDIR)/rabinKarpTest \
                 $(BINDIR)/bigintAddTest $(BINDIR)/convexHullTest

# ChunkSequence examples (dual-purpose: demo + a machine-readable CSV line).
# primitive_demos is one binary holding every per-primitive demo, dispatched on
# argv[1] (see ChunkSequence/examples/primitive_demos.cpp).
EXAMPLE_BINARIES := $(BINDIR)/primesExample $(BINDIR)/kmpExample \
                    $(BINDIR)/rabin_karpExample $(BINDIR)/bigint_addExample \
                    $(BINDIR)/linefitExample $(BINDIR)/convex_hullExample \
                    $(BINDIR)/samplesortExample $(BINDIR)/primitive_demosExample


.PHONY: all clean distclean deps test examples bench bench-full bench-examples bench-examples-full bench-summary trace clean-bench-data format format-check

all:
	$(MAKE) deps
	$(MAKE) $(TEST_BINARIES)

# ── formatting ──────────────────────────────────────────────────────────────────
# Google style via .clang-format at the repo root.  Formats this repo's own
# sources only — deps/ (fetched upstream) and results/ (generated) are excluded.
# clang-format comes from shell.nix (clang-tools); fall back to the Nix store if
# it is not on PATH.
CLANG_FORMAT := $(shell command -v clang-format 2>/dev/null || echo $(firstword $(wildcard /nix/store/*-clang-tools-*/bin/clang-format)))
FORMAT_FILES  = $(shell find . -path ./deps -prune -o -path ./results -prune -o \
                  \( -name '*.cpp' -o -name '*.h' -o -name '*.cc' -o -name '*.hpp' \) -print)

# Rewrite every source in place.
format:
	@test -n "$(CLANG_FORMAT)" || { echo "clang-format not found (enter nix-shell, or install clang-tools)"; exit 1; }
	$(CLANG_FORMAT) -i --style=file $(FORMAT_FILES)

# Non-mutating check: exits non-zero if anything is unformatted (hook-friendly).
format-check:
	@test -n "$(CLANG_FORMAT)" || { echo "clang-format not found (enter nix-shell, or install clang-tools)"; exit 1; }
	$(CLANG_FORMAT) --dry-run -Werror --style=file $(FORMAT_FILES)

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
	# Upstream bug: group_by_index/group_by_index_ pick their bucketing strategy
	# via `num_buckets*num_buckets`, computed in size_t with no overflow check.
	# Once num_buckets reaches 2^32 the square wraps to 0 (2^64 / 2^68 mod 2^64),
	# so the comparison always takes the group_by_small_int branch regardless of
	# the real size relationship -- confirmed to misroute large-vertex-count RMAT
	# graphs (external_bellman_ford at n=2^32+) into that far more memory-hungry
	# path, tripping its internal `assert(k < num_buckets)`. Fixed by comparing
	# via division instead of squaring (no wider standard integer type than
	# size_t exists to widen into here).
	sed -i 's/if (A.size() > static_cast<size_t>(num_buckets)\*num_buckets) {/if (A.size() \/ static_cast<size_t>(num_buckets) > static_cast<size_t>(num_buckets)) {/' \
	    deps/parlaylib/parlay/internal/group_by.h
	sed -i 's/if (n > static_cast<size_t>(num_buckets)\*num_buckets) {/if (n \/ static_cast<size_t>(num_buckets) > static_cast<size_t>(num_buckets)) {/' \
	    deps/parlaylib/parlay/internal/group_by.h

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
# -MMD -MP emits a .d sidecar per object listing the headers it pulled in; the
# -include below feeds those back so a header edit rebuilds what depends on it.
# This matters more than it used to: the whole library is four headers now, so
# without it essentially every edit would need a manual `rm -f bin/<target>`.
$(OBJDIR)/%.o: utils/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP $(INCLUDES) -c $< -o $@

# Header dependencies for the binaries: each link rule below also writes
# $(OBJDIR)/<target>.d via -MMD, so editing a header relinks its dependents.
DEPFLAGS = -MMD -MP -MF $(OBJDIR)/$(@F).d
# $< $(UTIL_OBJS), NOT $^: the -include below adds every header in the .d as a
# prerequisite of the binary, and $^ would then hand those headers to g++ as
# source inputs (which fails outright on e.g. liburing's barrier.h).  $< is the
# .cpp; the util objects are named explicitly.
LINKD = $(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $< $(UTIL_OBJS) -o $@ $(LDFLAGS) \
        -Wl,--start-group $(ABSL_LIBS) -Wl,--end-group

-include $(wildcard $(OBJDIR)/*.d)

# ── test binaries ──────────────────────────────────────────────────────────────

# primitivesTest: every case for chunk_seq.h / primitives.h / sort.h in one
# binary (iota, map, reduce, scan, segmented_reduce, find_if, histogram,
# scalar, filter, flat_tabulate, flat_map, partition, group_by,
# chunk_operation, combined, samplesort).
$(BINDIR)/primitivesTest: ChunkSequence/tests/primitives_test.cpp $(UTIL_OBJS)
	$(LINKD)

$(BINDIR)/delayedTest: ChunkSequence/tests/delayed_test.cpp $(UTIL_OBJS)
	$(LINKD)

$(BINDIR)/kmpTest: ChunkSequence/tests/kmp_test.cpp $(UTIL_OBJS)
	$(LINKD)

$(BINDIR)/rabinKarpTest: ChunkSequence/tests/rabin_karp_test.cpp $(UTIL_OBJS)
	$(LINKD)

# bigintAddTest includes an example header (examples/chunk_bigint_add.h); no
# order-only deps/parlaylib-examples prereq is needed (no upstream baseline).
$(BINDIR)/bigintAddTest: ChunkSequence/tests/bigint_add_test.cpp $(UTIL_OBJS)
	$(LINKD)

# convexHullTest includes upstream quickhull.h (its in-DRAM differential
# baseline), so it needs the order-only deps/parlaylib-examples prereq.
$(BINDIR)/convexHullTest: ChunkSequence/tests/convex_hull_test.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINKD)

# ── examples ───────────────────────────────────────────────────────────────────

# Build every example.  Each lives in ChunkSequence/examples/<name>.cpp and
# builds to bin/<name>Example via the generic pattern rule below -- there are no
# per-binary override rules any more.
examples: $(EXAMPLE_BINARIES)

# Order-only prereq: examples include upstream parlaylib example headers
# ("parlaylib-examples/…") as their in-memory baselines, and run_benches.py
# builds these targets directly (not via `make all`, which runs `deps` first).
$(BINDIR)/%Example: ChunkSequence/examples/%.cpp $(UTIL_OBJS) | deps/parlaylib-examples
	$(LINKD)

# ── benchmarks ─────────────────────────────────────────────────────────────────

# delayed_compare: one binary, swept over n at runtime.
$(BINDIR)/delayedCompare: benchmarks/delayed_compare.cpp $(UTIL_OBJS)
	$(LINKD)

# chunk_size_compare: compiled once per CHUNK_SIZE via -DCHUNK_SIZE_BYTES=<stem>.
# e.g. `make bin/chunkSizeCompare_2097152` bakes in a 2 MiB chunk size.
$(BINDIR)/chunkSizeCompare_%: benchmarks/chunk_size_compare.cpp $(UTIL_OBJS)
	$(CXX) $(CXXFLAGS) -DCHUNK_SIZE_BYTES=$* $(INCLUDES) $< $(UTIL_OBJS) -o $@ \
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

# Opt-in examples sweep: time each example across a sweep of input SIZE (each
# binary's element count is derived per example so all examples move the same
# bytes; see size_to_n in run_benches.py).  Kept separate from `bench`/`bench-full`
# (examples are heterogeneous and some are expensive).  `bench-examples` uses
# small dev-box (tmpfs) defaults (128MiB .. 1GiB).
bench-examples:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,linefit,convex_hull,samplesort" --outdir results

# Mid-scale examples sweep: input sizes up to 256 GiB.
bench-examples-mid:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,linefit,convex_hull,samplesort" --outdir results \
	    --example-sizes "1GiB 4GiB 16GiB 64GiB 256GiB"

# Full-scale examples sweep tuned for the benchmark machine (500 GiB RAM, 30x 1TB
# SSDs): input sizes up to 1 TiB.  Multi-TB of I/O — not for a tmpfs dev box.
bench-examples-full:
	python3 benchmarks/run_benches.py --example "primes,kmp,rabin_karp,bigint_add,linefit,convex_hull,samplesort" --outdir results \
	    --example-sizes "1GiB 4GiB 16GiB 64GiB 256GiB 1TiB"

# Combined bar chart: 19 primitives/examples, each run ONCE at the largest n
# where its own in-mem parlaylib baseline still fits DRAM, plotted as a
# relative-performance bar chart (in-mem pinned at 1.0). See
# benchmarks/summary_figure.py.  Real-scale run -- not for a tmpfs dev box
# (some entries' RAM-cliff n is large); dry-run with a small
# EXAMPLE_INMEM_BUDGET_BYTES override first (see the plan/verification notes).
bench-summary:
	python3 benchmarks/summary_figure.py --outdir results

# Single-run IO/CPU trace of one example (per-SSD read/write throughput + %util +
# CPU over time; build/op phases marked).  Meaningful on real block devices only.
#   make trace EXAMPLE=kmp SIZE=1GiB          (add ARGS='m ...' for extra positionals)
trace:
	python3 benchmarks/io_trace.py $(EXAMPLE) --size $(SIZE) $(if $(ARGS),-- $(ARGS),)

# ── cleanup ────────────────────────────────────────────────────────────────────

clean:
	rm -f $(UTIL_OBJS) $(OBJDIR)/*.d $(TEST_BINARIES) $(EXAMPLE_BINARIES) \
	      $(BINDIR)/delayedCompare $(BINDIR)/chunkSizeCompare_*

distclean: clean
	rm -rf deps $(BINDIR) $(OBJDIR)

# Remove files the benchmarks left behind: per-drive data (iota<drive> inputs +
# every pipeline's intermediates) under /mnt/ssd*, cleared between every sweep
# point already but worth a manual pass after a crash or an interrupted run.
# Leaves results/ (the actual figures/CSVs) alone unless CLEAN_BENCH_ARGS adds
# --results.  See benchmarks/clean_bench_data.py --help for all options.
#   make clean-bench-data
#   make clean-bench-data CLEAN_BENCH_ARGS=--dry-run
#   make clean-bench-data CLEAN_BENCH_ARGS=--results
clean-bench-data:
	python3 benchmarks/clean_bench_data.py $(CLEAN_BENCH_ARGS)
