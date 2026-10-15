# Build `euler`: Frobenius matrices for one curve at many primes, in parallel.
#
# Requirements: a C++17 compiler, NTL and GMP. If NTL is not under
# /usr/local, either copy config.mk.example to config.mk and edit it, or
# pass the settings on the command line:
#
#   make NTL_INC=$HOME/ntl/include NTL_LIB=$HOME/ntl/src NTL_LINK=$HOME/ntl/src/ntl.a
#
# Threads need NTL built with NTL_THREADS and NTL_THREAD_BOOST; without them
# use one thread (`euler 1`), which still batches.

BUILD    ?= build
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17
NTL_INC  ?= /usr/local/include
NTL_LIB  ?= /usr/local/lib
NTL_LINK ?= -lntl
NTL_RPATH ?= -Wl,-rpath,"$(NTL_LIB)"

GMP_INC  ?= $(NTL_INC)
GMP_LIB  ?= $(NTL_LIB)
CONFIG_MK ?= config.mk

-include $(CONFIG_MK)

OBJS := $(BUILD)/hypellfrob.o $(BUILD)/recurrences_ntl.o $(BUILD)/even_degree.o

all: $(BUILD)/euler

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) -I"$(NTL_INC)" -I"$(GMP_INC)" -Isrc $(CPPFLAGS) $(CXXFLAGS) -pthread -c $< -o $@

$(BUILD)/euler: euler.cpp $(OBJS) | $(BUILD)
	$(CXX) -I"$(NTL_INC)" -I"$(GMP_INC)" -Isrc $(CPPFLAGS) $(CXXFLAGS) -pthread $^ \
	    $(LDFLAGS) -L"$(NTL_LIB)" -L"$(GMP_LIB)" \
	    $(NTL_RPATH) -Wl,-rpath,"$(GMP_LIB)" $(NTL_LINK) -lgmpxx -lgmp -lpthread -o $@

# A smoke test only: it shows the driver runs and produces matrices. What
# actually checks the arithmetic is tests/test_lfunction.py, which compares
# every Euler factor against Sage's own, prime by prime.
#
# The primes here clear the fast path's precondition p >= (2N-1)*deg; below
# it matrix() declines, which is not an error -- the caller falls back.
check: $(BUILD)/euler
	@printf '1 0 0 1 0 1\n101 3\n103 3\n1009 3\n' | $(BUILD)/euler 1 | \
	  awk '{ n++; if ($$2 != 1) { print "FAILED: declined at p=" $$1; bad=1 } } \
	       END { if (bad || n != 3) exit 1; print "ok: " n " matrices" }'

clean:
	rm -rf $(BUILD)

.PHONY: all check clean
