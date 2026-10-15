# hypellfrob-threaded

Frobenius matrices of hyperelliptic curves over `F_p`, for one curve at many
primes, batched and run across threads.

This is **David Harvey's `hypellfrob`**. The mathematics and essentially all
of the code are his:

> Copyright (C) 2007, 2008, David Harvey
> `hypellfrob` version 2.1.1, as distributed with Sage in
> `sage/schemes/hyperelliptic_curves/hypellfrob`
>
> David Harvey, *Kedlaya's algorithm in larger characteristic*,
> International Mathematics Research Notices, 2007.
> [arXiv:math/0610973](https://arxiv.org/abs/math/0610973)

`hypellfrob` is free software under the **GNU General Public License,
version 2 or later**; the full text is in [COPYING](COPYING), and this
package is distributed under the same terms. Every file keeps its original
copyright header, and the files in `src/` carry notices saying whether and
how they were modified, as the GPL requires.

The copies here came from a working tree of `hypellfrob-fast`, which applies
three performance changes to the dyadic-evaluation recursion in
`recurrences_ntl.cpp`; its output is bit-identical to Harvey's original.
`even_degree.*` is not Harvey's: it is a direct implementation of Kedlaya's
algorithm, linear in `p` rather than `O(sqrt p)`, which handles the even
degree models `matrix()` rejects.

## Why it is here

An `L`-function needs the Euler factor at every good prime below a bound,
which for a genus 2 curve is thousands of `p`-adic Frobenius computations.
Sage will do them one at a time; `euler` takes the whole list as a single
job, so the setup is paid once and the primes run concurrently. On this
package's test curve that is about 5.6x faster at 2000 primes, and 2258
primes below 20000 take under four seconds.

Nothing here is required. `hyperell_regulator.frobenius` falls back to Sage
for any prime this program does not supply, so results never depend on
whether it is built -- only the speed does.

## Building

```
make                       # if NTL is under /usr/local
make NTL_INC=$HOME/ntl/include NTL_LIB=$HOME/ntl/src NTL_LINK=$HOME/ntl/src/ntl.a
make check
```

The package installer downloads the pinned, checksum-verified NTL 11.6.0 GitHub release,
builds it with `NTL_THREADS=on` and `NTL_THREAD_BOOST=on`, and statically links
it into `hyperell_regulator/bin/euler`. It discovers Sage's GMP prefix
(including `SAGE_LOCAL`); `GMP_INC`/`GMP_LIB` override those paths.
`HYPERELL_REGULATOR_BUILD_JOBS` sets build parallelism (default: logical CPU count).
Installation needs network access, Perl, Make, and a C++17 compiler.
Sage's NTL is never changed. Package builds ignore local `config.mk` and
`NTL_INC`/`NTL_LIB`/`NTL_LINK`; these remain available for manual builds.
The build compares serial, four-worker, and CPU-count results before installing.

For manual builds, copy `config.mk.example` to `config.mk` and edit it. Threads need NTL
built with `NTL_THREADS` and `NTL_THREAD_BOOST`; without them pass `1` for
the thread count, which still batches.

`make check` only shows that the driver runs. What checks the arithmetic is
`tests/test_lfunction.py`, which compares every Euler factor against Sage's
own, prime by prime.

## Using it

```
build/euler [nthreads]
```

reads from stdin

```
1 0 0 1 0 1        # Q's coefficients, ascending: Q = x^5 + x^3 + 1
11 3               # one line per job: p and the p-adic precision N
13 3
```

and writes one line per job, in order:

```
<p> 1 <dim> <entries...>      row-major, the 2g x 2g matrix mod p^N
<p> 0                         declined: bad reduction, or a precondition
                              of the fast path failed
```

Odd degree goes through `hypellfrob::matrix()`, which is `O(sqrt p)` and
needs `Q` monic with `p > (2g+1)(2N-1)`; even degree through
`matrix_reference()`, which is correct but linear in `p`.

The caller only sends an even degree model when it has to. `4f + h^2` has
even degree whenever `h` has degree `g+1`, but if the curve has a rational
Weierstrass point `r`, then `u^{\deg F} F(r + 1/u)` moves it to infinity and
has odd degree -- an isomorphism over `Q`, so the Frobenius characteristic
polynomial is unchanged -- and the fast path applies after monicising. Only
a curve with no rational Weierstrass point at all takes the slow route.

`hyperell_regulator.frobenius` drives this and lifts the matrices to
characteristic polynomials; it looks for the binary at `$HYPELLFROB_EULER`,
then at `hyperell_regulator/bin/euler`, then at
`hypellfrob-threaded/build/euler` beside the package.
