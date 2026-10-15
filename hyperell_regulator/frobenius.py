r"""
Frobenius polynomials at good primes, in bulk.

An `L`-function needs the Euler factor at every good prime below a bound,
and for a genus 2 curve that is thousands of `p`-adic Frobenius
computations.  Sage will do them one at a time through its own
``hypellfrob``; this module hands the whole list to David Harvey's
``hypellfrob`` as a single batched, threaded job, so the per-call overhead
is paid once and the primes run concurrently.

The mathematics is unchanged.  The external program returns the matrix of
Frobenius on `H^1_{\mathrm{dR}}` modulo `p^N`, and the characteristic
polynomial is lifted to `\ZZ[x]` exactly as
:meth:`~sage.schemes.hyperelliptic_curves.hyperelliptic_finite_field.HyperellipticCurve_finite_field.frobenius_polynomial_matrix`
does: reduce `a_g, \dots, a_0` into the symmetric range mod `p^N`, with `N`
Kedlaya's bound, and recover `a_{2g},\dots,a_{g+1}` from the functional
equation `a_{2g-i} = a_i q^{g-i}`.

The program is built from ``hypellfrob-threaded`` and ships with this
package: ``sage -pip install`` compiles it and puts it in
``hyperell_regulator/bin/euler``, and in a checkout ``make`` in that
directory does the same job.  It is looked for at ``$HYPELLFROB_EULER``
first, then inside the package, then in the source tree.
When it is missing, or declines a prime, that prime falls back to Sage, so
results never depend on whether it is built -- only the speed does.
"""

import os
import subprocess

__all__ = ["euler_binary", "odd_model", "frobenius_polynomials",
           "frobenius_polynomial"]


def euler_binary():
    r"""
    The batched Frobenius program, or ``None`` if it is not built.

    EXAMPLES::

        sage: from hyperell_regulator.frobenius import euler_binary
        sage: euler_binary() is None or euler_binary().endswith("euler")
        True
    """
    env = os.environ.get("HYPELLFROB_EULER")
    if env:
        return env if os.path.isfile(env) and os.access(env, os.X_OK) else None
    here = os.path.dirname(os.path.abspath(__file__))
    # installed inside the package by setup.py; then the source tree, so a
    # checkout that has run ``make`` in hypellfrob-threaded works uninstalled
    for rel in (("bin", "euler"),
                ("..", "hypellfrob-threaded", "build", "euler")):
        guess = os.path.normpath(os.path.join(here, *rel))
        if os.path.isfile(guess) and os.access(guess, os.X_OK):
            return guess
    return None


def odd_model(f):
    r"""
    A monic, integral, **odd** degree model of `y^2 = f(x)`, or ``None``.

    The fast path needs `Q` monic of odd degree `2g+1`; `4f + h^2` has even
    degree whenever `h` has degree `g+1`, and the only even degree routine
    available is linear in `p`.  An odd degree model exists exactly when the
    curve has a rational Weierstrass point, and then

    .. math:: u \mapsto u^{\deg f}\,f(r + 1/u)

    moves that point `r` to infinity and drops the degree by one, the
    coefficient of `u^{\deg f}` being `f(r) = 0`.  It is an isomorphism over
    `\QQ`, so the characteristic polynomial of Frobenius is unchanged at
    every prime where both models are good; at the finitely many primes
    dividing the scalings, the program declines and the caller falls back.

    The result is then made monic by `x \mapsto cx`, `y \mapsto c^{(d-1)/2}y`,
    and integral by `x \mapsto x/m^2`, `y \mapsto y/m^d`.

    EXAMPLES::

        sage: from hyperell_regulator.frobenius import odd_model
        sage: from sage.all import ZZ, PolynomialRing
        sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
        sage: odd_model(x**5 + x**3 + 1)             # already odd and monic
        x^5 + x^3 + 1
        sage: f = (x - 1) * (x**5 + x + 3)           # even, with a rational root
        sage: g = odd_model(f); g.degree(), g.is_monic()
        (5, True)

    A curve with no rational Weierstrass point has no odd model::

        sage: odd_model(x**6 + x + 1) is None
        True
    """
    from sage.all import QQ, PolynomialRing, ZZ, lcm

    R = PolynomialRing(QQ, "x")
    f = R(f)
    d = f.degree()
    if d % 2 == 0:
        roots = f.roots(QQ, multiplicities=False)
        if not roots:
            return None
        r = roots[0]
        u = R.gen()
        f = sum(f[i] * u ** (d - i) * (r * u + 1) ** i for i in range(d + 1))
        d = f.degree()
        if d % 2 == 0:                       # r was a multiple root
            return None
    c = f.leading_coefficient()
    q = [f[i] * c ** (d - 1 - i) for i in range(d + 1)]
    m = lcm([QQ(v).denominator() for v in q] + [1])
    q = [q[i] * m ** (2 * (d - i)) for i in range(d + 1)]
    return PolynomialRing(ZZ, "x")(q)


def _bound(p, g):
    r"""Kedlaya's `N` with `p^N \ge 2\binom{2g}{g}p^{g/2}`, as Sage uses."""
    from sage.all import RR, ZZ, binomial

    M = 2 * binomial(2 * g, g) * RR(p).sqrt() ** g
    B = ZZ(M.ceil()).exact_log(p)
    if p ** B < M:
        B += 1
    return int(B)


def _lift(M, p, g, N):
    r"""The charpoly of Frobenius in `\ZZ[x]` from the matrix mod `p^N`."""
    from sage.all import ZZ, matrix

    a = list(matrix(ZZ, M).charpoly().list()[g:2 * g + 1])
    ppow = p ** N
    a = [x % ppow for x in a]
    a = [x if 2 * x < ppow else x - ppow for x in a]
    a = [a[g - i] * p ** (g - i) for i in range(g)] + a
    return ZZ["x"](a)


def _run(fcoeffs, jobs, nthreads):
    """One call to the program: (ascending Q, [(p, N)]) -> {p: matrix}."""
    binary = euler_binary()
    if binary is None or not jobs:
        return {}
    inp = " ".join(str(int(c)) for c in fcoeffs) + "\n"
    inp += "".join("%d %d\n" % (p, N) for p, N in jobs)
    workers = (os.cpu_count() or 1) if nthreads is None else max(1, int(nthreads))
    args = [binary, str(workers)]
    try:
        out = subprocess.run(args, input=inp, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[1] != "1":
            continue
        p, dim = int(parts[0]), int(parts[2])
        vals = [int(v) for v in parts[3:3 + dim * dim]]
        got[p] = [vals[r * dim:(r + 1) * dim] for r in range(dim)]
    return got


def frobenius_polynomials(f, primes, nthreads=None):
    r"""
    `\{p : \text{charpoly of Frobenius on } y^2 = f(x) \bmod p\}`.

    ``f`` is a polynomial over `\QQ` with integral coefficients; ``primes``
    are the primes to do, which must be of good reduction.  Any prime the
    batched program declines falls back to Sage.

    An even degree ``f`` is replaced by :func:`odd_model` where one exists,
    since only odd degree has the `O(\sqrt p)` path; without a rational
    Weierstrass point it goes through the even degree routine, which is
    correct but linear in `p`, so an even degree curve with no such point
    cannot reach the same speed.

    The default uses one worker per logical CPU (``os.cpu_count()``), or one
    if the CPU count is unavailable. Pass ``nthreads`` to override this.
    The installed accelerator includes a private thread-safe NTL; custom
    binaries must also be linked to a thread-safe NTL.

    EXAMPLES::

        sage: from hyperell_regulator.frobenius import frobenius_polynomials
        sage: from sage.all import ZZ, PolynomialRing
        sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
        sage: P = frobenius_polynomials(x**5 + x**3 + 1, [11, 13, 17])
        sage: [P[p].degree() for p in (11, 13, 17)]
        [4, 4, 4]
        sage: all(P[p][0] == p**2 for p in (11, 13, 17))     # a_0 = q^g
        True
    """
    from sage.all import GF, HyperellipticCurve, PolynomialRing, ZZ

    R = PolynomialRing(ZZ, "x")
    fz = R(f)
    g = (fz.degree() - 1) // 2
    primes = [int(p) for p in primes]
    jobs = [(p, _bound(p, g)) for p in primes]
    # the fast path wants a monic odd degree model; when the curve has a
    # rational Weierstrass point there is one, isomorphic over QQ, and it is
    # what gets sent. Without one the even degree model goes as it is, which
    # is correct but linear in p.
    model = odd_model(fz)
    got = _run((model if model is not None else fz).list(), jobs, nthreads)

    out = {}
    for p, N in jobs:
        if p in got:
            out[p] = _lift(got[p], p, g, N)
        else:
            H = HyperellipticCurve(PolynomialRing(GF(p), "x")(fz))
            out[p] = R(H.frobenius_polynomial())
    return out


def frobenius_polynomial(f, p):
    r"""
    One prime's charpoly of Frobenius; a convenience over
    :func:`frobenius_polynomials`.

    EXAMPLES::

        sage: from hyperell_regulator.frobenius import frobenius_polynomial
        sage: from sage.all import ZZ, PolynomialRing
        sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
        sage: frobenius_polynomial(x**5 + x**3 + 1, 37).degree()
        4
    """
    return frobenius_polynomials(f, [p])[int(p)]


def _check_binary(binary):
    """Compare serial, four-worker, and CPU-count runs without Sage fallback."""
    primes = (101, 103, 107, 109)
    batch = "1 0 0 1 0 1\n" + "".join("%d 3\n" % p for p in primes)
    serial = None
    workers = tuple(dict.fromkeys((1, 4, os.cpu_count() or 1)))
    for count in workers:
        result = subprocess.run([binary, str(count)], input=batch, text=True,
                                capture_output=True, check=True, timeout=30)
        rows = [line.split() for line in result.stdout.splitlines()]
        if len(rows) != len(primes) or any(
                row[:3] != [str(p), "1", "4"] or len(row) != 19
                for p, row in zip(primes, rows)):
            raise RuntimeError("unexpected driver output: " + result.stdout)
        if serial is not None and rows != serial:
            raise RuntimeError("parallel matrices differ from serial")
        serial = rows
    return workers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check the installed Frobenius accelerator.")
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--binary", help="check this executable instead of the installed one")
    args = parser.parse_args()
    binary = args.binary or euler_binary()
    if binary is None:
        parser.exit(1, "Frobenius accelerator not found; computations use Sage.\n")
    try:
        workers = _check_binary(binary)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        parser.exit(1, "Accelerator failed: %s\n%s\n" %
                    (exc, getattr(exc, "stderr", "")))
    print("Frobenius accelerator runs successfully with worker counts %s: %s" %
          (", ".join(map(str, workers)), binary))
