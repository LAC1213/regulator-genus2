r"""Tests for the batched Frobenius path and the two `L`-functions.

The Euler factors at good primes now come from David Harvey's
``hypellfrob``, handed the whole list of primes as one threaded job rather
than one prime at a time.  The mathematics must be unchanged, so the first
thing checked is that the batched factors are *identical* to Sage's, prime
by prime; and that removing the program falls back to Sage rather than
failing.

An even degree model -- which is what `4f + h^2` gives when `h` has degree
`g+1` -- is replaced by its odd model when the curve has a rational
Weierstrass point to move to infinity, and otherwise goes through a routine
linear in `p`: correct, but not the same speed.

Then the two `L`-functions.  `L(C,s)` is checked against PARI's own
``lfungenus2`` -- coefficients and value both -- since that is an
independent implementation of the same object.  Finally, that asking for
more digits actually delivers them: returning ``RR(pari.lfun(...))`` caps
every value at 53 bits however high PARI's working precision is set, which
is the bug the arbitrary-precision path had to fix.
"""

import os
import subprocess

import pytest

from sage.all import (GF, HyperellipticCurve, PolynomialRing, QQ, RR, ZZ,
                      pari, prime_range)

from hyperell_regulator import HyperellipticCurveWithReduction
from hyperell_regulator.frobenius import (
    euler_binary,
    frobenius_polynomial,
    frobenius_polynomials,
    odd_model,
)
from hyperell_regulator.lfunction import (CurveLFunction, StandardLFunction,
                                          _real_field)

R = PolynomialRing(ZZ, "x")
X = R.gen()
F = X ** 5 + X ** 3 + 1                      # conductor 3233, bad 53 and 61
PRIMES = [p for p in prime_range(5, 300) if p not in (53, 61)]


def _sage_frobenius(f, p):
    H = HyperellipticCurve(PolynomialRing(GF(p), "x")(f))
    return R(H.frobenius_polynomial())


def test_batched_frobenius_matches_sage():
    got = frobenius_polynomials(F, PRIMES, nthreads=4)
    assert set(got) == set(PRIMES)
    for p in PRIMES:
        assert got[p] == _sage_frobenius(F, p), p


def test_frobenius_is_a_weil_polynomial():
    r"""
    Degree `2g`, `a_0 = q^g`, and the functional equation
    `a_{2g-i} = a_i q^{g-i}` -- which the lifting imposes, so this checks the
    lifting rather than the matrix.
    """
    for p, cp in frobenius_polynomials(F, PRIMES[:8]).items():
        c = cp.list()
        assert cp.degree() == 4 and c[4] == 1 and c[0] == p ** 2
        assert c[3] * p == c[1]


def test_falls_back_when_the_program_is_missing():
    r"""
    The answer must not depend on whether the program is installed.
    """
    old = os.environ.get("HYPELLFROB_EULER")
    os.environ["HYPELLFROB_EULER"] = "/nonexistent/euler"
    try:
        assert euler_binary() is None
        got = frobenius_polynomials(F, PRIMES[:6])
        for p in PRIMES[:6]:
            assert got[p] == _sage_frobenius(F, p), p
    finally:
        if old is None:
            del os.environ["HYPELLFROB_EULER"]
        else:
            os.environ["HYPELLFROB_EULER"] = old


def test_odd_model():
    r"""
    Monic, odd, integral; and ``None`` exactly when there is no rational
    Weierstrass point to move to infinity.
    """
    assert odd_model(F) == F                       # already monic and odd
    for f in ((X - 1) * (X ** 5 + X + 3), (X + 2) * (X ** 5 - X + 1)):
        g = odd_model(f)
        assert g.degree() == 5 and g.is_monic()
        assert all(c in ZZ for c in g.list())
    assert odd_model(X ** 6 + X + 1) is None
    assert odd_model(4 * (X ** 5 - X + 1) + (X ** 3 + X + 1) ** 2) is None


def test_even_degree_with_a_weierstrass_point_takes_the_fast_path():
    r"""
    An even degree model with a rational root is sent through its odd model,
    which is isomorphic over `\QQ`, so Sage's answer for the *original* must
    come back -- and quickly.
    """
    import time

    f = (X - 1) * (X ** 5 + X + 3)
    assert f.degree() % 2 == 0 and odd_model(f) is not None
    ps = [p for p in prime_range(7, 150) if f.discriminant() % p]
    t0 = time.time()
    got = frobenius_polynomials(f, ps, nthreads=4)
    assert time.time() - t0 < 30
    for p in ps:
        assert got[p] == _sage_frobenius(f, p), p


def test_even_degree_without_one_is_correct_if_slow():
    r"""
    `4f + h^2` here has no rational Weierstrass point, so it goes through the
    even degree routine: linear in `p`, and not able to reach the same speed,
    but it must still be right.
    """
    f = 4 * (X ** 5 - X + 1) + (X ** 3 + X + 1) ** 2
    assert odd_model(f) is None
    ps = [p for p in prime_range(7, 90) if f.discriminant() % p]
    got = frobenius_polynomials(f, ps, nthreads=4)
    for p in ps:
        assert got[p] == _sage_frobenius(f, p), p


def test_single_prime_convenience():
    assert frobenius_polynomial(F, 37) == _sage_frobenius(F, 37)


def test_curve_lfunction_matches_pari():
    r"""
    Coefficients and value both, against ``lfungenus2``.  The conductor,
    weight and gamma factors are the ones ``lfunparams`` reports.
    """
    C = HyperellipticCurveWithReduction(F)
    L = CurveLFunction(C)
    assert L.conductor() == 3233
    assert L.root_number() in (1, -1)
    ours = [ZZ(c) for c in L.dirichlet_coefficients(40)[1:]]
    pari.set_real_precision(20)
    theirs = [ZZ(c) for c in pari.lfunan(L._pari_L, 40)]
    assert ours == theirs
    params = list(pari.lfunparams(L._pari_L))
    assert ZZ(params[0]) == L.conductor() and ZZ(params[1]) == 2


def test_curve_lvalue_matches_pari():
    r"""
    Against PARI *at the same precision*.

    ``pari.set_real_precision`` does not reach ``lfun`` -- it answers at 64
    bits whatever that is set to -- so the comparison has to pass the
    precision to the call, on both sides, or it is comparing a 64-bit value
    with whatever we asked for.  How many digits either side is worth is a
    separate matter: ``lfungenus2``'s object for this curve satisfies its own
    functional equation only to `2^{-7}` (``lfuncheckfeq`` says so for PARI's
    object as much as for ours), so a few digits is all there is here.
    """
    C = HyperellipticCurveWithReduction(F)
    L = CurveLFunction(C)
    # the Dirichlet coefficients are the part that is ours, and they agree
    # with PARI's exactly
    ours = L.dirichlet_coefficients(40)
    theirs = pari.lfunan(L._pari_L, 40)
    assert [QQ(ours[n]) for n in range(1, 41)] == [QQ(v) for v in theirs]
    # the values agree only as well as the underlying object allows: PARI's
    # own lfungenus2 object for this curve satisfies its functional equation
    # to 2^-7, and ours -- built from the same coefficients and the same
    # conductor -- to the same, so a few digits is what either is worth
    bits = 128
    K = _real_field(30)
    feq = K(2) ** K(pari.lfuncheckfeq(L._pari_L, precision=bits))
    assert feq > 1e-4, feq            # PARI's own object, not ours, says so
    v_ours = L.value(2, 0, digits=int(bits * 0.301))
    v_pari = pari.lfun(L._pari_L, 2, precision=bits)
    assert abs(RR(v_ours) - RR(v_pari)) < 1e-3


def test_arbitrary_precision_really_is():
    r"""
    Asking for more digits must *deliver* more digits.

    Two things have to be right for that, and each was wrong once: the value
    must come back in a field wide enough to hold them (an ``RR`` conversion
    capped it at 53 bits), and the PARI call must be told the precision --
    ``pari.set_real_precision`` does not reach ``lfun`` or ``lfuncheckfeq``,
    which answered at 64 bits however much was asked for.  The second showed
    up as a functional equation stuck at `3.6\cdot10^{-15}` at every
    precision; the certificate now improves as it should, and it is the
    certificate that says the digits are real.
    """
    # LMFDB 249.a.249.1, whose bad primes 3 and 83 are both odd, so the
    # cluster pictures give its standard L-function outright
    R = PolynomialRing(ZZ, "x")
    x = R.gen()
    L = StandardLFunction(
        HyperellipticCurveWithReduction((x ** 3 + 1) ** 2 + 4 * (x ** 2 + x)))
    lo, hi = 15, 30
    nlo, nhi = L._coeffs_needed(lo), L._coeffs_needed(hi)
    vlo, vhi = L.value(1, 2, nlo, lo), L.value(1, 2, nhi, hi)
    assert vlo.parent().precision() < vhi.parent().precision()
    assert vhi.parent().precision() >= 100
    flo = L.check_functional_equation(nlo, lo)
    fhi = L.check_functional_equation(nhi, hi)
    assert flo < 1e-20, flo              # not the 3.6e-15 of the fixed value
    assert fhi < flo, (flo, fhi)         # and it improves with precision
    assert abs(RR(vlo) - RR(vhi)) < 1e-14


def test_real_field_is_wide_enough():
    for d in (10, 30, 60):
        assert _real_field(d).precision() >= int(d * 3.32)


@pytest.mark.parametrize("workers", [2, 4, 8, None])
@pytest.mark.parametrize("f", [F, 4 * (X**5 - X + 1) + (X**3 + X + 1)**2])
def test_native_parallel_matrices_match_serial_and_sage(workers, f):
    """Exercise the binary directly so fallback cannot mask native failures."""
    binary = euler_binary()
    if binary is None:
        pytest.skip("Frobenius accelerator not built")
    from hyperell_regulator.frobenius import _lift
    ps = [int(p) for p in prime_range(31, 50) if f.discriminant() % p]
    batch = " ".join(map(str, f.list())) + "\n"
    batch += "".join("%d 2\n" % p for p in ps)
    def run(n):
        command = [binary] if n is None else [binary, str(n)]
        result = subprocess.run(command, input=batch, text=True,
                                capture_output=True, check=True, timeout=30)
        rows = [list(map(int, line.split())) for line in result.stdout.splitlines()]
        assert len(rows) == len(ps)
        for p, row in zip(ps, rows):
            assert row[:3] == [p, 1, 4]
            assert len(row) == 19
            matrix = [row[3 + 4*i:7 + 4*i] for i in range(4)]
            assert _lift(matrix, p, 2, 2) == _sage_frobenius(f, p)
        return rows
    serial = run(1)
    for _ in range(3):
        assert run(workers) == serial
