r"""
The four things this package computes, each by two independent routes.

`F` is a polynomial over `\QQ` and a function on `C : y^2 = F(x)` is a pair
``phi = (A, B)`` of rational functions of `x`, standing for
`\varphi = A(x) + B(x)\,y`; :func:`~.functions.function_with_divisor` returns
one in that form.

The two routes are the plane integrator --- adaptive quadrature over
`\mathbf{C}` --- and Stokes' theorem on the sheets of `\varphi`.  They are tied
together by

.. math:: \operatorname{Im}\int_C \rho\,\omega_a \wedge \bar\omega_b
          = -4 \int_{\mathbf{C}} \rho\,\frac{G_{ab}}{|F|}\,dA ,
          \qquad G_{ab} = \operatorname{Re}(x^a \bar x^{\,b}),

with `\rho = 1` for the periods and `\rho = \log|\varphi|` for the regulator.
The second needs `\log|\varphi|` as a function of `x` alone, which it is after
pushing down: `\varphi\,\iota(\varphi) = \operatorname{N}(\varphi) = A^2 - B^2F`
and `\iota` fixes `\omega_a\wedge\bar\omega_b`, so

.. math:: \int_C \log|\varphi|\,\omega_a\wedge\bar\omega_b
          = \tfrac12 \int_C \log|\operatorname{N}(\varphi)|\,
            \omega_a\wedge\bar\omega_b ,

and the right side is a combination of the plane integrator's `\log|x-r|`
weights over the roots of the norm.
"""

import numpy as np

from . import plane
from .cover import Cover
from .stokes import _assemble, _assemble_both

__all__ = [
    "periods_plane",
    "periods_stokes",
    "periods_and_regulator_stokes",
    "regulator_plane",
    "regulator_stokes",
]


def _vector(H):
    r"""The `g(g+1)/2` independent entries of `-\operatorname{Im}H/4`."""
    g = H.shape[0]
    S = (H.imag + H.imag.T) / 2 / -4.0
    return np.array([S[a, b] for a in range(g) for b in range(a, g)])


def periods_plane(F, **kwds):
    r"""
    `\bigl(\int_{\mathbf{C}} G_{ab}/|F|\,dA\bigr)_{a\le b}` by quadrature over
    the plane, in the order `(1,1), (1,2), (2,2)` for genus two.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import periods_plane
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: v = periods_plane(x^5 + x^2 + 2*x + 1)     # long time
        sage: bool(v[0] > 0 and v[2] > 0)                # long time
        True
    """
    return np.asarray(plane.regulator_vector_2(F, **kwds)[0], float)


def periods_stokes(F, phi, prec=None, check_tol=1e-6, verbose=False):
    r"""
    The same vector by Stokes' theorem on the sheets of `\varphi`.

    `\varphi` enters only through the cut system: any function with
    `\operatorname{div}(\varphi) = N P - N Q` gives the same answer.
    ``prec`` is the number of decimal digits to work to; ``None`` is float64.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import periods_stokes
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: import numpy as np
        sage: x = ring().gen()
        sage: v = periods_stokes(x^5 + x^2 + 2*x + 1, (x + 1, 1))  # long time
        sage: bool(abs(v - np.array([6.05371274, -1.63861910, 8.17585620])).max() < 1e-6)  # long time
        True
    """
    if prec is not None:
        from .precision import periods_stokes_mp
        return periods_stokes_mp(F, phi, prec, check_tol=check_tol)
    cov = Cover(F, *phi)
    H, _, _ = _assemble(cov, False, False, check_tol, verbose)
    return _vector(H)


def c_invariants(F, prec=None):
    r"""
    `(c^+, c^-)`: the pivoted minors of the real and imaginary parts of the
    curve's period matrix.

    ``prec`` is the number of decimal digits; ``None`` is the float64
    quadrature of :mod:`~hyperell_regulator.regulator.plane`, whose default
    tolerances are worth about ten digits -- enough to cap a comparison that
    everything else carries fifteen digits into.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import c_invariants
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: cp, cm = c_invariants(x^5 + x^2 + 2*x + 1, prec=15)
        sage: cp0, cm0 = c_invariants(x^5 + x^2 + 2*x + 1)
        sage: bool(abs(float(cp) - cp0) < 1e-9 and abs(float(cm) - cm0) < 1e-9)
        True
    """
    if prec is None:
        return float(plane.cplus(F)), float(plane.cminus(F))
    from .precision import c_invariants_mp
    return c_invariants_mp(F, prec)


def regulator_plane(F, phi, **kwds):
    r"""
    `\bigl(\int_{\mathbf{C}} \log|\varphi|\,G_{ab}/|F|\,dA\bigr)_{a\le b}`, via
    the norm.

    `\operatorname{N}(\varphi) = A^2 - B^2F` is a rational function of `x`, so
    `\log|\varphi| = \tfrac12\log|\operatorname{N}(\varphi)|` integrates as a
    combination of the plane integrator's `\log|x-r|` weights over its zeros
    and poles.  Those come from :meth:`~.cover.Cover.norm_factors`, which
    factors over `\QQ`, and the multiplicities matter: the zero at `x_P` is
    `N`-fold, `\operatorname{N}(\varphi)` being `c\,(x - x_P)^N`, and taking
    the seven of them on LMFDB 249.a.249.1 apart as seven distinct roots --
    which is what a numerical root finder hands over, spread across
    `\varepsilon^{1/7}`, a hundredth -- costs seven quadratures in place of
    one *and* moves the answer by `5\cdot10^{-4}`, a thousand times the error
    the quadrature reports for itself.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import regulator_plane
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: v = regulator_plane(x^5 + x^2 + 2*x + 1, (x + 1, 1))     # long time
        sage: bool(v[0] < 0)                                           # long time
        True
    """
    cov = Cover(F, *phi)
    out = np.zeros(3 if cov.g == 2 else cov.g * (cov.g + 1) // 2)
    lead = cov.norm_num[0] / cov.norm_den[0]
    out = out + 0.5 * np.log(abs(lead)) * periods_plane(F, **kwds)
    zeros, poles = cov.norm_factors()
    for terms, sign in ((zeros, +1.0), (poles, -1.0)):
        for r, m in terms:
            out = out + 0.5 * sign * m * np.asarray(
                plane.regulator_vector(F, a=r, **kwds)[0], float)
    return out


def regulator_stokes(F, phi, fast=True, prec=None, check_tol=1e-6,
                     verbose=False):
    r"""
    The same vector by Stokes' theorem.

    With ``fast`` the second moments of the edges are not integrated.  They do
    not vanish, but in the symmetrised imaginary part -- the only part
    independent of the branch of `\log\varphi` -- they collapse to a closed
    form in the edge periods, because

    .. math:: \int_e A_a\,\bar\omega_b + \overline{\int_e A_b\,\bar\omega_a}
              = \int_e d(A_a\bar A_b) = u^{(a)}_e\,\overline{u^{(b)}_e},

    `A_a` being the running primitive, which vanishes at the start of the edge.
    The slow version integrates them instead, and must agree.

    ``prec`` is the number of decimal digits to work to; it needs ``fast``.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import regulator_stokes
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: import numpy as np
        sage: x = ring().gen()
        sage: v = regulator_stokes(x^5 + x^2 + 2*x + 1, (x + 1, 1))  # long time
        sage: bool(abs(v - np.array([-3.25424741, 1.26900145, 14.42730926])).max() < 1e-6)  # long time
        True
    """
    if prec is not None:
        if not fast:
            raise ValueError("arbitrary precision is only for the fast version")
        from .precision import regulator_stokes_mp
        return regulator_stokes_mp(F, phi, prec, check_tol=check_tol)
    cov = Cover(F, *phi)
    H, u, dlam = _assemble(cov, True, not fast, check_tol, verbose)
    return _regulator_vector(H, u, dlam, fast)


def _regulator_vector(H, u, dlam, fast):
    """Include the closed form for edge moments when they were not integrated."""
    if fast:
        M = 0.5 * dlam.imag * np.real(np.einsum("aim,bim->ab", u, np.conj(u)))
        g = H.shape[0]
        S = ((H.imag + H.imag.T) / 2 + M) / -4.0
        return np.array([S[a, b] for a in range(g) for b in range(a, g)])
    return _vector(H)


def periods_and_regulator_stokes(F, phi, fast=True, prec=None, check_tol=1e-6,
                                 verbose=False):
    r"""
    Return ``(periods, regulator)`` for one cover of the curve.

    Each cut edge is integrated once: its ordinary periods give
    the period vector, and its logarithmic periods give the regulator.  Both
    sheet-boundary closure checks are applied, with the same tolerances as
    the separate calls.  ``fast=False`` integrates the second moments.

    ``prec`` selects the same algorithm at arbitrary precision. Both vectors
    share the edge integrals and precision-dependent convergence checks.
    Supply exact rational coefficients. Results are Sage real numbers at
    ``prec`` decimal digits; cut topology and initial labels use float64.

    EXAMPLES::

        sage: from hyperell_regulator.regulator import periods_and_regulator_stokes
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: periods, regulator = periods_and_regulator_stokes(
        ....:     x^5 + x^2 + x, (x, 0))
        sage: bool(abs(periods[0] - 9.22548668) < 1e-7)
        True
        sage: bool(abs(regulator[0] + 5.80825813) < 1e-7)
        True
    """
    if prec is not None:
        if not fast:
            raise ValueError("arbitrary precision is only for the fast version")
        from .precision import periods_and_regulator_stokes_mp
        return periods_and_regulator_stokes_mp(F, phi, prec, check_tol=check_tol)
    cov = Cover(F, *phi)
    H0, H, u, dlam = _assemble_both(cov, not fast, check_tol, verbose)
    return _vector(H0), _regulator_vector(H, u, dlam, fast)
