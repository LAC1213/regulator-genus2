r"""
Periods and Beilinson regulators for `y^2 = F(x)`, each by two routes.

`F` is a polynomial over `\QQ`, and a function on the curve is a pair
``phi = (A, B)`` of rational functions of `x` standing for
`\varphi = A(x) + B(x)\,y` -- Sage objects throughout; see
:mod:`~hyperell_regulator.regulator.polynomials`, which also accepts
coefficient lists for the callers that predate this.  The regulator is taken
with respect to a `\varphi` whose divisor is `N P - N Q`.

The two routes are:

* the **plane integrator**, :mod:`~hyperell_regulator.regulator.plane` --
  adaptive quadrature over `\mathbf{C}` with a smooth partition of unity around
  the clusters of branch points.  No hypothesis on the branch points, float64
  accuracy, minutes per integral.  It is the reference.

* the **Stokes method**, :mod:`~hyperell_regulator.regulator.stokes` -- cut the
  sphere along the path `\Gamma` joining `\varphi(P) = 0` through the critical
  values of `\varphi` to `\varphi(Q) = \infty`, lift it to the `N` sheets of
  the cover, and apply Stokes' theorem on each.  One term per edge of the cut
  graph, with adaptive continuation and optional arbitrary precision.

They agree through

.. math:: \operatorname{Im}\int_C \rho\,\omega_a \wedge \bar\omega_b
          = -4 \int_{\mathbf{C}} \rho\,\frac{G_{ab}}{|F|}\,dA ,
          \qquad G_{ab} = \operatorname{Re}(x^a \bar x^{\,b}),

with `\rho = 1` giving the periods and `\rho = \log|\varphi|` the regulator.

What this package exports:

* :func:`periods_plane`, :func:`periods_stokes` -- the period vector
  `\bigl(\int G_{ab}/|F|\,dA\bigr)_{a\le b}`;
* :func:`periods_and_regulator_stokes` -- both vectors from one integration
  of the cut edges, in float64 or arbitrary precision;
* :func:`regulator_plane`, :func:`regulator_stokes` -- the same weighted by
  `\log|\varphi|`.  The Stokes version comes in a fast and a slow form,
  differing in whether the edges' second moments are integrated or taken in
  closed form;
* :func:`plot_cut_system` -- the critical values of `\varphi`, the cut system,
  and the branch of `\log\varphi`, which `\Gamma` also provides;
* :func:`function_with_divisor` -- `\varphi` from `P`, `Q` and `N`, via the
  Riemann-Roch space `L(NQ - NP)`; and :func:`torsion_order`, the least `N`
  for which that space is nonzero;
* :func:`c_invariants` -- `(c^+, c^-)`, the pivoted minors of the real and
  imaginary parts of the period matrix.

EXAMPLES::

    sage: from hyperell_regulator.regulator import function_with_divisor
    sage: from hyperell_regulator.regulator import periods_stokes, regulator_stokes
    sage: from hyperell_regulator.regulator.polynomials import ring
    sage: import numpy as np
    sage: x = ring().gen()
    sage: F = x^5 + (1 + x)^2                             # y^2 = x^5 + (1+x)^2
    sage: phi = function_with_divisor(F, (0, -1), "infinity", 5); phi
    (x + 1, 1)
    sage: bool(abs(periods_stokes(  # long time
    ....:          F, phi) - np.array([6.05371274, -1.63861910, 8.17585620])).max() < 1e-6)
    True
    sage: bool(abs(regulator_stokes(  # long time
    ....:          F, phi) - np.array([-3.25424741, 1.26900145, 14.42730926])).max() < 1e-6)
    True
"""

from .api import (
    c_invariants,
    periods_plane,
    periods_stokes,
    periods_and_regulator_stokes,
    regulator_plane,
    regulator_stokes,
)
from .diagnostics import IntegrationLimitError, IntegrationMonitor
from .functions import function_with_divisor, torsion_order
from .plots import plot_cut_system

__all__ = [
    "IntegrationMonitor",
    "IntegrationLimitError",
    "c_invariants",
    "periods_plane",
    "periods_stokes",
    "periods_and_regulator_stokes",
    "regulator_plane",
    "regulator_stokes",
    "plot_cut_system",
    "function_with_divisor",
    "torsion_order",
]
