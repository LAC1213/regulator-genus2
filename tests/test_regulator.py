r"""Tests for the regulator package.

Two curves, chosen so that between them they exercise every chart the Stokes
integrator has:

* `y^2 = x^5 + (1+x)^2` with `\varphi = y+1+x`, degree 5, `P = (0,-1)` and
  `Q = \infty` both totally ramified.  `B \neq 0`, so `x` separates the fibre
  and the end legs use the `x`- and `\tau`-charts.
* `y^2 = x(x^4+x+1)` with `\varphi = x`, degree 2, `P = (0,0)`.  `B = 0`, so
  `\varphi` factors through `x`, the cover polynomial is a perfect square and
  every ramification point of `\varphi` is a Weierstrass point of the curve:
  the `y`-chart.
* `y^2 = x(x-1)(x^3+x+1)` with `\varphi = x/(x-1)`, degree 2, `P = (0,0)` and
  `Q = (1,0)` -- a *finite* `Q`, so the leg out to `w = \infty` closes on a
  finite Weierstrass point instead of on the point at infinity.  Infinity is
  then an ordinary critical value, `\varphi(\infty) = A(\infty) = 1`, which
  the `\tau`-chart must handle at finite `w` as well.
* `y^2 = -4x^5+13x^4-18x^3+15x^2-6x+1` with
  `\varphi = \tfrac12(x^3-x^2-x) + \tfrac12(2-x)y`, degree 7, from
  `P = (1,1)` to `Q = \infty`: LMFDB 249.a.249.1, and the first cover here
  taken from a curve rather than written down.  The model is not monic, `B`
  has a zero -- so the fibre pinches in `x` over `w = A(2) = 1` although
  `\varphi` is unramified there -- and `P` is a sevenfold point, which
  ``np.roots`` cannot place to better than a hundredth.
* `y^2 = u^6 + 4u^5 + 4u^4 + 2u^3 + 1` with `\varphi` of degree 7 from
  `P = (-1, 0)` to one of the two points at infinity: the same curve as
  above in an *even* degree model, where infinity is two points rather than
  one Weierstrass point, `\tau = 1/x` rather than `x^{-1/2}`, and
  `\varphi` has a pole of order `N` at one of them and none at the other --
  its regular value there being a critical value with two sheets running
  into it.
* `y^2 = x^5 - 5x^3 + 4x` with `\varphi = x`, and
  `y^2 = x(x-1)(x^3+3x^2-4x+1)` with `\varphi = x/(x-1)`: both have all their
  critical values real, with `\varphi(P) = 0` in the middle of them, so no
  path of straight segments through them is embedded and the cut system has
  to route the closing step round a detour.  The second is finite-`Q` as
  well.

What is checked: the Stokes method against the plane integrator, the fast and
slow forms against each other, the arbitrary-precision path against itself at
two precisions, the structural identities the construction rests on, and the
helpers.
"""

import numpy as np

from hyperell_regulator.regulator import (
    function_with_divisor,
    periods_and_regulator_stokes,
    periods_stokes,
    plot_cut_system,
    regulator_stokes,
    torsion_order,
)
from hyperell_regulator.regulator.cover import Cover, check_path_embedded
from hyperell_regulator.regulator.polynomials import ring
from hyperell_regulator.regulator.stokes import (
    boundary_walks,
    cut_data,
    edge_data,
    gluing_permutations,
)

x = ring().gen()

DEG5 = (x**5 + (1 + x)**2, (x + 1, 1))                    # phi = y + 1 + x
DEG2 = (x**5 + x**2 + x, (x, 0))                          # phi = x
DEG2Q = (x**5 - x**4 + x**3 - x, (x / (x - 1), 0))        # phi = x/(x-1)
COLL = (x**5 - 5 * x**3 + 4 * x, (x, 0))                  # phi = x, all real
COLLQ = (x * (x - 1) * (x**3 + 3 * x**2 - 4 * x + 1),     # all real, Q finite
         (x / (x - 1), 0))
# LMFDB 249.a.249.1: not monic, B has a zero, N = 7
PINCH = (-4 * x**5 + 13 * x**4 - 18 * x**3 + 15 * x**2 - 6 * x + 1,
         ((x**3 - x**2 - x) / 2, (2 - x) / 2))
# the same curve in an even degree model, N = 7, infinity two points
EVEN = x**6 + 4 * x**5 + 4 * x**4 + 2 * x**3 + 1
CASES = (("deg5", DEG5), ("deg2", DEG2), ("deg2q", DEG2Q),
         ("coll", COLL), ("collq", COLLQ), ("pinch", PINCH))
INTEGRATION_CASES = (("deg5", DEG5), ("collq", COLLQ), ("pinch", PINCH))
STRUCTURAL_CASES = (("deg5", DEG5), ("deg2q", DEG2Q),
                    ("collq", COLLQ), ("pinch", PINCH))

# plane integrator values; minutes each, and the ground truth rather than the
# thing under test
PLANE = {
    "deg5": {"periods": [6.05371274, -1.63861909, 8.17585620],
             "regulator": [-3.25424741, 1.26900145, 14.42730926]},
    "deg2": {"periods": [9.22548668, -1.22108503, 8.98356248],
             "regulator": [-5.80825813, 0.45209334, 5.53626738]},
    "deg2q": {"periods": [8.82535356, 1.33719274, 9.05273869],
              "regulator": [-4.85618820, 3.06804338, 1.04959911]},
    "coll": {"periods": [3.35349283, 0.0, 6.70698566],
             "regulator": [-0.89959203, 0.0, 6.44811227]},
    # the plane quadrature reports a tolerance warning on this one, so it is
    # the less accurate side of the comparison
    "collq": {"periods": [14.39442346, 6.88772457, 9.29434140],
              "regulator": [0.31594979, 5.96962153, 5.50891600]},
    "pinch": {"periods": [5.17025230750167, 2.72707611005494, 4.28626293491068],
              "regulator": [-7.08144563, -5.33101930, 0.24694098]},
}


def _rel(got, want):
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_stokes_matches_plane():
    for key, (F, phi) in INTEGRATION_CASES:
        periods, regulator = periods_and_regulator_stokes(F, phi)
        assert _rel(periods, PLANE[key]["periods"]) < 1e-6, key
        assert _rel(regulator, PLANE[key]["regulator"]) < 1e-6, key


def test_fast_and_slow_regulator_agree():
    r"""
    The fast form takes the edges' second moments in closed form rather than
    integrating them.  They do not vanish -- on the diagonal the correction is
    `\tfrac12\operatorname{Im}(\Delta\lambda)\sum_e|u_e|^2`, a sum of positive
    terms -- so this is a real check, not a tautology.
    """
    F, phi = COLLQ
    fast = regulator_stokes(F, phi, fast=True)
    slow = regulator_stokes(F, phi, fast=False)
    # The slow form carries four more states through every leg.
    assert _rel(fast, slow) < 1e-6
def test_arbitrary_precision_converges():
    r"""
    The float64 pass fixes only the discrete data; raising the precision must
    then buy digits.  Twenty and thirty-four must agree to twenty.

    This includes detoured cuts and finite poles.
    """
    from hyperell_regulator.regulator import periods_and_regulator_stokes, IntegrationMonitor
    key, (F, phi) = "pinch", PINCH
    computed = []
    for digits in (20, 34):
        with IntegrationMonitor(max_edge_seconds=45, max_edge_steps=100000):
            computed.append(periods_and_regulator_stokes(F, phi, prec=digits))
    for lo, hi, what in zip(*computed, ("periods", "regulator")):
        for a, b in zip(lo, hi):
            assert abs(b.parent()(a) - b) < 1e-19 * max(abs(b), 1), (key, what)
        assert _rel([float(v) for v in lo], PLANE[key][what]) < 1e-7


def test_full_preimages_have_zero_period():
    r"""
    `\int_{\varphi^{-1}(\gamma)}\omega = \int_\gamma \varphi_*\omega = 0`,
    since `\varphi_*\omega \in H^0(\mathbf{P}^1,\Omega^1) = 0`.  This is the
    identity that kills any formula built from uniformly oriented preimages,
    and it is also a sharp check on every edge integral.
    """
    F, phi = PINCH
    cov = Cover(F, *phi)
    z, kinds, d = cov.cut_path()
    _, _, lam, _ = cut_data(cov, z, d)
    for log in (False, True):
        u, ell, _ = edge_data(cov, z, d, kinds, lam, log, False)
        for arr in (u, ell):
            s = np.abs(arr.sum(axis=2)).max()
            assert s < 1e-8 * np.abs(arr).max(), (log, s)


def test_even_degree_model():
    r"""
    Degree `2g+2`: infinity is two points, `\tau = 1/x` is the parameter at
    each, and `\varphi` tells them apart -- a pole of order `N` at the one it
    was built to have its pole at, and a finite value at the other, which is
    then a critical value like any other with two sheets running into it.

    The checks below verify how the two points at infinity are represented in
    the even-degree model.
    """
    from hyperell_regulator.regulator import function_with_divisor

    phi = function_with_divisor(EVEN, (-1, 0), "infinity", 7)
    cov = Cover(EVEN, *phi)
    assert cov.even and (cov.g, cov.N) == (2, 7)
    # one pole of order N, and a finite value at the other point
    orders = sorted(cov.pole_order_at_infinity(s) for s in (1, -1))
    assert orders == [0, 7], orders
    at = [cov.value_at_infinity(s) for s in (1, -1)]
    finite = [v for v in at if v is not None]
    assert len(finite) == 1 and abs(finite[0] + 1) < 1e-9, at
    # and that value is a critical value, with a fibre point at infinity
    assert min(abs(v - finite[0]) for v in cov.critical_values()) < 1e-9
    # two sheets run into it, and they come back as huge roots rather than
    # as roots at infinity: the leading coefficient of the cover polynomial
    # vanishes there in exact arithmetic but not in float64
    x, _ = cov.fibre(finite[0])
    assert np.count_nonzero(~np.isfinite(x) | (np.abs(x) > 1e6)) == 2

def test_a_zero_of_B_is_not_a_branch_value():
    r"""
    Over `w_0 = A(x_0)` with `B(x_0) = 0` the two points `(x_0, \pm y_0)` have
    the same image, so the cover polynomial has a double root in `x` there and
    `\operatorname{Res}_x(\mathcal{P},\mathcal{P}_x)` vanishes -- but
    `\varphi` is unramified over `w_0` and it is not a branch value.  Taking
    it for one puts a node of the cut system, and a chart leg, at a place
    where nothing comes together; `y = (w-A)/B` is `0/0` there as well.
    """
    cov = Cover(PINCH[0], *PINCH[1])
    assert abs(cov.B(2.0)) < 1e-14 and abs(cov.A(2.0) - 1) < 1e-14
    assert min(abs(v - 1) for v in cov.critical_values()) > 1e-3
    assert min(abs(v - 1) for v in cov.critical_values_hp(120)) > 1e-3

    x, y = cov.fibre(1.0)
    assert np.abs(y ** 2 - np.polyval(cov.F, x)).max() < 1e-10
    at = np.abs(x - 2) < 1e-6
    assert at.sum() == 2 and abs(y[at].sum()) < 1e-8      # the two are +-y_0


def test_the_N_fold_point_is_placed_to_the_last_digit():
    r"""
    `x_P` is an `N`-fold root of the cover polynomial, which ``np.roots``
    places only to `\varepsilon^{1/N}`: a hundredth at `N = 7`.  A leg aimed
    there would stop that far short, and since the sheets leave `P` turned by
    `\zeta_N` from one to the next, so is the piece each one drops -- which
    is why the sheet boundaries then fail to close by that much rather than
    the error cancelling.
    """
    from hyperell_regulator.regulator._legs import _target

    cov = Cover(PINCH[0], *PINCH[1])
    x, y = cov.fibre(1e-6)                 # just off phi(P) = 0
    spread = np.abs(x - x.mean()).max()
    assert spread > 1e-3, spread           # np.roots really is that bad here
    for i in range(cov.N):
        xt, yt = _target(cov, x[i], y[i], 0.0, "P")
        assert abs(xt - 1) < 1e-12 and abs(yt - 1) < 1e-12


def test_monodromy_matches_the_ramification():
    r"""
    `\tau_1` and `\tau_r` are the monodromies at the totally ramified `P` and
    `Q`, so they are `N`-cycles; every other critical value is a simple branch
    point, every detour corner carries no ramification at all, and
    Riemann-Hurwitz must come out.
    """
    for key, (F, phi) in STRUCTURAL_CASES:
        cov = Cover(F, *phi)
        z, kinds, d = cov.cut_path()
        up, lo, _, _ = cut_data(cov, z, d)
        tau = gluing_permutations(up, lo)
        assert _cycle_type(tau[0]) == [cov.N], key
        assert _cycle_type(tau[-1]) == [cov.N], key
        total = 2 * (cov.N - 1)
        for i in range(len(tau) - 1):
            loc = [tau[i + 1][_inv(tau[i])[k]] for k in range(cov.N)]
            if kinds[i + 1] == "none":
                assert _cycle_type(loc) == [1] * cov.N, (key, i)
                continue
            assert _cycle_type(loc) == [2] + [1] * (cov.N - 2), (key, i, loc)
            total += 1
        assert total == 2 * cov.g - 2 + 2 * cov.N, key

def test_log_jump_is_a_uniform_two_pi_i():
    r"""
    `\Gamma` joins `\varphi(P) = 0` to `\infty`, so it is already a cut for
    `\log\varphi`, and the jump is locally constant on it.
    """
    for _, (F, phi) in STRUCTURAL_CASES:
        cov = Cover(F, *phi)
        z, _, d = cov.cut_path()
        _, _, _, dlam = cut_data(cov, z, d)
        assert abs(abs(dlam) - 2 * np.pi) < 1e-9
        assert abs(dlam.real) < 1e-9


def test_boundary_walks_close():
    for _, (F, phi) in STRUCTURAL_CASES:
        cov = Cover(F, *phi)
        z, _, d = cov.cut_path()
        W = boundary_walks(*cut_data(cov, z, d)[:2])
        assert len(W) == cov.N and all(len(w) == 2 * len(z) for w in W)
        fwd = sorted((i, m) for w in W for (i, m, s) in w if s > 0)
        bwd = sorted((i, m) for w in W for (i, m, s) in w if s < 0)
        assert fwd == bwd and len(set(fwd)) == len(z) * cov.N


def test_cover_shape():
    c5, c2, cq = (Cover(DEG5[0], *DEG5[1]), Cover(DEG2[0], *DEG2[1]),
                  Cover(DEG2Q[0], *DEG2Q[1]))
    assert (c5.N, c5.g, c5.pure_x) == (5, 2, False)
    assert (c2.N, c2.g, c2.pure_x) == (2, 2, True)
    assert (cq.N, cq.g, cq.pure_x) == (2, 2, True)
    assert c5.pole_order_at_infinity() == 5
    assert c2.pole_order_at_infinity() == 2
    # Q is finite here, so phi has no pole at infinity at all
    assert cq.pole_order_at_infinity() == 0


def test_finite_Q_critical_values():
    r"""
    With `Q` finite, `\varphi(\infty) = A(\infty)` is an ordinary critical
    value and must be in the list, while the pole of `A` must not: evaluating
    `A` there gave a spurious value of order `10^{14}`.
    """
    cov = Cover(DEG2Q[0], *DEG2Q[1])
    v = np.sort_complex(cov.critical_values())
    assert len(v) == 5 and np.abs(v).max() < 10
    assert min(abs(v)) < 1e-12                       # phi(P) = 0
    assert min(abs(v - 1)) < 1e-12                   # A(infinity) = 1
    assert torsion_order(DEG2Q[0], (0, 0), (1, 0)) == 2


def test_straight_path_is_recognised_as_unusable():
    r"""
    The rotation that brings `\varphi(P)` to the front turns the closing edge
    of the cyclic order into a chord, and for collinear critical values that
    chord runs through the others.  The embeddedness check must see it.
    """
    check_path_embedded(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    try:
        check_path_embedded(np.array([0.0, 1.0, 2.0, -2.0, -1.0]))
    except NotImplementedError:
        return
    raise AssertionError("expected the embeddedness check to fire")


def test_detour_only_where_it_is_needed():
    r"""
    In general position the cut system is the straight one, with the radial
    ray; collinear critical values get the closing step routed round, and the
    ray then has to leave in some other direction.
    """
    nodes, kinds, d = Cover(DEG5[0], *DEG5[1]).cut_path()
    assert len(nodes) == 5 and "none" not in kinds
    for F, phi in (COLL, COLLQ):
        cov = Cover(F, *phi)
        nodes, kinds, d = cov.cut_path()
        crit = [n for n, k in zip(nodes, kinds) if k != "none"]
        assert len(crit) == 5 and kinds.count("none") > 0
        assert abs(crit[0]) < 1e-12                  # phi(P) still comes first
        check_path_embedded(nodes, d, np.array(crit))


def test_function_with_divisor():
    r"""
    The helper works in `\QQ(x)` and hands back `(A, B)` there, exactly --
    not a float coefficient list that the caller has to lift back.
    """
    assert function_with_divisor(DEG5[0], (0, -1), "infinity", 5) == (x + 1, 1)
    assert function_with_divisor(DEG2[0], (0, 0), "infinity", 2) == (x, 0)
    assert torsion_order(DEG5[0], (0, -1), "infinity") == 5
    try:
        function_with_divisor(DEG5[0], (0, -1), "infinity", 3)
    except ValueError:
        return
    raise AssertionError("expected N = 3 to be refused")


def test_coefficient_lists_still_work():
    r"""
    The old way in -- coefficients in descending degree, and a
    ``(numerator, denominator)`` pair for a rational function -- must still
    reach the same cover, since that is also how the *inexact* data of
    :meth:`~hyperell_regulator.regulator.cover.Cover.shifted` gets in.
    """
    old = Cover([1, -1, 1, 0, -1, 0], ([1, 0], [1, -1]), 0)
    new = Cover(*((DEG2Q[0],) + DEG2Q[1]))
    assert (old.N, old.g, old.pure_x) == (new.N, new.g, new.pure_x)
    assert np.abs(old.norm_num - new.norm_num).max() < 1e-12


def test_plot_cut_system_writes_a_file():
    import os
    import tempfile
    d = tempfile.mkdtemp()
    p = plot_cut_system(*DEG5, path=os.path.join(d, "cut.png"))
    assert os.path.getsize(p) > 5000


def _cycle_type(perm):
    seen, out = set(), []
    for s in range(len(perm)):
        if s in seen:
            continue
        n, t = 0, s
        while t not in seen:
            seen.add(t)
            n, t = n + 1, perm[t]
        out.append(n)
    return sorted(out, reverse=True)


def _inv(p):
    out = [0] * len(p)
    for i, v in enumerate(p):
        out[v] = i
    return out


def test_precision_supports_detoured_cuts():
    from hyperell_regulator.regulator import periods_and_regulator_stokes, IntegrationMonitor
    key, (F, phi) = "collq", COLLQ
    with IntegrationMonitor(max_edge_seconds=30, max_edge_steps=100000):
        per, reg = periods_and_regulator_stokes(F, phi, prec=20)
    assert _rel(per, PLANE[key]["periods"]) < 1e-7
    assert _rel(reg, PLANE[key]["regulator"]) < 1e-7


def test_walk_crosses_unramified_weierstrass_point():
    from hyperell_regulator.regulator.walks import Chart, walker

    phi = function_with_divisor(EVEN, (0, -1), "infinity", 14)
    cov = Cover(EVEN, *phi)
    xs, ys = cov.fibre(-2.1)
    i = np.argmin(abs(xs + 1))
    assert ys[i].real > 0
    # phi(-1,0)=-2 is unramified.  Along this path x turns around at
    # -1, while y crosses zero; nearest-square-root tracking stays behind.
    for kind in ("x", "inv"):
        for sigma in (False, True):
            chart = Chart(cov, kind, 0, sigma=sigma)
            coord = (lambda t: 1 / (-2.1 + 0.2*t)) if sigma else (
                lambda t: -2.1 + 0.2*t)
            wk = walker(cov, coord, xs[i], ys[i], chart)
            wk.to(1.0)
            xx, yy = wk.xy()
            assert yy.real < 0
            assert abs(cov.phi_stable(xx, yy) + 1.9) < 1e-10
            assert abs(yy*yy - np.polyval(cov.F, xx)) < 1e-12


def test_derivative_survives_cancellation():
    from sage.all import RealField

    phi = function_with_divisor(EVEN, (0, -1), "infinity", 14)
    cov = Cover(EVEN, *phi)
    RR = RealField(160)
    xx = RR(-14)
    yy = EVEN(xx).sqrt()
    A, B = phi
    expected = yy * (A.derivative()(xx) + B.derivative()(xx)*yy)
    expected += B(xx) * EVEN.derivative()(xx) / 2
    got = cov.y_dphi_dx(float(xx), float(yy))
    assert abs(got / float(expected) - 1) < 1e-11
    # The same product is finite where dphi/dx itself has a pole.
    assert abs(cov.y_dphi_dx(-1.0, 0.0)
               - float(B(-1)*EVEN.derivative()(-1)/2)) < 1e-12


def test_shift_preserves_simplified_cover_identities():
    phi = function_with_divisor(EVEN, (0, -1), "infinity", 14)
    cov = Cover(EVEN, *phi)
    center = -110.0 + 3j
    shifted = cov.shifted(center)
    assert shifted.N == cov.N == 14
    assert len(shifted.norm_num) == len(cov.norm_num)
    for u in (0.0, 0.01, 1j):
        assert abs(shifted.dlog_norm(u) - 14/(center+u)) < 1e-14
        assert abs(shifted.norm(u) / cov.norm(center+u) - 1) < 1e-13


def test_tanh_sinh_nodes_are_strictly_interior():
    from hyperell_regulator.regulator._legs import _tanh_sinh

    for level in (5, 6, 7):
        nodes, weights = _tanh_sinh(level)
        assert np.all((nodes > 0) & (nodes < 1))
        assert np.all(np.isfinite(-np.log1p(-nodes)))
        assert abs(weights.sum() - 1) < 1e-14


def test_finite_pole_chart_retains_small_displacements():
    from hyperell_regulator.regulator.walks import charts_over

    cov = Cover(DEG2Q[0], *DEG2Q[1])
    chart = charts_over(cov, None, sigma=True)[0]
    assert chart.kind == "x" and chart.c == 1
    sigma = 1e-30
    u = sigma / (1 - sigma)
    eta = np.sqrt(np.polyval(chart.Fu, u))
    num, den = chart.integrand(u, eta, sigma)
    # For s=sqrt(sigma), dx/y -> 2 ds/sqrt(F'(1)).  Although 1+u
    # rounds to 1, both holomorphic differentials retain this limit.
    expected = 2 / np.sqrt(float(DEG2Q[0].derivative()(1)))
    assert np.max(abs(num * (2*np.sqrt(sigma)/den) - expected)) < 1e-12


def test_chart_at_known_infinity_is_tried_first():
    from hyperell_regulator.regulator.walks import charts_over

    F = x**6 - 2*x**5 - x**4 + 4*x**3 + 3*x**2 + 2*x + 1
    cov = Cover(F, -x**3 + x**2 + x + 1, 1)
    assert any(q["infinite"] for q in cov.places_over(2))
    assert charts_over(cov, 2)[0].kind == "inv"
