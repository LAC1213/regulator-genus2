r"""
Beilinson regulator determinants from torsion points.

For each curve in ``same_component_results.json`` (written by
:mod:`hyperell_regulator.components`) this looks for a triple of
distinct rational points `P, Q, R` such that

1. `[P] - [R]` and `[Q] - [R]` are torsion in the Jacobian,
2. `(P, R)` and `(Q, R)` each reduce to a smooth point of the *same*
   irreducible component of the special fibre at *every* prime, and
3. no two of `P, Q, R` are hyperelliptic conjugates,

then ranks the triples by the sum of integration orders, breaking ties by cut conditioning,
and evaluates the
`3 \times 3` regulator determinant of Corollary "3x3 matrix".

When moving `R` to infinity would also move `P` or `Q` there, all three
points are kept affine instead. The logarithmic weight is then
`m_P = (x-x(P))/(x-x(R))`, and similarly for `Q`.

The construction (infinity normalization)
=======================================

With `\O^+ := R` moved to `[0:1:0]`, a torsion point `P^+ \in C(\QQ)` of order
`n` gives `n P^+ - n \O^+ = \div(f^+)`; applying the hyperelliptic involution
`F` gives `f^- := f^+ \circ F` with `\div(f^-) = n P^- - n \O^-`.  Setting
`\varphi := f^+ f^-`, so that
`\div(\varphi) = n(P^+ + P^- - \O^+ - \O^-)`, the cycle

.. MATH::

    \alpha_P = (\Delta C, \varphi \circ \pi_1) - (\Gamma_F, \varphi \circ \pi_1)
      - (\{P^+\} \times C, \psi \circ \pi_2) + (\{P^-\} \times C, \psi \circ \pi_2)
      - (C \times \{\O^+\}, \psi \circ \pi_1) + (C \times \{\O^-\}, \psi \circ \pi_1)

with `\psi := f^+/f^-` defines a class in `\CH^2(C \times C, 1)`.  Since
`\div(x - x(P)) = P^+ + P^- - \O^+ - \O^-`, one has
`(x - x(P))^n = c\,\varphi` for some `c \in \QQ^\times`, whence the regulator

.. MATH::

    \langle \reg(\alpha_P), \delta_{ij} \rangle
      = \frac{1}{2\pi i} \int_{\CC} \frac{\log|m_P(z)|}{|f(z)|} d_{ij}(z)\,
        dz\, d\overline z + \log|c|\, a_{ij},

`m_P` being the minimal polynomial of `x(P)` over `\QQ` -- here `x - x(P)`,
since `P` is rational.  The `\log|c| a_{ij}` terms are killed by row
operations against the third column, which is why the matrix below has the
`1/|f|` integrals as its last column and needs no knowledge of `c`.

What is computed
================

For `d_{11} = 1`, `d_{12} = \Re(z)`, `d_{22} = |z|^2`, the matrix

.. MATH::

    \begin{pmatrix}
      \int \frac{\log|m_P|}{|f|} d_{11} & \int \frac{\log|m_Q|}{|f|} d_{11} & \int \frac{d_{11}}{|f|} \\
      \int \frac{\log|m_P|}{|f|} d_{12} & \int \frac{\log|m_Q|}{|f|} d_{12} & \int \frac{d_{12}}{|f|} \\
      \int \frac{\log|m_P|}{|f|} d_{22} & \int \frac{\log|m_Q|}{|f|} d_{22} & \int \frac{d_{22}}{|f|}
    \end{pmatrix}

and its determinant, by the Stokes regulator
(:func:`hyperell_regulator.regulator.regulator_stokes`): the columns at `P`
and `Q` come from functions with divisor `n P - n\O`, the third is the period
vector.  A curve whose cut system the Stokes method cannot build is reported
and skipped -- nothing here integrates over `\CC`.
If the determinant is non-zero, Beilinson's conjecture predicts it is a
rational multiple of `(2\pi i) c^+(C) c^-(C) L'''(\wedge^2 C, 1)`; the periods
`c^\pm` are reported alongside, the cached standard `L`-value is computed when needed for the comparison.

Caveats
=======

* Condition 2 is decided by :mod:`hyperell_regulator.reduction`, which
  cannot answer at `p = 2`.
  A triple whose pairs are *undecided* at 2 is reported as such and skipped by
  default; pass ``allow_undecided=True`` to use it anyway.
* Triples are pairwise non-conjugate: conjugate `P,Q` give equal weights,
  while a point conjugate to `R` gives the constant weight 1.
* Integrality of `\alpha_P` -- the existence of a suitable `c` -- is exactly
  what condition 2 is a proxy for.  It is not verified independently here.

Run with::

    sage -python -m hyperell_regulator.experiments.beilinson [N]

The default is float64 on cached geometrically simple curves of conductor
strictly below N=1000. Use --help for comparison, cache and progress options.
"""

import json
import time

from sage.all import (
    RealField,
    Integer,
    PolynomialRing,
    QQ,
    RR,
    ZZ,
    gp,
    lcm,
    pi,
)

from hyperell_regulator.components import (
    CACHE,
    RESULTS,
    _fmt,
    curve_from_row,
    odd_degree_model,
)
from hyperell_regulator.lfunction import StandardLFunction
from hyperell_regulator.paths import data_path, ensure_parent
from hyperell_regulator.regulator import c_invariants

DEFAULT_CONDUCTOR_BOUND = 1000
REPORT_VERSION = 6  # minimize the sum of integration orders before conditioning

LVALUES = data_path("results", "lvalue_cache.json")
RESULTS_OUT = data_path("results", "beilinson_results.json")


# ---------------------------------------------------------------------------
# reading the results file
# ---------------------------------------------------------------------------

def _parse_point(v):
    r"""
    Undo :func:`same_component_enumeration._jsonable` on a point: a list
    ``[x, y]`` of rationals-as-strings, or ``["infinity", Y]``.
    """
    if v is None:
        return None
    a, b = v
    if a == "infinity":
        return ("infinity", QQ(b))
    return (QQ(a), QQ(b))


def load_results(path=RESULTS):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# points, divisors and the torsion classes
# ---------------------------------------------------------------------------

def involution(C, P):
    r"""
    The hyperelliptic involution `\iota(x, y) = (x, -y - h(x))`.

    On the points at infinity: the two are swapped when `\deg F = 2g+2`, their
    `Y`-values being the two roots of `Y^2 + h_{g+1}Y = f_{2g+2}` and so
    summing to `-h_{g+1}`; the single one is fixed when `\deg F = 2g+1`.
    """
    g = C.genus()
    if P[0] == "infinity":
        if QQ(C.F[2 * g + 2]) == 0:
            return P
        return ("infinity", -QQ(C.h[g + 1]) - QQ(P[1]))
    hQ = PolynomialRing(QQ, 'x')(C.h)
    return (P[0], -P[1] - hQ(P[0]))


def is_weierstrass_point(C, P):
    r"""
    Whether ``P`` is fixed by the involution.
    """
    return involution(C, P) == P


def pairwise_nonconjugate(C, P, Q, R):
    """No marked point is the hyperelliptic conjugate of another."""
    return involution(C, P) not in (Q, R) and involution(C, Q) != R


def lmfdb_rank(row):
    r"""
    The Mordell--Weil rank from LMFDB's ``mw_invs``, the invariants of
    `J(\QQ)` as an abstract abelian group: each `0` is a copy of `\ZZ`, so the
    rank is the number of zeros.  ``None`` if not recorded.
    """
    if row is None:
        return None
    inv = row.get("mw_invs")
    if inv is None:
        return None
    if isinstance(inv, str):
        try:
            inv = json.loads(inv)
        except ValueError:
            return None
    return sum(1 for t in inv if ZZ(t) == 0)


# ---------------------------------------------------------------------------
# searching for a usable triple
# ---------------------------------------------------------------------------

def find_triples(record, C, row=None, allow_undecided=False):
    r"""
    All triples `(P, Q, R)` of distinct points of ``record`` meeting the three
    conditions, as dictionaries.  ``R`` is the common point of the two pairs.

    A pair counts if its verdict is ``True`` (or ``None`` when
    ``allow_undecided``) and the order of the difference is finite.
    """
    # How is torsion certified?  With a rational Weierstrass point the order of
    # each difference was computed in J(QQ) and must be finite.  Without one
    # there is no Mumford arithmetic and the stored order is None, so instead
    # we accept LMFDB's Mordell-Weil rank being 0: then J(QQ) is all torsion
    # and every difference is torsion automatically.  This fallback is used
    # ONLY when there is no rational Weierstrass point.
    has_weierstrass = odd_degree_model(C) is not None
    rank = lmfdb_rank(row)
    rank_zero = (rank == 0)
    invariants = row.get("mw_invs", []) if row is not None else []
    if isinstance(invariants, str):
        invariants = json.loads(invariants)
    exponent = int(lcm([ZZ(v) for v in invariants])) if rank_zero else None
    via = "order in J(Q)" if has_weierstrass else (
        "LMFDB rank 0" if rank_zero else None)

    ok = {}                      # R -> list of S with (S, R) usable
    for d in record["pairs"]:
        verdict = d["same_component_everywhere"]
        if verdict is not True and not (allow_undecided and verdict is None):
            continue
        order = d["order_of_P_minus_Q"]
        if has_weierstrass:
            if order is None or order == "Infinity":
                continue         # not known to be torsion
        else:
            if not rank_zero:
                continue         # cannot certify torsion at all
            order = order if order not in (None, "Infinity") else "rank 0"
        P, Q = _parse_point(d["P"]), _parse_point(d["Q"])
        ok.setdefault(P, []).append((Q, verdict, order))
        ok.setdefault(Q, []).append((P, verdict, order))

    out = []
    for R, partners in ok.items():
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                (P, vP, nP), (Q, vQ, nQ) = partners[i], partners[j]
                if P == Q or P == R or Q == R:
                    continue
                if not pairwise_nonconjugate(C, P, Q, R):
                    continue
                out.append({"P": P, "Q": Q, "R": R,
                            "order_P": nP, "order_Q": nQ,
                            "verdict_P": vP, "verdict_Q": vQ,
                            "pairwise_nonconjugate": True,
                            "torsion_certified_by": via,
                            "mw_rank": rank,
                            "torsion_exponent": exponent})
    return out


# ---------------------------------------------------------------------------
# moving R to infinity
# ---------------------------------------------------------------------------

def model_with_R_at_infinity(C, R, points):
    r"""
    An isomorphic model `v^2 = G(u)` on which ``R`` is a point at infinity,
    together with the images of ``points``.

    If ``R`` is already at infinity the square model `Y^2 = F(x)` is returned
    unchanged.  Otherwise `u = 1/(x - x(R))`, `v = Y/(x - x(R))^{g+1}` with
    `Y = 2y + h(x)`, giving `G(u) = u^{2g+2} F(x(R) + 1/u)`; then `x = x(R)`
    goes to `u = \infty`, so `R` and its involution image are the points at
    infinity, and `\deg G = 2g+2` minus the multiplicity of `x(R)` as a root of
    `F` -- degree 5 exactly when `R` is a rational Weierstrass point.

    Returns ``(G, images, xR)`` with ``G`` in `\ZZ[u]` and ``images`` a list
    parallel to ``points``, each an affine `(u, v)` or ``"infinity"``.
    """
    g = C.genus()
    RQ = PolynomialRing(QQ, 'x')
    hQ, FQ = RQ(C.h), RQ(C.F)

    def Ycoord(P):
        return 2 * P[1] + hQ(P[0])

    if R[0] == "infinity":
        images = []
        for P in points:
            images.append("infinity" if P[0] == "infinity"
                          else (P[0], Ycoord(P)))
        return C.F, images, None

    xR = QQ(R[0])
    SQ = PolynomialRing(QQ, 'u')
    u = SQ.gen()
    shifted = FQ(RQ.gen() + xR)
    G = sum(QQ(c) * u ** (2 * g + 2 - i) for i, c in enumerate(list(shifted)))
    d = lcm([QQ(c).denominator() for c in G] + [1])
    G = (d ** 2 * G).change_ring(ZZ)

    images = []
    for P in points:
        if P[0] == "infinity":
            # x = oo -> u = 0, v = d*(2 Y_oo + h_{g+1})
            images.append((QQ(0), d * (2 * QQ(P[1]) + QQ(C.h[g + 1]))))
        elif QQ(P[0]) == xR:
            images.append("infinity")
        else:
            t = QQ(P[0]) - xR
            images.append((1 / t, d * Ycoord(P) / t ** (g + 1)))
    return G, images, xR


def model_with_finite_points(C, points):
    """A rational model keeping all specified points affine, including R."""
    if all(P[0] != "infinity" for P in points):
        H = PolynomialRing(QQ, "x")(C.h)
        return C.F, [(P[0], 2*P[1]+H(P[0])) for P in points], None
    # Put the new infinity fibre at a rational x-value containing none of
    # the marked points or Weierstrass points. Original infinities go to u=0.
    RQ = PolynomialRing(QQ, "x")
    F, H = RQ(C.F), RQ(C.h)
    marked = {QQ(P[0]) for P in points if P[0] != "infinity"}
    shift = QQ(0)
    while shift in marked or F(shift) == 0:
        shift += 1
    u = PolynomialRing(QQ, "u").gen()
    d = 2*C.genus()+2
    G = sum(c*u**(d-i) for i,c in enumerate(F(RQ.gen()+shift).list()))
    images = []
    for P in points:
        if P[0] == "infinity":
            images.append((QQ(0), 2*QQ(P[1])+QQ(C.h[C.genus()+1])))
        else:
            t = QQ(P[0])-shift
            images.append((1/t, (2*P[1]+H(P[0]))/t**(C.genus()+1)))
    return G, images, shift


def _model_for_triple(C, t):
    points = [t["P"], t["Q"], t["R"]]
    G, images, shift = model_with_R_at_infinity(C, t["R"], points)
    if images[0] != "infinity" and images[1] != "infinity":
        return G, images, shift, "infinity"
    G, images, shift = model_with_finite_points(C, points)
    return G, images, shift, images[2]


def _resolve_triple_orders(G, images, pole, t):
    """Recover missing orders by exact principal-divisor tests.

    Rank zero and the recorded finite group exponent give a proven multiple;
    testing its divisors avoids an arbitrary torsion-search cutoff.
    """
    from hyperell_regulator.regulator import function_with_divisor
    for key, point in zip(("order_P", "order_Q"), images[:2]):
        if str(t.get(key)).isdigit():
            continue
        exponent = t.get("torsion_exponent")
        if not exponent:
            raise ValueError("no exact torsion order or proven exponent")
        for n in ZZ(exponent).divisors():
            try:
                function_with_divisor(G, point, pole, int(n))
            except ValueError:
                continue
            t[key] = int(n)
            break
        else:
            raise ArithmeticError("recorded torsion exponent is not principal")


# ---------------------------------------------------------------------------
# choosing which triple to integrate
# ---------------------------------------------------------------------------

def cut_conditioning(G, point, n, pole="infinity"):
    r"""
    ``(N, nodes, span, gap, detour)`` for the cut system of the `\varphi` with
    `\operatorname{div}(\varphi) = nP - n\O` on `v^2 = G(u)`.

    Everything the Stokes legs do is measured in ``span`` and ``gap`` --- the
    largest and smallest distance between two nodes --- so their ratio is what
    says how hard the cut system will be to integrate.  It costs a
    :meth:`~hyperell_regulator.regulator.cover.Cover.cut_path` and **no
    integral at all**, which is what makes it usable as a selection rule.
    """
    from hyperell_regulator.regulator import function_with_divisor
    from hyperell_regulator.regulator.cover import Cover
    from hyperell_regulator.regulator.stokes import cut_scale

    cov = Cover(G, *function_with_divisor(G, point, pole, int(n)))
    z, kinds, _ = cov.cut_path()
    span, gap = cut_scale(z)
    return cov.N, len(z), span, gap, "none" in kinds


def triple_conditioning(C, t, mp=False):
    r"""
    How well conditioned a triple's two cut systems are, as a sort key, or
    ``None`` when the triple has no usable model.

    The key is ``(n_P + n_Q, worst span/gap)`` for the orders of `P-R`
    and `Q-R`. Thus the smaller sum of integration orders always wins;
    among triples with the same sum, prefer the better-conditioned
    cut systems. The rule is independent of the order of the two columns.

    ``mp`` is retained for caller compatibility; both precision backends
    support detoured cuts and use the same ranking.
    """
    try:
        G, images, xR, pole = _model_for_triple(C, t)
        _resolve_triple_orders(G, images, pole, t)
    except Exception:                                          # noqa: BLE001
        return None
    if images[0] == "infinity" or images[1] == "infinity":
        return None
    if G.degree() not in (5, 6):
        return None
    out = []
    for im, n in ((images[0], t.get("order_P")), (images[1], t.get("order_Q"))):
        if not (str(n).isdigit() or isinstance(n, (int, Integer))):
            return None
        try:
            out.append(cut_conditioning(G, im, int(n), pole=pole))
        except Exception:                                      # noqa: BLE001
            return None
    order_sum = int(t["order_P"]) + int(t["order_Q"])
    ratio = max(r[2] / r[3] for r in out)
    return (order_sum, ratio), out


def rank_triples(C, triples, verbose=False, mp=False, usable_only=False):
    r"""
    ``triples`` best first, by :func:`triple_conditioning`.

    With ``usable_only=True``, discard triples whose covers cannot be built;
    otherwise keep them in their original order at the end.
    """
    scored, rest = [], []
    for k, t in enumerate(triples):
        got = triple_conditioning(C, t, mp=mp)
        (scored if got is not None else rest).append(
            (got[0], k, t, got[1]) if got is not None else (k, t))
    scored.sort(key=lambda r: r[0])
    if verbose and scored:
        print("  cut systems, best first (no integrals):")
        for key, k, t, cols in scored[:4]:
            print("    %-46s N = %-6s span/gap %9.1f%s"
                  % ("%s / %s" % (_fmt(t["P"]), _fmt(t["Q"])),
                     "%d, %d" % (cols[0][0], cols[1][0]), key[1],
                     "   (detour)" if any(col[4] for col in cols) else ""))
        if len(scored) > 4:
            print("    ... and %d more" % (len(scored) - 4))
    return [t for _, _, t, _ in scored] + ([] if usable_only else [t for _, t in rest])


# ---------------------------------------------------------------------------
# the regulator determinant
# ---------------------------------------------------------------------------

def regulator_column_stokes(f, point, order, digits=None, pole="infinity"):
    r"""
    The column `\bigl(\int \log|x - x(P)|\,d_{ij}/|f|\bigr)_{ij}`, by the
    Stokes regulator rather than by quadrature over `\CC`.

    The Stokes method needs a function whose divisor is `N A - N B`, and
    `x - x(P)` instead has divisor `P^+ + P^- - \O^+ - \O^-`.
    We use `f^+` with `\div(f^+) = n P^+ - n\O^+`, the function defining
    the cycle `\alpha_P`, and `f^- = f^+ \circ \iota`.  Pointwise,

    .. math:: \log|f^+| + \log|f^-|
              = \log|\operatorname{N}(f^+)|
              = n\log|x - x(P)| - \log|c| ,

    where `(x - x(P))^n = c\,f^+f^-`.  The differential weights are
    invariant under the hyperelliptic involution `\iota`, so the integrals
    weighted by `\log|f^+|` and `\log|f^-|` agree.  Consequently the column
    is `\frac{2}{n}` times the `\log|f^+|` regulator plus
    `\frac{\log|c|}{n}` times the periods.  The `\log|c|` term is the same
    one the third column removes by row operations, but it costs nothing to
    put it in and makes this column equal the plane integrator's outright.

    ``point`` is `(x(P), y(P))` on `y^2 = f(x)` and ``order`` is `n`.
    ``pole`` is the common pole `R`, either ``"infinity"`` (default) or an
    affine pair. For finite `R`, replace `x-x(P)` throughout by
    `(x-x(P))/(x-x(R))`. The same exact norm identity fixes the constant.
    ``digits`` asks for that many decimal digits; ``None`` is float64.
    """
    return _regulator_column_and_periods(f, point, order, digits=digits, pole=pole)[0]


def _regulator_column_and_periods(f, point, order, digits=None, pole="infinity"):
    """The normalized logarithmic column and its already-computed periods."""
    from sage.all import QQ, log

    from hyperell_regulator.regulator import (
        function_with_divisor, periods_and_regulator_stokes)
    from hyperell_regulator.regulator.cover import Cover

    n = int(order)
    phi = function_with_divisor(f, point, pole, n)
    per, reg = periods_and_regulator_stokes(f, phi, prec=digits)

    # N(phi) = A^2 - B^2 f, and weight^n = c N(phi), c constant.
    num, den = Cover(f, *phi).norm_exact()
    x = num.parent().gen()
    weight = x - QQ(point[0])
    if pole != "infinity":
        weight /= x - QQ(pole[0])
    c = weight**n * den / num
    if (c == 0 or c.numerator().degree() > 0
            or c.denominator().degree() > 0):
        raise ArithmeticError("the requested normalized weight does not match N(phi)")
    field = RR if digits is None else per[0].parent()
    logc = field(abs(QQ(c))).log()
    column = [field(2) / n * reg[i] + logc / n * per[i] for i in range(3)]
    return column, per


def phi_string(f, point, n, width=200, pole="infinity"):
    r"""
    `\varphi = A(u) + B(u)v` with `\operatorname{div}(\varphi) = nP - nR`,
    as a string over a common denominator, or the reason there is none.

    Long ones are cut off at ``width``: the function for a point of order 14
    has degree 7 with coefficients over a denominator of `2\cdot10^{11}`, and
    printing it in full says less than saying how big it is.
    """
    from sage.all import QQ, PolynomialRing, lcm

    from hyperell_regulator.regulator import function_with_divisor

    try:
        A, B = function_with_divisor(f, point, pole, n)
    except Exception as exc:                                   # noqa: BLE001
        return "unavailable (%s)" % str(exc)[:50]
    R = PolynomialRing(QQ, "u")
    common = A.denominator().lcm(B.denominator())
    An, Bn = R(A * common), R(B * common)
    out = str(An) if An else ""
    if Bn:
        out += "%s(%s)*v" % (" + " if out else "", Bn)
    if common != 1:
        out = "(%s) / (%s)" % (out, R(common))
    if len(out) > width:
        out = "%s... [numerator degrees %s, %s; denominator degree %s]" % (
            out[:width], An.degree(), Bn.degree(), common.degree())
    return out or "0"


def _sage_real(v, digits):
    r"""
    ``v`` as a Sage real carrying ``digits`` decimals, whatever it came as.

    Conversion keeps Sage values and saved decimal strings at guarded precision.
    """
    from sage.all import RealField

    R = RealField(int(3.33 * int(digits)) + 24)
    return R(v)


def beilinson_determinant_stokes(G, imP, nP, imQ, nQ, digits=None, pole="infinity"):
    r"""
    The `3\times 3` determinant by the Stokes regulator.

    Columns one and two are :func:`regulator_column_stokes` at `P` and `Q`;
    the third is the period vector `\int d_{ij}/|f|`.  In float64 each of
    the two covers is integrated once, reusing its ordinary edge periods.
    ``digits`` selects the existing arbitrary-precision implementation.

    Returns ``(det, matrix, None)``, the ``None`` standing where the plane
    integrator reports its per-entry error estimates.
    """
    from sage.all import Matrix

    colP, per = _regulator_column_and_periods(G, imP, nP, digits=digits, pole=pole)
    colQ, _ = _regulator_column_and_periods(G, imQ, nQ, digits=digits, pole=pole)
    # The unweighted period vector depends only on G, so P already supplies
    # the third column as well as the correction to its logarithmic column.
    f = (lambda v: v) if digits is None else (lambda v: _sage_real(v, digits))
    M = [[f(colP[i]), f(colQ[i]), f(per[i])] for i in range(3)]
    return Matrix(3, 3, [x for r in M for x in r]).determinant(), M, None


# ---------------------------------------------------------------------------
# cached standard L-values
# ---------------------------------------------------------------------------

def load_lvalues(path=LVALUES):
    r"""
    The standard `L`-values already computed, keyed by label.
    """
    try:
        with open(path) as fh:
            return json.load(fh)
    except (IOError, OSError, ValueError):
        return {}


def save_lvalue(label, data, path=LVALUES):
    r"""
    Merge one curve's `L`-data into ``path``, keyed by label.
    """
    have = load_lvalues(path)
    have[label] = data
    with open(ensure_parent(path), "w") as fh:
        json.dump(have, fh, indent=2, sort_keys=True, default=str)
    return have


def standard_lvalue_cached(label, C, digits=15, path=LVALUES, force=False,
                           progress=True):
    r"""
    `L''(\mathrm{std}, 1)` for ``C``, reusing ``path`` when it already holds a
    value computed with at least as many coefficients as ``digits`` now
    needs.

    ``L_digits >= digits`` alone is a *claim*, not a guarantee: a record made
    while ``StandardLFunction._coeffs_needed`` used a smaller (or buggier)
    coefficient-count formula would satisfy that check while being nowhere
    near ``digits`` accurate -- which is exactly how a stale ``L_digits=15``
    entry survived undetected here for a while, computed with only ~13% of
    the coefficients PARI's own ``lfuncost`` says `15` digits requires.  So
    the real check is against ``L_num_coeffs``, cross-checked against what
    ``_coeffs_needed`` says is required *now* (cheap: a cost estimate, no
    coefficients computed).

    The `L`-value is an invariant of the curve, not of the model used for the
    regulator, so caching it by label is safe.  Returns a dictionary with keys
    ``L_second_derivative``, ``standard_conductor``, ``feq_check``,
    ``L_digits``, ``L_num_coeffs`` and, on failure, ``L_status``.
    """
    have = load_lvalues(path)
    LF = StandardLFunction(C)
    if not force and label in have:
        rec = have[label]
        # ``L_pari_precision`` marks a record computed after the precision
        # of the PARI calls was fixed: before that, ``lfun`` answered at 64
        # bits whatever was asked of it, so a cached "25 digit" value has
        # sixteen good ones and noise after them.  Records without the mark
        # are recomputed.
        if rec.get("L_second_derivative") is not None and \
                rec.get("L_pari_precision") and \
                (digits <= 15 or isinstance(rec.get("L_second_derivative"), str)) and \
                ZZ(rec.get("L_digits", 0)) >= digits and \
                ZZ(rec.get("L_num_coeffs", 0)) >= LF._coeffs_needed(digits):
            rec = dict(rec)
            rec["L_cached"] = True
            return rec
    try:
        num_coeffs = LF._coeffs_needed(digits)
        # warm the coefficient cache first, reporting progress: this is the
        # slow step and everything after it is instant
        LF.lfun(num_coeffs, digits, progress=progress)
        rec = {"standard_conductor": int(LF.conductor()),
               "feq_check": float(LF.check_functional_equation(num_coeffs,
                                                               digits)),
               "L_second_derivative": LF.value(1, 2, num_coeffs,
                                                      digits).str(truncate=False),
               "L_digits": int(digits), "L_num_coeffs": int(num_coeffs),
               "L_pari_precision": True}
        # the functional equation is the only independent check on the value
        # here, and it is a sharp one: with too few coefficients it fails
        # first.  A ``digits`` that the coefficients do not support is worth
        # saying out loud rather than recording as achieved.
        if rec["feq_check"] > 10.0 ** (-int(digits)):
            rec["L_precision_warning"] = (
                "the functional equation holds only to %.1e, short of the %d "
                "digits asked for" % (rec["feq_check"], int(digits)))
            if progress:
                print("   WARNING: %s" % rec["L_precision_warning"])
    except NotImplementedError as exc:
        rec = {"L_second_derivative": None, "L_status": str(exc),
               "L_digits": int(digits)}
    except Exception as exc:
        rec = {"L_second_derivative": None,
               "L_status": "%s: %s" % (type(exc).__name__, exc),
               "L_digits": int(digits)}
    save_lvalue(label, rec, path)
    rec = dict(rec)
    rec["L_cached"] = False
    return rec


# ---------------------------------------------------------------------------
# comparison with the L-value
# ---------------------------------------------------------------------------

def beilinson_comparison(det, c_plus, c_minus, lval, bound=8, digits=None):
    r"""
    Compare the regulator determinant with `2\pi\, c^+(C)\, c^-(C)\, L''`.

    Corollary "3x3 matrix" predicts a non-zero determinant is a *rational*
    multiple of `(2\pi i)\, c^+(C)\, c^-(C)` times the leading `L`-derivative;
    with `c^-` returned as the determinant of the imaginary part of the period
    matrix, the normalisation that the data actually picks out is
    `2\pi c^+ c^- L''(\mathrm{std}, 1)`.

    Returns ``{}`` if any ingredient is missing, else a dictionary with the
    ratio, the integer relation ``[m, n]`` from ``gp.lindep`` with
    ``m*det + n*norm = 0``, the implied rational ``-n/m`` and the residual.
    """
    if lval is None or c_plus is None or c_minus is None:
        return {}
    # in RR the residual could not be smaller than about 1e-14 whatever the
    # ingredients are worth, so the field follows ``digits``
    F = RR if digits is None else RealField(int(3.33 * int(digits)) + 24)
    cv = (lambda v: RR(v)) if digits is None else (lambda v: _sage_real(v, digits))
    det, c_plus, c_minus, lval = (cv(det), cv(c_plus), cv(c_minus), cv(lval))
    norm = 2 * F(pi) * c_plus * c_minus * lval
    if norm == 0:
        return {}
    out = {"normalisation": "2pi c+ c- L''(std,1)",
           "norm_value": norm, "ratio": det / norm}
    try:
        rel = [Integer(t) for t in gp.lindep([det, norm], bound)]
        out["relation"] = rel
        if rel and rel[0] != 0:
            out["rational"] = -QQ(rel[1]) / QQ(rel[0])
            out["residual"] = abs(rel[0] * det + rel[1] * norm)
            if out["rational"]:
                out["relative_agreement"] = abs(out["ratio"] / out["rational"] - 1)
    except Exception as exc:
        out["relation_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _analyse_triple(record, C, t, n_triples, cache=CACHE, verbose=True,
                    with_periods=True, with_lvalue=True, digits=15,
                    lindep_bound=8, reg_digits=None):
    r"""
    :func:`analyse` for one triple `(P, Q, R)`.

    Separate because a triple can fail on its own account -- the model it
    gives may have poorly separated critical values, or the cut system for its
    `\\varphi` may not close -- while another triple on the same curve is
    perfectly good, and a curve is worth more than its first triple.
    """
    G, images, xR, pole = _model_for_triple(C, t)
    _resolve_triple_orders(G, images, pole, t)
    imP, imQ = images[0], images[1]
    if imP == "infinity" or imQ == "infinity":
        return {"label": record["label"], "cond": record["cond"],
                "status": "P or Q lands at infinity with R"}
    if G.degree() not in (5, 6):
        return {"label": record["label"], "cond": record["cond"],
                "status": "model has degree %s, outside the integrator's "
                          "range" % G.degree()}
    uP, uQ = imP[0], imQ[0]
    GQ = G.change_ring(QQ)
    weier = [GQ(uP) == 0, GQ(uQ) == 0]

    res = {"label": record["label"], "cond": record["cond"],
           "status": "ok", "triple": t, "model": str(G),
           "model_degree": int(G.degree()), "shift": xR, "pole": pole,
           "u_P": uP, "u_Q": uQ, "P_weierstrass": weier[0],
           "Q_weierstrass": weier[1],
           "n_triples": n_triples}

    nP, nQ = t.get("order_P"), t.get("order_Q")
    if verbose:
        print("%s  cond %s  %s" % (record["label"], record["cond"], C))
        print("  P %s  Q %s  R %s   orders %s, %s%s"
              % (_fmt(t["P"]), _fmt(t["Q"]), _fmt(t["R"]), nP, nQ,
                 "   WARNING: %s is a Weierstrass point of this model"
                 % ("P" if weier[0] else "Q") if any(weier) else ""))
        print("  model  v^2 = %s   (deg %s%s)   u(P) = %s  u(Q) = %s"
              % (G, G.degree(),
                 "" if xR is None else ", u = 1/(x - %s)" % xR, uP, uQ))
        print("  common pole R = %s" % (pole,))
        for who, im, n in (("P", imP, nP), ("Q", imQ, nQ)):
            if str(n).isdigit() or isinstance(n, (int, Integer)):
                print("  phi_%s = %s" % (who, phi_string(G, im, int(n), pole=pole)))

    usable = all(isinstance(n, (int, Integer)) or
                 (isinstance(n, str) and n.isdigit()) for n in (nP, nQ))
    if not usable:
        # the Stokes route needs the order of each torsion point, to build a
        # function with divisor n P - n O
        res["status"] = "no order for P or Q, so no function with that divisor"
        if verbose:
            print("   skipped: %s" % res["status"])
        return res
    if verbose:
        print("  regulator: integrating (%s) ..."
              % ("float64" if reg_digits is None else "%s digits" % reg_digits),
              flush=True)
    clock = time.time()
    try:
        det, M, Err = beilinson_determinant_stokes(
            G, imP, int(nP), imQ, int(nQ), digits=reg_digits, pole=pole)
        res["regulator_digits"] = reg_digits
    except Exception as exc:
        res.update(status="numerical_failure",
                   error="%s: %s" % (type(exc).__name__, exc))
        if verbose:
            print("   failed: %s" % res["error"])
        return res
    res["regulator_method"] = "stokes"
    res["regulator_seconds"] = round(time.time() - clock, 1)
    number = float if reg_digits is None else lambda v: _sage_real(v, reg_digits).str(truncate=False)
    res["det"] = number(det)
    res["matrix"] = [[number(x) for x in row_] for row_ in M]
    res["errors"] = (None if Err is None
                     else [[float(x) for x in row_] for row_ in Err])

    cp = cm = None
    if with_periods:
        try:
            # at the regulator's precision: taken at the float64 quadrature's
            # default tolerance instead, these are worth about ten digits and
            # nothing else in the comparison can then be worth more
            cp, cm = c_invariants(G, prec=res["regulator_digits"])
            res["c_plus"], res["c_minus"] = number(cp), number(cm)
        except Exception as exc:
            cp = cm = None
            res["c_plus"] = res["c_minus"] = None
            res["periods_error"] = str(exc)

    # --- the L-value: an invariant of the curve, not of the model, so it is
    # --- cached by label and reused rather than recomputed
    if with_lvalue:
        if verbose:
            print("  L-value: %s coefficients ..."
                  % StandardLFunction(C)._coeffs_needed(digits), flush=True)
        clock = time.time()
        res.update(standard_lvalue_cached(record["label"], C, digits=digits,
                                           progress=verbose))
        res["lvalue_seconds"] = round(time.time() - clock, 1)

    res["comparison"] = beilinson_comparison(
        det, cp, cm, res.get("L_second_derivative"), bound=lindep_bound,
        digits=res["regulator_digits"])

    if verbose:
        print("  regulator  %-9s %6.1fs   det = % .14e"
              % ("float64" if res.get("regulator_digits") is None
                 else "%s digits" % res["regulator_digits"],
                 res.get("regulator_seconds", 0.0), float(res["det"])))
        if res.get("c_plus") is not None:
            print("  periods                      c+ = % .10e  c- = % .10e"
                  % (float(res["c_plus"]), float(res["c_minus"])))
        if res.get("L_second_derivative") is not None:
            print("  L''(std,1) %-2s digits %6.1fs%s  % .14e   (N = %s, feq %.1e)"
                  % (res.get("L_digits"), res.get("lvalue_seconds", 0.0),
                     " [cached]" if res.get("L_cached") else "         ",
                     float(res["L_second_derivative"]), res["standard_conductor"],
                     res["feq_check"]))
        if res.get("L_status"):
            print("  L''(std,1)  not computed -- %s" % res["L_status"])
        if res.get("L_precision_warning"):
            print("  WARNING: %s" % res["L_precision_warning"])
        e = res.get("comparison") or {}
        if "rational" in e:
            print("  => det = (%s) * 2pi c+ c- L''(std,1)    residual %.2e"
                  "    [ratio % .14e]"
                  % (e["rational"], float(e["residual"]), float(e["ratio"])))
        elif "ratio" in e:
            print("  => det / (2pi c+ c- L) = % .12e   (no relation found)"
                  % float(e["ratio"]))
        print("")
    return res


def analyse(record, cache=CACHE, allow_undecided=False, verbose=True,
            with_periods=True, with_lvalue=True, digits=15,
            lindep_bound=8, geom_simple=True, reg_digits=None, tries=1):
    r"""
    Find the first usable triple for one curve and compute its regulator
    determinant.  Returns a dictionary, with ``"status"`` saying what happened.

    The regulator is the Stokes one throughout.  It needs the order of each
    torsion point, to build a function with divisor `n P - n\O`; a curve
    without one, or whose cut system will not close, is reported in
    ``"status"`` and skipped rather than integrated over `\CC`.

    ``geom_simple`` restricts to curves whose Jacobian is geometrically
    simple, which is LMFDB's ``is_simple_geom``.

    ``digits`` is the `L`-value's precision and ``reg_digits`` the
    regulator's. The default regulator uses float64 (``reg_digits=None``);
    the L-value uses fifteen digits. Arbitrary precision is explicit and
    failures never silently switch implementations.

    Which triple is integrated is decided by :func:`rank_triples`: minimize
    the sum of the two integration orders, then the worst
    span/gap. Only ``tries`` triples are integrated, best first; the default
    is one.
    """
    rows = {r["label"]: r for r in json.load(open(cache))}
    row = rows.get(record["label"])
    if row is None:
        return {"label": record["label"], "status": "not in the curve cache"}
    if geom_simple and not row.get("is_simple_geom"):
        # a Jacobian that splits over Qbar is a product of elliptic curves up
        # to isogeny, and its standard L-function factors accordingly; the
        # comparison this script makes is not the one to make there
        return {"label": record["label"], "cond": record["cond"],
                "status": "Jacobian is not geometrically simple"}
    C = curve_from_row(row)

    triples = find_triples(record, C, row=row, allow_undecided=allow_undecided)
    if not triples:
        return {"label": record["label"], "cond": record["cond"],
                "status": "no_admissible_triple"}
    n_triples = len(triples)
    triples = rank_triples(C, triples, verbose=verbose,
                           mp=reg_digits is not None, usable_only=True)
    if not triples:
        return {"label": record["label"], "cond": record["cond"],
                "status": "no_usable_cover"}
    first = None
    for k, t in enumerate(triples[:max(1, int(tries))]):
        res = _analyse_triple(record, C, t, n_triples, cache=cache,
                              verbose=verbose, with_periods=with_periods,
                              with_lvalue=with_lvalue, digits=digits,
                              lindep_bound=lindep_bound,
                              reg_digits=reg_digits)
        if res.get("status") == "ok":
            if k:
                res["triples_skipped"] = k
            return res
        first = first or res
        if verbose and k + 1 < min(len(triples), max(1, int(tries))):
            print("   trying the next triple (%d of %d) ..."
                  % (k + 2, len(triples)))
    return first


def load_stored(out=RESULTS_OUT):
    """Saved reports keyed by label; old schemas are not reused by run()."""
    if out is None:
        return {}
    try:
        with open(out) as fh:
            return {r["label"]: r for r in json.load(fh)}
    except (OSError, ValueError):
        return {}


def ensure_records(max_cond=DEFAULT_CONDUCTOR_BOUND, cache=CACHE,
                   results=RESULTS, geom_simple=True, fetch=False, verbose=False):
    """Component records for cached curves with conductor strictly below N.

    Enumerate missing component records. Only an explicit fetch requests
    more curves from LMFDB; cache-only reports do not claim completeness.
    """
    from hyperell_regulator.components import (
        fetch_lmfdb, load_curves, report_curve, save_results)

    if fetch:
        fetch_lmfdb(max_cond=max_cond-1, path=cache, geom_simple=geom_simple)
    rows = load_curves(path=cache, max_cond=max_cond-1,
                       simple_only=True, geom_simple=geom_simple)
    try:
        records = {r["label"]: r for r in load_results(results)}
    except OSError:
        records = {}
    goodrecords = []
    for row in rows:
        if row["label"] not in records:
            try:
                merged = save_results([report_curve(row, verbose=verbose)], results)
                records.update({r["label"]: r for r in merged})
            except NotImplementedError as exc:
                if verbose:
                    print("   %s: %s" % (row["label"], exc))
                continue
        goodrecords.append(records[row["label"]])
    return goodrecords


def _stored_result_satisfies(old, settings):
    """Reuse successful reports at equal or higher requested precision.

    Float64 can satisfy only float64 requests; arbitrary precision of at
    least 16 decimal digits can also satisfy a float64 request. Keep the
    original result and settings, without rounding its saved decimal strings.
    """
    if old.get("status") != "ok" or not isinstance(old.get("settings"), dict):
        return False
    stored = old["settings"]
    precision_keys = {"regulator_digits", "lvalue_digits"}
    execution_keys = {"max_edge_seconds", "max_edge_steps", "tries"}
    if any(stored.get(k) != v for k, v in settings.items()
           if k not in precision_keys | execution_keys):
        return False
    if not precision_keys <= stored.keys():
        return False
    have, want = stored["regulator_digits"], settings["regulator_digits"]
    if have is None:
        if want is not None:
            return False
    elif int(have) < (16 if want is None else int(want)):
        return False
    if settings.get("with_lvalue"):
        if int(stored["lvalue_digits"]) < int(settings["lvalue_digits"]):
            return False
    return True


def _report_row(result, stored=False):
    comparison = result.get("comparison") or {}
    def number(value):
        return "-" if value is None else "%.8g" % float(value)
    print("%-18s %5s %-21s %14s %12s %10s %10s%s" % (
        result["label"], result.get("cond", "?"), result["status"],
        number(result.get("det")), number(comparison.get("ratio")),
        str(comparison.get("rational", "-")),
        number(comparison.get("relative_agreement")),
        " [stored]" if stored else ""), flush=True)
    for key in ("error", "periods_error", "L_status", "L_precision_warning"):
        if result.get(key):
            print("  %s: %s" % (key, result[key]), flush=True)


def run(max_cond=DEFAULT_CONDUCTOR_BOUND, path=RESULTS, cache=CACHE,
        out=RESULTS_OUT, force=False, digits=15, reg_digits=None,
        allow_undecided=False, geom_simple=True, fetch=False,
        with_lvalue=True, tries=1, verbose=False, max_edge_seconds=15,
        max_edge_steps=100000):
    """Report cached curves with conductor < max_cond (default 1000).

    The pairwise non-conjugate triple with minimal sum of integration orders is
    tried first, using cut conditioning to break ties.
    Float64 and per-edge budgets are the defaults. Successful results are
    reused for the same report policy at equal or higher precision; failures
    are saved with their reasons but retried on the next run.
    """
    from collections import Counter
    from hyperell_regulator.regulator import IntegrationMonitor

    max_cond = int(max_cond)
    if max_cond < 1 or int(digits) < 1 or int(tries) < 1:
        raise ValueError("conductor bound, digits and tries must be positive")
    records = ensure_records(max_cond, cache=cache, results=path,
                             geom_simple=geom_simple, fetch=fetch, verbose=verbose)
    settings = dict(report_version=REPORT_VERSION, regulator_digits=reg_digits,
                    lvalue_digits=int(digits), with_lvalue=with_lvalue,
                    allow_undecided=allow_undecided, geom_simple=geom_simple,
                    tries=int(tries), max_edge_seconds=max_edge_seconds,
                    max_edge_steps=max_edge_steps)
    have, results = load_stored(out), []
    print("%d cached curves with conductor < %d; %s; pairwise non-conjugate triples"
          % (len(records), max_cond, "float64" if reg_digits is None
             else "%s-digit regulator" % reg_digits))
    print("%-18s %5s %-21s %14s %12s %10s %10s" %
          ("curve", "cond", "status", "determinant", "q", "rational", "rel. error"))
    for record in records:
        label = record["label"]
        old = have.get(label, {})
        if not force and _stored_result_satisfies(old, settings):
            results.append(old)
            _report_row(old, stored=True)
            continue
        def progress(event):
            if verbose and event["event"] != "start":
                print("  %s edge %d %s: %.1fs, %s" %
                      (label, event["index"], event["event"],
                       event["seconds"], event["phase"]), flush=True)
        started = time.monotonic()
        monitor = IntegrationMonitor(max_edge_seconds=max_edge_seconds,
                                     max_edge_steps=max_edge_steps,
                                     callback=progress, progress_interval=5)
        if verbose:
            print("Computing %s ..." % label, flush=True)
        try:
            with monitor:
                res = analyse(record, cache=cache,
                              allow_undecided=allow_undecided, verbose=verbose,
                              with_periods=with_lvalue, with_lvalue=with_lvalue,
                              digits=digits, geom_simple=geom_simple,
                              reg_digits=reg_digits, tries=tries)
        except Exception as exc:
            res = dict(label=label, cond=record["cond"],
                       status="preparation_error", error="%s: %s" % (type(exc).__name__, exc))
        res.update(settings=settings, seconds=time.monotonic()-started,
                   edges_attempted=len(monitor.records),
                   slowest_edge_seconds=max((e["seconds"] for e in monitor.records), default=0))
        results.append(res)
        have[label] = res
        if out is not None:
            with open(ensure_parent(out), "w") as fh:
                json.dump(sorted(have.values(), key=lambda r: (int(r["cond"]), r["label"])),
                          fh, indent=2, default=str)
        _report_row(res)
    counts = Counter(r["status"] for r in results)
    print("\n" + "; ".join("%s: %d" % item for item in sorted(counts.items())))
    print("q = det / (2*pi*c+*c-*L''(std,1)); rel. error = abs(q/rational - 1).")
    print("no_admissible_triple: no pairwise non-conjugate torsion triple passes the component test.")
    print("no_usable_cover: triples exist, but their models or covering functions could not be constructed.")
    print("numerical_failure: a selected cover failed continuation, quadrature, or an integration limit.")
    if out is not None:
        print("Results: %s" % out)
    return results


def main(argv=None):
    """Command line: beilinson [N], with conductor strictly below N."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("The construction")[0])
    parser.add_argument("N", nargs="?", type=int, default=DEFAULT_CONDUCTOR_BOUND,
                        help="exclusive conductor bound (default: 1000)")
    parser.add_argument("--force", action="store_true", help="recompute saved successful regulators")
    parser.add_argument("--fetch", action="store_true", help="extend the curve cache from LMFDB")
    parser.add_argument("--no-lvalue", action="store_true", help="only compute regulator determinants")
    parser.add_argument("--verbose", action="store_true", help="show triples and per-edge progress")
    parser.add_argument("--allow-undecided", action="store_true")
    parser.add_argument("--digits", type=int, default=15, help="L-value precision (default: 15 digits)")
    parser.add_argument("--reg-digits", "--prec", type=int, default=None,
                        help="explicit arbitrary-precision regulator (default: float64)")
    parser.add_argument("--edge-seconds", type=float, default=15, help="time limit per edge (default: 15 seconds)")
    parser.add_argument("--edge-steps", type=int, default=100000, help="continuation step limit per edge")
    parser.add_argument("--tries", type=int, default=1, help="number of ranked triples to try per curve")
    parser.add_argument("--out", default=RESULTS_OUT, help="JSON report path")
    args = parser.parse_args(argv)
    if args.N < 1 or args.digits < 1 or args.tries < 1 or (args.reg_digits is not None and args.reg_digits < 5):
        parser.error("N, digits and tries must be positive; regulator precision must be at least 5")
    if args.edge_seconds <= 0 or args.edge_steps < 1:
        parser.error("edge budgets must be positive")
    return run(args.N, out=args.out, force=args.force, fetch=args.fetch,
               with_lvalue=not args.no_lvalue, verbose=args.verbose,
               allow_undecided=args.allow_undecided, digits=args.digits,
               reg_digits=args.reg_digits, tries=args.tries,
               max_edge_seconds=args.edge_seconds, max_edge_steps=args.edge_steps)


if __name__ == "__main__":
    main()
