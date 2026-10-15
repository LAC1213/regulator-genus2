r"""
Semistable reduction of genus 2 curves over `\QQ`, and the component
question for their rational points.

Input a genus 2 curve

.. MATH::

    C : y^2 + h(x) y = f(x),   f, h \in \ZZ[x],

and

1. decide whether `C` has semistable reduction at every prime,
2. if so, describe the irreducible components of the special fibre at each
   prime of bad reduction,
3. given two points `P, Q \in C(\QQ)`, decide at each bad prime whether `P`
   and `Q` reduce to a *smooth* point of the *same* component.

Mathematical basis
==================

For odd `p` we complete the square: `Y = 2y + h(x)` gives `Y^2 = F(x)` with
`F = 4f + h^2`, an isomorphism over `\ZZ_p`.  Everything is then read off the
*cluster picture* of `F` over `\QQ_p` following

    T. Dokchitser, V. Dokchitser, C. Maistret, A. Morgan,
    "Arithmetic of hyperelliptic curves over local fields",
    Math. Ann. 385 (2023).  (referred to as [DDMM] below)

via the ``sage_cluster_pictures`` package.  Specifically:

* semistability is [DDMM, Thm 1.8]: the splitting field of `F` has `e \le 2`,
  every proper cluster is inertia stable, and every principal cluster
  `\mathfrak{s}` has `d_\mathfrak{s} \in \ZZ` and `\nu_\mathfrak{s} \in 2\ZZ`;

* the components of the special fibre are `\Gamma_\mathfrak{s}` for the
  principal clusters `\mathfrak{s}` (two components
  `\Gamma_\mathfrak{s}^{\pm} \cong \mathbb{P}^1` when `\mathfrak{s}` is
  übereven), joined by chains of `\mathbb{P}^1`'s [DDMM, Thm 8.5].  The
  components of the *stable* model are exactly the `\Gamma_\mathfrak{s}`; the
  chains are contracted to nodes.

* the reduction map onto `\Gamma_\mathfrak{s}` is [DDMM, Def 6.x], i.e.
  ``Cluster.red``.

Residue characteristic 2 is *not* covered by cluster pictures.  There we fall
back on Liu's algorithm (``genus2reduction``, i.e. PARI's ``genus2red``):
the Namikawa--Ueno type of the special fibre of the minimal regular model
determines semistability (semistable `\iff` every component of the special
fibre has multiplicity one `\iff` the type is built only from `I_n` pieces),
and Grothendieck's formula gives the rigorous necessary condition
`f_2 \le 2` on the conductor exponent.  When PARI does not return a
Namikawa--Ueno type at 2 (its genus 2 reduction algorithm is incomplete there)
the answer is reported as *unknown* rather than guessed.

Usage
=====

::

    sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
    sage: R.<x> = ZZ[]
    sage: C = HyperellipticCurveWithReduction(f=x^5 + x^3 + 1)
    sage: C.bad_primes()
    [2, 53, 61]
    sage: C.is_semistable_at(53)
    True
    sage: [(c.kind, c.genus) for c in C.components(53)]
    [('principal', 1)]
    sage: C.same_component((0, 1), (0, -1), 53)["answer"]
    True

or, for everything at once,

::

    sage: from contextlib import redirect_stdout
    sage: from io import StringIO
    sage: output = StringIO()
    sage: with redirect_stdout(output):
    ....:     C.report(P=(0, 1), Q=(0, -1))
    sage: 'conductor' in output.getvalue()
    True

Correctness
===========

``self_test()`` cross-checks the answers at every odd bad prime of a list of
test curves against Liu's independent implementation: the semistability
verdict, the conductor exponent, the identity
`\sum_i g(\Gamma_i) + b_1(\Gamma) = f_p` with
`\sum_i g(\Gamma_i) + b_1(\Gamma) = 2`, and the order of the group of
connected components of the Néron model (which pins down the lengths of the
linking chains, not just the topology of the special fibre).

Requirements
============

This needs a ``sage_cluster_pictures`` in which ``Cluster.red`` on points and
``Cluster.dual_graph`` are correct.  Released 1.0 is not: ``red`` multiplies by
`\pi^{\nu/2}` instead of dividing, does not scale that exponent by the
ramification index, and computes another as ``-s.size()//2``, which Python
evaluates as `\lfloor -|s|/2 \rfloor` rather than `-\lfloor |s|/2 \rfloor`;
``dual_graph`` raises ``NameError`` for a non-principal top cluster with two
odd children, and uses `2\delta_S` in place of `2\delta_\mathfrak{s}` for the
chain of a cotwin.  Fixes are on the ``fix-red-and-dual-graph`` branch of
https://github.com/alexjbest/cluster-pictures; install with
``sage -pip install -e /path/to/cluster-pictures``.  :func:`self_test` fails
loudly against an unpatched copy.

The one thing still computed here rather than delegated is the component of an
übereven cluster: `\Gamma_\mathfrak{s}` then splits as
`\Gamma_\mathfrak{s}^{\pm}`, a pair of `\mathbb{P}^1`'s that
``component_special_fibre`` cannot represent (its polynomial
`\bar\theta^2 G^2` degenerates to a constant when `\mathfrak{s}` has no twin
of relative depth 1/2), and correspondingly ``red`` cannot reduce onto it.

.. SEEALSO::

    :mod:`hyperell_regulator.components` runs the component question over the
    LMFDB list of curves, and :mod:`hyperell_regulator.lfunction` builds the
    standard `L`-function out of the cluster pictures computed here.
"""

import re

from sage.all import (
    Graph,
    HyperellipticCurve,
    PolynomialRing,
    QQ,
    Qp,
    SageObject,
    ZZ,
    ascii_art,
    factor,
    gcd,
    genus2reduction,
    prod,
)
from sage.schemes.hyperelliptic_curves.constructor import (
    HyperellipticCurve as _sage_hyperelliptic_curve)
from sage.structure.dynamic_class import dynamic_class

from sage_cluster_pictures.cluster_pictures import Cluster


# ---------------------------------------------------------------------------
# Parsing Liu / PARI ``genus2red`` local data
# ---------------------------------------------------------------------------

_NU_TYPE_RE = re.compile(r"reduction at p:\s*(.*?)\s*page", re.S)
_F_RE = re.compile(r"\bf\s*=\s*(\d+)")
_STABLE_RE = re.compile(r"\(potential\) stable reduction:\s*\(([^)]*)\)")


def _parse_local_data(text):
    r"""
    Pull the useful pieces out of one entry of ``genus2reduction(...).local_data``.

    Returns a dictionary with keys ``stable_type`` (Liu's type (I)--(VII) of the
    potential stable reduction, a string or ``None``), ``nu_type`` (the
    Namikawa--Ueno type of the special fibre of the minimal regular model, a
    string or ``None``) and ``conductor_exponent`` (an integer or ``None``).
    """
    out = {"stable_type": None, "nu_type": None, "conductor_exponent": None,
           "raw": text}
    m = _STABLE_RE.search(text)
    if m:
        out["stable_type"] = m.group(1).strip()
    m = _NU_TYPE_RE.search(text)
    if m:
        out["nu_type"] = m.group(1).strip()
    m = _F_RE.search(text)
    if m:
        out["conductor_exponent"] = ZZ(m.group(1))
    return out


# Liu's classification of the *potential* stable reduction of a genus 2 curve
# (see the documentation of ``ReductionData``), together with the toric rank of
# the semi-abelian reduction of the Jacobian that it forces:
#
#   (I)   smooth genus 2                                      -> t = 0
#   (II)  elliptic curve with one node                        -> t = 1
#   (III) projective line with two nodes                      -> t = 2
#   (IV)  two projective lines meeting at three points        -> t = 2
#   (V)   two elliptic curves meeting at one point            -> t = 0
#   (VI)  elliptic curve + nodal line meeting at one point    -> t = 1
#   (VII) as (VI) with both components singular               -> t = 2
#
_POTENTIAL_TORIC_RANK = {"I": 0, "II": 1, "III": 2, "IV": 2,
                         "V": 0, "VI": 1, "VII": 2}


def potential_toric_rank(stable_type):
    r"""
    The toric rank of the potential semi-abelian reduction of `J(C)`, read off
    Liu's type ``(I)``--``(VII)`` of the potential stable reduction.

    EXAMPLES::

        sage: from hyperell_regulator.reduction import potential_toric_rank
        sage: potential_toric_rank("II")
        1
        sage: potential_toric_rank("V")
        0
    """
    if stable_type is None:
        return None
    key = stable_type.split(",")[0].strip()
    if key not in _POTENTIAL_TORIC_RANK:
        return None
    return ZZ(_POTENTIAL_TORIC_RANK[key])


def nu_type_is_semistable(label):
    r"""
    Decide whether a Namikawa--Ueno type is semistable, i.e. whether every
    component of the corresponding special fibre has multiplicity one.

    The fibres of multiplicity one are exactly those whose Namikawa--Ueno
    symbol is assembled from `I_n` pieces only: no other Roman numeral, no
    ``*``, and no multiplicity prefix such as ``2I_0``.

    INPUT:

    - ``label`` -- string, e.g. ``"[I{1-0-0}]"``, ``"[II-II-0]"``, ``"[V]"``.

    EXAMPLES::

        sage: from hyperell_regulator.reduction import nu_type_is_semistable
        sage: nu_type_is_semistable("[I{1-0-0}]")
        True
        sage: nu_type_is_semistable("[I{1-1-0}]")
        True
        sage: nu_type_is_semistable("[V]")
        False
        sage: nu_type_is_semistable("[II-II-0]")
        False
        sage: nu_type_is_semistable("[VIII-1]")
        False
        sage: nu_type_is_semistable("[I*{1-0-0}]")
        False
    """
    if label is None:
        return None
    if "*" in label:
        return False
    for m in re.finditer(r"[IVXK]+", label):
        if m.group(0) != "I":
            return False
        # a digit in front of a numeral is a multiplicity, e.g. "2I{0}-0"
        if m.start() > 0 and label[m.start() - 1].isdigit():
            return False
    return True


# ---------------------------------------------------------------------------
# Components of the special fibre
# ---------------------------------------------------------------------------

class SpecialFibreComponent(SageObject):
    r"""
    One irreducible component of the special fibre.

    Attributes:

    - ``label`` -- a printable name, e.g. ``"Gamma_s1"`` or ``"L(s1,s2)[0]"``
    - ``kind`` -- ``"principal"`` (a component `\Gamma_\mathfrak{s}` of the
      stable model) or ``"chain"`` (a `\mathbb{P}^1` in a linking chain,
      contracted in the stable model)
    - ``cluster`` -- the corresponding cluster, or ``None`` for chain pieces
    - ``sign`` -- ``+1``/``-1`` for the two components of an übereven cluster,
      else ``None``
    - ``genus`` -- geometric genus
    - ``equation`` -- a string giving the equation of the component over the
      residue field, or ``None`` for a chain `\mathbb{P}^1`
    - ``multiplicity`` -- always 1 (the fibre is semistable)
    """

    def __init__(self, label, kind, cluster=None, sign=None, genus=None,
                 equation=None):
        self.label = label
        self.kind = kind
        self.cluster = cluster
        self.sign = sign
        self.genus = genus
        self.equation = equation
        self.multiplicity = ZZ(1)

    def _repr_(self):
        s = "%s: %s, genus %s" % (self.label, self.kind, self.genus)
        if self.equation is not None:
            s += ", %s" % (self.equation,)
        else:
            s += ", P^1"
        return s


class PointReduction(SageObject):
    r"""
    The reduction of a point of `C(\QQ)` in the special fibre at one prime.

    Attributes:

    - ``prime`` -- the prime
    - ``is_smooth`` -- whether the reduction is a smooth point of the special
      fibre of the *stable* model (equivalently: it lands on a component
      `\Gamma_\mathfrak{s}` away from the nodes)
    - ``component`` -- the ``SpecialFibreComponent`` it lands on, or ``None``
    - ``point`` -- the pair `(X, Y)` over the residue field giving the reduced
      point on that component, the string ``"infinity"``, or ``None``
    - ``cluster`` -- the principal cluster of that component, or ``None``
    - ``reason`` -- explanation when ``is_smooth`` is ``False``

    The private attribute ``_y_datum`` holds, for an übereven cluster, the
    branch value `\pm\bar\theta_\mathfrak{s}` distinguishing
    `\Gamma_\mathfrak{s}^+` from `\Gamma_\mathfrak{s}^-`.
    """

    def __init__(self, prime, is_smooth, component=None, point=None,
                 reason=None, cluster=None, y_datum=None):
        self.prime = prime
        self.is_smooth = is_smooth
        self.component = component
        self.point = point
        self.reason = reason
        self.cluster = cluster
        self._y_datum = y_datum
        self._raw = None        # the (x, y) that produced this, in K(R)

    def _repr_(self):
        if self.is_smooth:
            return "smooth point %s on %s (p=%s)" % (
                self.point, self.component.label, self.prime)
        return "singular point of the special fibre at p=%s (%s)" % (
            self.prime, self.reason)


# ---------------------------------------------------------------------------
# The main class
# ---------------------------------------------------------------------------

def HyperellipticCurveWithReduction(f, h=0, prec=250):
    r"""
    The hyperelliptic curve `y^2 + h(x) y = f(x)` over `\QQ`, with `f, h \in
    \ZZ[x]`, extended with the reduction methods of
    :class:`HyperellipticReduction`.

    The result is a genuine Sage hyperelliptic curve -- Sage's own constructor
    builds it, and the reduction methods are grafted on with
    ``dynamic_class`` -- so ``genus``, ``jacobian``, ``igusa_clebsch_invariants``,
    point construction and everything else continue to work.

    INPUT:

    - ``f`` -- integral polynomial, of degree at most `2g+2`
    - ``h`` -- integral polynomial, of degree at most `g+1` (default 0)
    - ``prec`` -- `p`-adic working precision for the cluster pictures
      (default 250)

    EXAMPLES::

        sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
        sage: C
        Hyperelliptic Curve over Rational Field defined by y^2 = x^5 + x^3 + 1
        sage: C.genus()
        2
        sage: C.bad_primes()
        [2, 53, 61]
        sage: C.is_tame(53), C.is_semistable_at(53)
        (True, True)

    Sage's own methods are untouched::

        sage: C.jacobian()
        Jacobian of Hyperelliptic Curve over Rational Field defined by y^2 = x^5 + x^3 + 1

    Any genus is allowed::

        sage: HyperellipticCurveWithReduction(x^7 - x^5 + x + 1).genus()
        3
    """
    S = PolynomialRing(ZZ, 'x')
    try:
        f = S(f)
        h = S(h)
    except TypeError:
        raise TypeError("f and h must be polynomials with integer coefficients")
    F = 4 * f + h ** 2
    if F.degree() < 3:
        raise ValueError("4f + h^2 has degree %s; a hyperelliptic curve needs "
                         "degree at least 3" % F.degree())
    if F.discriminant() == 0:
        raise ValueError("4f + h^2 is not separable: the curve is singular")
    RQ = PolynomialRing(QQ, 'x')
    H = _sage_hyperelliptic_curve(RQ(f), RQ(h))
    H.__class__ = dynamic_class("HyperellipticCurveWithReduction",
                                (HyperellipticReduction,), type(H))
    H.f, H.h, H.F = f, h, F
    H.prec = ZZ(prec)
    H._cache = {}
    return H


#: Short alias for :func:`HyperellipticCurveWithReduction`, for interactive use.
ClusterCurve = HyperellipticCurveWithReduction


class HyperellipticReduction:
    r"""
    Reduction methods added to Sage's hyperelliptic curves by
    :func:`HyperellipticCurveWithReduction`.

    This is a mixin: instances are genuine Sage hyperelliptic curves, so every
    Sage method (``genus``, ``jacobian``, ``igusa_clebsch_invariants``,
    ``change_ring``, point construction, ...) remains available alongside the
    methods defined here.

    The extra attributes are ``f``, ``h`` and ``F = 4f + h^2`` as integral
    polynomials, ``prec`` (the `p`-adic working precision) and ``_cache``.
    """

    # -- basic objects ------------------------------------------------------

    def curve(self):
        r"""
        ``self``.  Kept so that code written against the old wrapper class
        still works; these objects *are* Sage hyperelliptic curves.
        """
        return self

    def square_model(self):
        r"""
        The isomorphic model `Y^2 = F(x)`, `F = 4f + h^2`, valid over
        `\ZZ[1/2]`.
        """
        RQ = PolynomialRing(QQ, 'x')
        return HyperellipticCurve(RQ(self.F))

    def to_square_model(self, P):
        r"""
        Send a point of `C` to the model `Y^2 = F(x)`, via `Y = 2y + h(x)`.

        INPUT:

        - ``P`` -- a pair ``(x0, y0)`` of rationals, or a point of
          ``self.curve()``.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 - x, x)
            sage: C.to_square_model((1, 0))
            (1, 1)
        """
        x0, y0 = self._coords(P)
        return (x0, 2 * y0 + self.f.parent()(self.h)(x0))

    def _coords(self, P):
        r"""
        Normalise a user supplied point to a pair of rationals, checking that
        it lies on `C`.  Points at infinity are rejected.
        """
        if isinstance(P, (tuple, list)) and len(P) == 2:
            x0, y0 = QQ(P[0]), QQ(P[1])
        else:
            # (X : Y : Z) on Sage's plane model, which is the degree deg(F)
            # homogenisation y^2 z^{d-2} + ... ; setting Z = 1 gives x = X/Z,
            # y = Y/Z.
            if len(tuple(P)) != 3:
                raise ValueError("a point must be given by 2 or 3 coordinates")
            if QQ(P[2]) == 0:
                raise NotImplementedError(
                    "points at infinity are not supported; translate the "
                    "model so that the points of interest are affine")
            x0, y0 = QQ(P[0]) / QQ(P[2]), QQ(P[1]) / QQ(P[2])
        RQ = PolynomialRing(QQ, 'x')
        if y0 ** 2 + RQ(self.h)(x0) * y0 != RQ(self.f)(x0):
            raise ValueError("the point (%s, %s) is not on %s" % (x0, y0, self))
        return (x0, y0)

    # -- global reduction data ---------------------------------------------

    def reduction_data(self):
        r"""
        Liu's reduction data for `C`, via ``genus2reduction``.
        """
        if self.genus() != 2:
            raise ValueError(
                "genus2reduction (Liu's algorithm) applies only to genus 2; "
                "this curve has genus %s.  Use the cluster picture methods, "
                "which work at every odd prime of tame reduction."
                % self.genus())
        if "reduction_data" not in self._cache:
            RQ = PolynomialRing(QQ, 'x')
            self._cache["reduction_data"] = genus2reduction(RQ(self.h), RQ(self.f))
        return self._cache["reduction_data"]

    def conductor(self):
        r"""
        The conductor of the Jacobian of `C`.
        """
        return ZZ(self.reduction_data().conductor)

    def minimal_discriminant(self):
        r"""
        The minimal discriminant of `C` (Liu's `\Delta_{\min}`).
        """
        return ZZ(self.reduction_data().minimal_disc)

    def bad_primes(self):
        r"""
        The primes of bad reduction of `C`, i.e. the primes dividing the
        minimal discriminant.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: HyperellipticCurveWithReduction(x^5 + x^3 + 1).bad_primes()
            [2, 53, 61]
        """
        if "bad_primes" not in self._cache:
            primes = None
            if self.genus() == 2:
                try:
                    primes = [q for q, _ in
                              ZZ(self.minimal_discriminant()).factor()]
                except Exception:
                    primes = None
            if primes is None:
                # No minimal discriminant available: start from the primes
                # dividing the discriminant of the model, a superset of the
                # bad primes, and sieve the odd ones with the cluster picture.
                D = self.F.discriminant() * self.F.leading_coefficient()
                primes = []
                for q, _ in ZZ(2 * D).factor():
                    if q == 2:
                        primes.append(q)      # undecidable here; keep it
                        continue
                    try:
                        if not self.cluster_picture(q).has_good_reduction(
                                Qp(q, self.prec)):
                            primes.append(q)
                    except NotImplementedError:
                        primes.append(q)      # wild: cannot rule it out
            self._cache["bad_primes"] = sorted(primes)
        return list(self._cache["bad_primes"])

    def local_data(self, p):
        r"""
        Liu's local reduction data at ``p``, parsed into a dictionary with keys
        ``stable_type``, ``nu_type``, ``conductor_exponent``, ``raw``.
        """
        p = ZZ(p)
        ld = self.reduction_data().local_data
        if p not in ld:
            return {"stable_type": None, "nu_type": "[I{0-0-0}]",
                    "conductor_exponent": ZZ(0), "raw": "good reduction"}
        return _parse_local_data(ld[p])

    # -- cluster pictures (odd p) -------------------------------------------

    def cluster_picture(self, p):
        r"""
        The cluster picture of `Y^2 = 4f + h^2` over `\QQ_p`, for odd ``p``.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
            sage: C.cluster_picture(53)
            Cluster with 5 roots and 4 children
        """
        p = ZZ(p)
        if p == 2:
            raise ValueError("cluster pictures require odd residue "
                             "characteristic; p = 2 is handled by Liu's "
                             "algorithm, see local_data(2)")
        key = ("cluster", p)
        if key not in self._cache:
            K = Qp(p, self.prec)
            RK = PolynomialRing(K, 'x')
            self._cache[key] = Cluster.from_polynomial(RK(self.F))
        return self._cache[key]

    # -- tameness -----------------------------------------------------------

    def _require_cluster_picture(self, p):
        r"""
        The cluster picture at ``p``, or a clear error.  This is the
        *operational* precondition for the reduction methods -- `p` odd and the
        splitting field computable -- as opposed to :meth:`is_tame`, which is a
        statement about the `\ell`-adic representation.
        """
        p = ZZ(p)
        if p == 2:
            raise NotImplementedError(
                "cluster pictures need odd residue characteristic")
        try:
            return self.cluster_picture(p)
        except NotImplementedError:
            raise NotImplementedError(
                "the splitting field of 4f + h^2 over Q_%s could not be built; "
                "sage_cluster_pictures refuses it as possibly wildly ramified, "
                "and is_tame(%s) = %s" % (p, p, self.is_tame(p)))

    def is_tame(self, p):
        r"""
        Whether the `\ell`-adic representation of `J = \mathrm{Jac}(C)` is
        tamely ramified at ``p``: the wild inertia subgroup `P \subset I_p`
        acts trivially on `V_\ell = T_\ell J \otimes \QQ_\ell` for
        `\ell \ne p`.  Equivalently the Swan conductor at `p` vanishes,
        equivalently `C` acquires semistable reduction over a *tamely* ramified
        extension of `\QQ_p`.

        Returns ``True``, ``False``, or ``None`` if undecided.

        The criteria used, in order:

        * If `C` is semistable at `p` then `\rho_\ell` is tame, at every `p`
          including 2: inertia acts unipotently, so its image is pro-`\ell`,
          while `P` is pro-`p`, and a pro-`p` group has trivial image in a
          pro-`\ell` group.  Good reduction is the special case where
          `\rho_\ell` is unramified.

        * For `p > 2g+1` the representation is always tame: `\QQ_p(J[2])` is
          the splitting field of `F`, whose Galois group embeds in `S_{2g+2}`,
          and such a `p` does not divide `(2g+2)!`.

        * If every irreducible factor `h` of `F` over `\QQ_p` has degree prime
          to `p`, the representation is tame: `e(\QQ_p[x]/(h))` divides
          `\deg h`, so each root generates a tamely ramified extension, and a
          compositum of tamely ramified extensions is tamely ramified, hence so
          is the splitting field `\QQ_p(J[2])`.

        * Otherwise, for odd `p`, the Swan conductor is computed directly from
          the cluster picture (``Cluster.n_wild``) and tested against 0.

        For odd `p` these all agree with the concrete test
        `p \nmid e(\QQ_p(R)/\QQ_p)`: taking `\ell = 2`, which is allowed since
        `p` is odd, tameness of `\rho_\ell` forces `\QQ_p(J[2]) = \QQ_p(R)` to
        be tame; conversely if `p \nmid e(\QQ_p(R))` then `C` is semistable
        over an extension of ramification `e` or `2e`, both prime to `p`.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^6 - 27)
            sage: C.is_tame(3), C.is_semistable_at(3)
            (True, False)

        Tame but not semistable, and tame at 2 because the reduction there is
        semistable::

            sage: HyperellipticCurveWithReduction(x^5 - x, x).is_tame(2)
            True
        """
        p = ZZ(p)
        if not p.is_prime():
            raise ValueError("%s is not prime" % p)

        # The conductor exponent of J bounds things both ways, and for genus 2
        # Liu gives it -- but NOT always at 2.  When PARI's genus2red cannot
        # determine the reduction at 2 it reports only the prime-to-2
        # conductor, and does so without any caveat, so v_2 of it is 0 by
        # construction rather than because J has good reduction there.  The
        # detectable signal is that the local data at 2 carries no
        # Namikawa-Ueno type and no conductor exponent; in that case f_2 is
        # simply unknown.  (Seen on 388.a.776.1, where PARI reports conductor
        # 97 while the true conductor is 388 = 4*97.)
        if self.genus() == 2:
            try:
                fp = ZZ(self.conductor()).valuation(p)
                if p == 2 and self.local_data(2)["conductor_exponent"] is None:
                    fp = None
            except Exception:
                fp = None
            if fp is not None:
                if fp == 0:
                    # J has good reduction, so rho_l is unramified
                    # (Neron-Ogg-Shafarevich), hence tame
                    return True
                if fp > 2 * self.genus():
                    # tame would force f_p = 2g - dim V^I <= 2g
                    return False

        # semistable (in particular good) reduction implies tame, for every p
        if self.genus() == 2 or p != 2:
            try:
                if self.is_semistable_at(p) is True:
                    return True
            except Exception:
                pass

        if p == 2:
            return None       # no criterion below applies in residue char 2

        if p > 2 * self.genus() + 1:
            return True

        K = Qp(p, self.prec)
        if all(ZZ(fac.degree()) % p != 0
               for fac, _ in PolynomialRing(K, 'x')(self.F).factor()):
            return True

        try:
            return ZZ(self.cluster_picture(p).n_wild()) == 0
        except (NotImplementedError, ValueError, ArithmeticError):
            return None

    def semistabilising_index(self, p):
        r"""
        A ramification index `m` such that `C` is semistable over a tamely
        ramified extension `L/\QQ_p` with `e(L/\QQ_p) = m`.

        Over `L \supseteq K(R)` conditions (1) and (2) of [DDMM, Thm 1.8] are
        automatic, and the depths and `\nu` scale by `e(L)`, so it suffices to
        take `m` a multiple of `e(K(R))` making `m d_\mathfrak{s} \in \ZZ` and
        `m \nu_\mathfrak{s} \in 2\ZZ` for every principal `\mathfrak{s}`.
        Returns 1 exactly when `C` is already semistable over `\QQ_p`.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: HyperellipticCurveWithReduction(x^6 - 27).semistabilising_index(3)
            4
            sage: HyperellipticCurveWithReduction(x^5 + x^3 + 1).semistabilising_index(53)
            1
        """
        p = ZZ(p)
        R = self._require_cluster_picture(p)
        if R.is_semistable(Qp(p, self.prec)):
            return ZZ(1)
        e = ZZ(R.roots()[0].parent().absolute_e())
        principal = [s for s in R.all_descendants() if s.is_principal()]
        k = ZZ(1)
        while True:
            m = e * k
            if all(m * s.depth() in ZZ and m * s.nu() / 2 in ZZ
                   for s in principal):
                return m
            k += 1

    # -- semistability ------------------------------------------------------

    def semistability_at(self, p):
        r"""
        Decide semistability of `C` at ``p``.

        OUTPUT:

        a dictionary with keys

        - ``prime``
        - ``semistable`` -- ``True``, ``False``, or ``None`` if undecided
        - ``method`` -- ``"cluster picture"``, ``"Liu/Namikawa-Ueno"`` or
          ``"good reduction"``
        - ``reason`` -- a human readable justification
        - plus, for odd ``p``, ``good`` (whether the reduction is good)

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: HyperellipticCurveWithReduction(x^5 - x, x).semistability_at(65563)["semistable"]
            True
        """
        p = ZZ(p)
        if not p.is_prime():
            raise ValueError("%s is not prime" % p)
        if p not in self.bad_primes():
            return {"prime": p, "semistable": True, "method": "good reduction",
                    "reason": "p does not divide the minimal discriminant",
                    "good": True}

        if self.genus() == 2:
            liu_ss, liu_reason = self._liu_semistability(p)
        else:
            liu_ss, liu_reason = None, None

        if p != 2:
            try:
                R = self.cluster_picture(p)
            except NotImplementedError:
                # ``Cluster.from_polynomial`` refuses wildly ramified
                # splitting fields.  Wild ramification means p | e, hence
                # e >= p >= 3 > 2, so [DDMM, Thm 1.8](1) already fails and the
                # curve is not semistable -- but the package's test for
                # wildness is only a sufficient condition for it, so we defer
                # to Liu's data instead of concluding.
                return {"prime": p, "semistable": liu_ss,
                        "method": "Liu/Namikawa-Ueno (cluster pictures "
                                  "unavailable: possibly wild extension)",
                        "reason": liu_reason, "liu": liu_ss,
                        "liu_reason": liu_reason}
            K = Qp(p, self.prec)
            ss = bool(R.is_semistable(K))
            reason = self._semistability_reason(R, K, ss)
            rec = {"prime": p, "semistable": ss, "method": "cluster picture",
                   "reason": reason, "good": bool(R.has_good_reduction(K)),
                   "liu": liu_ss, "liu_reason": liu_reason}
            if liu_ss is not None and liu_ss != ss:
                rec["warning"] = ("cluster pictures say semistable = %s but "
                                  "Liu's data say %s (%s)"
                                  % (ss, liu_ss, liu_reason))
            return rec

        # ---- p = 2: cluster pictures do not apply ------------------------
        if self.genus() != 2:
            return {"prime": p, "semistable": None, "method": "none",
                    "reason": "cluster pictures need odd residue "
                              "characteristic, and Liu's algorithm applies "
                              "only to genus 2 (this curve has genus %s)"
                              % self.genus()}
        return {"prime": p, "semistable": liu_ss, "method": "Liu/Namikawa-Ueno",
                "reason": liu_reason, "liu": liu_ss, "liu_reason": liu_reason}

    def _liu_semistability(self, p):
        r"""
        Decide semistability at ``p`` from Liu's reduction data alone.  Returns
        a pair ``(answer, reason)`` with ``answer`` in ``True``, ``False``,
        ``None``.

        Two ingredients are used.

        * Grothendieck's formula: for a semistable abelian variety the
          conductor exponent equals the toric rank, and the toric rank of the
          potential semi-abelian reduction is read off Liu's potential stable
          reduction type.  So `f_p \ne t` *proves* that `C` is not semistable
          at `p`.

        * The Namikawa--Ueno type of the special fibre of the minimal regular
          model at `p`: semistable means precisely that this fibre is reduced,
          i.e. that every component has multiplicity one, which happens exactly
          for the types built from `I_n` pieces (see
          :func:`nu_type_is_semistable`).
        """
        ld = self.local_data(p)
        t = potential_toric_rank(ld["stable_type"])
        fexp = ld["conductor_exponent"]
        nu = ld["nu_type"]

        if fexp is not None and t is not None and fexp != t:
            return (False,
                    "the potential stable reduction has type (%s), forcing "
                    "toric rank %s, while the conductor exponent is f_%s = %s; "
                    "a semistable abelian surface has f_p equal to its toric "
                    "rank (Grothendieck), so C is not semistable at %s"
                    % (ld["stable_type"], t, p, fexp, p))

        by_nu = nu_type_is_semistable(nu)
        if by_nu is None:
            extra = ""
            if fexp is not None and t is not None:
                extra = ("; the necessary condition f_%s = %s = toric rank "
                         "does hold" % (p, fexp))
            return (None,
                    "PARI's genus2red returned no Namikawa-Ueno type at %s "
                    "(its algorithm is incomplete at 2), and cluster pictures "
                    "need odd residue characteristic; potential stable "
                    "reduction type (%s)%s"
                    % (p, ld["stable_type"], extra))
        return (by_nu,
                "the special fibre of the minimal regular model has "
                "Namikawa-Ueno type %s, whose components %s multiplicity one"
                % (nu, "all have" if by_nu else "do not all have"))

    def _semistability_reason(self, R, K, ss):
        r"""
        Spell out which clause of [DDMM, Thm 1.8] fails (or that none does).
        """
        if ss:
            return ("all proper clusters are inertia stable, the splitting "
                    "field has e <= 2, and every principal cluster has "
                    "integral depth and even nu")
        bad = []
        for s in R.all_descendants():
            if s.is_principal():
                if s.depth() not in ZZ:
                    bad.append("principal cluster of size %s has depth %s "
                               "which is not an integer" % (s.size(), s.depth()))
                elif s.nu() / 2 not in ZZ:
                    bad.append("principal cluster of size %s has nu = %s "
                               "which is not even" % (s.size(), s.nu()))
        try:
            Kr = R.roots()[0].parent()
            e = Kr.absolute_e() / gcd(Kr.absolute_e(), K.absolute_e())
            if e > 2:
                bad.append("the splitting field of 4f + h^2 has ramification "
                           "degree %s > 2 over Q_p" % e)
            for s in R.all_descendants():
                if s.is_proper() and s.inertia() != s:
                    bad.append("the cluster of size %s is not inertia stable"
                               % s.size())
                    break
        except Exception as exc:
            bad.append("(while checking the splitting field: %s)" % exc)
        return "; ".join(bad) if bad else "not semistable"

    def is_semistable_at(self, p):
        r"""
        Whether `C` is semistable at ``p`` (``None`` if undecided).
        """
        return self.semistability_at(p)["semistable"]

    def is_semistable(self):
        r"""
        Whether `C` has semistable reduction at *every* prime.

        Returns ``True``, ``False``, or ``None`` if the answer is undecided at
        some prime (only possible at `p = 2`).

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: HyperellipticCurveWithReduction(x^5 - x, x).is_semistable()
            True
        """
        answers = [self.is_semistable_at(p) for p in self.bad_primes()]
        if any(a is False for a in answers):
            return False
        if any(a is None for a in answers):
            return None
        return True

    def semistability(self):
        r"""
        The dictionary ``p -> semistability_at(p)`` over all bad primes.
        """
        return {p: self.semistability_at(p) for p in self.bad_primes()}

    # -- components of the special fibre -----------------------------------

    def components(self, p, model="stable"):
        r"""
        The irreducible components of the special fibre at ``p``.

        INPUT:

        - ``p`` -- an odd prime of bad reduction
        - ``model`` -- ``"stable"`` (default) for the components of the stable
          model, i.e. the components `\Gamma_\mathfrak{s}` attached to the
          principal clusters; or ``"regular"`` for the special fibre of the
          minimal regular model with normal crossings, which additionally
          contains the `\mathbb{P}^1`'s of the linking chains.

        OUTPUT: a list of :class:`SpecialFibreComponent`.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
            sage: [(c.kind, c.genus) for c in C.components(53)]
            [('principal', 1)]
        """
        p = ZZ(p)
        if p == 2:
            raise NotImplementedError(
                "the components at p = 2 are not computed: cluster pictures "
                "need odd residue characteristic.  Liu's Namikawa-Ueno type "
                "at 2 is %r" % (self.local_data(2)["nu_type"],))
        ss = self.is_semistable_at(p)
        if ss is not True:
            raise ValueError("C is not semistable at %s, so the special fibre "
                             "is not a stable curve; the Namikawa-Ueno type is "
                             "%r" % (p, self.local_data(p)["nu_type"]))
        R = self.cluster_picture(p)
        if model == "stable":
            return self._principal_components(R)
        if model == "regular":
            return self._regular_components(R)
        raise ValueError('model must be "stable" or "regular"')

    def _cluster_names(self, R):
        r"""
        Assign short names to the clusters of ``R``: ``R`` for the top cluster,
        ``s1, s2, ...`` for the proper descendants in tree order.
        """
        names = {}
        i = 0
        for s in R.all_descendants():
            if not s.is_proper():
                continue
            if s.is_top_cluster():
                names[s] = "R"
            else:
                i += 1
                names[s] = "s%s" % i
        return names

    def _principal_components(self, R):
        names = self._cluster_names(R)
        out = []
        for s in R.all_descendants():
            if not s.is_principal():
                continue
            name = names[s]
            poly = self._component_polynomial(s)
            if s.is_ubereven():
                # Gamma_s : Y^2 = theta^2 G(X)^2 splits into two copies of P^1
                th = self._theta_bar(s)
                G = prod((PolynomialRing(poly.base_ring(), 'X').gen()
                          - self._red_cluster(s, c)) for c in s.children()
                         if c.is_twin() and c.relative_depth() == QQ(1) / 2)
                for eps in (1, -1):
                    if th is None:
                        eqn = ("Y = %ssqrt(%s) * (%s), conjugate over the "
                               "residue field"
                               % ("" if eps > 0 else "-",
                                  s.theta_squared().unit_part().residue(), G))
                    else:
                        eqn = "Y = %s" % (eps * th * G)
                    out.append(SpecialFibreComponent(
                        "Gamma_%s^%s" % (name, "+" if eps > 0 else "-"),
                        "principal", cluster=s, sign=eps, genus=ZZ(0),
                        equation=eqn))
            else:
                out.append(SpecialFibreComponent(
                    "Gamma_%s" % name, "principal", cluster=s, sign=None,
                    genus=ZZ(s.genus()), equation="Y^2 = %s" % poly))
        return out

    def _regular_components(self, R):
        r"""
        All components of the special fibre of the minimal regular model,
        read off :meth:`dual_graph`.
        """
        principal = {}
        for comp in self._principal_components(R):
            key = comp.cluster if comp.sign is None else (comp.cluster, comp.sign)
            principal[key] = comp
        out = []
        for v in R.dual_graph().vertices(sort=False):
            if v in principal:
                out.append(principal[v])
            else:
                out.append(SpecialFibreComponent(
                    self._chain_label(v, self._cluster_names(R)), "chain",
                    cluster=None, sign=None, genus=ZZ(0), equation=None))
        return out

    @staticmethod
    def _chain_label(v, names):
        r"""
        A printable name for a chain vertex of ``Cluster.dual_graph``.  Such a
        vertex is a tuple whose last entry is the position along the chain and
        whose earlier entries record which components the chain joins, each a
        cluster, a (cluster, sign) pair, or a bare sign.
        """
        parts = []
        for u in v[:-1]:
            if isinstance(u, Cluster):
                parts.append("Gamma_%s" % names.get(u, "?"))
            elif isinstance(u, tuple):
                parts.append("Gamma_%s^%s" % (names.get(u[0], "?"),
                                              "+" if u[1] > 0 else "-"))
            else:
                parts.append("+" if u > 0 else "-")
        return "L(%s)[%s]" % (", ".join(parts), v[-1])

    def special_fibre(self, p):
        r"""
        A dictionary describing the special fibre at the bad prime ``p``:
        the cluster picture, the components of the stable model, the
        components of the minimal regular model, the dual graph, the number of
        nodes, and the toric rank.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: d = HyperellipticCurveWithReduction(x^5 + x^3 + 1).special_fibre(53)
            sage: d["toric_rank"]
            1
        """
        p = ZZ(p)
        R = self.cluster_picture(p)
        out = {"prime": p,
               "cluster_picture": R,
               "cluster_picture_ascii": ascii_art(R),
               "stable_components": self.components(p, "stable"),
               "toric_rank": ZZ(R.potential_toric_rank()),
               "conductor_exponent": ZZ(R.conductor_exponent()),
               "namikawa_ueno_type": self.local_data(p)["nu_type"],
               "regular_components": None,
               "dual_graph": None,
               "nodes": None,
               "tamagawa_number": None,
               "note": None}
        try:
            G = R.dual_graph()
            out["dual_graph"] = G
            out["nodes"] = ZZ(len(G.edges(sort=False)))
            out["regular_components"] = self.components(p, "regular")
        except ArithmeticError as exc:
            out["note"] = ("the chains of the minimal regular model were not "
                           "determined: %s" % exc)
        try:
            out["tamagawa_number"] = ZZ(R.tamagawa_number())
        except Exception:
            pass
        return out

    # -- reduction of points -----------------------------------------------

    @staticmethod
    def _valuation(a):
        r"""
        The valuation of ``a``, normalised so that ``v(p) = 1``, for ``a`` in
        any `p`-adic field.
        """
        try:
            return a.normalized_valuation()
        except AttributeError:
            return QQ(a.valuation()) / a.parent().absolute_e()

    def _locate(self, R, x0):
        r"""
        Locate ``x0`` in the cluster picture ``R``.

        Walking down the cluster tree, return

        - ``("component", s)`` if ``x0`` lies in the disc of ``s`` and its
          image in `\Gamma_\mathfrak{s}` is distinct from the images of all
          proper children of ``s`` (so, if ``s`` is principal, a smooth point
          of `\Gamma_\mathfrak{s}`);
        - ``("link", s, c, m)`` if ``x0`` lies strictly between the discs of
          ``s`` and of its proper child ``c``, at level ``m``: the point
          reduces onto the chain joining the two, i.e. to a node of the stable
          model;
        - ``("outside", R)`` if ``x0`` lies outside the disc of the top
          cluster, so that the point reduces towards infinity on
          `\Gamma_R`.
        """
        v = self._valuation
        if v(x0 - R.center()) < R.depth():
            return ("outside", R)
        s = R
        while True:
            nxt, lev = None, None
            for c in s.children():
                if not c.is_proper():
                    continue
                m = v(x0 - c.center())
                if m > s.depth():
                    nxt, lev = c, m
                    break
            if nxt is None:
                return ("component", s)
            if lev >= nxt.depth():
                s = nxt
                continue
            return ("link", s, nxt, lev)

    # The reduction map onto a component.  We do not use ``Cluster.red`` for
    # points: at the time of writing it multiplies by `\pi^{\nu/2}` instead of
    # dividing, and computes the exponent as ``-s.size()//2``, which Python
    # evaluates as ``(-|s|)//2`` rather than ``-(|s|//2)`` -- wrong for odd
    # children of relative depth > 1/2.  The formulas below are derived from
    #
    #     F(x)/\pi^{\nu_s} = \bar\theta_s^2 \prod_{s' child}(X - red(s'))^{|s'|},
    #     X = (x - z_s)/\pi^{d_s},
    #
    # which follows from `\nu_s = v(\theta_s^2) + |s| d_s`; the component is
    # `\Gamma_s : Y^2 = \bar\theta_s^2 G_s(X)` with `G_s` the product of
    # `X - red(s')` over the odd children and `(X - red(s'))^2` over the twins
    # of relative depth 1/2, so `Y = (y/\pi^{\nu_s/2}) /
    # \prod_{\delta_{s'} > 1/2} (X - red(s'))^{\lfloor |s'|/2 \rfloor}`.

    @staticmethod
    def _pi_pow(Kr, t):
        r"""
        `\pi^{t}` in ``Kr``, where ``t`` is normalised so that `v(p) = 1`
        (i.e. the exponent of the uniformiser is ``t * e``).
        """
        n = QQ(t) * Kr.absolute_e()
        if n not in ZZ:
            raise ValueError("exponent %s is not in the value group of %s"
                             % (t, Kr))
        return Kr.uniformiser() ** ZZ(n)

    def _red_x(self, s, xK):
        r"""
        `red_\mathfrak{s}` of an element of the disc of ``s``: the residue of
        `(x - z_\mathfrak{s})/\pi^{d_\mathfrak{s}}`.
        """
        Kr = xK.parent()
        a = (xK - Kr(s.center())) / self._pi_pow(Kr, s.depth())
        if a.valuation() > 0:
            return Kr.residue_field()(0)
        return a.residue()

    def _red_cluster(self, s, c):
        r"""
        `red_\mathfrak{s}` of a child ``c`` of ``s``.
        """
        Kr = s.roots()[0].parent()
        return self._red_x(s, Kr(c.roots()[0]))

    def _component_polynomial(self, s):
        r"""
        The polynomial `\bar\theta_\mathfrak{s}^2 G_\mathfrak{s}(X)` cutting
        out `\Gamma_\mathfrak{s}`.
        """
        return s.component_polynomial()

    def _theta_bar(self, s):
        r"""
        The square root of `\bar\theta_\mathfrak{s}^2` in the residue field, or
        ``None`` when it is not a square there.  It is used only to label the
        two components `\Gamma_\mathfrak{s}^{\pm}` of an übereven cluster: they
        are `Y = \pm\bar\theta G(X)`.  If `\bar\theta^2` is not a square in the
        residue field the two components are conjugate over it and carry no
        rational point, so no point of `C(\QQ)` reduces onto them.
        """
        Kr = s.roots()[0].parent()
        th2 = Kr.residue_field()(s.theta_squared().unit_part().residue())
        return th2.sqrt() if th2.is_square() else None

    def _branch_value(self, s, X, Y):
        r"""
        For an übereven cluster, `\Gamma_\mathfrak{s} : Y^2 =
        \bar\theta^2 G(X)^2` splits as `Y = \pm\bar\theta G(X)` with
        `G = \prod (X - red(\mathfrak{s}'))` over the twins of relative depth
        1/2.  Return `Y/G(X) = \pm\bar\theta`: it is a residue field element
        (no square root needed), and two points lie on the same component
        precisely when their branch values agree.
        """
        G = prod(X - self._red_cluster(s, c) for c in s.children()
                 if c.is_twin() and c.relative_depth() == QQ(1) / 2)
        return Y / G

    def _reduce_affine(self, s, xK, yK):
        r"""
        Reduce `(x, y)` with `x` in the disc of the principal cluster ``s``
        onto `\Gamma_\mathfrak{s}`; returns the pair `(X, Y)` over the residue
        field.  Raises ``ArithmeticError`` if the point is not on the
        component.
        """
        try:
            pt = s.red((xK, yK))
        except ValueError as exc:
            raise ArithmeticError(str(exc))
        return (pt[0], pt[1])

    def _reduce_at_infinity(self, R, xK, yK):
        r"""
        Reduce a point whose `x`-coordinate lies *outside* the disc of the top
        cluster ``R``: it reduces to a point at infinity of `\Gamma_R`.

        In the chart `U = 1/X`, `V = Y/X^{\deg G/2}` at infinity one finds
        `V^2 = \bar\theta_R^2`, with

        .. MATH::

            V = \mathrm{res}\bigl(y \,/\, \pi^{v(c)/2} (x - z_R)^{N/2}\bigr),
            \qquad N = \deg F,

        when `N` is even (two points at infinity, `V = \pm\bar\theta_R`), while
        for `N` odd there is a single point at infinity.  Returns ``None`` for
        `N` odd, and `V` for `N` even.
        """
        N = self.F.degree()
        if N % 2:
            return None
        Kr = xK.parent()
        c = R.leading_coefficient()
        V = yK / (self._pi_pow(Kr, self._valuation(Kr(c)) / 2)
                  * (xK - Kr(R.center())) ** (N // 2))
        if V.valuation() != 0:
            raise ArithmeticError("V = %s is not a unit at infinity" % V)
        return V.residue()

    @staticmethod
    def _infinity_child(s):
        r"""
        If ``s`` is a non-principal top cluster, even with exactly two
        children of which one is not proper, return the other child.

        In that configuration the proper child is principal of odd size
        `2g+1`, so its component has odd degree and a *single* point at
        infinity, and the whole complement of its disc -- including the lone
        root outside it and `x = \infty` -- maps into that component rather
        than onto a linking chain.  (This is the same configuration in which
        :meth:`dual_graph` glues nothing over the top.)  Otherwise return
        ``None``.
        """
        if not (s.is_top_cluster() and not s.is_principal() and s.is_even()):
            return None
        ch = s.children()
        if len(ch) != 2:
            return None
        proper = [c for c in ch if c.is_proper()]
        if len(proper) != 1:
            return None
        c = proper[0]
        return c if (c.is_principal() and c.is_odd()) else None

    def reduce_point(self, p, P):
        r"""
        Reduce the point ``P`` into the special fibre of the stable model at
        the odd bad prime ``p``.

        The special fibre of the stable model is the union of the components
        `\Gamma_\mathfrak{s}` (or `\Gamma_\mathfrak{s}^{\pm}`) attached to the
        principal clusters, glued at nodes.  ``P`` reduces to a smooth point
        exactly when `x(P)` lands in the disc of a principal cluster
        `\mathfrak{s}` away from the discs of all proper children of
        `\mathfrak{s}` (or outside the top disc, with `R` principal, giving a
        point at infinity of `\Gamma_R`); otherwise `P` reduces to a node.

        INPUT:

        - ``p`` -- an odd prime of bad reduction at which `C` is semistable
        - ``P`` -- a pair ``(x0, y0)`` of rationals on `C`, or a point of
          ``self.curve()``

        OUTPUT: a :class:`PointReduction`.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
            sage: C.reduce_point(53, (0, 1)).is_smooth
            True
        """
        p = ZZ(p)
        if p == 2:
            raise NotImplementedError(
                "reduction of points at p = 2 is not implemented (cluster "
                "pictures need odd residue characteristic)")
        R0 = self._require_cluster_picture(p)
        semistable = bool(R0.is_semistable(Qp(p, self.prec)))
        x0, y0 = self.to_square_model(P)
        R = self.cluster_picture(p)
        Kr = R.roots()[0].parent()
        xK, yK = Kr(x0), Kr(y0)

        loc = self._locate(R, xK)
        names = self._cluster_names(R)
        raw = (xK, yK)
        comps = self._principal_components(R) if semistable else []

        def pick(s, sign=None):
            if not semistable:
                # Without semistability over Q_p there is no special fibre over
                # F_p to label; the components live over the residue field of
                # the semistabilising extension.
                return SpecialFibreComponent(
                    "Gamma_%s" % names[s], "principal", cluster=s, sign=sign,
                    genus=ZZ(s.genus()), equation=None)
            return [c for c in comps if c.cluster == s and c.sign == sign][0]

        if loc[0] == "link":
            _, s, c, m = loc
            return PointReduction(
                p, False,
                reason="the x-coordinate lies strictly between the discs of "
                       "the clusters %s (depth %s) and %s (depth %s), at "
                       "level %s: the point reduces onto the chain of P^1's "
                       "joining them, i.e. to a node of the stable model"
                       % (names[s], s.depth(), names[c], c.depth(), m))

        if loc[0] == "outside":
            if not R.is_principal():
                c = self._infinity_child(R)
                if c is not None:
                    comp = pick(c)
                    r = PointReduction(p, True, component=comp,
                                       point="infinity", cluster=c)
                    r._raw = raw
                    return r
                return PointReduction(
                    p, False,
                    reason="the x-coordinate lies outside the disc of the top "
                           "cluster, which is not principal: the region at "
                           "infinity is part of a linking chain, so the point "
                           "reduces to a node of the stable model")
            try:
                V = self._reduce_at_infinity(R, xK, yK)
            except ArithmeticError as exc:
                return PointReduction(
                    p, False,
                    reason="could not reduce to infinity on Gamma_R: %s" % exc)
            if R.is_ubereven() and semistable:
                # V = +- theta, the same branch invariant as in the affine case
                comp = self._ubereven_component(R, V, comps)
                r = PointReduction(p, True, component=comp,
                                   point="infinity", cluster=R, y_datum=V)
                r._raw = raw
                return r
            comp = pick(R)
            r = PointReduction(p, True, component=comp, point="infinity",
                               cluster=R)
            r._raw = raw
            return r

        s = loc[1]
        if not s.is_principal():
            c = self._infinity_child(s)
            if c is not None:
                # the complement of c's disc maps to the single point at
                # infinity of Gamma_c, not onto a chain
                comp = pick(c)
                r = PointReduction(p, True, component=comp, point="infinity",
                                   cluster=c)
                r._raw = raw
                return r
            kind = ("twin" if s.is_twin()
                    else "cotwin" if s.is_cotwin()
                    else "not proper" if not s.is_proper()
                    else "non-principal")
            return PointReduction(
                p, False,
                reason="the x-coordinate lands in the %s cluster %s of size "
                       "%s: the corresponding piece of the special fibre is a "
                       "chain of P^1's, contracted to a node in the stable "
                       "model"
                       % (kind, names[s], s.size()))

        if not semistable:
            # C is tame but not semistable over Q_p: it becomes semistable over
            # a tamely ramified L, and the stable model lives there.  The
            # cluster combinatorics, and hence which component a point lands
            # on, are unchanged; only the depths and nu scale by e(L).  The
            # x-coordinate of the reduction is still computable over Q_p, since
            # red_s divides by an element of valuation exactly d_s.  The
            # y-coordinate is not (it needs pi^(nu/2), which may not exist over
            # Q_p), but it is never needed on its own -- only branch
            # *comparisons* between two points, in which that factor cancels.
            X = self._red_x(s, xK)
            r = PointReduction(p, True, component=pick(s), point=(X, None),
                               cluster=s)
            r._raw = raw
            return r

        try:
            X, Y = self._reduce_affine(s, xK, yK)
        except ArithmeticError as exc:
            return PointReduction(p, False,
                                  reason="could not reduce onto Gamma_%s: %s"
                                         % (names[s], exc))

        if s.is_ubereven():
            b = self._branch_value(s, X, Y)
            comp = self._ubereven_component(s, b, comps)
            r = PointReduction(p, True, component=comp, point=(X, Y),
                               cluster=s, y_datum=b)
            r._raw = raw
            return r
        r = PointReduction(p, True, component=pick(s), point=(X, Y), cluster=s)
        r._raw = raw
        return r

    def _ubereven_component(self, s, branch, comps):
        r"""
        Which of `\Gamma_\mathfrak{s}^{\pm}` a point with branch value
        ``branch`` `= \pm\bar\theta_\mathfrak{s}` lies on.
        """
        th = self._theta_bar(s)
        if th is None:
            raise ArithmeticError(
                "theta^2 = %s is not a square in the residue field, so "
                "Gamma_s^+ and Gamma_s^- are conjugate and carry no rational "
                "point, yet a point reduced onto them"
                % s.theta_squared().unit_part().residue())
        sign = 1 if branch == th else -1
        return [c for c in comps if c.cluster == s and c.sign == sign][0]

    def same_component(self, P, Q, p):
        r"""
        At the odd bad prime ``p``, decide whether `P` and `Q` reduce to smooth
        points of the *same* irreducible component of the special fibre of the
        stable model.

        OUTPUT: a dictionary with keys ``prime``, ``answer`` (a boolean),
        ``P``, ``Q`` (the two :class:`PointReduction`'s) and ``reason``.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
            sage: C.same_component((0, 1), (0, -1), 53)["answer"]
            True
        """
        p = ZZ(p)
        rP = self.reduce_point(p, P)
        rQ = self.reduce_point(p, Q)
        if not rP.is_smooth:
            return {"prime": p, "answer": False, "P": rP, "Q": rQ,
                    "reason": "P does not reduce to a smooth point: %s"
                              % rP.reason}
        if not rQ.is_smooth:
            return {"prime": p, "answer": False, "P": rP, "Q": rQ,
                    "reason": "Q does not reduce to a smooth point: %s"
                              % rQ.reason}
        if rP.cluster != rQ.cluster:
            return {"prime": p, "answer": False, "P": rP, "Q": rQ,
                    "reason": "P reduces to %s and Q reduces to %s"
                              % (rP.component.label, rQ.component.label)}
        if rP.cluster.is_ubereven():
            # Same übereven cluster: Gamma_s = Gamma_s^+ u Gamma_s^-, the two
            # branches Y = +- theta*G(X).  Decide whether the branches agree.
            same = self._same_branch(rP.cluster, rP, rQ)
            lab = rP.component.label.rsplit("^", 1)[0]
            if same is None:
                return {"prime": p, "answer": None, "P": rP, "Q": rQ,
                        "reason": "both reduce onto the übereven pair %s^+, "
                                  "%s^-, but C is not semistable at %s and one "
                                  "point is at infinity while the other is "
                                  "affine, so the two branch invariants cannot "
                                  "be compared without the semistabilising "
                                  "extension" % (lab, lab, p)}
            return {"prime": p, "answer": same, "P": rP, "Q": rQ,
                    "reason": "both reduce onto the übereven cluster's pair of "
                              "components %s^+, %s^-, on the %s branch"
                              % (lab, lab, "same" if same else "opposite")}
        return {"prime": p, "answer": True, "P": rP, "Q": rQ,
                "reason": "both reduce to smooth points of %s"
                          % rP.component.label}

    def _same_branch(self, s, rP, rQ):
        r"""
        Do two smooth reductions onto an übereven `\Gamma_\mathfrak{s}` lie on
        the same one of `\Gamma_\mathfrak{s}^{\pm}`?

        When `C` is semistable over `\QQ_p` both carry an absolute branch
        invariant `\pm\bar\theta_\mathfrak{s}` and we compare those.

        Otherwise the invariant `b = Y/G(X)` involves `\pi^{\nu/2}`, which need
        not exist over `\QQ_p`; but that factor is common to both points, so it
        cancels from the comparison.  For two affine points

        .. MATH::

            b_P = b_Q \iff \mathrm{res}(y_P/y_Q)\, G(X_Q) = G(X_P),

        and for two points at infinity, where `b = \mathrm{res}\bigl(y /
        \pi^{v(c)/2}(x - z_R)^{N/2}\bigr)`,

        .. MATH::

            b_P = b_Q \iff
            \mathrm{res}\bigl((y_P/y_Q)((x_Q - z_R)/(x_P - z_R))^{N/2}\bigr) = 1.

        Returns ``None`` for the one case this leaves open: `C` not semistable
        over `\QQ_p` with one point affine and the other at infinity.
        """
        if rP._y_datum is not None and rQ._y_datum is not None:
            return bool(rP._y_datum == rQ._y_datum)
        xP, yP = rP._raw
        xQ, yQ = rQ._raw
        infP = (rP.point == "infinity")
        infQ = (rQ.point == "infinity")
        if infP != infQ:
            return None
        if infP:
            N = self.F.degree()
            if N % 2:
                return True             # a single point at infinity
            z = xP.parent()(s.center())
            t = (yP / yQ) * ((xQ - z) / (xP - z)) ** (N // 2)
            return bool(t.residue() == 1)
        twins = [c for c in s.children()
                 if c.is_twin() and c.relative_depth() == QQ(1) / 2]
        XP, XQ = self._red_x(s, xP), self._red_x(s, xQ)
        GP = prod(XP - self._red_cluster(s, c) for c in twins)
        GQ = prod(XQ - self._red_cluster(s, c) for c in twins)
        return bool((yP / yQ).residue() * GQ == GP)

    def same_component_everywhere(self, P, Q):
        r"""
        Run :meth:`same_component` at every bad prime.

        OUTPUT: a dictionary ``p -> record``, where the record for `p = 2` (and
        for any prime where the computation is not available) has
        ``answer = None``.
        """
        out = {}
        for p in self.bad_primes():
            if p == 2:
                out[p] = {"prime": p, "answer": None, "P": None, "Q": None,
                          "reason": "not computed: cluster pictures need odd "
                                    "residue characteristic"}
                continue
            try:
                self._require_cluster_picture(p)
            except NotImplementedError as exc:
                out[p] = {"prime": p, "answer": None, "P": None, "Q": None,
                          "reason": str(exc)}
                continue
            out[p] = self.same_component(P, Q, p)
        return out

    # -- reporting ---------------------------------------------------------

    def report(self, P=None, Q=None):
        r"""
        Print everything: the bad primes, semistability at each of them, the
        components of the special fibre, and (if ``P`` and ``Q`` are given)
        whether `P` and `Q` reduce to a smooth point of the same component.

        EXAMPLES::

            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: HyperellipticCurveWithReduction(x^5 - x, x).report(P=(0, 0), Q=(1, 0))  # random
        """
        print(self)
        print("  4f + h^2 = %s" % self.F)
        try:
            print("  conductor            = %s" % factor(self.conductor()))
            print("  minimal discriminant = %s" % factor(self.minimal_discriminant()))
        except Exception as exc:
            print("  (Liu's algorithm failed: %s)" % exc)
        bad = self.bad_primes()
        print("  bad primes           = %s" % bad)

        ss = self.is_semistable()
        print("")
        print("Semistable at all primes: %s" % ss)
        for p in bad:
            d = self.semistability_at(p)
            print("  p = %-8s semistable: %-5s  [%s]" % (p, d["semistable"],
                                                         d["method"]))
            print("      %s" % d["reason"])

        if ss is False:
            print("")
            print("Not semistable everywhere: skipping the special fibres.")
        else:
            for p in bad:
                print("")
                print("-" * 70)
                print("Special fibre at p = %s" % p)
                if p == 2:
                    ld = self.local_data(2)
                    print("  not computed (cluster pictures need p odd).")
                    print("  Liu: potential stable reduction type (%s), "
                          "Namikawa-Ueno type %s"
                          % (ld["stable_type"], ld["nu_type"]))
                    continue
                if self.is_semistable_at(p) is not True:
                    print("  C is not semistable at %s." % p)
                    continue
                d = self.special_fibre(p)
                print("  cluster picture:      %s"
                      % d["cluster_picture_ascii"])
                print("  Namikawa-Ueno type:   %s (Liu)"
                      % d["namikawa_ueno_type"])
                print("  toric rank %s, conductor exponent %s, "
                      "Tamagawa number %s"
                      % (d["toric_rank"], d["conductor_exponent"],
                         d["tamagawa_number"]))
                print("  components of the stable model (%s):"
                      % len(d["stable_components"]))
                for c in d["stable_components"]:
                    print("    %s" % c)
                if d["regular_components"] is None:
                    print("  minimal regular model: %s" % d["note"])
                else:
                    print("  components of the minimal regular model "
                          "(%s components, %s nodes):"
                          % (len(d["regular_components"]), d["nodes"]))
                    for c in d["regular_components"]:
                        print("    %s" % c)

        if P is None or Q is None:
            return
        print("")
        print("=" * 70)
        print("Reduction of P = %s and Q = %s" % (self._coords(P),
                                                  self._coords(Q)))
        for p, rec in self.same_component_everywhere(P, Q).items():
            print("  p = %-8s same smooth component: %s" % (p, rec["answer"]))
            print("      %s" % rec["reason"])
            for nm in ("P", "Q"):
                if rec[nm] is not None:
                    print("      %s -> %s" % (nm, rec[nm]))


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------

def genus2_semistable_report(f, h=0, P=None, Q=None, prec=250):
    r"""
    One shot: build the curve `y^2 + h y = f`, check semistability everywhere,
    describe the special fibres, and compare the reductions of ``P`` and ``Q``.

    EXAMPLES::

        sage: from hyperell_regulator.reduction import genus2_semistable_report
        sage: R.<x> = ZZ[]
        sage: C = genus2_semistable_report(x^5 - x, x, P=(0, 0), Q=(1, 0))  # random
    """
    C = HyperellipticCurveWithReduction(f, h, prec=prec)
    C.report(P=P, Q=Q)
    return C


def _liu_component_group_order(raw):
    r"""
    The order of Liu's group of connected components of the Néron model over
    `\bar\GF{p}`, parsed out of one entry of ``local_data``.  It is printed
    between the page reference and ``f=`` as a product of cyclic factors
    ``(n)``, possibly with exponents ``(n)^k``; ``Hn`` denotes a group of
    order 4.  Returns ``None`` if absent.
    """
    m = re.search(r"page[^,]*,\s*(.*?),\s*f\s*=", raw, re.S)
    if not m:
        return None
    seg = m.group(1)
    if "H" in seg:
        return ZZ(4)
    order, found = ZZ(1), False
    for mm in re.finditer(r"\((\d+)\)(?:\^(\d+))?", seg):
        found = True
        n = max(ZZ(1), ZZ(mm.group(1)))
        k = ZZ(mm.group(2)) if mm.group(2) else ZZ(1)
        order *= n ** k
    return order if found else None


def self_test(verbose=True):
    r"""
    Consistency checks against Liu's independent implementation
    (``genus2reduction``, i.e. PARI's ``genus2red``).

    For every odd prime of bad reduction of a list of test curves we check:

    * the cluster picture criterion for semistability agrees with what Liu's
      data force (see :meth:`HyperellipticCurveWithReduction._liu_semistability`);
    * the conductor exponent computed from the cluster picture equals Liu's;
    * the special fibre satisfies `\sum_i g(\Gamma_i) + b_1(\Gamma) = 2` and
      `b_1(\Gamma) = f_p`;
    * the number of spanning trees of the dual graph equals the order of Liu's
      group of connected components of the Néron model over `\bar\GF{p}` --
      this pins down the *lengths* of the linking chains, not merely the
      topology;
    * every point reduction lands on its component (checked inside
      :meth:`HyperellipticCurveWithReduction._reduce_affine`), and ``same_component`` is reflexive
      and symmetric.

    EXAMPLES::

        sage: from hyperell_regulator.reduction import self_test
        sage: self_test(verbose=False)
        True
    """
    S = PolynomialRing(ZZ, 'x')
    x = S.gen()
    tests = [
        (x**5 + x**3 + 1, S(0)),
        (x**6 + x**2 + 2, S(0)),
        (x**5 - x, x),
        (x**5 + 2*x**4 + 4, x + 1),
        (x**5 + 8, S(0)),
        (x**6 - 16, S(0)),
        (x*(x - 1)*(x - 2)*(x - 3)*(x - 4), S(0)),
        ((x**2 - 9)*(x**2 - 25)*(x**2 - 49), S(0)),
        ((x**2 - 5**4)*(x + 1)*(x + 2)*(x + 3), S(0)),
        ((x**4 - 5**8)*(x**2 + 2*x + 1 - 5**2), S(0)),
        (2*x**6 - 3*x**5 - 3*x**4 - 2*x**3 + 5*x + 1, S(2)),
        # two triples far apart: two genus 1 components and a chain
        (x*(x - 3**4)*(x + 3**4)*(x - 1)*(x - 1 - 3**4)*(x - 1 + 3**4), S(0)),
        # a long chain
        ((x - 1)*(x - 2)*(x - 3)*(x - 5**2)*(x - 5**5)*(x + 5**5), S(0)),
        # cotwin top cluster (child of size 2g)
        ((x - 1)*x*(x - 5**2)*(x - 2*5**2)*(x - 3*5**2), S(0)),
        ((x - 1)*x*(x - 5**4)*(x - 2*5**4)*(x - 3*5**4), S(0)),
        # cotwin that is not the top cluster
        ((x - 1)*(x - 5**2)*x*(x - 5**6)*(x - 2*5**6)*(x - 3*5**6), S(0)),
        ((x - 1)*(x - 5**4)*x*(x - 5**6)*(x - 2*5**6)*(x - 3*5**6), S(0)),
    ]
    failures = []
    for (f, h) in tests:
        C = HyperellipticCurveWithReduction(f, h, prec=300)
        for p in C.bad_primes():
            if p == 2:
                continue
            rec = C.semistability_at(p)
            if rec.get("liu") is not None and rec["liu"] != rec["semistable"]:
                failures.append((f, h, p, "semistability disagrees with Liu"))
            if rec["semistable"] is not True:
                continue
            d = C.special_fibre(p)
            fl = C.local_data(p)["conductor_exponent"]
            if fl is not None and d["conductor_exponent"] != fl:
                failures.append((f, h, p, "conductor exponent %s vs Liu %s"
                                 % (d["conductor_exponent"], fl)))
            G = d["dual_graph"]
            if G is None:
                continue
            nv = len(G.vertices(sort=False))
            ne = len(G.edges(sort=False))
            b1 = ne - nv + G.connected_components_number()
            gsum = sum(c.genus for c in d["regular_components"])
            if gsum + b1 != 2:
                failures.append((f, h, p, "genus formula: %s + %s != 2"
                                 % (gsum, b1)))
            if b1 != d["conductor_exponent"]:
                failures.append((f, h, p, "b_1 = %s but f_p = %s"
                                 % (b1, d["conductor_exponent"])))
            order = _liu_component_group_order(C.local_data(p)["raw"])
            if order is not None:
                H = Graph(multiedges=True)
                H.add_vertices(G.vertices(sort=False))
                H.add_edges([e[:2] for e in G.edges(sort=False)
                             if e[0] != e[1]])
                ntrees = (ZZ(1) if len(H.vertices(sort=False)) == 1
                          else ZZ(H.spanning_trees_count()))
                if ntrees != order:
                    failures.append((f, h, p, "component group %s vs Liu %s"
                                     % (ntrees, order)))
            # points
            RQ = PolynomialRing(QQ, 'x')
            fQ, hQ = RQ(f), RQ(h)
            pts = []
            for x0 in [QQ(a) for a in range(-6, 7)]:
                disc = hQ(x0) ** 2 + 4 * fQ(x0)
                if disc >= 0 and QQ(disc).is_square():
                    r = QQ(disc).sqrt()
                    for y0 in {(-hQ(x0) + r) / 2, (-hQ(x0) - r) / 2}:
                        pts.append((x0, y0))
            pts = pts[:6]
            for A in pts:
                rA = C.reduce_point(p, A)
                if rA.is_smooth and \
                        C.same_component(A, A, p)["answer"] is not True:
                    failures.append((f, h, p, "same_component not reflexive "
                                              "at %s" % (A,)))
                for B in pts:
                    if (C.same_component(A, B, p)["answer"]
                            != C.same_component(B, A, p)["answer"]):
                        failures.append((f, h, p, "same_component asymmetric "
                                                  "at %s, %s" % (A, B)))
        if verbose:
            print("checked %s" % HyperellipticCurveWithReduction(f, h))
    if verbose:
        print("")
        print("%s failure(s)" % len(failures))
        for t in failures:
            print("   ", t)
    return not failures


