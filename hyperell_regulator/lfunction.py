r"""
The standard `L`-function of a genus 2 curve over `\QQ`.

For a genus 2 curve `C/\QQ` the *standard* motive is the degree 5 piece

.. MATH::

    \mathrm{std} = \ker\bigl(\wedge^2 h^1(C) \to h^2(C) = \QQ(-1)\bigr),

so that `L(\wedge^2 h^1, s) = L(\mathrm{std}, s)\,\zeta(s-1)`.  This is the
`L`-function that appears in Beilinson's conjecture for `K_2` of a genus 2
curve, and the one :mod:`hyperell_regulator.experiments.beilinson` compares
the regulator against.

What is computed here
=====================

* :func:`wedge2_euler_factor` -- the degree 6 Euler factor of `\wedge^2 h^1`
  from the degree 4 Euler factor of `h^1`;

* :class:`CurveLFunction` -- `L(C,s)` itself, degree 4; and
  :class:`StandardLFunction` -- the Euler factors (good ones batched through
  :mod:`~hyperell_regulator.frobenius`, bad ones from the cluster picture of
  :mod:`hyperell_regulator.reduction`, which supplies the monodromy
  filtration that the `h^1` factor alone does not determine), the conductor,
  the Dirichlet coefficients, a numerical check of the functional equation,
  and values and derivatives at `s = 1`;

* :func:`standard_lvalue` -- the convenience wrapper returning
  `L''(\mathrm{std}, 1)` together with everything that went into it.

EXAMPLES::

    sage: from hyperell_regulator.lfunction import StandardLFunction
    sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
    sage: R.<x> = ZZ[]
    sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
    sage: L = StandardLFunction(C, bad_factors={2: None})  # doctest: +ELLIPSIS
    ...Warning: unknown valuation of conductor at 2.
    sage: L.euler_factor(53).degree()
    3

:func:`standard_lvalue` runs the whole computation -- Euler factors,
conductor, functional equation check, `L''(\mathrm{std}, 1)` -- in one call,
and is what the experiments use.
"""

from sage.all import (
    prime_range,
    GF,
    HyperellipticCurve,
    Matrix,
    PolynomialRing,
    PowerSeriesRing,
    QQ,
    RR,
    SageObject,
    ZZ,
    gp,
    identity_matrix,
    pari,
)

from hyperell_regulator.reduction import potential_toric_rank


def _bits(digits):
    r"""
    Bits for a PARI call asked for ``digits`` decimals, with room to spare.

    ``pari.set_real_precision`` does **not** reach ``lfun`` or
    ``lfuncheckfeq``: with it alone they answer at 64 bits whatever it is set
    to, so a value asked for at 25 or 50 digits came back with 19 correct
    ones and noise after them, and the functional equation checked to
    `3.6\cdot10^{-15}` and no better however much precision was requested.
    The precision has to be passed to the call itself, which is what this is
    for.  (The same is true of ``lfuncost``, and
    :meth:`StandardLFunction._coeffs_needed` says so.)
    """
    return int(3.33 * int(digits)) + 64


def _real_field(digits):
    r"""
    A real field wide enough for ``digits`` decimal digits.

    ``RR`` is 53 bits whatever PARI's working precision is set to, so
    returning ``RR(pari.lfun(...))`` silently caps every value at about
    fifteen digits -- which is not what "arbitrary precision" means.

    EXAMPLES::

        sage: from hyperell_regulator.lfunction import _real_field
        sage: _real_field(30).precision() >= 100
        True
    """
    from sage.all import RealField

    return RealField(int(RR(digits) * RR(10).log2()) + 12)


def wedge2_euler_factor(P):
    r"""
    The degree 6 Euler factor of `\wedge^2 h^1` from the degree 4 Euler factor
    ``P`` of `h^1`.

    If `P(T) = \prod_{i=1}^4 (1 - \alpha_i T) = 1 + c_1T + c_2T^2 + c_3T^3 +
    c_4T^4`, so that `c_1 = -s_1`, `c_2 = s_2`, `c_3 = -s_3`, `c_4 = s_4` in the
    elementary symmetric functions of the `\alpha_i`, then
    `\prod_{i<j}(1 - \alpha_i\alpha_j T)` has coefficients

    .. MATH::

        d_1 = -s_2,\quad d_2 = s_1s_3 - s_4,\quad
        d_3 = -(s_1^2s_4 - 2s_2s_4 + s_3^2),\quad
        d_4 = s_1s_3s_4 - s_4^2,\quad d_5 = -s_2s_4^2,\quad d_6 = s_4^3.

    EXAMPLES:

    The identity is checked symbolically against the definition::

        sage: from hyperell_regulator.lfunction import wedge2_euler_factor
        sage: S = PolynomialRing(QQ, ['a1','a2','a3','a4'])
        sage: a1, a2, a3, a4 = S.gens()
        sage: RT = PolynomialRing(S, 'T'); T = RT.gen()
        sage: P = prod(1 - a*T for a in (a1, a2, a3, a4))
        sage: W = prod(1 - a*b*T for a, b in Combinations([a1,a2,a3,a4], 2))
        sage: wedge2_euler_factor(P) == W
        True
    """
    RT = P.parent()
    T = RT.gen()
    c = [P[i] for i in range(5)]
    s1, s2, s3, s4 = -c[1], c[2], -c[3], c[4]
    d = [1,
         -s2,
         s1 * s3 - s4,
         -(s1 ** 2 * s4 - 2 * s2 * s4 + s3 ** 2),
         s1 * s3 * s4 - s4 ** 2,
         -s2 * s4 ** 2,
         s4 ** 3]
    return sum(d[i] * T ** i for i in range(7))


class StandardLFunction(SageObject):
    r"""
    The *standard* `L`-function of a genus 2 curve `C/\QQ`: the degree 5 motive

    .. MATH::

        \mathrm{std} = \ker\bigl(\wedge^2 h^1(C) \to h^2(C) = \QQ(-1)\bigr),

    so that `L(\wedge^2 h^1, s) = L(\mathrm{std}, s)\,\zeta(s-1)` and, locally,
    `P_p^{\wedge^2}(T) = P_p^{\mathrm{std}}(T)\,(1 - pT)` -- the `\QQ(-1)`
    summand being unramified everywhere with Frobenius `p`.

    Euler factors
    =============

    The `h^1` factors `P_p(T) = \det(1 - \mathrm{Frob}_p T \mid V_\ell^{I_p})`
    come from PARI's ``lfungenus2``, at good and bad primes alike.

    At a **good** prime `P_p` has degree 4 and `P_p^{\wedge^2}` is its
    alternating square, :func:`wedge2_euler_factor`.

    At a **bad** prime this is not enough: `(\wedge^2 V)^{I} \ne \wedge^2(V^I)`,
    so the `h^1` factor alone does not determine the `\wedge^2` one.  What is
    missing is the monodromy filtration, and that is what the cluster picture
    supplies.  For `C` semistable at `p` write `t` for the toric rank, so that
    `V_\ell` has graded pieces `U` (weight 0, dimension `t`), `B` (weight 1,
    dimension `2a`, `a = 2 - t`) and `U' \cong U^*(-1)` (weight 2), with the
    monodromy `N` killing `U \oplus B` and mapping `U'` isomorphically onto
    `U`.  Then `V^I = \ker N = U \oplus B`, and
    `(\wedge^2 V)^I = \ker N_{\wedge^2}` works out as follows.

    * `t = 0`: `A` has good reduction, `V^I = V`, and the alternating square is
      the full degree 6 factor.

    * `t = 1`: `\ker N_{\wedge^2} = (U \wedge B) \oplus \wedge^2 B \oplus
      (U \wedge U')` has dimension `2 + 1 + 1 = 4`, giving

      .. MATH::

          P_p^{\wedge^2}(T) = A(\gamma T)\,(1 - b_2 T)\,(1 - \gamma^2 p T),

      where `U(T) = 1 - \gamma T` is the toric factor, `A(T) = 1 + b_1T + b_2T^2`
      the abelian one (so `b_2 = p`), and `\gamma = \pm 1`.

    * `t = 2`: `\ker N_{\wedge^2} = \wedge^2 U \oplus
      \ker(U \otimes U' \to \wedge^2 U)` has dimension `1 + 3 = 4`, giving

      .. MATH::

          P_p^{\wedge^2}(T) = (1 - u_2 T)(1 - pT)^2(1 - u_2 p T),
          \qquad u_2 = \det(\mathrm{Frob} \mid U) = \pm 1.

    The toric factor is read off the Frobenius action on
    `H_1` of the dual graph, via ``Cluster.homology_of_special_fibre``; the
    abelian factor is then `P_p / U(T)`.

    Tame but not semistable: potentially good reduction
    ===================================================

    If `C` is tame at `p` but not semistable, `V_\ell` still becomes
    unramified-up-to-monodromy only over the minimal semistabilising extension
    `K`, of degree `e` prime to `p`.  Inertia acts through the finite cyclic
    quotient `\Phi = I_p/I_K`, and being of order prime to `p` it acts
    *semisimply*.

    When the potential stable reduction has **toric rank 0** -- Liu's types
    (I) and (V), i.e. `J` acquires good reduction over `K` -- the monodromy `N`
    vanishes, so `V^{I_K} = V` and `V^{I_p} = V^{\Phi}`.  Writing
    `V = V^{I} \perp V'` for the splitting into the trivial and the non-trivial
    `\Phi`-isotypic parts (orthogonal for the Weil pairing, since `\Phi`
    preserves it), the cross terms carry no invariants because `(V')^{I} = 0`:

    .. MATH::

        (\wedge^2 V)^{I} = \wedge^2\bigl(V^{I}\bigr) \oplus
                           \bigl(\wedge^2 V'\bigr)^{I}.

    Two sub-cases are then completely determined by `P_p` alone.

    * `f_p = 0`: `V' = 0`, `V^I = V`, and the factor is the full degree 6
      alternating square, as in the semistable `t = 0` case.

    * `f_p = 2`: `\dim V' = 2`, so `(\wedge^2 V')^{I} = \wedge^2 V'` is a line
      on which Frobenius acts by `\det(F \mid V)/\det(F \mid V^I) = p^2/d`,
      where `d = \det(F \mid V^I)` is the leading coefficient of `P_p`.  The
      polarisation is nondegenerate on each summand, so the class `\omega` of
      `\QQ(-1)` is `\omega_1 + \omega_2` with both parts non-zero; being a
      Frobenius eigenvector of eigenvalue `p` forces `d = p`.  Hence

      .. MATH::

          P_p^{\wedge^2}(T) = (1 - pT)^2, \qquad
          P_p^{\mathrm{std}}(T) = 1 - pT.

    `f_p` odd cannot occur here: `\Phi` acts trivially on
    `\wedge^4 V = \QQ_\ell(-2)`, so `\det(\sigma \mid V') = 1`, and `V'` is
    symplectic, hence even dimensional.  `f_p = 4` is not implemented: with
    `V^I = 0` the answer depends on the eigenvalues of `\Phi` on `V`, which
    `P_p = 1` does not record.

    Toric rank `> 0` together with `e > 1` -- potentially multiplicative but
    not semistable, Liu's types (II), (III), (IV), (VI), (VII) with a non-trivial
    `\Phi` -- needs `(\ker N_{\wedge^2})^{\Phi}` and is not implemented either.

    Assumptions, all checked at runtime
    ===================================

    * `p` odd, and `C` either semistable at `p` or tame with potentially good
      reduction -- otherwise the prime must be supplied in ``bad_factors``.  In
      particular `p = 2` always needs an override, cluster pictures needing odd
      residue characteristic.
    * Frobenius acts on `H_1` of the dual graph with `M^2 = 1`, so that its
      eigenvalues are `\pm 1`.  This is what makes the `t = 2` bookkeeping above
      valid; a higher-order graph automorphism raises.
    * `(1 - pT)` divides `P_p^{\wedge^2}` exactly.

    The real check on all of this is the functional equation: see
    :meth:`check_functional_equation`.  A wrong bad factor or conductor will
    show up there.

    EXAMPLES::

        sage: from hyperell_regulator.lfunction import StandardLFunction
        sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
        sage: L = StandardLFunction(C, bad_factors={2: None})   # doctest: +ELLIPSIS
        ...Warning: unknown valuation of conductor at 2.
        sage: L.euler_factor(53).degree()
        3
    """

    def __init__(self, C, bad_factors=None, prec=None):
        self.C = C
        if C.genus() != 2:
            raise ValueError("the standard L-function here is for genus 2; "
                             "this curve has genus %s" % C.genus())
        self.bad_factors = dict(bad_factors or {})
        RQ = PolynomialRing(QQ, 'x')
        self._pari_L = pari.lfungenus2([pari(str(RQ(C.f))), pari(str(RQ(C.h)))])
        self._RT = PolynomialRing(QQ, 'T')
        self._cache = {}
        self._h1cache = {}

    def _repr_(self):
        return "Standard L-function of %s" % self.C

    # -- Euler factors -----------------------------------------------------

    def h1_euler_factor(self, p):
        r"""
        The `h^1` Euler factor at ``p``.

        For odd good primes this uses Sage's own
        :meth:`~sage.schemes.hyperelliptic_curves.hyperelliptic_finite_field.HyperellipticCurve_finite_field.frobenius_polynomial`
        on the complete-the-square model `Y^2=F(x)`, `F=4f+h^2` (an isomorphism
        over `\ZZ[1/2]`, so the charpoly of Frobenius on `H^1` is unchanged).
        That dispatches to David Harvey's ``hypellfrob``, a purpose-built
        implementation of the same `p`-adic algorithm PARI's ``lfuneuler``
        uses -- but PARI pays a large, growing fixed cost per prime (measured:
        0.49s at a single `p=2999`, matching a direct call to
        ``hyperellcharpoly``), confirmed *not* to be Sage/PARI call overhead
        (looping entirely inside one ``gp`` call gave the same total time),
        whereas ``hypellfrob`` takes 0.01-0.03s even at `p` in the hundred
        thousands. Verified to reproduce PARI's answer exactly at several
        primes. Bad primes and `p=2` fall back to PARI, where the fast model
        may be singular or Sage's dispatch conditions do not apply.
        """
        p = ZZ(p)
        if p in self._h1cache:
            return self._h1cache[p]
        if p != 2 and p not in self.C.bad_primes():
            H = HyperellipticCurve(self.C.F.change_ring(GF(p)))
            cp = list(H.frobenius_polynomial())        # ascending, leading coeff 1
            out = self._RT(list(reversed(cp)))          # -> Euler-factor convention
            self._h1cache[p] = out
            return out
        e = self._pari_L.lfuneuler(p)
        num, den = pari(e).numerator(), pari(e).denominator()
        # lfuneuler returns 1/P_p(x); take the denominator
        poly = den if str(num).strip() in ("1", "1.0") else den / num
        return self._RT([QQ(c) for c in pari(poly).Vecrev()])

    def precompute_h1(self, primes, nthreads=None):
        r"""
        Fill the `h^1` cache for the good odd primes in ``primes``, in one
        batched call.

        This is where the time goes.  Obtaining the Euler factors one prime
        at a time dominates everything else -- the loop that expands them into
        Dirichlet coefficients runs in 0.1 s where the factors themselves took
        59 s -- and the cost per prime grows with `p`.  Handing the whole list
        to :func:`~hyperell_regulator.frobenius.frobenius_polynomials` pays
        the setup once and runs the primes concurrently.

        Primes the batched program declines fall back to Sage when they are
        asked for, so this only ever makes things faster.

        EXAMPLES::

            sage: from sage.all import ZZ, PolynomialRing
            sage: from hyperell_regulator import HyperellipticCurveWithReduction
            sage: from hyperell_regulator.lfunction import StandardLFunction
            sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
            sage: L = StandardLFunction(HyperellipticCurveWithReduction(x**5 + x**3 + 1))
            ...Warning: unknown valuation of conductor at 2.
            sage: L.precompute_h1([11, 13, 17])
            sage: sorted(L._h1cache)
            [11, 13, 17]
        """
        from .frobenius import frobenius_polynomials

        bad = set(self.C.bad_primes())
        todo = [int(p) for p in primes
                if int(p) != 2 and ZZ(p).is_prime() and ZZ(p) not in bad
                and ZZ(p) not in self._h1cache]
        if not todo:
            return
        for p, cp in frobenius_polynomials(self.C.F, todo, nthreads).items():
            self._h1cache[ZZ(p)] = self._RT(list(reversed(cp.list())))

    def toric_factor(self, p):
        r"""
        `\det(1 - \mathrm{Frob}_p T \mid H_1(\Gamma))`, the weight 0 part, from
        the Frobenius action on the dual graph of the special fibre.
        """
        R = self.C.cluster_picture(p)
        H1, M, frob = R.homology_of_special_fibre()
        basis = list(H1.basis())
        t = len(basis)
        if t == 0:
            return self._RT(1)
        rows = []
        for b in basis:
            image = frob(b)
            rows.append([QQ(image.coefficient(c.support()[0]))
                         if hasattr(image, "coefficient") else QQ(0)
                         for c in basis])
        Mfrob = Matrix(QQ, t, t, rows)
        if Mfrob ** 2 != identity_matrix(QQ, t):
            raise NotImplementedError(
                "Frobenius acts on H_1 of the dual graph at %s with a matrix "
                "of order > 2 (%s); the wedge^2 bookkeeping implemented here "
                "assumes eigenvalues +-1" % (p, Mfrob.list()))
        return self._RT(Mfrob.charpoly().reverse())

    def wedge2_euler_factor(self, p):
        r"""
        The `\wedge^2 h^1` Euler factor at ``p``.
        """
        if p in self.bad_factors and self.bad_factors[p] is not None:
            return self._RT(self.bad_factors[p])
        T = self._RT.gen()
        p = ZZ(p)
        P = self.h1_euler_factor(p)
        if p not in self.C.bad_primes():
            return wedge2_euler_factor(P)

        if p == 2:
            raise NotImplementedError(
                "p = 2 is a bad prime and cluster pictures need odd residue "
                "characteristic; supply the wedge^2 factor in bad_factors")
        if self.C.is_semistable_at(p) is not True:
            return self._wedge2_potentially_good(p, P)

        U = self.toric_factor(p)
        t = U.degree()
        if t == 0:
            W = wedge2_euler_factor(P)
        elif t == 1:
            gamma = -U[1]                       # U(T) = 1 - gamma T
            A = P // U                          # abelian factor, degree 2
            if A * U != P:
                raise ArithmeticError(
                    "at %s the toric factor does not divide the h^1 factor "
                    "(%s does not divide %s)" % (p, U, P))
            b2 = A[2]
            W = A(gamma * T) * (1 - b2 * T) * (1 - gamma ** 2 * p * T)
        elif t == 2:
            u2 = U[2]                           # det(Frob | U) = +-1
            W = (1 - u2 * T) * (1 - p * T) ** 2 * (1 - u2 * p * T)
        else:
            raise NotImplementedError("toric rank %s at %s" % (t, p))
        return W

    def _wedge2_potentially_good(self, p, P):
        r"""
        The `\wedge^2` factor at a prime where `C` is tame but not semistable
        and the potential stable reduction has toric rank 0.

        See the class docstring: `(\wedge^2 V)^I = \wedge^2(V^I) \oplus
        (\wedge^2 V')^I`, which `P_p` determines when `f_p \in \{0, 2\}`.
        """
        T = self._RT.gen()
        if self.C.is_tame(p) is not True:
            raise NotImplementedError(
                "C is neither semistable nor known to be tame at %s "
                "(is_tame = %s); supply bad_factors[%s]"
                % (p, self.C.is_tame(p), p))
        stable_type = self.C.local_data(p)["stable_type"]
        t = potential_toric_rank(stable_type)
        if t != 0:
            raise NotImplementedError(
                "C is not semistable at %s and its potential stable reduction "
                "has toric rank %s (Liu type (%s)); only potentially good "
                "reduction is implemented in the non-semistable case, since "
                "toric rank > 0 needs (ker N)^Phi.  Supply bad_factors[%s]"
                % (p, t, stable_type, p))

        fp = ZZ(self.C.conductor()).valuation(p)
        if fp != 4 - P.degree():
            raise ArithmeticError(
                "at %s the conductor exponent %s and deg P_p = %s disagree; "
                "for tame reduction f_p = 4 - dim V^I" % (p, fp, P.degree()))
        if fp == 0:
            return wedge2_euler_factor(P)
        if fp == 2:
            d = P[2]                        # det(Frob | V^I), P_p of degree 2
            if d != p:
                raise ArithmeticError(
                    "at %s the potentially good wedge^2 factor needs "
                    "det(Frob | V^I) = %s, but P_p = %s has leading "
                    "coefficient %s" % (p, p, P, d))
            return (1 - p * T) ** 2
        raise NotImplementedError(
            "potentially good reduction at %s with f_%s = %s: the wedge^2 "
            "invariants are not determined by P_p (which is %s) -- they need "
            "the eigenvalues of inertia on V_ell.  Supply bad_factors[%s]"
            % (p, p, fp, P, p))

    def euler_factor(self, p):
        r"""
        The standard Euler factor `P_p^{\wedge^2}(T)/(1 - pT)`.
        """
        p = ZZ(p)
        if p in self._cache:
            return self._cache[p]
        T = self._RT.gen()
        W = self.wedge2_euler_factor(p)
        q, r = W.quo_rem(1 - p * T)
        if r != 0:
            raise ArithmeticError(
                "(1 - %s T) does not divide the wedge^2 factor %s at %s; the "
                "Q(-1) summand of wedge^2 h^1 must contribute it"
                % (p, W, p))
        self._cache[p] = q
        return q

    def conductor(self):
        r"""
        `\prod_p p^{5 - \deg P_p^{\mathrm{std}}}`, the conductor of the standard
        motive (no wild part, the tame case being assumed).
        """
        N = ZZ(1)
        for p in self.C.bad_primes():
            N *= ZZ(p) ** (5 - self.euler_factor(p).degree())
        return N

    # -- the L-function ----------------------------------------------------

    def dirichlet_coefficients(self, num, progress=None):
        r"""
        The first ``num`` Dirichlet coefficients of `L(\mathrm{std}, s)`,
        expanded from the Euler product.

        ``progress`` is a number of coefficients between reports, or ``True``
        for a sensible default.  Without one this looks like a hang: obtaining
        the Euler factors dominates, at roughly 138 ms per prime on a
        conductor 5497 curve, and the cost grows with `p` (about 10 ms at
        `p \approx 250`, 456 ms at `p \approx 2750`), so a run over tens of
        thousands of coefficients spends nearly all of its time here.
        """
        import sys
        import time
        if progress is True:
            progress = max(500, num // 20)
        a = [ZZ(0)] * (num + 1)
        a[1] = ZZ(1)
        t0 = time.time()
        self.precompute_h1(prime_range(num + 1))

        # 1/P_p(x) expanded once per prime, to the greatest power of p below
        # num, rather than re-inverted for every n divisible by p.  Measured on
        # a conductor 5497 curve this is *not* where the time goes: with the
        # Euler factors already cached the whole loop below runs in 0.1 s for
        # 3000 coefficients, against 59 s to obtain the factors themselves
        # (138 ms per prime, in PARI's lfuneuler).  Kept because it is free.
        inv = {}

        def local(p, e):
            if p not in inv:
                depth = ZZ(num).exact_log(p)
                RS = PowerSeriesRing(QQ, 'x', default_prec=depth + 1)
                x = RS.gen()
                inv[p] = (1 / RS(self.euler_factor(p)(x))).padded_list(depth + 1)
            return inv[p][e]

        for n in range(2, num + 1):
            coeff = QQ(1)
            for p, e in ZZ(n).factor():
                coeff *= local(p, e)
            a[n] = coeff
            if progress and n % progress == 0:
                el = time.time() - t0
                # the remaining primes are the *expensive* ones: the cost of a
                # single Euler factor grows roughly like p^1.6, so the total to
                # N scales about like N^2.6.  A linear extrapolation
                # underestimates badly -- at the half way point of a 13842
                # coefficient run it claimed 667 s left where the truth is
                # nearer 3400 s.
                left = el * ((float(num) / n) ** 2.6 - 1.0)
                sys.stdout.write(
                    "\r      Euler factors: %d/%d (%.0f%%)  %.0fs elapsed, "
                    "~%.0fs left   " % (n, num, 100.0 * n / num, el, left))
                sys.stdout.flush()
        if progress:
            sys.stdout.write("\r      Euler factors: %d/%d done in %.0fs"
                             "%s\n" % (num, num, time.time() - t0, " " * 20))
            sys.stdout.flush()
        return a[1:]

    def _coeffs_needed(self, digits, der=2):
        r"""
        How many Dirichlet coefficients PARI's own ``lfuncost`` says are
        needed for `L^{(\mathrm{der})}(\mathrm{std}, 1)` at ``digits`` decimal
        digits of working precision.

        Replaces a hand-tuned heuristic (`\sim\mathrm{digits}\cdot\sqrt N`)
        that was checked to undershoot substantially: at `\mathrm{digits}=15`
        on a conductor `3\times10^7` curve it supplied `20713`-`82555`
        coefficients (depending on which version of the heuristic), while
        ``lfuncost`` says `160925` are actually needed there, and PARI's own
        runtime warning on the resulting ``lfun`` call
        (``#an = ... < 160925, results may be imprecise``) confirms the
        heuristic's count was not enough -- the requested ``digits`` was
        recorded as achieved without actually being achieved.

        Uses ``gp.eval('default(realprecision, ...)')`` to set the precision
        context for this call, not :func:`pari.set_real_precision`: the
        latter was found empirically *not* to move ``lfuncost``'s answer at
        all across `\mathrm{digits}=5..60` on the same example, while the
        former scales correctly and matches the coefficient count that live
        ``lfun``/``lfuncheckfeq`` calls actually warn about needing. ``der=2``
        (matching :meth:`value`'s default) is used regardless of the caller,
        since a higher derivative needs more terms and :meth:`lfun`'s result
        is shared between :meth:`value` and :meth:`check_functional_equation`.
        """
        N = self.conductor()
        prev = gp.eval('default(realprecision)')
        gp.eval('default(realprecision, %d)' % int(digits))
        try:
            L0 = pari.lfuncreate([[QQ(0)], 0, [-1, -1, 0, 0, 1], 3, N, 1])
            cost = gp.eval('lfuncost(%s, [1,0,0], %d)' % (L0, int(der)))
            return int(ZZ(str(cost).strip('[]').split(',')[0].strip()))
        finally:
            gp.eval('default(realprecision, %s)' % prev)

    def lfun(self, num_coeffs=None, digits=10, progress=None):
        r"""
        The PARI `L`-function object, with gamma factors `[-1,-1,0,0,1]`,
        weight 3 and root number `+1`, matching ``standardlfuntame.gp``.

        ``digits`` is the working precision in decimal digits. The number of
        Dirichlet coefficients needed grows with it (and with the conductor);
        it is taken from :meth:`_coeffs_needed`, PARI's own estimate, rather
        than a hand-tuned formula -- see that method for why. Each coefficient
        costs a Frobenius-polynomial evaluation, so this is genuinely more
        expensive than the previous heuristic, not merely more careful: PARI's
        own number can be several times larger.
        """
        key = ("lfun", num_coeffs, digits)          # progress is cosmetic
        if key in self._cache:
            return self._cache[key]
        N = self.conductor()
        if num_coeffs is None:
            num_coeffs = max(200, self._coeffs_needed(digits))
        an = self.dirichlet_coefficients(num_coeffs, progress=progress)
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            L = pari.lfuncreate([[QQ(c) for c in an], 0,
                                 [-1, -1, 0, 0, 1], 3, N, 1])
        finally:
            pari.set_real_precision(prev)
        self._cache[key] = L
        return L

    def _at_precision(self, digits, fn):
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            return fn()
        finally:
            pari.set_real_precision(prev)

    def check_functional_equation(self, num_coeffs=None, digits=10):
        r"""
        ``2^lfuncheckfeq``: the size of the failure of the functional equation,
        so *small is good*.  This is the real test of the bad Euler factors and
        of the conductor -- get either wrong and this will not be small.  At
        ``digits`` digits of working precision, do not expect it below about
        ``10^-digits``.
        """
        L = self.lfun(num_coeffs, digits)
        K = _real_field(digits)
        return self._at_precision(
            digits,
            lambda: K(2) ** K(pari.lfuncheckfeq(L, precision=_bits(digits))))

    def vanishing_order(self, s=1, max_order=4, num_coeffs=None, digits=10,
                        tol=None):
        r"""
        The order of vanishing of `L(\mathrm{std}, \cdot)` at ``s``, found by
        testing successive derivatives against ``tol`` (by default
        `10^{-\text{digits}/2}`).

        Worth reporting alongside the value: since
        `L(\wedge^2 h^1, s) = L(\mathrm{std}, s)\zeta(s-1)` and
        `\zeta(0) = -1/2 \ne 0`, the two `L`-functions vanish to the same
        order and their leading derivatives differ by the rational factor
        `-1/2`.  So taking `L''(\mathrm{std}, 1)` in place of the leading
        derivative of `\wedge^2` is harmless *provided the order really is 2*,
        which is what this checks.
        """
        if tol is None:
            tol = _real_field(digits)(10) ** (-_real_field(digits)(digits) / 2)
        for k in range(max_order + 1):
            if abs(self.value(s, k, num_coeffs, digits)) > tol:
                return k
        return None

    def value(self, s=1, derivative=2, num_coeffs=None, digits=10):
        r"""
        `L^{(k)}(\mathrm{std}, s)`, by default the **second derivative at
        `s = 1`** -- the quantity appearing in the Beilinson conjecture for
        `\wedge^2 h^1`.

        EXAMPLES::

            sage: from hyperell_regulator.lfunction import StandardLFunction
            sage: from hyperell_regulator.reduction import HyperellipticCurveWithReduction
            sage: R.<x> = ZZ[]
            sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
        """
        L = self.lfun(num_coeffs, digits)
        K = _real_field(digits)
        return self._at_precision(
            digits,
            lambda: K(pari.lfun(L, s, derivative, precision=_bits(digits))))


def standard_lvalue(C, bad_factors=None, derivative=2, num_coeffs=None,
                    digits=10, verbose=True, progress=None):
    r"""
    `L''(\mathrm{std}, 1)` for the genus 2 curve ``C``, with the bad Euler
    factors computed from the cluster picture wherever `C` is semistable at an
    odd prime.

    Returns a dictionary with the value, the conductor, the Euler factors used
    and the functional equation check.
    """
    L = StandardLFunction(C, bad_factors=bad_factors)
    out = {"curve": str(C), "bad_primes": C.bad_primes(),
           "euler_factors": {}, "conductor": None, "value": None,
           "derivative": derivative, "feq_check": None}
    for p in C.bad_primes():
        out["euler_factors"][p] = str(L.euler_factor(p))
    out["conductor"] = L.conductor()
    out["digits"] = digits
    # build (and cache) the coefficients once, with progress; everything below
    # then hits the cache
    L.lfun(num_coeffs, digits, progress=progress)
    out["feq_check"] = L.check_functional_equation(num_coeffs, digits)
    out["value"] = L.value(1, derivative, num_coeffs, digits)
    out["L_at_1"] = L.value(1, 0, num_coeffs, digits)
    out["Lprime_at_1"] = L.value(1, 1, num_coeffs, digits)
    out["vanishing_order"] = L.vanishing_order(1, num_coeffs=num_coeffs,
                                               digits=digits)
    if verbose:
        print(C)
        print("  bad primes %s, standard conductor %s"
              % (C.bad_primes(), out["conductor"]))
        for p, e in out["euler_factors"].items():
            print("    P_%s^std(T) = %s" % (p, e))
        print("  functional equation check (small is good): %.3e"
              % out["feq_check"])
        print("  L(std,1) = %.3e   L'(std,1) = %.3e   order of vanishing %s"
              % (out["L_at_1"], out["Lprime_at_1"], out["vanishing_order"]))
        print("  L^(%s)(std, 1) = %s   (%s digits)"
              % (derivative, out["value"], digits))
    return out


# ---------------------------------------------------------------------------
# L(C, s) itself
# ---------------------------------------------------------------------------

class CurveLFunction(SageObject):
    r"""
    `L(C,s) = L(h^1(C), s)` for the genus 2 curve: degree 4, motivic weight 1,
    so the functional equation is `s \leftrightarrow 2-s` and the gamma factor
    is `\Gamma_{\mathbf{C}}(s)^2`.

    PARI's ``lfungenus2`` already knows this `L`-function, and is used for the
    conductor, the root number and the bad Euler factors.  What is rebuilt
    here is the Dirichlet series, from Euler factors obtained in bulk through
    :mod:`~hyperell_regulator.frobenius` rather than one prime at a time, and
    the whole thing is then evaluated at whatever precision is asked for.

    EXAMPLES::

        sage: from sage.all import ZZ, PolynomialRing
        sage: from hyperell_regulator import HyperellipticCurveWithReduction
        sage: from hyperell_regulator.lfunction import CurveLFunction
        sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
        sage: L = CurveLFunction(HyperellipticCurveWithReduction(x**5 + x**3 + 1))  # doctest: +ELLIPSIS
        ...Warning: unknown valuation of conductor at 2.
        sage: L.euler_factor(11).degree()
        4
        sage: L.conductor()
        3233
    """

    def __init__(self, C):
        self.C = C
        self._RT = PolynomialRing(QQ, 'T')
        self._cache = {}
        self._h1cache = {}
        RQ = PolynomialRing(QQ, 'x')
        self._pari_L = pari.lfungenus2([pari(str(RQ(C.f))), pari(str(RQ(C.h)))])
        self._eps = None

    def _repr_(self):
        return "L(C, s) for %s" % (self.C,)

    def precompute_h1(self, primes, nthreads=None):
        r"""Fill the cache for the good odd primes in one batched call."""
        from .frobenius import frobenius_polynomials

        bad = set(self.C.bad_primes())
        todo = [int(p) for p in primes
                if int(p) != 2 and ZZ(p).is_prime() and ZZ(p) not in bad
                and ZZ(p) not in self._h1cache]
        if not todo:
            return
        for p, cp in frobenius_polynomials(self.C.F, todo, nthreads).items():
            self._h1cache[ZZ(p)] = self._RT(list(reversed(cp.list())))

    def euler_factor(self, p):
        r"""
        `P_p(T)` with `L(C,s) = \prod_p P_p(p^{-s})^{-1}`.

        Good odd primes come from the batched Frobenius computation; `p = 2`
        and the bad primes from ``lfungenus2``, where the complete-the-square
        model may be singular.

        EXAMPLES::

            sage: from sage.all import ZZ, PolynomialRing
            sage: from hyperell_regulator import HyperellipticCurveWithReduction
            sage: from hyperell_regulator.lfunction import CurveLFunction
            sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
            sage: L = CurveLFunction(HyperellipticCurveWithReduction(x**5 + x**3 + 1))
            ...Warning: unknown valuation of conductor at 2.
            sage: L.euler_factor(11)[0]
            1
        """
        p = ZZ(p)
        if p in self._h1cache:
            return self._h1cache[p]
        if p != 2 and p not in self.C.bad_primes():
            H = HyperellipticCurve(self.C.F.change_ring(GF(p)))
            out = self._RT(list(reversed(list(H.frobenius_polynomial()))))
            self._h1cache[p] = out
            return out
        e = self._pari_L.lfuneuler(p)
        num, den = pari(e).numerator(), pari(e).denominator()
        poly = den if str(num).strip() in ("1", "1.0") else den / num
        out = self._RT([QQ(c) for c in pari(poly).Vecrev()])
        self._h1cache[p] = out
        return out

    def conductor(self):
        r"""The conductor of `C`, which is the one in the functional equation."""
        return ZZ(self.C.conductor())

    def dirichlet_coefficients(self, num):
        r"""The first ``num`` Dirichlet coefficients, from the Euler product."""
        self.precompute_h1(prime_range(num + 1))
        a = [ZZ(0)] * (num + 1)
        a[1] = ZZ(1)
        inv = {}

        def local(p, e):
            if p not in inv:
                depth = ZZ(num).exact_log(p)
                RS = PowerSeriesRing(QQ, 'x', default_prec=depth + 1)
                x = RS.gen()
                inv[p] = (1 / RS(self.euler_factor(p)(x))).padded_list(depth + 1)
            return inv[p][e]

        for n in range(2, num + 1):
            coeff = QQ(1)
            for p, e in ZZ(n).factor():
                coeff *= local(p, e)
            a[n] = coeff
        return a

    def root_number(self, num_coeffs=200, digits=10):
        r"""
        The sign in the functional equation, `\pm 1`, chosen as the one that
        satisfies it.

        ``lfungenus2`` determines it internally, but does not hand it over, so
        it is recovered here by building the `L`-function both ways and
        keeping whichever passes ``lfuncheckfeq``.
        """
        if self._eps is not None:
            return self._eps
        an = self.dirichlet_coefficients(num_coeffs)[1:]
        N = self.conductor()
        best, score = 1, None
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            for eps in (1, -1):
                L = pari.lfuncreate([[QQ(c) for c in an], 0, [0, 0, 1, 1],
                                     2, N, eps])
                K = _real_field(digits)
                v = K(2) ** K(pari.lfuncheckfeq(L, precision=_bits(digits)))
                if score is None or v < score:
                    best, score = eps, v
        finally:
            pari.set_real_precision(prev)
        self._eps = best
        return best

    def _coeffs_needed(self, digits, der=0):
        r"""
        PARI's own estimate of how many Dirichlet coefficients are needed.

        As for :meth:`StandardLFunction._coeffs_needed`, the precision context
        has to be set through ``gp.eval('default(realprecision, ...)')``:
        :func:`pari.set_real_precision` does not move ``lfuncost``'s answer.
        A hand-rolled `\sqrt N` heuristic is not good enough -- it left the
        functional equation failing at `8\times10^{-3}`.
        """
        N = self.conductor()
        prev = gp.eval('default(realprecision)')
        gp.eval('default(realprecision, %d)' % int(digits))
        try:
            L0 = pari.lfuncreate([[QQ(0)], 0, [0, 0, 1, 1], 2, N, 1])
            cost = gp.eval('lfuncost(%s, [1,0,0], %d)' % (L0, int(der)))
            return int(ZZ(str(cost).strip('[]').split(',')[0].strip()))
        finally:
            gp.eval('default(realprecision, %s)' % prev)

    def lfun(self, num_coeffs=None, digits=10, der=2):
        r"""
        The PARI `L`-function object: gamma factors `[0,0,1,1]`, weight 2,
        conductor `N`, and the root number from :meth:`root_number`.
        """
        if num_coeffs is None:
            num_coeffs = self._coeffs_needed(digits, der)
        key = ("lfun", int(num_coeffs), int(digits))
        if key in self._cache:
            return self._cache[key]
        an = self.dirichlet_coefficients(num_coeffs)[1:]
        eps = self.root_number(min(num_coeffs, 400), digits)
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            L = pari.lfuncreate([[QQ(c) for c in an], 0, [0, 0, 1, 1],
                                 2, self.conductor(), eps])
        finally:
            pari.set_real_precision(prev)
        self._cache[key] = L
        return L

    def check_functional_equation(self, num_coeffs=None, digits=10):
        r"""``2^lfuncheckfeq``: small is good."""
        L = self.lfun(num_coeffs, digits)
        K = _real_field(digits)
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            return K(2) ** K(pari.lfuncheckfeq(L, precision=_bits(digits)))
        finally:
            pari.set_real_precision(prev)

    def value(self, s=1, derivative=0, num_coeffs=None, digits=10):
        r"""
        `L^{(k)}(C, s)` at ``digits`` decimal digits.

        EXAMPLES::

            sage: from sage.all import ZZ, PolynomialRing
            sage: from hyperell_regulator import HyperellipticCurveWithReduction
            sage: from hyperell_regulator.lfunction import CurveLFunction
            sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
            sage: L = CurveLFunction(HyperellipticCurveWithReduction(x**5 + x**3 + 1))  # doctest: +ELLIPSIS
            ...Warning: unknown valuation of conductor at 2.
            sage: bool(L.value(2, digits=10) > 0)            # long time
            True
        """
        L = self.lfun(num_coeffs, digits)
        K = _real_field(digits)
        prev = pari.get_real_precision()
        pari.set_real_precision(digits)
        try:
            return K(pari.lfun(L, s, derivative, precision=_bits(digits)))
        finally:
            pari.set_real_precision(prev)


def curve_lvalue(C, s=1, derivative=0, num_coeffs=None, digits=10,
                 verbose=False):
    r"""
    `L^{(k)}(C, s)` for the genus 2 curve ``C``, to ``digits`` decimal digits.

    Returns a dictionary with the value, the conductor, the root number and
    the functional equation check, in the shape :func:`standard_lvalue`
    returns for the standard `L`-function.

    EXAMPLES::

        sage: from sage.all import ZZ, PolynomialRing
        sage: from hyperell_regulator import HyperellipticCurveWithReduction
        sage: from hyperell_regulator.lfunction import curve_lvalue
        sage: R = PolynomialRing(ZZ, 'x'); x = R.gen()
        sage: d = curve_lvalue(HyperellipticCurveWithReduction(x**5 + x**3 + 1),  # doctest: +ELLIPSIS
        ....:                  s=2, digits=12)                      # long time
        ...Warning: unknown valuation of conductor at 2.
        ...
        sage: d["conductor"]                                         # long time
        3233
    """
    L = CurveLFunction(C)
    out = {"curve": str(C), "conductor": L.conductor(),
           "root_number": L.root_number(),
           "s": s, "derivative": derivative, "digits": digits}
    out["feq_check"] = L.check_functional_equation(num_coeffs, digits)
    out["value"] = L.value(s, derivative, num_coeffs, digits)
    if verbose:
        print(C)
        print("  conductor %s, root number %+d"
              % (out["conductor"], out["root_number"]))
        print("  L^(%d)(C, %s) = %s" % (derivative, s, out["value"]))
        print("  functional equation check %s" % out["feq_check"])
    return out
