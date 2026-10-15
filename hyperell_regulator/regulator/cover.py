r"""
The branched cover `\varphi : C \to \mathbf{P}^1` a function on `y^2 = F(x)`
defines.

A function on the hyperelliptic curve is `\varphi = A(x) + B(x)\,y` with `A, B`
rational.  Clearing `y` through `y^2 = F` gives a polynomial relation

.. math:: \mathcal{P}(x,w) = p_0(x) + p_1(x)\,w + p_2(x)\,w^2 = 0,

whose degree in `x` is the degree `N` of `\varphi`.  Everything the Stokes
integrator needs is read off it:

.. math:: \frac{dx}{dw} = -\frac{\mathcal{P}_w}{\mathcal{P}_x}, \qquad
          \frac{dw}{dx} = -\frac{\mathcal{P}_x}{\mathcal{P}_w},
          \qquad \mathcal{P}_w = 2p_2 B\,y ,

so the first is singular exactly at the ramification of `\varphi` and the
second exactly where `y = 0` or `B = 0`.  The two roots in `w` are
`A \pm B\sqrt F`, with product

.. math:: \operatorname{N}(\varphi) = A^2 - B^2 F ,

the norm down to `\mathbf{P}^1`; it is how `w` is recovered without
cancellation when `w` is small, and how the plane integrator is given a
`\log|\varphi|` weight.
"""

import numpy as np

from .polynomials import evaluate as _eval

from .polynomials import array as _poly, arrays as _ratio, polynomial, rational

__all__ = ["Cover", "check_path_embedded"]


def rounded(c):
    r"""
    A float64 coefficient array back over `\QQ`, highest degree first.

    This is the last resort: the exact input is kept from the start and used
    wherever it is there, and rounding floats back to rationals is only for
    the data :meth:`Cover.shifted` produces, which never was exact.
    """
    from sage.all import QQ

    from .polynomials import ring

    return ring()([QQ(v.real if abs(v.imag) < 1e-12 else v)
                   for v in np.asarray(c, complex).ravel()[::-1].tolist()])


def _vanishes(c, x, tol=1e-10):
    r"""
    Where the polynomial ``c`` vanishes at ``x``, relative to the size of the
    terms that make up the value there.
    """
    x = np.asarray(x, dtype=complex)
    return (np.abs(_eval(c, x))
            <= tol * _eval(np.abs(c), np.abs(x)))


def _polish(c, x, rounds=2):
    r"""
    Newton on the roots ``x`` of the polynomial ``c``.

    ``np.roots`` is an eigenvalue solve on the companion matrix, and it
    places two roots that are close to each other only to about the square
    root of what the arithmetic allows -- `10^{-8}` where a fibre point
    carries `10^{-15}` everywhere else.  The legs start and end at these
    points, so those seven digits are lost to the answer as well.  A step is
    taken only when it is small against the distance to the nearest other
    root, which leaves a genuine multiple root alone: there the pair is
    handled by the charts, not by polishing.
    """
    c, x = np.asarray(c, complex), np.asarray(x, complex)
    if len(x) < 1 or len(c) < 2:
        return x
    dc = np.polyder(c)
    for _ in range(rounds):
        d = np.abs(x[:, None] - x[None, :])
        np.fill_diagonal(d, np.inf)
        gap = d.min(axis=1)
        dp = _eval(dc, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = _eval(c, x) / dp
        ok = np.isfinite(step) & (np.abs(step) < 0.25 * gap)
        x = np.where(ok, x - step, x)
    return x


def _padsum(polys):
    n = max(len(p) for p in polys)
    return sum(np.concatenate([np.zeros(n - len(p), complex), p]) for p in polys)


class Cover:
    r"""
    The cover defined by `\varphi = A + B y` on `y^2 = F(x)`.

    `F` is a polynomial over `\QQ`; `A` and `B` are rational functions of
    `x`, and `\varphi = A + By`.  Coefficient lists in descending degree are
    accepted as well -- see
    :mod:`~hyperell_regulator.regulator.polynomials`.

    EXAMPLES::

        sage: from hyperell_regulator.regulator.cover import Cover
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: C = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1)   # phi = y + 1 + x
        sage: C.N, C.g
        (5, 2)
        sage: C = Cover(x^5 + x^2 + x, x, 0)             # phi = x
        sage: C.N
        2
    """

    def __init__(self, F, A, B):
        # The exact objects are kept beside the float64 arrays: the cover
        # polynomial's coefficients are products of F, A and B's, and for a
        # phi of high degree those span a wide range -- the degree-14 one on
        # LMFDB 249.a.249.1 has coefficients of size 1e12 over a common
        # denominator of 2e11.  Multiplied out in float64 the cancellations
        # that should leave a zero coefficient leave noise instead: a leading
        # 9e-13 beside coefficients of 1.8e3, whose reciprocal then appears
        # as a "critical value" of 1e123, and multiple roots split apart.
        # ``shifted`` produces genuinely inexact data, and there ``exact`` is
        # None and the float64 route is all there is.
        Fq, Aq, Bq = polynomial(F), rational(A), rational(B)
        exact = not any(q is None for q in (Fq, Aq, Bq))
        self.exact = ((Fq, Aq.numerator(), Aq.denominator(),
                       Bq.numerator(), Bq.denominator()) if exact else None)

        self.F = _poly(Fq if exact else F)
        self.Fp = np.polyder(self.F)
        deg = len(self.F) - 1
        # odd degree 2g+1: infinity is one point, a Weierstrass point, and
        # x^{-1/2} is the local parameter.  Even degree 2g+2: infinity is two
        # points, neither of them Weierstrass, and 1/x is the parameter at
        # each.  Both are handled; ``even`` says which is in hand.
        self.even = deg % 2 == 0
        if deg < 5:
            raise NotImplementedError("F must have degree at least five")
        self.g = (deg - 2) // 2 if self.even else (deg - 1) // 2
        self.a, self.ad = _ratio(A)
        self.b, self.bd = _ratio(B)
        self.ap, self.adp = np.polyder(self.a), np.polyder(self.ad)
        self.bp, self.bdp = np.polyder(self.b), np.polyder(self.bd)

        # P = (bd ad)^2 [ (A + B y)^2 - 2 A (A + By) + A^2 - B^2 F ] cleared:
        #   p2 w^2 + p1 w + p0  with roots A +- B sqrt(F)
        if exact:
            Fx, a, ad, b, bd = self.exact
            polynomials = [b ** 2 * ad ** 2 * Fx - bd ** 2 * a ** 2,
                           2 * bd ** 2 * ad * a, -(bd * ad) ** 2]
            # Clearing rational denominators introduces fixed x-factors.
            # They are not sheets of phi and must be cancelled over QQ,
            # just as in exact_polynomial(), before numerical root finding.
            common = polynomials[0].gcd(polynomials[1]).gcd(polynomials[2])
            self.p = [_poly(q // common) for q in polynomials]
            self.norm_num, self.norm_den = (_poly(q) for q in self.norm_exact())
        else:
            aa, bb = np.polymul(self.a, self.a), np.polymul(self.b, self.b)
            add, bdd = np.polymul(self.ad, self.ad), np.polymul(self.bd, self.bd)
            self.p = [_poly(np.polysub(np.polymul(bb, np.polymul(add, self.F)),
                                       np.polymul(bdd, aa))),
                      _poly(2 * np.polymul(np.polymul(bdd, self.ad), self.a)),
                      _poly(-np.polymul(bdd, add))]
            self.norm_num = _poly(np.polysub(np.polymul(aa, bdd),
                                             np.polymul(bb, np.polymul(add, self.F))))
            self.norm_den = _poly(np.polymul(add, bdd))
        self.pd = [np.polyder(q) for q in self.p]
        self.N = max(len(q) for q in self.p) - 1
        if self.N < 2:
            raise ValueError("phi is constant")
        # B == 0 means phi factors through x, so x does not separate the fibre
        # (both points over an x are in it) and P is a perfect square.  The
        # fibre is then indexed by the sign of y, and every ramification point
        # of phi over a Weierstrass point is one of the curve as well.
        self.pure_x = bool(np.all(self.b == 0))
        self._factors = None
        self._factors_hp = {}
        self._rs = {}
        self._dd = None
        self._degen = None
        self._dds = None
        self._places = {}

    def riemann_surface(self, prec=53):
        r"""
        Sage's :class:`RiemannSurface` for `\mathcal{P}(x,w) = 0`, tracking
        `x` over `w`, or ``None`` without exact coefficients.

        Sage's variables are the other way round -- it follows the roots of
        `f(z,w)` in `w` over `z` -- so they are swapped on the way in.  What
        it is wanted for is :meth:`_compute_delta`, a step along a path that
        is certified from `\varepsilon`, the least distance between the
        points of the fibre divided by three, together with
        :meth:`_determine_new_w`, which Newton-iterates the whole fibre and
        refuses the step if any point leaves its `\varepsilon` ball.  That
        pair is a guarantee no sheet has been exchanged for another, which is
        what choosing a stand-off by a fixed fraction of an edge never was.
        """
        if self.exact is None:
            return None
        if getattr(self, "_rs", None) is None:
            self._rs = {}
        if prec not in self._rs:
            from sage.all import PolynomialRing, QQ
            from sage.schemes.riemann_surfaces.riemann_surface import (
                RiemannSurface)

            P = self.exact_polynomial()
            xx, ww = P.parent().gens()
            R = PolynomialRing(QQ, ["z", "w"])
            z, w = R.gens()
            self._rs[prec] = RiemannSurface(P.subs({xx: w, ww: z}), prec=prec)
        return self._rs[prec]

    def norm_exact(self):
        r"""
        `\operatorname{N}(\varphi) = A^2 - B^2F` as a ``(numerator,
        denominator)`` pair over `\QQ`, or ``None`` without exact
        coefficients.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).norm_exact()
            (-x^5, 1)
        """
        if self.exact is None:
            return None
        if hasattr(self, "_norm_exact"):
            return self._norm_exact
        F, a, ad, b, bd = self.exact
        norm = (a / ad) ** 2 - (b / bd) ** 2 * F
        self._norm_exact = (norm.numerator(), norm.denominator())
        return self._norm_exact

    def shifted(self, c):
        r"""
        The same cover with every polynomial Taylor-shifted about `c`, so that
        it is evaluated at `u = x - c`.

        Near a point where a denominator vanishes this is the difference
        between an answer and nothing: forming `x = c + u` in floating point
        rounds back to `c` as soon as `|u|` drops below the spacing there, and
        `a_d(x)` then evaluates to exactly zero.  Shifting keeps the small
        quantity small throughout.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: import numpy as np
            sage: C = Cover(x^5 + x^2 + x, x, 0).shifted(1.0)
            sage: bool(abs(np.polyval(C.F, 1e-30)
            ....:          - np.polyval([1, 0, 0, 1, 1, 0], 1 + 1e-30)) < 1e-12)
            True
        """
        if c == 0:
            return self
        if self.exact is not None and complex(c).imag == 0:
            from sage.all import QQ

            F, a, ad, b, bd = self.exact
            shifted = F.parent().gen() + QQ(complex(c).real)
            return Cover(F(shifted), a(shifted) / ad(shifted),
                         b(shifted) / bd(shifted))
        sh = lambda q: _shift_poly(q, c)
        out = Cover(sh(self.F), (sh(self.a), sh(self.ad)),
                    (sh(self.b), sh(self.bd)))
        # Translate identities already simplified before conversion to
        # float64.  Re-forming A^2-B^2 F from shifted coefficients loses
        # its cancellations (degree 28 down to 14 in the first example).
        out.p = [sh(q) for q in self.p]
        out.pd = [np.polyder(q) for q in out.p]
        out.N = self.N
        out.norm_num, out.norm_den = sh(self.norm_num), sh(self.norm_den)
        out._factors = tuple([(r - c, m) for r, m in terms]
                             for terms in self.norm_factors())
        return out

    def reciprocal(self, c=0):
        r"""Read the cover in u = 1/(x-c), eta = u^(g+1) y.

        The transformed equation is eta^2 = u^(2g+2) F(c+1/u).
        Its rational functions and norm are simplified exactly when the
        center and the original cover are rational. In particular, a
        cancelling branch at infinity retains its finite value. The
        already simplified cover polynomial is transformed separately:
        rebuilding it from the rational functions introduces powers of
        u that do not represent points of the fibre.

        This supplies the regular product eta*dphi/du even at a
        Weierstrass point, where -eta*P_u/P_w has the form 0/0.
        """
        from sage.all import QQ

        center = complex(c)
        if self.exact is not None and center.imag == 0:
            F, a, ad, b, bd = self.exact
            R = F.parent()
            u = R.fraction_field()(R.gen())
            xx = QQ(center.real) + 1 / u
            out = Cover(R(u ** (2 * self.g + 2) * F(xx)),
                        a(xx) / ad(xx),
                        b(xx) / bd(xx) / u ** (self.g + 1))
            ps = (b ** 2 * ad ** 2 * F - bd ** 2 * a ** 2,
                  2 * bd ** 2 * ad * a, -(bd * ad) ** 2)
            common = ps[0].gcd(ps[1]).gcd(ps[2])
            out.p = [_poly(R(u ** self.N * (q // common)(xx))) for q in ps]
        else:
            sh = lambda q: np.asarray(_shift_poly(q, center), complex)

            def reverse_ratio(num, den, extra=0):
                n, d = sh(num)[::-1], sh(den)[::-1]
                power = len(den) - len(num) - extra
                if power >= 0:
                    n = np.pad(n, (0, power))
                else:
                    d = np.pad(d, (0, -power))
                return _poly(n), _poly(d)

            degree = len(self.F) - 1
            Fu = np.pad(sh(self.F)[::-1],
                        (0, 2 * self.g + 2 - degree))
            out = Cover(Fu, reverse_ratio(self.a, self.ad),
                        reverse_ratio(self.b, self.bd, self.g + 1))
            out.p = [np.pad(sh(q)[::-1], (0, self.N - len(q) + 1))
                     for q in self.p]
            out.norm_num, out.norm_den = reverse_ratio(
                self.norm_num, self.norm_den)
        out.pd = [np.polyder(q) for q in out.p]
        out.N = self.N
        return out

    # ---------------------------------------------------------------- fibres

    def poly_in_x(self, w):
        r"""`\mathcal{P}(\cdot, w)` as a polynomial in `x`."""
        return _padsum([self.p[0], w * self.p[1], w * w * self.p[2]])

    def fibre(self, w):
        r"""
        The `N` points of `\varphi^{-1}(w)`, as `(x, y)` pairs.

        When `B \neq 0` the `x`-coordinates are the roots of
        `\mathcal{P}(\cdot,w)` and `y = (w-A)/B`.  When `B \equiv 0` those
        roots come in coincident pairs and the two points over each are
        `(x, \pm\sqrt{F})`.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: import numpy as np
            sage: C = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1)
            sage: x, y = C.fibre(2.0)
            sage: bool(abs(y ** 2 - np.polyval(C.F, x)).max() < 1e-12)
            True
        """
        if self.exact is not None:
            if not hasattr(self, "_centered_fibre"):
                self._centered_fibre = None
                zeros = self.norm_factors()[0]
                if len(zeros) == 1:
                    center = complex(zeros[0][0])
                    if center != 0 and center.imag == 0:
                        local = self.shifted(center)
                        local._centered_fibre = None
                        self._centered_fibre = (center, local)
            if self._centered_fibre is not None:
                center, local = self._centered_fibre
                x, y = local.fibre(w)
                return x + center, y
        if self.pure_x:
            num = np.polysub(w * self.ad, self.a)
            x = np.roots(num) if len(num) > 1 else np.zeros(0, complex)
            x = np.repeat(x, 2)
            y = np.sqrt(_eval(self.F, x).astype(complex))
            y[1::2] *= -1
        else:
            c = self.poly_in_x(w)
            x = _polish(c, np.roots(c))
            x, y = self._split_pinches(x, self.y_of(x, w))
        if len(x) < self.N:
            pad = self.N - len(x)
            x = np.concatenate([x, np.full(pad, np.inf)])
            y = np.concatenate([y, np.full(pad, np.inf)])
        return x, y

    def Px(self, x, w):
        return (_eval(self.pd[0], x) + w * _eval(self.pd[1], x)
                + w * w * _eval(self.pd[2], x))

    def Pw(self, x, w):
        return _eval(self.p[1], x) + 2 * w * _eval(self.p[2], x)

    # ------------------------------------------------------------- functions

    def A(self, x):
        return _eval(self.a, x) / _eval(self.ad, x)

    def B(self, x):
        return _eval(self.b, x) / _eval(self.bd, x)

    def dA(self, x):
        n, d = _eval(self.a, x), _eval(self.ad, x)
        return (_eval(self.ap, x) * d
                - n * _eval(self.adp, x)) / d ** 2

    def dB(self, x):
        n, d = _eval(self.b, x), _eval(self.bd, x)
        return (_eval(self.bp, x) * d
                - n * _eval(self.bdp, x)) / d ** 2

    def tau_powers(self):
        r"""
        `(p_A, p_B)`: the orders in `1/\tau` that `A` and `B\,y` reach at
        infinity, `\tau` being the local parameter there.

        In odd degree `x = \tau^{-2}` and `y = \eta\tau^{-(2g+1)}`, so they
        are `2(d_a - d_{a_d})` and `2(d_b - d_{b_d}) + 2g+1`.  In even degree
        `x = \tau^{-1}` and `y = \eta\tau^{-(g+1)}`, and they are
        `d_a - d_{a_d}` and `d_b - d_{b_d} + g + 1`.
        """
        da, db = len(self.a) - len(self.ad), len(self.b) - len(self.bd)
        if self.even:
            return da, (db + self.g + 1 if np.any(self.b) else None)
        return 2 * da, (2 * db + 2 * self.g + 1 if np.any(self.b) else None)

    def pole_order_at_infinity(self, branch=1):
        r"""
        The order of the pole of `\varphi` at the point at infinity.

        In even degree there are two of them and ``branch`` says which:
        `y \sim \mathrm{branch}\sqrt{f_{2g+2}}\,x^{g+1}`.  The two orders
        need not agree -- when `\varphi` has divisor `N P - N Q` with `Q` one
        of them, it has a pole of order `N` there and none at the other, the
        leading terms of `A` and `B y` cancelling.  That cancellation is why
        the order is read off the growth of `\varphi` rather than off the
        degrees: the degrees give the larger of the two, which is right at
        only one of the points.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).pole_order_at_infinity()
            5
            sage: Cover(x^5 + x^2 + x, x, 0).pole_order_at_infinity()
            2
        """
        pA, pB = self.tau_powers()
        p = pA if pB is None else max(pA, pB)
        if not self.even:
            return p
        # the growth of |phi| along the branch, which sees the cancellation
        v = []
        for t in (1e4, 1e8):
            x = np.array([t])
            y = branch * np.sqrt(_eval(self.F, x).astype(complex))
            v.append(abs(complex(self.phi(x, y)[0])))
        if not v[0] or not v[1]:
            return p
        q = int(round((np.log(v[1]) - np.log(v[0])) / np.log(1e4)))
        return min(p, q)

    def value_at_infinity(self, branch=1):
        r"""
        `\varphi` at the point at infinity, or ``None`` when it has a pole
        there.  Even degree only; in odd degree there is one such point and
        :meth:`_A_at_infinity` covers the case `B = 0`.
        """
        if not self.even or self.pole_order_at_infinity(branch) > 0:
            return None
        x = np.array([1e8])
        y = branch * np.sqrt(_eval(self.F, x).astype(complex))
        return complex(self.phi(x, y)[0])

    def norm(self, x):
        r"""`\operatorname{N}(\varphi) = A^2 - B^2 F` at `x`."""
        return _eval(self.norm_num, x) / _eval(self.norm_den, x)

    def phi(self, x, y):
        r"""`\varphi` at `(x,y)`, without cancellation when it is small."""
        big = self.A(x) - self.B(x) * y
        small = self.A(x) + self.B(x) * y
        use_norm = np.abs(small) < np.abs(big)
        out = np.where(use_norm, self.norm(x) / np.where(big == 0, 1, big), small)
        return out

    def dphi_dx(self, x, y):
        r"""`d\varphi/dx` along the curve, using `y' = F'/2y`."""
        return self.dA(x) + self.dB(x) * y + self.B(x) * _eval(self.Fp, x) / (2 * y)

    def y_dphi_dx(self, x, y):
        r"""
        `y\,d\varphi/dx = y(A' + B'y) + B F'/2`.

        `d\varphi/dx` itself has a pole at a Weierstrass point while
        `\omega_a = x^a dx/y` has none, so every rate is formed against this
        product rather than against the two factors separately -- otherwise
        `x^a\,(dx/dw)/y` is computed as `0/0` there.
        """
        A, B = self.A(x), self.B(x)
        small, big = A + B * y, A - B * y
        da, db = y * self.dA(x), self.dB(x) * y * y
        bf = B * _eval(self.Fp, x) / 2
        direct = da + db + bf
        # Differentiate phi = Norm(phi)/(A - By) when A + By cancels.
        # Multiplying through by y keeps this form valid at Weierstrass
        # points too, unlike y * dphi_dx_stable (which first divides by y).
        if np.ndim(x) == 0:
            if abs(small) > abs(big) or big == 0 or self.pure_x:
                return direct
            alt = self.norm_value(x) / big * (
                y * self.dlog_norm(x) - (da - db - bf) / big)
            return alt if np.isfinite(alt) else direct
        with np.errstate(divide="ignore", invalid="ignore"):
            alt = self.norm_value(x) / big * (
                y * self.dlog_norm(x) - (da - db - bf) / big)
        return np.where((np.abs(small) <= np.abs(big)) & np.isfinite(alt),
                        alt, direct)

    def dlogphi(self, x, y, dx, dy):
        r"""
        `d\log\varphi` along a path with the given `dx, dy`.

        Computing it as `(d\varphi)/\varphi` cancels catastrophically where
        `\varphi` is small: at `P` both `d\varphi` and `\varphi` vanish to
        high order while their ingredients are `O(1)`, so the quotient comes
        out as noise divided by an underflowed number.  There
        `\varphi = \operatorname{N}(\varphi)/(A - By)` instead, and

        .. math:: d\log\varphi = \frac{\operatorname{N}'}{\operatorname{N}}\,dx
                  - \frac{d(A - By)}{A - By},

        where the first term is a ratio of polynomials that carries the zero
        exactly and the second has a nonvanishing denominator.
        """
        A, B = self.A(x), self.B(x)
        small, big = A + B * y, A - B * y
        if abs(small) <= abs(big):
            dlogN = self.dlog_norm(x) * dx
            dbig = (self.dA(x) - self.dB(x) * y) * dx - B * dy
            return dlogN - dbig / big
        dsmall = (self.dA(x) + self.dB(x) * y) * dx + B * dy
        return dsmall / small

    def norm_factors(self):
        r"""
        The zeros and poles of `\operatorname{N}(\varphi)` with their
        multiplicities, as ``(zeros, poles)``.

        Exactly, from a squarefree decomposition over `\QQ`: the zero at
        `x_P` has multiplicity `N` and a numerical root finder would place it
        only to `\varepsilon^{1/N}`.
        """
        if self._factors is not None:
            return self._factors
        from sage.all import CC

        out = []
        if self.exact is None:
            for c in (self.norm_num, self.norm_den):
                out.append([(complex(r), 1) for r in np.roots(c)]
                           if len(c) > 1 else [])
        else:
            for q in self.norm_exact():
                terms = []
                if q.degree() > 0:
                    for fac, m in q.squarefree_decomposition():
                        for r in fac.roots(CC, multiplicities=False):
                            terms.append((complex(r), int(m)))
                out.append(terms)
        self._factors = (out[0], out[1])
        return self._factors

    def norm_factors_hp(self, bits):
        r"""
        :meth:`norm_factors` with the roots to ``bits`` bits, as Sage complex
        numbers, or ``None`` without exact coefficients.
        """
        from sage.all import ComplexField

        if self.exact is None:
            return None
        key = int(bits)
        if key in self._factors_hp:
            return self._factors_hp[key]
        C = ComplexField(key)
        out = []
        for q in self.norm_exact():
            terms = []
            if q.degree() > 0:
                for fac, m in q.squarefree_decomposition():
                    for r in fac.roots(C, multiplicities=False):
                        terms.append((r, int(m)))
            out.append(terms)
        self._factors_hp[key] = (out[0], out[1])
        return self._factors_hp[key]

    def norm_leading(self):
        r"""
        The leading coefficient of `\operatorname{N}(\varphi)`'s numerator
        over that of its denominator, exactly, or ``None``.

        With :meth:`norm_factors_hp` this turns the norm into a product,
        which is what makes it computable near its `N`-fold zero.
        """
        if self.exact is None:
            return None
        if not hasattr(self, "_norm_lead"):
            num, den = self.norm_exact()
            self._norm_lead = (None if num == 0 or den == 0 else
                               num.leading_coefficient() / den.leading_coefficient())
        return self._norm_lead

    def norm_root_hp(self, seed, bits, pole=False):
        r"""
        The zero (or pole) of `\operatorname{N}(\varphi)` nearest ``seed``,
        to ``bits`` bits, or ``None`` without exact coefficients.

        At `P` that zero has multiplicity `N`.  Differentiating `N-1` times
        to make it simple is fine in exact arithmetic and useless in
        floating point once `N` is large, so the squarefree part is taken
        over `\QQ` and its roots -- all simple -- are found at the working
        precision.
        """
        from sage.all import ComplexField

        if self.exact is None:
            return None
        num, den = self.norm_exact()
        q = den if pole else num
        if q.degree() < 1:
            return None
        q = q // q.gcd(q.derivative())
        r = q.roots(ComplexField(bits), multiplicities=False)
        return min(r, key=lambda t: abs(complex(t) - complex(seed))) if r else None

    def norm_value(self, x):
        r"""
        `\operatorname{N}(\varphi)(x)` from its factorisation, so that its
        `N`-fold zero costs nothing.

        Evaluated from the expanded polynomial the value at `x` near `x_P`
        is `c\,(x-x_P)^N` recovered by cancellation among terms of size one:
        at `|x - x_P| = 0.135`, which is where the fibre sits over
        `|w| = 2.5\cdot10^{-6}` on the degree-6 model of LMFDB 249.a.249.1,
        that is six digits gone.  As `c\prod_j (x-r_j)^{m_j}` there is
        nothing to cancel.
        """
        lead = self.norm_leading()
        if lead is None:
            return (_eval(self.norm_num, x)
                    / _eval(self.norm_den, x))
        zeros, poles = self.norm_factors()
        v = (np.complex128(complex(lead)) if np.ndim(x) == 0
             else np.full(np.shape(x), complex(lead), complex))
        for r, m in zeros:
            v = v * (x - r) ** m
        for r, m in poles:
            v = v / (x - r) ** m
        return v

    def dlog_norm(self, x):
        r"""`(\log\operatorname{N}(\varphi))'` at `x`, as a sum over the
        zeros and poles -- :meth:`dlog_norm_about` with the base at zero."""
        zeros, poles = self.norm_factors()
        out = (np.complex128(0) if np.ndim(x) == 0
               else np.zeros(np.shape(x), complex))
        for r, m in zeros:
            out = out + m / (x - r)
        for r, m in poles:
            out = out - m / (x - r)
        return out

    def phi_stable(self, x, y):
        r"""
        `\varphi = A + By`, evaluated without cancellation where it is small.

        `\varphi\,\iota(\varphi) = \operatorname{N}(\varphi)` and
        `\iota(\varphi) = A - By` is the large one there, so
        `\varphi = \operatorname{N}(\varphi)/(A - By)` carries the zero
        exactly while `A + By` computes it by subtraction.  It is the same
        substitution :meth:`dlogphi` already makes for the logarithmic
        derivative, and the tracker needs it for `\varphi` itself: on the
        approach to the `N`-fold point the direct form loses every digit --
        `|\varphi(x,y) - w|/|w|` reaches `9\cdot10^{-4}` at
        `|w| = 2.5\cdot10^{-6}` -- and Newton then converges to the root of
        a polynomial that is not the one intended.
        """
        A, B = self.A(x), self.B(x)
        small, big = A + B * y, A - B * y
        with np.errstate(divide="ignore", invalid="ignore"):
            alt = self.norm_value(x) / big
        return np.where((np.abs(small) <= np.abs(big)) & np.isfinite(alt),
                        alt, small)

    def dphi_dx_stable(self, x, y):
        r"""`d\varphi/dx` along the curve, by the same substitution."""
        A, B = self.A(x), self.B(x)
        small, big = A + B * y, A - B * y
        dA, dB = self.dA(x), self.dB(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            dy = _eval(self.Fp, x) / (2 * y)
            direct = dA + dB * y + B * dy
            dbig = dA - dB * y - B * dy
            alt = self.phi_stable(x, y) * (self.dlog_norm(x) - dbig / big)
        return np.where((np.abs(small) <= np.abs(big)) & np.isfinite(alt),
                        alt, direct)

    def dlog_norm_about(self, base, tol=1e-8):
        r"""
        `u \mapsto (\log\operatorname{N}(\varphi))'` at `x = \mathrm{base}
        + u`, as a sum over the zeros and poles.

        Evaluating `\operatorname{N}(\varphi)'/\operatorname{N}(\varphi)`
        from the polynomials cancels catastrophically near a zero of high
        order: at `x_P` the value is of size `|u|^{N}` while the terms are of
        size one, so below `|u| = \varepsilon^{1/N}` -- a tenth at `N = 15`
        -- the quotient is noise.  Taylor-shifting the polynomial does not
        help, since the shift then has to produce `N` leading coefficients
        that cancel to zero and in float64 they do not.  As a sum
        `\sum_j m_j/(u + \mathrm{base} - r_j)` there is nothing to cancel:
        the factor at ``base`` contributes exactly `N/u`, the rest is regular,
        and the whole thing is as accurate at `u = 10^{-17}` as at `u = 1`.
        """
        zeros, poles = self.norm_factors()
        terms = [(base - r, m) for r, m in zeros]
        terms += [(base - r, -m) for r, m in poles]
        scale = tol * max(1.0, abs(base))
        terms = [(0.0 if abs(d) <= scale else d, m) for d, m in terms]

        def f(u):
            return sum(m / (d + u) for d, m in terms)

        return f

    def y_of(self, x, w):
        r"""
        `y` from `(x,w)`, i.e. `(w - A)/B`.

        This is `0/0` at a zero of `B`, where :meth:`_split_pinches` takes
        over; everywhere else it is what tells the two points over an `x`
        apart.
        """
        x = np.asarray(x, dtype=complex)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (w - self.A(x)) / self.B(x)

    def _split_pinches(self, x, y):
        r"""
        The fibre with the two points over a zero of `B` told apart.

        A zero `x_0` of `B` forces `w = A(x_0)`, so such a root of
        `\mathcal{P}(\cdot,w)` is a double one and always comes with its
        partner: the fibre contains both of `(x_0, \pm\sqrt{F(x_0)})` and
        `\varphi` is unramified over `A(x_0)`.  There `(w-A)/B` is the ratio
        of two vanishing numbers, which tends to `-A'/B'` -- a perfectly
        finite value belonging to neither point -- so `y` is taken from the
        curve instead, with the two roots of the pair given opposite signs.
        """
        at = list(np.nonzero(_vanishes(self.b, x, 1e-6))[0])
        used = set()
        for i, k in enumerate(at):
            if k in used:
                continue
            for l in at[i + 1:]:
                if l in used or abs(x[k] - x[l]) > 1e-6 * (1.0 + abs(x[k])):
                    continue
                # the mean of a double root is accurate where each of the
                # two is only accurate to the square root of the epsilon,
                # and the principal square root of F would flip between them
                c = (x[k] + x[l]) / 2
                r = np.sqrt(complex(_eval(self.F, c)))
                x[k], x[l], y[k], y[l] = c, c, r, -r
                used.update((k, l))
                break
        return x, y

    # ----------------------------------------------------- critical values

    def exact_polynomial(self):
        r"""`\mathcal{P}(x,w)` in `\QQ[x,w]`, for the resultant."""
        from sage.all import QQ, PolynomialRing, gcd

        R = PolynomialRing(QQ, "x")
        S = PolynomialRing(QQ, ["x", "w"])
        xx, ww = S.gens()
        if self.exact is not None:
            # multiplied out over QQ: see __init__ for what the float route
            # does to a phi of high degree
            F, a, ad, b, bd = self.exact
            p2 = -(bd * ad) ** 2
            p1 = 2 * bd ** 2 * ad * a
            p0 = b ** 2 * ad ** 2 * F - bd ** 2 * a ** 2
        else:
            p0, p1, p2 = (rounded(q) for q in self.p)
        g = gcd(gcd(p0, p1), p2)
        if g.degree() > 0:
            p0, p1, p2 = p0 // g, p1 // g, p2 // g
        conv = lambda q: S(q.subs({R.gen(): xx}))
        return conv(p0) + conv(p1) * ww + conv(p2) * ww ** 2

    def _branch_polynomial(self):
        r"""
        The squarefree part of `\operatorname{disc}_x\mathcal{P}`, whose
        roots are the candidate branch values.

        The discriminant has a multiple root wherever two branch points
        share a value or one of them is worse than simple -- the degree-14
        cover of LMFDB 249.a.249.1 has a fourfold one -- and a numerical root
        finder spreads such a root over `\varepsilon^{1/m}`, which at 53 bits
        is a hundredth: four "critical values" a hundredth apart in place of
        one, each with a cut of its own.  Dividing by
        `\gcd(f, f')` leaves every distinct value a simple root, which is
        then found to the last digit at whatever precision is asked for.

        It is the discriminant and not the resultant because

        .. math:: \operatorname{Res}_x(\mathcal{P}, \mathcal{P}_x)
                  = \pm\, a_N\, \operatorname{disc}_x \mathcal{P},

        and the extra `a_N` is not a branch value at all: where the leading
        coefficient vanishes the fibre's degree in `x` drops and a point
        runs off to `x = \infty`, which the resultant reports and the cut
        system must not.  A node there is spurious -- on the degree-6 model
        of LMFDB 249.a.249.1, `a_N(w) = 2(w-2)` and the function field says
        every place over `w = 2` has `e = 1` -- and it used to cost fifty-
        three seconds on a leg into a value at which nothing happens.  What
        does happen at `w = \infty` in the fibre is still caught, since
        `\operatorname{disc}` specialises at a root of `a_N` to
        `\pm a_{N-1}^2\operatorname{disc}` of the reduced polynomial: the
        value survives exactly when a second point goes there too.

        This is the same correction :meth:`_pinch_polynomials` makes for the
        zeros of `B`, where the fibre degenerates in `x` while `\varphi`
        stays unramified, and for the same reason.
        """
        from sage.all import QQ, PolynomialRing

        P = self.exact_polynomial()
        x, w = P.parent().gens()
        T = PolynomialRing(QQ, "W")
        res = T(P.resultant(P.derivative(x), x).polynomial(w))
        lead = T(P.coefficient({x: P.degree(x)}).polynomial(w))
        f, rem = res.quo_rem(lead)
        if rem != 0 or f.degree() < 1:        # not the expected identity
            f = res
        f = f // f.gcd(f.derivative())
        pinch, ramified = self._pinch_polynomials(P.parent(), T)
        if pinch is None or pinch.degree() < 1:
            return f
        f = f // f.gcd(pinch)
        return f if ramified is None else f * (ramified // f.gcd(ramified))

    def _pinch_polynomials(self, S, T):
        r"""
        `(\text{the values over the zeros of } B, \text{those of them that
        ramify})`, as polynomials in `w`, or ``(None, None)``.

        A zero `x_0` of `B` forces `\varphi(x_0, \pm y_0) = A(x_0)`, so
        `\mathcal{P}(\cdot, A(x_0))` has a double root in `x` although
        `\varphi` is unramified over it -- the two points are distinct.
        `\operatorname{Res}_x(\mathcal{P},\mathcal{P}_x)` reports such a
        value all the same, and the cut system must not: a node there puts a
        chart leg where nothing comes together.  Those values are
        `\operatorname{Res}_x(B_{\mathrm{num}},\, w\,a_d - a)`.

        A zero of `B` can be a ramification point as well: there
        `d\varphi = (A' \pm B'y)\,dx` with `y^2 = F`, so it is one exactly
        when `(A')^2 = (B')^2F`, and its value must be kept.  Both conditions
        are polynomial identities over `\QQ`, so no tolerance decides this --
        which matters, since a zero of `B` of multiplicity two still accounts
        for a single coincidence, and counting the fibre numerically cannot
        tell that from a coincidence with a branch point in it.
        """
        from sage.all import gcd

        if self.pure_x or self.exact is None or len(self.b) < 2:
            return None, None
        x, w = S.gens()
        F, a, ad, b, bd = self.exact
        R = F.parent()
        if b.degree() < 1:
            return None, None
        into = lambda q: S(q.subs({R.gen(): x}))

        def values(q):
            r = T(into(q).resultant(w * into(ad) - into(a), x).polynomial(w))
            return r // r.gcd(r.derivative()) if r.degree() > 0 else r

        g = gcd(b, ((a.derivative() * ad - a * ad.derivative()) ** 2 * bd ** 2
                    - b.derivative() ** 2 * ad ** 4 * F))
        return values(b), (values(g) if g.degree() >= 1 else None)

    def places_over(self, z, tol=1e-9):
        r"""
        The places of the curve over `\varphi = z`, exactly, as a list of
        dictionaries with

        * ``"e"`` -- the ramification index `v_p(\varphi - z)`;
        * ``"degree"`` -- the degree of the place, so `e \cdot \deg` points;
        * ``"infinite"`` -- whether it lies over `x = \infty`;
        * ``"uniformizer"`` -- a local uniformizer at it, from Sage;
        * ``"poles"`` -- the `x` where that uniformizer has a pole.

        ``z`` may be ``None``, which asks for the places over `\varphi =
        \infty` -- the poles of `\varphi`, with `e = -v_p(\varphi)`.  That
        is where the ray leg ends, and it is `Q`.

        This is what the float64 tests around it were approximating.  `e` was
        read off the cut system's monodromy, which is right but indirect;
        whether a point sits at infinity was `|a_N(z)|` against a threshold,
        and which of `x`, `y`, `\tau` is a coordinate there was `|y|` against
        another.  All three are valuations, and the function field returns
        them rather than estimating them -- on the degree-6 model of LMFDB
        249.a.249.1 it says in a tenth of a second that `\varphi^{-1}(2)`
        contains a place at infinity, a place at `x = 0`, a Weierstrass
        place and a place of degree eleven, which is the whole reason no
        single naive coordinate serves there.

        ``None`` without exact coefficients.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: C = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1)
            sage: sorted(p["e"] for p in C.places_over(0))
            [5]
            sage: sorted(p["e"] for p in C.places_over(None))
            [5]
        """
        if self.exact is None:
            return None
        from sage.all import QQ

        from .functions import _field

        key = ("places", None if z is None else complex(z))
        if getattr(self, "_places", None) is None:
            self._places = {}
        if key in self._places:
            return self._places[key]
        Fq, a, ad, b, bd = self.exact
        K, L = _field(Fq)
        y = L.gen()
        into = lambda q: K(q.list())
        phi = into(a) / into(ad) + (into(b) / into(bd)) * y
        try:
            if z is None:
                div = phi.divisor_of_poles()
            else:
                zz = QQ(complex(z).real) if abs(complex(z).imag) < tol else None
                if zz is None:
                    return None
                div = (phi - zz).divisor_of_zeros()
        except Exception:                                      # noqa: BLE001
            return None
        inf = set(L.places_infinite())
        out = []
        for pl, m in div.list():
            u = pl.local_uniformizer()
            out.append({"e": int(m), "degree": int(pl.degree()),
                        "infinite": pl in inf, "uniformizer": u,
                        "place": pl})
        self._places[key] = out
        return out

    def endpoint_ramification(self, z, kind="simple"):
        r"""Ramification used to desingularize a cut endpoint.

        Cut vertices are numerical approximations of critical values.
        Converting such a value to QQ generally gives a nearby *regular*
        fibre, so its valuations cannot determine the endpoint's index.

        P and Q have index N under the divisor hypothesis; a detour corner
        has index one. For the remaining vertices, a simple zero of the
        exact discriminant certifies index two. At an exactly rational
        critical value the exact places also support higher ramification.
        The common substitution uses the least common multiple when
        several places have different indices. Other higher degeneracies
        are refused rather than guessed.
        """
        from math import lcm

        if kind == "none":
            return 1
        if kind == "P":
            return self.N
        if kind == "Q" or z is None:
            places = self.places_over(None)
            return lcm(*(int(p["e"]) for p in places)) if places else self.N
        if kind != "simple":
            raise ValueError("unknown cut endpoint kind %r" % (kind,))
        if self.exact is None:
            raise NotImplementedError(
                "endpoint ramification needs exact coefficients or monodromy")
        from sage.all import CC, QQ, PolynomialRing

        value = complex(z)
        if self.pure_x:
            # A degree-one rational map is unramified on the x-line.
            # Its composition with the hyperelliptic map has index two
            # at each branch point of a smooth hyperelliptic curve.
            if max(len(self.a), len(self.ad)) == 2:
                return 2
            places = self.places_over(z) if value.imag == 0 else None
            if places and max(p["e"] for p in places) > 1:
                return lcm(*(int(p["e"]) for p in places))
            raise NotImplementedError(
                "ramification of this composite x-map needs exact places")

        if not hasattr(self, "_endpoint_discriminant"):
            P = self.exact_polynomial()
            x, w = P.parent().gens()
            T = PolynomialRing(QQ, "W")
            res = T(P.resultant(P.derivative(x), x).polynomial(w))
            lead = T(P.coefficient({x: P.degree(x)}).polynomial(w))
            disc, remainder = res.quo_rem(lead)
            if remainder:
                raise ArithmeticError("cover resultant is not divisible by its leading coefficient")
            self._endpoint_discriminant = disc
            self._simple_endpoint_values = np.array([
                complex(root)
                for factor, multiplicity in disc.squarefree_decomposition()
                if multiplicity == 1
                for root in factor.roots(CC, multiplicities=False)], complex)

        disc = self._endpoint_discriminant
        if value.imag == 0 and disc(QQ(value.real)) == 0:
            # Equality is exact, with no snapping to a nearby rational.
            places = self.places_over(z)
            if places and max(p["e"] for p in places) > 1:
                return lcm(*(int(p["e"]) for p in places))
        roots = self._simple_endpoint_values
        error = 64 * np.finfo(float).eps * max(1.0, abs(value))
        if len(roots) and np.min(abs(roots - value)) <= error:
            return 2
        raise NotImplementedError(
            "endpoint %r has no verified simple ramification; "
            "an exact endpoint or its monodromy is required" % (z,))

    def degenerate_values(self):
        r"""
        Every `w` over which the fibre degenerates *in the `x`-coordinate*:
        the roots of the squarefree part of
        `\operatorname{Res}_x(\mathcal{P}, \mathcal{P}_x)`.

        This is **not** :meth:`critical_values`.  That one is the branch locus
        of `\varphi`, with the pinches taken back out because `\varphi` is
        unramified over them and they must not become nodes of the cut system.
        Here they must be put back: over a pinch two points of the fibre share
        an `x`, so the polynomial whose roots are being tracked has a double
        one, and a step bound that ignored it would step straight through a
        collision.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: len(Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).degenerate_values())
            5
        """
        if getattr(self, "_degen", None) is None:
            from sage.all import CC, PolynomialRing, QQ

            if self.pure_x:
                # P is a perfect square there, so Res_x(P, P_x) vanishes
                # identically and has nothing to say.  The fibre in x is the
                # roots of w a_d - a, which collide exactly over the critical
                # values of A -- and those are what critical_values computes
                # for this case, together with the Weierstrass points where
                # the two sheets over an x come together.
                self._degen = np.asarray(self.critical_values(), complex)
                return self._degen
            P = self.exact_polynomial()
            xx, _ = P.parent().gens()
            T = PolynomialRing(QQ, "W")
            res = P.resultant(P.derivative(xx), xx)
            f = T(res.polynomial(P.parent().gens()[1]))
            if f.degree() > 0:
                f = f // f.gcd(f.derivative())
                out = [complex(r) for r in f.roots(CC, multiplicities=False)]
            else:
                out = []
            self._degen = np.array(out, complex)
        return self._degen

    def critical_values(self, tol=1e-9):
        r"""
        The finite critical values of `\varphi`, from
        `\operatorname{Res}_x(\mathcal{P}, \mathcal{P}_x)`, clustered and
        snapped to zero.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: v = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).critical_values()
            sage: len(v), bool(min(abs(v)) < 1e-12)
            (5, True)
        """
        from sage.all import CC

        if self.pure_x:
            return self._critical_values_pure(tol)
        f = self._branch_polynomial()
        out = []
        for r, m in f.roots(CC, multiplicities=True):
            v = complex(r)
            if abs(v) < tol:
                v = 0.0
            if not any(abs(v - u) < tol * max(1.0, abs(v)) for u in out):
                out.append(v)
        return np.array(out)

    def _critical_values_pure(self, tol):
        r"""
        Critical values when `\varphi = A(x)` factors through `x`.

        The resultant vanishes identically there, `\mathcal{P}` being a
        square.  `\varphi` is `A` composed with the hyperelliptic map, so it
        ramifies over the Weierstrass points, over `\infty`, and over the
        critical points of `A` -- and their images are the critical values.
        """
        pts = list(np.roots(self.F))
        num = np.polysub(np.polymul(np.polyder(self.a), self.ad),
                         np.polymul(self.a, np.polyder(self.ad)))
        if len(num) > 1:
            pts += list(np.roots(num))
        vals = []
        for x in pts:
            if abs(_eval(self.ad, x)) < tol * max(1.0, abs(x) ** len(self.ad)):
                continue                      # a pole of A, so over infinity
            vals.append(complex(self.A(x)))
        vals += self._A_at_infinity()
        out = []
        for v in vals:
            if abs(v) < tol:
                v = 0.0
            if not any(abs(v - u) < tol * max(1.0, abs(v)) for u in out):
                out.append(v)
        return np.array(out)

    def _A_at_infinity(self):
        r"""
        `[A(\infty)]` when that is a finite critical value, else ``[]``.

        `x` ramifies at the point at infinity -- it is a Weierstrass point --
        so `\varphi = A \circ x` ramifies there too and `A(\infty)` is a
        critical value, unless it is `\infty` itself, which is `\varphi(Q)`.
        """
        da, dad = len(self.a) - 1, len(self.ad) - 1
        if da > dad:
            return []
        return [complex(self.a[0] / self.ad[0]) if da == dad else 0.0]

    def critical_values_hp(self, bits):
        r"""
        The finite critical values as Sage complex numbers to ``bits`` bits.

        The float64 values cap everything downstream at about fifteen digits,
        the cut system's vertices being exactly these.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: v = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).critical_values_hp(120)
            sage: len(v)
            5
        """
        from sage.all import ComplexField

        C = ComplexField(bits)
        if self.pure_x:
            if self.exact is None:
                F, a, ad = (rounded(c) for c in (self.F, self.a, self.ad))
            else:
                F, a, ad, _b, _bd = self.exact
            pts = list(F.roots(C, False))
            num = a.derivative() * ad - a * ad.derivative()
            if num.degree() > 0:
                pts += list(num.roots(C, False))
            vals = [a(x) / ad(x) for x in pts if abs(ad(x)) > 1e-9]
            vals += [C(v) for v in self._A_at_infinity()]
            out = []
            for v in vals:
                if abs(v) < 1e-9:
                    v = C(0)
                if not any(abs(v - u) < 1e-9 * max(1, abs(v)) for u in out):
                    out.append(v)
            return out
        f = self._branch_polynomial()
        out = []
        for r in f.roots(C, False):
            v = C(0) if abs(r) < 1e-9 else r
            if not any(abs(v - u) < 1e-9 * max(1, abs(v)) for u in out):
                out.append(v)
        return out

    def _angular_order(self, eps):
        r"""
        The critical values by argument about the perturbed centroid, rotated
        so that `\varphi(P) = 0` comes first, with the centroid and the amount
        of the rotation.

        The centroid is displaced by `\varepsilon(\pi + e\,i)`.  A rational
        displacement can leave a symmetric configuration tied, and collinear
        critical values tie outright; an irrational direction orders them
        deterministically.
        """
        v = self.critical_values()
        if min(abs(v)) > 1e-12:
            raise ValueError("0 is not a critical value: phi is not N P - N Q")
        c = v.mean() + eps * (np.pi + np.e * 1j)
        order = sorted(range(len(v)), key=lambda i: np.angle(v[i] - c))
        k = min(range(len(v)), key=lambda j: abs(v[order[j]]))
        return v[order[k:] + order[:k]], c, k

    def ordered_critical_values(self, eps=1e-13):
        r"""
        `z_1 = \varphi(P) = 0, z_2, \dots, z_r` in the order the cut system
        visits them, and the outward unit direction of the ray to `\infty`.

        This is the *straight* cut system; :meth:`cut_path` is what the
        integrator uses, and falls back to a detour when this one is not
        embedded.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: z, d = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).ordered_critical_values()
            sage: bool(abs(z[0]) < 1e-14), len(z)
            (True, 5)
        """
        z, c, _ = self._angular_order(eps)
        check_path_embedded(z)
        d = z[-1] - c
        return z, d / abs(d)

    def cut_path(self, eps=1e-13, radius=1.7, arc=24, fan=64):
        r"""
        The cut system `\Gamma`: its vertices, what each one is, and the
        outward direction of the ray to `\infty`.

        `\Gamma` must be an embedded arc from `\varphi(P) = 0` to `\infty`
        through every critical value, so that
        `\mathbf{P}^1\smallsetminus\Gamma` is a disc.  Joining the critical
        values by straight segments in the order of their argument about the
        centroid does that whenever they are in general position -- the order
        makes them the vertices of a star-shaped polygon, and dropping the
        closing edge leaves an embedded path.  But the rotation that brings
        `\varphi(P)` to the front puts that closing edge *back*, as the step
        from the last of the cyclic order to the first, and it is a chord: for
        collinear critical values with `\varphi(P)` between them it runs
        straight through the others.

        No straight-segment path can fix that, so the closing step is replaced
        by a detour out to radius ``radius`` times the configuration's, round
        through the angular gap the cyclic order leaves empty, and back in.
        The ray is then taken in the first of ``fan`` directions that keeps the
        whole thing embedded.  For five real critical values this is exactly
        the semicircle round one side with the ray leaving perpendicular on
        the other.

        The detour's corners are returned as ordinary vertices of `\Gamma`
        carrying no ramification, so the cut graph merely gets finer: the
        sheets glue across them by the identity and every formula is unchanged.
        Their position is immaterial -- any embedded `\Gamma` gives the same
        answer -- which is why they need no precision.

        Returns ``(nodes, kinds, direction)``, ``kinds[i]`` being ``"P"`` at
        `\varphi(P)`, ``"simple"`` at the other critical values and ``"none"``
        at a detour corner.

        EXAMPLES::

            sage: from hyperell_regulator.regulator.cover import Cover
            sage: from hyperell_regulator.regulator.polynomials import ring
            sage: x = ring().gen()
            sage: import numpy as np
            sage: n, k, d = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).cut_path()
            sage: len(n), k[0], set(k[1:])        # general position: no detour
            (5, 'P', {'simple'})
            sage: C = Cover(x^5 - 5*x^3 + 4*x, x, 0)   # all real
            sage: n, k, d = C.cut_path()
            sage: bool(len(n) > 5), 'none' in k, bool(abs(d + 1j) < 1e-12)
            (True, True, True)
        """
        z, c, k = self._angular_order(eps)
        r = len(z)
        nodes = list(z)
        kinds = ["P"] + ["simple"] * (r - 1)
        d = self._pick_ray(nodes, kinds, c, fan)
        if d is not None:
            return np.array(nodes), kinds, d
        if k == 0:
            raise NotImplementedError(
                "no embedded cut system: the critical values are in general "
                "position but no ray escapes")
        j = r - 1 - k                       # the closing edge, put back by the
        R = radius * max(np.abs(z - c))     # rotation, is the one to route round
        bend = _detour(nodes[j], nodes[j + 1], c, R, arc)
        nodes = nodes[:j + 1] + bend + nodes[j + 1:]
        kinds = kinds[:j + 1] + ["none"] * len(bend) + kinds[j + 1:]
        d = self._pick_ray(nodes, kinds, c, fan)
        if d is None:
            raise NotImplementedError(
                "no embedded cut system even with the closing edge routed "
                "round: no ray escapes")
        return np.array(nodes), kinds, d

    def _pick_ray(self, nodes, kinds, c, fan):
        r"""
        The direction in which the ray to `\infty` leaves the last vertex,
        or ``None`` if none keeps `\Gamma` embedded.

        """
        a = np.array(nodes)
        last = a[-1]
        scale = max(np.abs(a - c).max(), 1.0)
        crit = a[[i for i, t in enumerate(kinds) if t != "none"]]
        if abs(last - c) > 0:                  # radial: right whenever it works
            d = (last - c) / abs(last - c)
            try:
                check_path_embedded(a, d, crit)
                return d
            except NotImplementedError:
                pass
        best, room = None, -1.0
        for d in [np.exp(2j * np.pi * m / fan) for m in range(fan)]:
            try:
                check_path_embedded(a, d, crit)
            except NotImplementedError:
                continue
            probe = last + d * np.linspace(0.02, 6.0, 60) * scale
            gap = min(_dist_to_path(p, a[:-1]) for p in probe)
            if gap > room:
                best, room = d, gap
        return best


def _dist_to_path(p, nodes):
    """Distance from `p` to the polyline through `nodes`."""
    best = np.inf
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        t = np.clip(((p - u) * np.conj(v - u)).real / abs(v - u) ** 2, 0.0, 1.0)
        best = min(best, abs(u + t * (v - u) - p))
    return best


def _detour(a, b, c, R, n):
    r"""
    Vertices routing the step from `a` to `b` out round radius `R` about `c`.

    The sweep runs in the direction of increasing argument, which is the gap
    the cyclic order leaves empty, so the detour stays outside everything.
    The first and last vertices sit radially outward from `a` and `b`, so the
    segments joining them to `a` and `b` are radial.
    """
    ta, tb = np.angle(a - c), np.angle(b - c)
    sweep = np.mod(tb - ta, 2 * np.pi)
    t = np.linspace(0.0, 1.0, n)
    return list(c + R * np.exp(1j * (ta + sweep * t)))


def _shift_poly(p, c):
    """The coefficients of `p(c+u)` in `u`, highest degree first."""
    r = np.asarray(p, complex).copy()
    div = np.array([1.0, -complex(c)])
    out = []
    while len(r) > 1:
        r, rem = np.polydiv(r, div)
        out.append(complex(rem[-1]) if len(rem) else 0.0)
    out.append(complex(r[0]) if len(r) else 0.0)
    return np.array(out[::-1], complex)


def check_path_embedded(e, ray=None, crit=None, tol=1e-9):
    r"""
    Check that `\Gamma = [z_1,z_2] \cup \dots \cup [z_r,\infty]` is embedded.

    The derivation needs `\mathbf{P}^1 \smallsetminus \Gamma` to be a disc, so
    `\Gamma` must not meet itself and must not run through a critical value
    other than at its ends.  Sorting by argument about the centroid makes the
    `z_i` the vertices of a star-shaped polygon, but the rotation that brings
    `\varphi(P)` to the front deletes the edge `z_r z_1` and replaces it by the
    ray, and the result need not stay embedded -- for collinear critical values
    it typically does not.

    EXAMPLES::

        sage: from hyperell_regulator.regulator.cover import check_path_embedded
        sage: import numpy as np
        sage: check_path_embedded(np.array([-2., -1., 0., 1., 2.]))
        sage: check_path_embedded(np.array([0., 1., 2., -2., -1.]))
        Traceback (most recent call last):
        ...
        NotImplementedError: ...
    """
    e = np.asarray(e, complex)
    scale = max(np.abs(e).max(), 1.0)
    if ray is None:
        ray = e[-1] - e.mean()
        if abs(ray) < tol * scale:
            raise NotImplementedError(
                "the last critical value sits at the centroid")
    ray = ray / abs(ray)
    if crit is None:
        crit = e
    crit = np.asarray(crit, complex)
    pts = list(e) + [e[-1] + ray * 10.0 * scale]
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def side(a, b, c):
        return ((b - a).conjugate() * (c - a)).imag

    for i, (a, b) in enumerate(segs):
        for c in crit:
            if abs(c - a) < tol * scale or abs(c - b) < tol * scale:
                continue
            t = ((c - a).conjugate() * (b - a)).real / abs(b - a) ** 2
            if -tol < t < 1 + tol and abs(a + t * (b - a) - c) < tol * scale:
                raise NotImplementedError(
                    "the critical value %r lies on the segment %r--%r" % (c, a, b))
        for j in range(i + 2, len(segs)):
            c, d = segs[j]
            if (side(a, b, c) * side(a, b, d) < 0
                    and side(c, d, a) * side(c, d, b) < 0):
                raise NotImplementedError(
                    "the segments %r--%r and %r--%r cross" % (a, b, c, d))
            # collinear segments never register as crossing, but they can lie
            # on top of one another, which is just as fatal
            if (abs(side(a, b, c)) < tol * scale
                    and abs(side(a, b, d)) < tol * scale):
                u = (b - a) / abs(b - a)
                s0, s1 = 0.0, abs(b - a)
                t0 = ((c - a) / u).real
                t1 = ((d - a) / u).real
                if min(max(t0, t1), s1) - max(min(t0, t1), s0) > tol * scale:
                    raise NotImplementedError(
                        "the segments %r--%r and %r--%r overlap" % (a, b, c, d))
