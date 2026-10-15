r"""
`\int_C \omega_a \wedge \bar\omega_b` and `\int_C \log\varphi\,\omega_a
\wedge \bar\omega_b` by Stokes' theorem on the sheets of `\varphi`.

Let `\varphi : C \to \mathbf{P}^1` have degree `N` with
`\operatorname{div}(\varphi) = N P - N Q`.  Order the finite critical values of
`\varphi` by argument about their perturbed centroid, starting at
`\varphi(P) = 0`, join consecutive ones by segments and the last to
`\varphi(Q) = \infty` by a ray, and call the resulting path `\Gamma`.  Its
complement is a disc, so its `N` lifts `D_1,\dots,D_N` -- the sheets -- are
discs too, and on each of them

.. math:: g(z) = \int_P^z \lambda\,\omega_a, \qquad \lambda \in \{1, \log\varphi\}

is single valued, with `d(g\,\bar\omega_b) = \lambda\,\omega_a\wedge
\bar\omega_b`.  Stokes then gives one term per edge of the cut graph
`\varphi^{-1}(\Gamma)`, each edge bounding two sheets and being traversed once
each way:

.. math:: \int_C \lambda\,\omega_a \wedge \bar\omega_b
          = \sum_e \Bigl[ C_e \int_e \bar\omega_b
            + \Delta\lambda \int_e \Bigl(\int_{\mathrm{start}(e)}^y \omega_a
              \Bigr)\bar\omega_b \Bigr],

where `C_e = U^{(k)}_j - U^{(l)}_{j'} + \int_e \lambda\,\omega_a` is built from
the partial sums of `\int\lambda\,\omega_a` along the two sheet boundaries, and
`\Delta\lambda` is the jump of `\lambda` across the cut.  For `\lambda = 1` the
jump vanishes and the formula is purely bilinear in the periods; for
`\lambda = \log\varphi` it is `2\pi i`, since `\Gamma` joins the zero of
`\varphi` to its pole and so is already a branch cut for the logarithm.

`\partial D_k` runs up the left side of `\sigma_1,\dots,\sigma_r` and back down
the right; which lift of `\sigma_i` is which side in sheet `k` is read off by
following the fibre round a racetrack hugging `\Gamma`, which stays in the
complement and so must close up with the identity permutation.

The link to :mod:`~hyperell_regulator.regulator.plane` is

.. math:: \operatorname{Im}\int_C \rho\,\omega_a \wedge \bar\omega_b
          = -4 \int_{\mathbf{C}} \rho\,\frac{G_{ab}}{|F|}\,dA,
          \qquad G_{ab} = \operatorname{Re}\bigl(x^a \bar x^{\,b}\bigr),

for `\rho = 1` and, after symmetrising in `(a,b)`, for `\rho = \log|\varphi|`.
"""

import numpy as np

from ._legs import LegFailure, edge_leg, ray_leg

__all__ = [
    "racetrack",
    "follow",
    "cut_data",
    "boundary_walks",
    "hold_radii",
    "gluing_permutations",
    "wedge_matrix",
    "log_wedge_matrix",
]


# ------------------------------------------------------------- the racetrack

def _line(a, b, n):
    return list(a + (b - a) * np.linspace(0, 1, n)[1:])


def _arc(c, rad, th0, sweep, n):
    return list(c + rad * np.exp(1j * (th0 + sweep * np.linspace(0, 1, n)[1:])))


def _miter(v, eps, ea, eb):
    r"""
    Where the two offset lines meet at a corner concave for the side in hand.

    The offset is always `\varepsilon` to the left of the direction of travel,
    so on the return pass it is `i\varepsilon e_a` for the *reversed* direction.
    Putting an arc here instead would carry the path round the vertex and pick
    up its monodromy spuriously.
    """
    rhs = 1j * eps * (eb - ea)
    A = np.array([[ea.real, -eb.real], [ea.imag, -eb.imag]])
    s = np.linalg.solve(A, np.array([rhs.real, rhs.imag]))[0]
    return v + 1j * eps * ea + s * ea


def cut_scale(z):
    r"""
    `(\mathrm{span}, \mathrm{gap})`: the largest and smallest distance
    between two nodes of the cut system.

    Everything the racetrack and the ray do is measured in these rather than
    in absolute units.  A cover need not be of size one: the degree-15 one on
    LMFDB 277.a.277.1 has its critical values at `0`, `0.035`, `-2.6 \pm
    16.2i` and `8.4\cdot 10^4`.  With a fixed offset the racetrack would
    swallow the two that are a hundredth apart; with a fixed reach the ray to
    `\infty` would stop well inside the configuration; and with a fixed
    number of points per unit length the tracking would take two million
    steps along the longest segment, which is what made that curve appear to
    hang.  The sheet tracking bisects wherever a step is too big for it, so a
    discretisation in units of the span costs nothing in accuracy.
    """
    a = np.asarray(z)
    d = np.abs(a[:, None] - a[None, :])
    span = float(d.max()) or 1.0
    gap = float(d[d > 0].min()) if np.any(d > 0) else span
    return span, gap


def racetrack(z, d, eps=0.02, M=60.0, dens=30.0):
    r"""
    The boundary of the `\varepsilon`-neighbourhood of `\Gamma`, and the index
    at which it passes each segment's reference point.

    Out along the left of every segment, once around `\infty`, back along the
    right, round the cap at `z_1`.  At a corner convex for the side being
    offset the boundary runs round an arc; at a concave one the offset lines
    simply meet.

    EXAMPLES::

        sage: from hyperell_regulator.regulator.cover import Cover
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: from hyperell_regulator.regulator.stokes import racetrack
        sage: z, d = Cover(x^5 + x^2 + 2*x + 1, x + 1, 1).ordered_critical_values()
        sage: path, marks, ref = racetrack(z, d)
        sage: bool(abs(path[0] - path[-1]) < 1e-12)
        True
    """
    span, gap = cut_scale(z)
    eps, M = eps * gap, M * span          # see cut_scale
    pts = list(z) + [z[-1] + M * d]
    r = len(z)
    e = [(pts[k + 1] - pts[k]) / abs(pts[k + 1] - pts[k]) for k in range(r)]
    ref = [(z[k] + z[k + 1]) / 2 for k in range(r - 1)] + [z[-1] + span * d]

    path, marks = [pts[0] + 1j * eps * e[0]], {}
    for s in (+1, -1):
        for k in (range(r) if s > 0 else range(r - 1, -1, -1)):
            end = (pts[k + 1] if s > 0 else pts[k]) + 1j * s * eps * e[k]
            mid = ref[k] + 1j * s * eps * e[k]
            path += _line(path[-1], mid,
                          max(8, int(dens * abs(mid - path[-1]) / span)))
            marks[(k, s)] = len(path) - 1
            path += _line(mid, end, max(8, int(dens * abs(end - mid) / span)))
            nxt = k + 1 if s > 0 else k - 1
            if 0 <= nxt < r:
                v = pts[k + 1] if s > 0 else pts[k]
                ea, eb = (e[k], e[nxt]) if s > 0 else (-e[k], -e[nxt])
                phi = np.angle(eb / ea)
                if abs(phi) < 1e-12:
                    pass              # collinear: the offsets already meet
                elif s * phi < 0:
                    path += _arc(v, eps, np.angle(path[-1] - v), phi, 60)
                else:
                    path += _line(path[-1], _miter(v, eps, ea, eb), 6)
        if s > 0:
            R = abs(path[-1] - z[-1])
            th0 = np.angle(path[-1] - z[-1])
            th1 = np.angle(pts[-1] - 1j * eps * e[-1] - z[-1])
            path += _arc(z[-1], R, th0, np.mod(th1 - th0, 2 * np.pi), 900)
    v = pts[0]
    th0, th1 = np.angle(path[-1] - v), np.angle(1j * e[0])
    sweep = np.mod(th1 - th0, 2 * np.pi)
    if np.mod(np.angle(e[0]) - th0, 2 * np.pi) < sweep:
        sweep -= 2 * np.pi
    path += _arc(v, eps, th0, sweep, 60)
    return np.array(path), marks, ref


def follow(cov, wa, wb, x, y, ratio=0.4, depth=40):
    r"""
    Continue the fibre from `w_a` to `w_b`.

    A step is accepted only when the nearest-candidate assignment is a
    bijection and every point's choice is at least `1/\mathrm{ratio}` times
    closer than its runner-up; otherwise the step is bisected.  Points are
    matched on `(x,y)`, not on `x` alone, which is what distinguishes them when
    `\varphi` factors through `x`.
    """
    def step(a, b, x, y, k):
        cx, cy = cov.fibre(b)
        D = np.abs(cx[None, :] - x[:, None]) + np.abs(cy[None, :] - y[:, None])
        j = np.argmin(D, axis=1)
        if len(set(j.tolist())) == len(x):
            best = D[np.arange(len(x)), j]
            second = np.partition(D, 1, axis=1)[:, 1]
            if np.all(best <= ratio * second):
                return cx[j], cy[j]
        if k >= depth:
            raise LegFailure("cannot follow the fibre near %r" % (b,))
        m = (a + b) / 2
        xm, ym = step(a, m, x, y, k + 1)
        return step(m, b, xm, ym, k + 1)

    return step(wa, wb, x, y, 0)


def cut_data(cov, z, d, dens=30.0, tries=4, **kwds):
    r"""
    Sheet labels and the branch of `\log\varphi` on each side of each cut.

    Returns `(\mathrm{upper}, \mathrm{lower}, \lambda^-, \Delta\lambda)`.
    `\Gamma` runs from `\varphi(P) = 0` to `\infty`, so it is already a cut for
    `\log\varphi`; the jump function is locally constant on the connected
    `\Gamma`, hence the same `2\pi i` on every edge, which is checked.

    The tracking bisects a step it cannot resolve, but a step coarse enough
    to carry two sheets past each other can still look resolved, and then the
    labels come out permuted -- which the checks here catch rather than pass
    on.  There is no density that is right for every cover, so the answer to
    a failed check is to take the racetrack four times finer and try again.
    """
    last = None
    for k in range(tries):
        try:
            return _cut_data(cov, z, d, dens=dens * 4 ** k, **kwds)
        except RuntimeError as exc:
            last = exc
    raise last


def _cut_data(cov, z, d, **kwds):
    """One pass of :func:`cut_data`, at the density it is given."""
    path, marks, ref = racetrack(z, d, **kwds)
    want = {v: k for k, v in marks.items()}
    theta = np.unwrap(np.angle(path))
    x, y = cov.fibre(path[0])
    rec, ang = {}, {}
    for j in range(1, len(path)):
        x, y = follow(cov, path[j - 1], path[j], x, y)
        if j in want:
            rec[want[j]] = (x.copy(), y.copy())
            ang[want[j]] = theta[j]
    x0, y0 = cov.fibre(path[0])
    perm = [int(np.argmin(np.abs(x0 - xi) + np.abs(y0 - yi)))
            for xi, yi in zip(x, y)]
    if perm != list(range(cov.N)):
        raise RuntimeError("the racetrack did not close up: %s" % perm)

    upper, lower, lam, jumps = [], [], [], []
    for k in range(len(z)):
        ex, ey = cov.fibre(ref[k])
        for side, out in ((+1, upper), (-1, lower)):
            rx, ry = rec[(k, side)]
            out.append([int(np.argmin(np.abs(ex - v) + np.abs(ey - u)))
                        for v, u in zip(rx, ry)])
            if sorted(out[-1]) != list(range(cov.N)):
                raise RuntimeError(
                    "sheet labels at sigma_%d are not a bijection" % (k + 1))
        a0 = np.angle(ref[k])
        n = {s: round((ang[(k, s)] - a0) / (2 * np.pi)) for s in (+1, -1)}
        lam.append(np.log(abs(ref[k])) + 1j * (a0 + 2 * np.pi * n[-1]))
        jumps.append(2j * np.pi * (n[+1] - n[-1]))
    if len(set(jumps)) != 1 or abs(abs(jumps[0]) - 2 * np.pi) > 1e-9:
        raise RuntimeError("the jump of log(phi) is not a uniform 2*pi*i: %s"
                           % jumps)
    return upper, lower, np.array(lam), jumps[0]


def gluing_permutations(upper, lower):
    r"""
    `\tau_i`, where the upper side of `\sigma_i` in sheet `k` is its lower side
    in sheet `\tau_i(k)`.  `\tau_1` and `\tau_r` are the monodromies at `P` and
    `Q`, and `\tau_{i+1}\tau_i^{-1}` the one at `z_{i+1}`.
    """
    n = len(upper[0])
    return [[lower[i].index(upper[i][k]) for k in range(n)]
            for i in range(len(upper))]


def boundary_walks(upper, lower):
    r"""
    The `2r` entries `(\sigma_i, \text{lift}, \pm)` of each `\partial D_k`: up
    the left side of `\sigma_1,\dots,\sigma_r`, back down the right.
    """
    r, n = len(upper), len(upper[0])
    return [[(i, upper[i][k], +1) for i in range(r)]
            + [(i, lower[i][k], -1) for i in range(r - 1, -1, -1)]
            for k in range(n)]


# ----------------------------------------------------------------- assembly

def hold_radii(z, frac=0.25):
    r"""
    How far from each node a chart leg may start.

    The chart at a node describes the cover near *that* node, so a leg must
    be handed over inside the node's own neighbourhood -- a quarter of the
    way to the nearest other node.  Where the edges are all of a size this
    is what a fixed fraction of the edge already gives; where they are not,
    it is the difference between a leg that is right to `10^{-11}` and one
    that is wrong in the first digit.
    """
    a = np.asarray(z)
    out = []
    for i in range(len(a)):
        gaps = np.abs(a - a[i])
        gaps[i] = np.inf
        out.append(frac * float(gaps.min()))
    return out


def edge_data(cov, z, d, kinds, lam_minus, log, moments):
    r"""`\int\omega_a`, `\int\lambda\,\omega_a` and the moments on every edge."""
    r, g, n = len(z), cov.g, cov.N
    hold = hold_radii(z)
    u = np.zeros((g, r, n), complex)
    ell = np.zeros((g, r, n), complex)
    K = np.zeros((g, g, r, n), complex) if moments else None
    for i in range(r):
        ref = complex(lam_minus[i]) if log else 0.0
        if i < r - 1:
            _, L = edge_leg(cov, z[i], z[i + 1], ref, kinds[i], kinds[i + 1],
                            moments, hold_a=hold[i], hold_b=hold[i + 1])
        else:
            _, L = ray_leg(cov, z[-1], d, ref, moments, kind_in=kinds[-1],
                           span=cut_scale(z)[0], hold=hold[-1])
        u[:, i, :], ell[:, i, :] = L.dI, L.dL
        if moments:
            K[:, :, i, :] = L.k
    return u, ell, K


def _assembly_data(cov, log, moments):
    r"""Integrate the cut once, keeping both ordinary and weighted periods."""
    z, kinds, d = cov.cut_path()
    upper, lower, lam_minus, dlam = cut_data(cov, z, d)
    W = boundary_walks(upper, lower)
    u, ell, K = edge_data(cov, z, d, kinds, lam_minus, log, moments)
    return W, u, ell, K, dlam


def _assemble(cov, log, moments, check_tol, verbose):
    r"""The Stokes sum, and the edge periods the caller may still need."""
    W, u, ell, K, dlam = _assembly_data(cov, log, moments)
    if not log:
        ell, dlam = u, 0.0
    H = _stokes_sum(W, u, ell, K, dlam, log, check_tol, verbose)
    return H, u, dlam


def _assemble_both(cov, moments, check_tol, verbose):
    r"""Period and logarithmic Stokes sums from the same edge integrals."""
    W, u, ell, K, dlam = _assembly_data(cov, True, moments)
    H0 = _stokes_sum(W, u, u, None, 0.0, False, check_tol, verbose)
    H = _stokes_sum(W, u, ell, K, dlam, True, check_tol, verbose)
    return H0, H, u, dlam


def _stokes_sum(W, u, ell, K, dlam, log, check_tol, verbose):
    r"""Assemble and check a Stokes sum without reintegrating its edges."""
    g, r, n = u.shape

    # Both primitives must close around every sheet. A logarithmic
    # defect can expose incorrect endpoint valuations even when it cancels
    # from the final real regulator; apply the same tolerance to both.
    H = np.zeros((g, g), complex)
    for a in range(g):
        plus = ell[a] + dlam * u[a]
        U = np.zeros((n, 2 * r + 1), complex)
        for k in range(n):
            for j, (i, m, sg) in enumerate(W[k]):
                U[k, j + 1] = U[k, j] + (plus[i][m] if sg > 0 else -ell[a][i][m])
        defect = np.abs(U[:, -1]).max() / max(np.abs(plus).max(), 1e-300)
        tol = check_tol
        if verbose:
            print("closing defect for omega_%d: %.2e" % (a + 1, defect))
        if not np.isfinite(defect) or defect > tol:
            raise RuntimeError("the sheet boundaries do not close: %.2e" % defect)
        fwd, bwd = {}, {}
        for k in range(n):
            for j, (i, m, sg) in enumerate(W[k]):
                (fwd if sg > 0 else bwd)[(i, m)] = (k, j)
        for i in range(r):
            for m in range(n):
                k, j = fwd[(i, m)]
                l, jp = bwd[(i, m)]
                C = U[k, j] - U[l, jp] + ell[a][i][m]
                for b in range(g):
                    H[a, b] += C * np.conj(u[b][i][m])
                    if K is not None:
                        H[a, b] += dlam * K[a][b][i][m]
    return H


def wedge_matrix(cov, check_tol=1e-6, verbose=False):
    r"""
    `\bigl(\int_C \omega_a \wedge \bar\omega_b\bigr)_{a,b}`, anti-Hermitian.

    EXAMPLES::

        sage: from hyperell_regulator.regulator.cover import Cover
        sage: from hyperell_regulator.regulator.polynomials import ring
        sage: x = ring().gen()
        sage: from hyperell_regulator.regulator.stokes import wedge_matrix
        sage: H = wedge_matrix(Cover(x^5 + x^2 + 2*x + 1, x + 1, 1))  # long time
        sage: bool(abs(H + H.conj().T).max() < 1e-6 * abs(H).max())  # long time
        True
    """
    return _assemble(cov, False, False, check_tol, verbose)[0]


def log_wedge_matrix(cov, check_tol=1e-6, verbose=False):
    r"""
    `\bigl(\int_C \log\varphi\;\omega_a \wedge \bar\omega_b\bigr)_{a,b}`,
    with the second moments integrated.

    Not anti-Hermitian: `\arg\varphi` contributes an antisymmetric real part,
    so only the mean of the two off-diagonal imaginary parts is pinned down.
    """
    return _assemble(cov, True, True, check_tol, verbose)[0]
