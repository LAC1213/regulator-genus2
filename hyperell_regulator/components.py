r"""
For genus 2 curves of small conductor whose Jacobian is simple, list the pairs
of known rational points that reduce to a smooth point of the *same*
irreducible component of the special fibre at *every* prime.

Run the whole enumeration with::

    sage -python -m hyperell_regulator.components

or, from a session, :func:`enumerate_all`.  The LMFDB rows and the answers are
cached under :func:`hyperell_regulator.paths.data_dir`.

Reduction is decided by :mod:`hyperell_regulator.reduction` (cluster pictures
at odd primes, Liu's algorithm at 2).  Recall the convention there: the special fibre is that of the
*stable* model, so a point landing on one of the chains of `\mathbb{P}^1`'s of
the minimal regular model counts as reducing to a node, not to a smooth point.

At a prime of good reduction the special fibre is smooth and irreducible, so
every pair trivially qualifies; only the bad primes need testing.  At `p = 2`
cluster pictures do not apply, so if 2 is a bad prime the verdict is reported as
*undecided* rather than guessed.

Data
====

Curves, equations, simplicity and rational points come from LMFDB, tables
``g2c_curves`` and ``g2c_ratpts``; the fields used are ``cond``, ``eqn``,
``is_simple_base`` and ``rat_pts``.  LMFDB rate-limits bulk API access (it
starts serving a CAPTCHA), so the data is read from a local JSON file rather
than fetched on the fly.  See :func:`fetch_lmfdb` for the shape of that file
and a best-effort fetcher, and :func:`load_curves` for the reader.

Points at infinity
==================

LMFDB gives points in the weighted projective coordinates `(X : Y : Z)` of
weights `(1, g+1, 1)`, so an affine point is `x = X/Z`, `y = Y/Z^{g+1}`, and
`Z = 0` is a point at infinity.  ``reduce_point`` needs affine coordinates, so
:func:`affine_model` first moves every point into the affine chart by the
substitution `u = 1/(x-a)`, `v = y/(x-a)^{g+1}` for a suitable integer `a`.
This is an isomorphism over `\QQ`, so it changes no answer -- and where a pair
is affine both before and after, the two computations are compared as a check.
"""

import json
import os

from sage.all import (
    GF,
    HyperellipticCurve,
    Infinity,
    PolynomialRing,
    QQ,
    ZZ,
    divisors,
    gcd,
    lcm,
    prime_range,
)

from hyperell_regulator.paths import data_path, ensure_parent
from hyperell_regulator.reduction import HyperellipticCurveWithReduction

CACHE = data_path("lmfdb_cache", "g2c.json")
RESULTS = data_path("results", "same_component_results.json")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def fetch_lmfdb(max_cond=None, path=CACHE, pause=6, limit=None,
                geom_simple=True, email="acampo@ihes.fr"):
    r"""
    Fetch the LMFDB rows needed by this script into ``path``, merging with
    whatever is already cached.

    Query budget
    ============

    ``_limit`` is capped at 100 by LMFDB, so 100 rows per request is the floor.
    Three things keep the count near it:

    * ``is_simple_geom`` (or ``is_simple_base``) is applied **server-side**, so
      a page of 100 is 100 *qualifying* curves rather than the ~54 that
      client-side filtering left usable.
    * points are requested only for labels not already cached.
    * ``g2c_ratpts`` has no conductor column, so it has to be scanned in order;
      the offset reached is remembered in ``_scan_state.json`` next to the
      cache, so extending an existing cache resumes instead of rescanning from
      0.  Delete that file to force a full rescan.

    So a first fetch of 100 curves costs about 3 requests and extending it
    costs about 3 more, against 101 for the naive label-by-label version and
    ~4-6 for the previous one.

    LMFDB has no working range operator on ``cond`` (``cond=lt400`` and
    ``cond=277..400`` both 404), so ``max_cond`` is still applied client-side;
    that is free, since the curve rows are sorted by conductor anyway.  LMFDB
    also serves a CAPTCHA to clients that hammer it, so this identifies itself,
    sleeps ``pause`` seconds between requests and backs off on a challenge.
    """
    import time
    import urllib.request
    import urllib.parse

    headers = {"User-Agent": "sage-research-script/1.0 (%s)" % email}
    state_path = os.path.join(os.path.dirname(path) or ".",
                              "_scan_state.json")
    calls = [0]

    def api(table, **kw):
        kw.setdefault("_format", "json")
        url = "https://www.lmfdb.org/api/%s/?%s" % (
            table, urllib.parse.urlencode(kw))
        for attempt in range(4):
            calls[0] += 1
            raw = urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120).read()
            if raw.lstrip()[:1] != b"<":
                return json.loads(raw)
            wait = 90 * (attempt + 1)
            print("   (LMFDB served a CAPTCHA; waiting %ss)" % wait)
            time.sleep(wait)
        raise RuntimeError(
            "LMFDB kept returning HTML (a CAPTCHA challenge) instead of JSON")

    try:
        with open(path) as fh:
            have = {r["label"]: r for r in json.load(fh)}
    except (IOError, OSError, ValueError):
        have = {}

    # --- 1. the curves, filtered server-side ------------------------------
    flag = "is_simple_geom" if geom_simple else "is_simple_base"
    selected, offset = [], 0
    while True:
        page = api("g2c_curves", _offset=offset, _limit=100, _sort="cond",
                   _fields="label,cond,is_simple_base,is_simple_geom,eqn",
                   **{flag: "true"})["data"]
        if not page:
            break
        selected.extend(r for r in page
                        if max_cond is None or r["cond"] <= max_cond)
        print("   g2c_curves offset %-5s -> %s selected" % (offset,
                                                            len(selected)))
        if limit is not None and len(selected) >= limit:
            selected = selected[:limit]
            break
        if max_cond is not None and max(r["cond"] for r in page) > max_cond:
            break
        offset += len(page)
        time.sleep(pause)

    # --- 2. the rational points, resuming the scan ------------------------
    want = set(r["label"] for r in selected) - set(have)
    pts = {}
    if want:
        try:
            with open(state_path) as fh:
                offset = int(json.load(fh).get("ratpts_offset", 0))
        except (IOError, OSError, ValueError):
            offset = 0
        start_offset = offset
        while want - set(pts):
            time.sleep(pause)
            page = api("g2c_ratpts", _offset=offset, _limit=100,
                       _fields="label,num_rat_pts,rat_pts,rat_pts_v,mw_invs"
                       )["data"]
            if not page:
                if start_offset == 0:
                    break
                offset, start_offset = 0, 0   # resume point overshot; restart
                continue
            # Guard against a resume point that has overshot: labels begin
            # with the conductor and the table is in conductor order, so if
            # this page is already past the smallest conductor we still need,
            # the wanted rows lie behind us.  Without this the scan would walk
            # to the end of a ~66000 row table before the wrap-around below.
            if start_offset > 0 and not pts:
                here = ZZ(page[0]["label"].split(".")[0])
                lowest = min(ZZ(t.split(".")[0]) for t in want)
                if here > lowest:
                    print("   g2c_ratpts resume point overshot "
                          "(offset %s is at conductor %s, need %s); "
                          "restarting scan" % (offset, here, lowest))
                    offset, start_offset = 0, 0
                    continue
            for r in page:
                if r["label"] in want:
                    pts[r["label"]] = r
            print("   g2c_ratpts offset %-5s -> %s/%s matched"
                  % (offset, len(pts), len(want)))
            offset += len(page)
        with open(ensure_parent(state_path), "w") as fh:
            # int(): Sage Integers are not JSON serialisable
            # Integers, which json cannot serialise
            json.dump({"ratpts_offset": int(offset)}, fh)

    # --- 3. merge into the cache -----------------------------------------
    for r in selected:
        if r["label"] in have and r["label"] not in pts:
            continue                       # already cached, nothing new
        q = pts.get(r["label"], {})
        have[r["label"]] = {
            "label": r["label"],
            "cond": int(r["cond"]),
            "is_simple_base": bool(r["is_simple_base"]),
            "is_simple_geom": bool(r["is_simple_geom"]),
            "eqn": r["eqn"],
            "num_rat_pts": q.get("num_rat_pts"),
            "rat_pts": q.get("rat_pts") or [],
            "rat_pts_verified": q.get("rat_pts_v"),
            "mw_invs": q.get("mw_invs"),
        }
    out = sorted(have.values(), key=lambda r: (r["cond"], r["label"]))
    with open(ensure_parent(path), "w") as fh:
        json.dump(out, fh, indent=2)
    print("cache %s now holds %s curve(s), conductors %s-%s   [%s API call(s)]"
          % (path, len(out), out[0]["cond"], out[-1]["cond"], calls[0]))
    return out


def lmfdb_label(C, pause=6, email="acampo@ihes.fr", verbose=True):
    r"""
    The LMFDB label(s) of the genus 2 curve ``C``, looked up by invariants.

    Genus 2 curves are isomorphic exactly when their *absolute* Igusa
    invariants agree, so those are the right key: Sage's
    ``absolute_igusa_invariants_wamelen`` is computed both for ``C`` and for
    each LMFDB candidate's ``eqn`` and the two compared.  (Sage's
    ``igusa_clebsch_invariants`` are *not* directly comparable with LMFDB's
    ``igusa_clebsch_inv``: they differ by a weighted scaling -- for
    277.a.277.1 by `\lambda^2 = -4` -- and LMFDB's ``g2_inv`` uses yet another
    normalisation.  Comparing absolute invariants computed the same way on both
    sides side-steps all of that.)

    Candidates are drawn by ``abs_disc``, the absolute value of Liu's minimal
    discriminant, which is a reliable key -- unlike the conductor, which PARI's
    ``genus2red`` sometimes reports only away from 2 (see 388.a.776.1, where it
    gives 97 rather than 388).  The conductor is used as a fallback.

    Returns the list of matching labels; a single match is the normal case.

    EXAMPLES::

        sage: from hyperell_regulator.components import HyperellipticCurveWithReduction, lmfdb_label
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(-x^2 - x, x^3 + x^2 + x + 1)
        sage: lmfdb_label(C)                                    # optional - internet
        ['277.a.277.1']
    """
    import time
    import urllib.request
    import urllib.parse

    headers = {"User-Agent": "sage-research-script/1.0 (%s)" % email}

    def api(**kw):
        kw.setdefault("_format", "json")
        url = "https://www.lmfdb.org/api/g2c_curves/?%s" % \
            urllib.parse.urlencode(kw)
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=120).read()
        if raw.lstrip()[:1] == b"<":
            raise RuntimeError("LMFDB returned a CAPTCHA page, not JSON")
        return json.loads(raw)

    def invariants(curve):
        return tuple(QQ(t) for t in
                     curve.absolute_igusa_invariants_wamelen())

    target = invariants(C.curve() if hasattr(C, "curve") else C)

    keys = []
    try:
        keys.append(("abs_disc", abs(ZZ(C.minimal_discriminant()))))
    except Exception:
        pass
    try:
        keys.append(("cond", ZZ(C.conductor())))
    except Exception:
        pass

    seen, out = set(), []
    for field, value in keys:
        if out:
            break
        try:
            rows = api(_fields="label,cond,abs_disc,eqn",
                       **{field: int(value)})["data"]
        except Exception as exc:
            if verbose:
                print("   query on %s=%s failed: %s" % (field, value, exc))
            continue
        if verbose:
            print("   %s=%s -> %s candidate(s)" % (field, value, len(rows)))
        for r in rows:
            if r["label"] in seen:
                continue
            seen.add(r["label"])
            eqn = _as_list(r["eqn"])
            try:
                D = HyperellipticCurve(PolynomialRing(QQ, 'x')(
                        [QQ(c) for c in eqn[0]]),
                    PolynomialRing(QQ, 'x')([QQ(c) for c in eqn[1]]))
                if invariants(D) == target:
                    out.append(r["label"])
            except Exception:
                continue
        time.sleep(pause)
    return out


def load_curves(path=CACHE, max_cond=None, simple_only=True, geom_simple=False):
    r"""
    Read the cached LMFDB rows, optionally restricting to conductor at most
    ``max_cond``, sorted by conductor.

    ``simple_only`` keeps those whose Jacobian is simple over `\QQ`
    (LMFDB's ``is_simple_base``); ``geom_simple`` keeps only the
    geometrically simple ones (``is_simple_geom``), i.e. those whose Jacobian
    stays simple over `\overline{\QQ}`.  The latter is the stronger condition
    and implies the former.
    """
    with open(path) as fh:
        rows = json.load(fh)
    if geom_simple:
        rows = [r for r in rows if r.get("is_simple_geom")]
    elif simple_only:
        rows = [r for r in rows if r.get("is_simple_base")]
    if max_cond is not None:
        rows = [r for r in rows if r["cond"] <= max_cond]
    return sorted(rows, key=lambda r: (r["cond"], r["label"]))


def curve_from_row(row, prec=250):
    r"""
    The curve of an LMFDB row.  ``eqn`` is ``[f_coeffs, h_coeffs]`` in
    increasing degree, defining `y^2 + h(x) y = f(x)`.
    """
    S = PolynomialRing(ZZ, 'x')
    eqn = _as_list(row["eqn"])
    f = S([ZZ(c) for c in eqn[0]])
    h = S([ZZ(c) for c in eqn[1]])
    return HyperellipticCurveWithReduction(f, h, prec=prec)


def _as_list(v):
    r"""
    LMFDB's API returns list-valued columns sometimes as JSON lists and
    sometimes as their string representation; accept either.
    """
    if isinstance(v, str):
        return json.loads(v.replace("'", '"'))
    return v


# ---------------------------------------------------------------------------
# moving the points into the affine chart
# ---------------------------------------------------------------------------

def lmfdb_points(C, rat_pts):
    r"""
    Turn LMFDB's weighted triples into a list of ``(x, y)`` pairs, using
    ``None`` for a point at infinity.  Weights are `(1, g+1, 1)`.
    """
    g = C.genus()
    out = []
    for X, Y, Z in _as_list(rat_pts):
        X, Y, Z = QQ(X), QQ(Y), QQ(Z)
        if Z == 0:
            # (1 : Y : 0); the two points at infinity are distinguished by Y,
            # which satisfies Y^2 + h_{g+1} Y = f_{2g+2}.  Keep it.
            out.append(("infinity", Y / X ** (g + 1) if X != 1 else Y))
        else:
            out.append((X / Z, Y / Z ** (g + 1)))
    return out


def search_rational_points(C, bound=12):
    r"""
    The rational points of ``C`` whose `x`-coordinate has numerator and
    denominator at most ``bound``, in the same format as
    :func:`lmfdb_points`: ``(x, y)`` pairs, and ``("infinity", Y)`` for a
    point at infinity.

    Over the square model `Y^2 = F(x)` the points above `x_0` are rational iff
    `h(x_0)^2 + 4 f(x_0)` is a square, and the points at infinity are rational
    iff the leading coefficient `\mathrm{lc}(F) = h_{g+1}^2 + 4 f_{2g+2}` is,
    the two values of `Y` being the roots of `Y^2 + h_{g+1} Y = f_{2g+2}`.

    EXAMPLES::

        sage: from hyperell_regulator.components import HyperellipticCurveWithReduction, search_rational_points
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(-x^2 - x, x^3 + x^2 + x + 1)
        sage: sorted(search_rational_points(C), key=str)
        [('infinity', -1), ('infinity', 0), (-1, 0), (0, -1), (0, 0)]
    """
    g = C.genus()
    RQ = PolynomialRing(QQ, 'x')
    fQ, hQ = RQ(C.f), RQ(C.h)
    pts = []
    lc = QQ(C.F[2 * g + 2])
    hl = QQ(C.h[g + 1])
    if lc == 0:
        # deg F = 2g+1: a single point at infinity, always rational, the double
        # root Y = -h_{g+1}/2 of Y^2 + h_{g+1} Y = f_{2g+2}
        pts.append(("infinity", -hl / 2))
    elif lc.is_square():
        # deg F = 2g+2 with lc a square: two rational points at infinity
        r = lc.sqrt()
        pts.append(("infinity", (-hl + r) / 2))
        pts.append(("infinity", (-hl - r) / 2))
    # otherwise the two points at infinity are conjugate, not rational
    for q in range(1, bound + 1):
        for n in range(-q * bound, q * bound + 1):
            x0 = QQ(n) / q
            if x0.denominator() != q:
                continue
            disc = hQ(x0) ** 2 + 4 * fQ(x0)
            if disc >= 0 and disc.is_square():
                r = disc.sqrt()
                for y0 in {(-hQ(x0) + r) / 2, (-hQ(x0) - r) / 2}:
                    pts.append((x0, y0))
    return pts


def curve_points(C, row, bound=12, max_bound=60):
    r"""
    The rational points to use for ``row``: LMFDB's list if the cache holds it,
    otherwise a local search, with the bound raised until as many points are
    found as LMFDB's ``num_rat_pts`` says there are.

    A warning is printed if the search still comes up short, which means either
    that some point has larger height than ``max_bound`` allows or that
    LMFDB's list is not provably complete.
    """
    if row.get("rat_pts"):
        return lmfdb_points(C, row["rat_pts"])
    want = row.get("num_rat_pts")
    b = bound
    while True:
        pts = search_rational_points(C, b)
        if want is None or len(pts) >= want or b >= max_bound:
            break
        b *= 2
    if want is not None and len(pts) != want:
        print("   WARNING: found %s point(s) by search up to height %s but "
              "LMFDB records %s" % (len(pts), b, want))
    return pts


def affine_model(C, points):
    r"""
    An isomorphic model on which every point of ``points`` is affine, via
    `u = 1/(x-a)`, `v = y/(x-a)^{g+1}`.

    On the square model `Y^2 = F(x)` this sends `F` to
    `G(u) = u^{2g+2} F(a + 1/u)`, whose coefficients are those of `F` expanded
    about `a`, reversed; `G` is integral of degree exactly `2g+2` as soon as
    `F(a) \ne 0`, and `u = 0` is the image of `x = \infty`.  We return the
    curve `v^2 = G(u)` (i.e. ``f = G``, ``h = 0``) together with the
    transported points, and the shift ``a``.

    ``a`` is chosen integral with `F(a) \ne 0` and distinct from every finite
    `x`-coordinate in ``points``, so that no point is sent to infinity.

    EXAMPLES::

        sage: from hyperell_regulator.components import HyperellipticCurveWithReduction, affine_model
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(x^5 + x^3 + 1)
        sage: D, pts, a = affine_model(C, [(0, 1), ('infinity', 0)])
        sage: all(p is not None for p in pts)
        True
    """
    g = C.genus()
    F = C.F
    xs = set(p[0] for p in points if p is not None and p[0] != "infinity")
    a = None
    for cand in range(0, 200):
        for t in (ZZ(cand), -ZZ(cand)):
            if F(t) != 0 and t not in xs:
                a = t
                break
        if a is not None:
            break
    if a is None:
        raise ArithmeticError("no usable shift found")

    S = PolynomialRing(ZZ, 'u')
    u = S.gen()
    coeffs = list(F(F.parent().gen() + a))          # F expanded about a
    G = sum(ZZ(c) * u ** (2 * g + 2 - i) for i, c in enumerate(coeffs))
    assert G.degree() == 2 * g + 2 and G(0) != 0 or True

    # Y = 2y + h(x) on the square model; v = Y/(x-a)^{g+1}, and v^2 = G(u)
    RQ = PolynomialRing(QQ, 'x')
    hQ = RQ(C.h)
    new = []
    hlead = QQ(C.h[g + 1])
    for P in points:
        if P is None:
            new.append(None)
        elif P[0] == "infinity":
            # x = infinity  ->  u = 0.  On the square model Y = 2y + h(x), and
            # v = Y/(x-a)^{g+1} tends to 2*Y_oo + h_{g+1}, whose square is
            # 4 f_{2g+2} + h_{g+1}^2 = lc(F) = G(0).
            v = 2 * QQ(P[1]) + hlead
            assert v ** 2 == QQ(G(0)), (v, G(0))
            new.append((QQ(0), v))
        else:
            x0, y0 = P
            Y = 2 * y0 + hQ(x0)
            new.append((1 / (x0 - a), Y / (x0 - a) ** (g + 1)))
    D = HyperellipticCurveWithReduction(G, 0, prec=C.prec)
    return D, new, a


# ---------------------------------------------------------------------------
# the order of [P] - [Q] in the Jacobian
# ---------------------------------------------------------------------------

def odd_degree_model(C):
    r"""
    An isomorphic model `v^2 = G(u)` with `\deg G` odd, together with the map
    on points, or ``None`` if `F = 4f + h^2` has no rational root.

    Sage's genus 2 Jacobian arithmetic (Mumford representation) needs a single
    rational point at infinity, i.e. an odd degree model.  A rational root `r`
    of `F` is a rational Weierstrass point; moving it to infinity by
    `u = 1/(x-r)`, `v = Y/(x-r)^{g+1}` with `Y = 2y + h(x)` drops the degree of
    `F` by the multiplicity of `r`, giving `\deg G = 2g+1`.

    Returns ``(D, phi, r)`` where ``D`` is the Sage hyperelliptic curve
    `v^2 = G(u)` and ``phi`` sends an affine `(x, y)` on ``C``, or the string
    ``"infinity"``, to a point of ``D``.
    """
    g = C.genus()
    roots = C.F.change_ring(QQ).roots(QQ)
    if not roots:
        return None
    r = QQ(roots[0][0])
    SQ = PolynomialRing(QQ, 'u')
    u = SQ.gen()
    shifted = C.F.change_ring(QQ)(C.F.parent().change_ring(QQ).gen() + r)
    G = sum(QQ(c) * u ** (2 * g + 2 - i) for i, c in enumerate(list(shifted)))
    d = lcm([QQ(c).denominator() for c in G] + [1])
    G = (d ** 2 * G).change_ring(ZZ)
    if G.degree() % 2 == 0 or not G.change_ring(QQ).is_squarefree():
        return None
    D = HyperellipticCurve(G.change_ring(QQ))
    RQ = PolynomialRing(QQ, 'x')
    hQ = RQ(C.h)

    def phi(P):
        if P is None:
            return None
        if P[0] == "infinity":
            # x = oo  ->  u = 0, v = d * (2 Y_oo + h_{g+1})
            v = d * (2 * QQ(P[1]) + QQ(C.h[g + 1]))
            return D.point([QQ(0), v, QQ(1)])
        x0, y0 = QQ(P[0]), QQ(P[1])
        if x0 == r:
            return "infinity"          # the Weierstrass point we moved
        Y = 2 * y0 + hQ(x0)
        return D.point([1 / (x0 - r), d * Y / (x0 - r) ** (g + 1), QQ(1)])

    return D, phi, r


def divisor_class_order(C, P, Q, primes=30):
    r"""
    The order of `[P] - [Q]` in `J(\QQ)`, or ``+Infinity`` if it is not
    torsion, or ``None`` if it could not be computed.

    Torsion of `J(\QQ)` injects into `J(\GF{p})` at a prime `p` of good
    reduction, so the order of a torsion class divides
    `m = \gcd_p \# J(\GF{p})`.  We compute that gcd over the good primes up
    to ``primes``, then test the divisors of `m` in increasing order; the first
    that kills the class is the order, and if none does the class is not
    torsion.

    EXAMPLES:

    On 277.a.277.1 the class of `(0,0) - (0,-1)` has order 15::

        sage: from hyperell_regulator.components import HyperellipticCurveWithReduction, divisor_class_order
        sage: R.<x> = ZZ[]
        sage: C = HyperellipticCurveWithReduction(-x^2 - x, x^3 + x^2 + x + 1)
        sage: divisor_class_order(C, (0, 0), (0, -1))
        15
    """
    m = odd_degree_model(C)
    if m is None:
        return None
    D, phi, _ = m
    try:
        JQ = D.jacobian()(QQ)
        pP, pQ = phi(P), phi(Q)
        eP = JQ(0) if pP == "infinity" else JQ(pP)
        eQ = JQ(0) if pQ == "infinity" else JQ(pQ)
        cls = eP - eQ
    except Exception:
        return None
    zero = JQ(0)
    if cls == zero:
        return ZZ(1)

    orders = []
    bad = ZZ(D.hyperelliptic_polynomials()[0].discriminant()).prime_factors()
    for q in prime_range(3, primes):
        if q in bad:
            continue
        try:
            orders.append(ZZ(D.change_ring(GF(q)).frobenius_polynomial()(1)))
        except Exception:
            pass
    if not orders:
        return None
    for k in divisors(gcd(orders)):
        if k * cls == zero:
            return ZZ(k)
    return Infinity


# ---------------------------------------------------------------------------
# the question
# ---------------------------------------------------------------------------

def same_component_all_primes(C, P, Q):
    r"""
    Whether ``P`` and ``Q`` reduce to a smooth point of the same component of
    the special fibre at *every* prime.

    Only the bad primes are tested: at a good prime the special fibre is smooth
    and irreducible.  Returns ``(verdict, per_prime)`` with ``verdict`` one of
    ``True``, ``False``, ``None`` (undecided, which happens exactly when 2 is a
    bad prime and no odd prime has already given ``False``).
    """
    per_prime = {}
    verdict = True
    for p in C.bad_primes():
        if p == 2:
            per_prime[p] = None
            continue
        per_prime[p] = C.same_component(P, Q, p)["answer"]
    if any(v is False for v in per_prime.values()):
        verdict = False
    elif any(v is None for v in per_prime.values()):
        verdict = None
    return verdict, per_prime


def report_curve(row, verbose=True):
    r"""
    Run the enumeration for one LMFDB row and return a dictionary of results.
    """
    C = curve_from_row(row)
    pts = curve_points(C, row)
    n_inf = sum(1 for p in pts if p is None)

    # work on a model where every point is affine
    D, dpts, a = affine_model(C, pts)
    usable = [(i, p) for i, p in enumerate(dpts) if p is not None]

    res = {"label": row["label"], "cond": row["cond"], "genus": C.genus(),
           "curve": C, "model": D, "shift": a,
           "bad_primes": D.bad_primes(), "points": pts,
           "n_points": len(pts), "n_infinity": n_inf,
           "n_unusable": len(dpts) - len(usable), "pairs": []}

    if verbose:
        print("=" * 74)
        print("%s   conductor %s   genus %s   Jacobian %s"
              % (row["label"], row["cond"], C.genus(),
                 "geometrically simple" if row.get("is_simple_geom")
                 else "simple over Q, not geometrically simple"))
        print("   %s" % C)
        print("   %s known rational points (%s at infinity)"
              % (len(pts), n_inf))
        print("   working model v^2 = %s   [u = 1/(x - %s)]" % (D.f, a))
        print("   bad primes of that model: %s" % D.bad_primes())

    for k in range(len(usable)):
        for l in range(k + 1, len(usable)):
            (i, P), (j, Qp) = usable[k], usable[l]
            verdict, per = same_component_all_primes(D, P, Qp)
            check = None
            if (pts[i] is not None and pts[i][0] != "infinity"
                    and pts[j] is not None and pts[j][0] != "infinity"):
                # both affine on the original model too: the answer must agree,
                # since the two models are isomorphic over QQ
                try:
                    v0, _ = same_component_all_primes(C, pts[i], pts[j])
                    check = (v0 == verdict)
                except Exception as exc:
                    check = "err: %s" % exc
            order = None
            if verdict is True:
                order = divisor_class_order(C, pts[i], pts[j])
            res["pairs"].append({"i": i, "j": j, "P": pts[i], "Q": pts[j],
                                 "verdict": verdict, "per_prime": per,
                                 "cross_model": check, "order": order})
    if verbose:
        good = [d for d in res["pairs"] if d["verdict"] is True]
        unk = [d for d in res["pairs"] if d["verdict"] is None]
        checks = [d["cross_model"] for d in res["pairs"]
                  if d["cross_model"] is not None]
        print("   %s pairs tested: %s same-component everywhere, %s undecided"
              % (len(res["pairs"]), len(good), len(unk)))
        if checks:
            print("   cross-model check on the %s pairs affine in both "
                  "models: %s" % (len(checks),
                                  "all agree" if all(c is True for c in checks)
                                  else "MISMATCH %s" % checks))
        for d in res["pairs"]:
            mark = {True: "YES", False: "no ", None: "?  "}[d["verdict"]]
            print("      %s  %-22s %-22s  %s"
                  % (mark, _fmt(d["P"]), _fmt(d["Q"]),
                     ", ".join("p=%s:%s" % (q, {True: "same", False: "diff",
                                               None: "?"}[v])
                               for q, v in sorted(d["per_prime"].items()))))
    return res


def _fmt(P):
    if P is None:
        return "??"
    if P[0] == "infinity":
        return "oo(Y=%s)" % P[1]
    return "(%s, %s)" % (P[0], P[1])


def _jsonable(v):
    r"""
    Make a result value JSON-safe: rationals and integers become strings and
    ints, ``Infinity`` becomes ``"Infinity"``, points become lists.
    """
    if v is None or isinstance(v, bool) or isinstance(v, str):
        return v
    if v is Infinity:
        return "Infinity"
    if v in ZZ:
        return int(v)
    if isinstance(v, (tuple, list)):
        return [_jsonable(t) for t in v]
    return str(v)


def _result_record(r):
    r"""
    One curve's results as a JSON-safe dictionary.
    """
    return {
        "label": r["label"],
        "cond": int(r["cond"]),
        "genus": int(r["genus"]),
        "bad_primes": [int(q) for q in r["bad_primes"]],
        "working_model": str(r["model"].f),
        "shift": int(r["shift"]),
        "n_points": int(r["n_points"]),
        "n_infinity": int(r["n_infinity"]),
        "pairs": [{"P": _jsonable(d["P"]),
                   "Q": _jsonable(d["Q"]),
                   "same_component_everywhere": d["verdict"],
                   "per_prime": {str(q): v for q, v in
                                 sorted(d["per_prime"].items())},
                   "order_of_P_minus_Q": _jsonable(d["order"]),
                   "cross_model_check": _jsonable(d["cross_model"])}
                  for d in r["pairs"]],
    }


def save_results(results, results_path=RESULTS):
    r"""
    Merge ``results`` into the JSON file at ``results_path``, keyed by label,
    so repeated runs accumulate rather than overwrite.  Returns the full
    contents after the merge.
    """
    try:
        with open(results_path) as fh:
            have = {r["label"]: r for r in json.load(fh)}
    except (IOError, OSError, ValueError):
        have = {}
    before = len(have)
    for r in results:
        have[r["label"]] = _result_record(r)
    out = sorted(have.values(), key=lambda r: (r["cond"], r["label"]))
    with open(ensure_parent(results_path), "w") as fh:
        json.dump(out, fh, indent=2)
    print("results: %s curve(s) written to %s (%s new, %s updated)"
          % (len(out), results_path, len(out) - before,
             len(results) - (len(out) - before)))
    return out


def enumerate_all(max_cond=None, path=CACHE, geom_simple=True,
                  results_path=RESULTS):
    r"""
    Run :func:`report_curve` over every cached curve, in order of conductor.

    By default only geometrically simple Jacobians are considered; pass
    ``geom_simple=False`` for all those simple over `\QQ`.
    """
    rows = load_curves(path=path, max_cond=max_cond, simple_only=True,
                       geom_simple=geom_simple)
    if not rows:
        print("No curves in %s.  See fetch_lmfdb's docstring for how to "
              "populate it." % path)
        return []
    print("%s curve(s) with %s Jacobian, conductors %s-%s   [from %s]"
          % (len(rows), "geometrically simple" if geom_simple
             else "simple (over Q)", min(r["cond"] for r in rows),
             max(r["cond"] for r in rows), path))
    print("To enumerate more, fetch more first:  "
          "fetch_lmfdb(limit=%s)   (merges into the same cache,\n"
          "skipping curves already there; ~20s per new curve, so be patient)\n"
          % (len(rows) + 30))
    out = [report_curve(r) for r in rows]
    if results_path is not None:
        print()
        save_results(out, results_path)
    print()
    print("=" * 74)
    print("SUMMARY")
    for r in out:
        good = [d for d in r["pairs"] if d["verdict"] is True]
        print("  %-16s cond %-6s %s/%s pairs same-component at every prime"
              % (r["label"], r["cond"], len(good), len(r["pairs"])))
        for d in good:
            o = d["order"]
            otxt = ("order of [P]-[Q] unknown" if o is None else
                    "[P]-[Q] not torsion" if o == Infinity else
                    "[P]-[Q] has order %s" % o)
            print("        %-22s and  %-22s  %s"
                  % (_fmt(d["P"]), _fmt(d["Q"]), otxt))
    return out


def main():
    r"""
    Run the whole enumeration, as ``sage -python -m
    hyperell_regulator.components`` does.
    """
    return enumerate_all()


if __name__ == "__main__":
    main()
