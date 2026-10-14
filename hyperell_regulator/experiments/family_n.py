r"""Beilinson reports for y^2+x(x-1)y=x(x-1)(x^3+n*x^2-(n+1)*x+1).

The marked points P=(0,0), Q=(1,0), R=infinity are distinct Weierstrass
points, hence pairwise non-conjugate, and P-R and Q-R have order two.
The same-component condition holds throughout this family; the optional
``verify_same_component`` helper checks individual members.

Like ``beilinson``, the runner uses float64 by default, shares ordinary and
logarithmic edge integrals, supports ``--prec`` for arbitrary precision, and
saves precision-preserving JSON with per-edge limits and retryable failures.
The standard conductor can be large: ``--no-lvalue`` skips L-values, and
``--max-conductor`` caps their cost. Run ``--help`` for parameter selection.
"""

import json

from sage.all import (
    Infinity,
    PolynomialRing,
    QQ,
    ZZ,
)

from hyperell_regulator.components import affine_model
from hyperell_regulator.experiments import beilinson as b
from hyperell_regulator.lfunction import StandardLFunction
from hyperell_regulator.paths import data_path, ensure_parent
from hyperell_regulator.reduction import HyperellipticCurveWithReduction
from hyperell_regulator.regulator import IntegrationMonitor

RESULTS = data_path("results", "family_n_beilinson.json")

# Integer parameters where the cubic factor of F=4f+h^2 has three real roots.
REAL_SLIT_LO, REAL_SLIT_HI = -4, 3


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------

def family_polynomial(n):
    r"""
    The pair `(f, h)` of `y^2 + h(x) y = f(x)` for the parameter ``n``.
    """
    S = PolynomialRing(ZZ, 'x')
    x = S.gen()
    f = x * (x - 1) * (x ** 3 + n * x ** 2 - (n + 1) * x + 1)
    h = x * (x - 1)
    return f, h


def family_curve(n, prec=250):
    r"""
    The curve for the parameter ``n``, or ``None`` if it is singular.
    """
    f, h = family_polynomial(n)
    F = 4 * f + h ** 2
    if F.degree() != 5 or F.discriminant() == 0:
        return None
    try:
        return HyperellipticCurveWithReduction(f, h, prec=prec)
    except ValueError:
        return None


def family_points():
    r"""
    `P = (0,0)`, `Q = (1,0)`, `R = \infty`, in the coordinates of
    :mod:`hyperell_regulator.reduction`.  `\deg F = 5`, so there is a single point at infinity and
    `Y = -h_{g+1}/2 = 0` since `\deg h = 2 < g + 1`.
    """
    return [(QQ(0), QQ(0)), (QQ(1), QQ(0)), ("infinity", QQ(0))]


def is_real_slit_member(n):
    r"""
    Whether all five branch points of member ``n`` are real.

    ``0`` and ``1`` are branch points for every ``n`` (they are roots of
    `F = 4f + h^2 = x(x-1)(4x^3 + (4n+1)x^2 - (4n+5)x + 4)`), so the condition
    is only that the cubic factor have three real roots.
    """
    return int(n) <= REAL_SLIT_LO or int(n) >= REAL_SLIT_HI


def verify_same_component(n, prec=250, verbose=True):
    r"""
    Re-check that `P`, `Q`, `R` reduce to a smooth point of the same component
    at every bad prime.

    Not called by :func:`run` -- the condition holds identically in ``n``, see
    the module docstring -- but useful as a spot check.  ``reduce_point`` wants
    affine coordinates, so the test runs on the shifted model `u = 1/(x-a)`,
    over the bad primes of `C` itself.
    """
    C = family_curve(n, prec=prec)
    if C is None:
        return {"n": int(n), "status": "singular"}
    bad = C.bad_primes()
    D, dpts, shift = affine_model(C, family_points())
    names = ["(0,0)", "(1,0)", "oo"]
    out = {"n": int(n), "status": "ok", "bad_primes": [int(p) for p in bad],
           "pairs": {}, "undecided_primes": []}
    verdict, undecided = True, set()
    for i in range(3):
        for j in range(i + 1, 3):
            per = {}
            for p in bad:
                try:
                    per[int(p)] = D.same_component(dpts[i], dpts[j], p)["answer"]
                    if per[int(p)] is None:
                        undecided.add(int(p))
                except Exception as exc:
                    per[int(p)] = None
                    undecided.add(int(p))
                    out.setdefault("errors", []).append(
                        "%s vs %s at %s: %s" % (names[i], names[j], p, exc))
            out["pairs"]["%s|%s" % (names[i], names[j])] = per
            if any(v is False for v in per.values()):
                verdict = False
            elif any(v is None for v in per.values()) and verdict is not False:
                verdict = None
    out["verdict"] = verdict
    out["undecided_primes"] = sorted(undecided)
    if verbose:
        print("n = %-4s bad %-22s same component everywhere: %s%s"
              % (n, out["bad_primes"], verdict,
                 "  (undecided at %s)" % out["undecided_primes"]
                 if out["undecided_primes"] else ""))
    return out


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

_STD_CONDUCTOR = {}


def standard_conductor(n):
    r"""
    The conductor of the standard motive, memoised; ``None`` if some Euler
    factor is not available.

    This governs the cost; the actual coefficient count comes from PARI
    through ``StandardLFunction._coeffs_needed``.
    """
    key = int(n)
    if key not in _STD_CONDUCTOR:
        try:
            C = family_curve(n)
            _STD_CONDUCTOR[key] = ZZ(StandardLFunction(C).conductor())
        except Exception:
            _STD_CONDUCTOR[key] = None
    return _STD_CONDUCTOR[key]


# ---------------------------------------------------------------------------
# the test
# ---------------------------------------------------------------------------

def _regulator_precision(reg_digits, dps):
    """Keep the old dps keyword as an alias, without ambiguous precedence."""
    if dps is not None:
        if reg_digits is not None and reg_digits != dps:
            raise ValueError("reg_digits and dps specify different precisions")
        reg_digits = dps
    if reg_digits is not None and (int(reg_digits) != reg_digits or reg_digits < 5):
        raise ValueError("regulator precision must be an integer of at least 5 digits")
    return reg_digits


def family_determinant(G, dps=None, *, reg_digits=None):
    """Compute both regulator columns and periods using two shared cover passes."""
    precision = _regulator_precision(reg_digits, dps)
    return b.beilinson_determinant_stokes(G, (0, 0), 2, (1, 0), 2,
                                        digits=precision)


def beilinson_for_n(n, digits=15, lindep_bound=8, verbose=False, prec=250,
                    dps=None, *, reg_digits=None, with_lvalue=True,
                    max_edge_seconds=15, max_edge_steps=100000):
    """One family member, using the same numerical/report policy as beilinson.

    ``prec`` retains its historical meaning: reduction arithmetic in bits.
    ``reg_digits`` (or the legacy ``dps`` alias) controls regulator decimals.
    The CLI's ``--prec`` means regulator decimals, as in ``beilinson``.
    """
    import time
    reg_digits = _regulator_precision(reg_digits, dps)
    if int(digits) != digits or digits < 1:
        raise ValueError("L-value digits must be positive")
    label = "family_n(%s)" % int(n)
    result = dict(n=int(n), label=label)
    def progress(event):
        if verbose and event['event'] != 'start':
            print("  %s edge %d %s: %.1fs, %s" %
                  (label, event['index'], event['event'], event['seconds'],
                   event['phase']), flush=True)
    monitor = IntegrationMonitor(max_edge_seconds=max_edge_seconds,
        max_edge_steps=max_edge_steps, callback=progress, progress_interval=5)
    start = time.monotonic()
    try:
        C = family_curve(n, prec=prec)
        if C is None:
            result['status'] = 'singular'
        else:
            P, Q, R = family_points()
            triple = dict(P=P, Q=Q, R=R, order_P=2, order_Q=2,
                          pairwise_nonconjugate=True)
            with monitor:
                result.update(b._analyse_triple(
                    dict(label=label, cond=int(C.conductor())), C, triple, 1,
                    verbose=verbose, with_periods=with_lvalue,
                    with_lvalue=with_lvalue, digits=digits,
                    reg_digits=reg_digits, lindep_bound=lindep_bound))
            result.update(f=str(C.f), h=str(C.h), conductor=result['cond'],
                          bad_primes=[int(p) for p in C.bad_primes()])
    except Exception as exc:
        result.update(status='preparation_error',
                      error="%s: %s" % (type(exc).__name__, exc))
    result.update(seconds=time.monotonic()-start,
                  edges_attempted=len(monitor.records),
                  slowest_edge_seconds=max((e['seconds'] for e in monitor.records), default=0))
    return result


def load_stored(path=RESULTS):
    """Read previous results, keyed by the integer parameter."""
    if path is None:
        return {}
    try:
        with open(path) as fh:
            return {int(r['n']): r for r in json.load(fh)}
    except (OSError, ValueError):
        return {}


def save_stored(have, path=RESULTS):
    if path is not None:
        with open(ensure_parent(path), 'w') as fh:
            json.dump(sorted(have.values(), key=lambda r: r['n']), fh,
                      indent=2, default=str)


def run(ns=None, out=RESULTS, limit=None, digits=15, force=False,
        order='cost', max_conductor=10**8, verbose=False, dps=None, *,
        reg_digits=None, with_lvalue=True, max_edge_seconds=15,
        max_edge_steps=100000):
    """Report family members, caching successes only under matching settings.

    The default parameter range is -4 through 2, including non-real-slit
    members. ``max_conductor`` is an inclusive bound on the standard motive's
    conductor, rather than the curve conductor. Failures are saved and retried.
    ``limit`` counts attempted members, including failures and cached results.
    """
    from collections import Counter
    reg_digits = _regulator_precision(reg_digits, dps)
    if order not in ('cost', 'lex') or (limit is not None and limit < 1):
        raise ValueError("order must be cost or lex, and limit must be positive")
    if digits < 1 or max_edge_seconds <= 0 or max_edge_steps < 1:
        raise ValueError("digits and edge budgets must be positive")
    if max_conductor is not None and max_conductor < 1:
        raise ValueError("standard conductor bound must be positive")
    ns = sorted(set(int(n) for n in (range(-4, 3) if ns is None else ns)))
    if max_conductor is not None:
        ns = [n for n in ns if (standard_conductor(n) or Infinity) <= max_conductor]
    if order == 'cost':
        ns.sort(key=lambda n: (standard_conductor(n) is None,
                               standard_conductor(n) or 0, n))
    if limit is not None:
        ns = ns[:limit]
    settings = dict(report_version=b.REPORT_VERSION, family_report_version=1,
                    regulator_digits=reg_digits, lvalue_digits=int(digits),
                    with_lvalue=with_lvalue, max_edge_seconds=max_edge_seconds,
                    max_edge_steps=max_edge_steps)
    have, results = load_stored(out), []
    print("Family n: %s; %s" % (ns, 'float64' if reg_digits is None
                               else '%s-digit regulator' % reg_digits))
    print("%-18s %5s %-21s %14s %12s %10s %10s" %
          ('curve', 'cond', 'status', 'determinant', 'q', 'rational', 'rel. error'))
    for n in ns:
        old = have.get(n, {})
        cached = not force and b._stored_result_satisfies(old, settings)
        if cached:
            result = old
        else:
            result = beilinson_for_n(n, digits=digits, verbose=verbose,
                reg_digits=reg_digits, with_lvalue=with_lvalue,
                max_edge_seconds=max_edge_seconds, max_edge_steps=max_edge_steps)
            result['settings'] = settings
            have[n] = result
            save_stored(have, out)
        results.append(result)
        b._report_row(result, stored=cached)
    print("; ".join('%s: %d' % item for item in sorted(Counter(r['status'] for r in results).items())))
    print("q = det / (2*pi*c+*c-*L''(std,1)); rel. error = abs(q/rational - 1).")
    if out is not None:
        print('Results: %s' % out)
    return results


def print_stored(path=RESULTS):
    """Print saved successes using the common report columns."""
    rows = [r for _, r in sorted(load_stored(path).items()) if r.get('status') == 'ok']
    for row in rows:
        b._report_row(row, stored=True)
    return rows


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('The marked')[0])
    parser.add_argument('ns', nargs='*', type=int, help='integer parameters (default: -4 through 2)')
    parser.add_argument('--max-conductor', type=int, default=10**8,
                        help='inclusive standard-motive conductor bound (default: 100000000)')
    parser.add_argument('--order', choices=['cost', 'lex'], default='cost')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--digits', type=int, default=15, help='L-value decimal precision')
    parser.add_argument('--reg-digits', '--prec', type=int, default=None,
                        help='regulator decimal precision (default: float64)')
    parser.add_argument('--edge-seconds', type=float, default=15)
    parser.add_argument('--edge-steps', type=int, default=100000)
    parser.add_argument('--no-lvalue', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--out', default=RESULTS)
    args = parser.parse_args(argv)
    if (args.max_conductor < 1 or args.digits < 1 or args.edge_seconds <= 0
            or args.edge_steps < 1 or (args.limit is not None and args.limit < 1)
            or (args.reg_digits is not None and args.reg_digits < 5)):
        parser.error('bounds and budgets must be positive; regulator precision must be at least 5')
    return run(args.ns or None, out=args.out, limit=args.limit, digits=args.digits,
        force=args.force, order=args.order, max_conductor=args.max_conductor,
        verbose=args.verbose, reg_digits=args.reg_digits,
        with_lvalue=not args.no_lvalue, max_edge_seconds=args.edge_seconds,
        max_edge_steps=args.edge_steps)


if __name__ == '__main__':
    main()
