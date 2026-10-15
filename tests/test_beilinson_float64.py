"""Bounded float64 checks against plane integrals and conjugate symmetry."""

from collections import Counter

import numpy as np

from hyperell_regulator.experiments.beilinson import beilinson_determinant_stokes
from hyperell_regulator.regulator import IntegrationMonitor
from hyperell_regulator.regulator.polynomials import ring


def test_conductor_249_determinant_matches_plane_with_bounded_edges():
    x = ring().gen()
    F = x**6 + 4*x**5 + 4*x**4 + 2*x**3 + 1
    # The plane determinant was assembled from independently integrated
    # periods and logarithmic weights, with roughly eight recorded decimals.
    plane_determinant = 16.743865748432064
    with IntegrationMonitor(max_edge_seconds=15) as monitor:
        det, matrix, errors = beilinson_determinant_stokes(
            F, (-1, 0), 7, (0, -1), 14, digits=None)
    values = np.asarray(matrix, dtype=float)
    assert np.isfinite(values).all()
    # Compare the individual weights too: a determinant alone can conceal
    # compensating errors or a spurious multiple of the period column.
    per = np.array([4.00236302275639, -1.55918682479619, 4.28626293488223])
    assert np.max(abs(values[:, 2] - per)) < 1e-8
    assert np.max(abs(3.5 * values[:, 0]
                      - [-3.82753395, 5.57796028, -0.24694098])) < 1e-8
    # Norm(phi_Q)=4*x^14, hence its normalization subtracts log(2)/7.
    assert np.max(abs(7 * values[:, 1] + np.log(2) * values[:, 2]
                      - [-5.21685044, -4.00181129, 13.80481861])) < 1e-8
    assert abs(float(det) - plane_determinant) < 1e-7
    assert errors is None
    # Two covers of five edges each: the ordinary periods must be reused,
    # including the third determinant column, rather than integrated again.
    assert len(monitor.records) == 10
    assert Counter(record['degree'] for record in monitor.records) == {7: 5, 14: 5}
    assert all(record['status'] == 'ok' for record in monitor.records)
    assert all(record['seconds'] <= 15 for record in monitor.records)
