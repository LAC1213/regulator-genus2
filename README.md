# hyperell-regulator

A SageMath package for numerical tests of Beilinson's conjecture for the
standard motive of genus 2 curves over the rationals. This is the rank-5
submotive of the wedge-square of h^1(C) given by the kernel of the cup
product. This packages computes regulator
integrals, periods, standard L-values, and reduction/component data.
Regulators, periods and L-values support both float64 and arbitrary precision.

## Installation

Install into an existing SageMath environment:

```sh
sage -pip install git+https://github.com/LAC1213/hyperell-regulator.git
```

From a checkout, use `sage -pip install .`, or
`sage -pip install -e '.[test]'` for development and tests.
The patched `sage_cluster_pictures` dependency is installed automatically.

Installation also attempts to build the bundled Frobenius accelerator using
NTL and GMP. If that build is unavailable, the package uses Sage's implementation.
To require a working accelerator during installation and then check it:

```sh
HYPERELL_REGULATOR_REQUIRE_EULER=1 sage -pip install -v git+https://github.com/LAC1213/hyperell-regulator.git
sage -python -m hyperell_regulator.frobenius --check
```

See [hypellfrob-threaded](hypellfrob-threaded/README.md) for manual build instructions.

## Python API

Run in Sage or with `sage -python`:

```python
from sage.all import QQ, PolynomialRing
from hyperell_regulator.regulator import (
    periods_and_regulator_stokes, c_invariants,
)

x = PolynomialRing(QQ, "x").gen()
F = x**5 + (x + 1)**2              # curve y^2 = F(x)
phi = (x + 1, 1)                  # function (x + 1) + y

periods, regulator = periods_and_regulator_stokes(F, phi)  # float64
periods, regulator = periods_and_regulator_stokes(F, phi, prec=30)
c_plus, c_minus = c_invariants(F, prec=30)
```

Use exact rational coefficients for arbitrary precision. `prec` is a decimal
accuracy target and convergence failures raise exceptions. 

## Beilinson experiments

```sh
# Fetch curve data and test conductors below 1000; float64 by default.
sage -python -m hyperell_regulator.experiments.beilinson 1000 --fetch

# Reuse cached data and request 30-digit regulators and L-values.
sage -python -m hyperell_regulator.experiments.beilinson 250 --prec 30 --digits 30 --edge-seconds 120

# Test the parameter n=2 in the one-parameter family, without an L-value.
sage -python -m hyperell_regulator.experiments.family_n 2 --no-lvalue
```

Both experiments support `--prec`, `--digits`, `--no-lvalue`, `--verbose`,
`--edge-seconds`, `--force`, and `--out`; use `--help` for all options.
The default edge limit is 15 seconds. For `family_n`, `--max-conductor`
bounds the standard motive's conductor, rather than the curve conductor.

Reports compare `q = det / (2*pi*c_plus*c_minus*L''(std,1))` with a rational
guess. Data and results are cached under
`lmfdb_cache/` and `results/` in the working directory; set
`HYPERELL_REGULATOR_DATA` to choose another location.

The [saved 100-digit survey](reports/README.md) includes LaTeX tables,
full-precision values, exact functions, and regeneration instructions.

## Tests

Install the `test` extra as shown above, then run:

```sh
sage -python -m pytest
sage -t hyperell_regulator
```
