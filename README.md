# Wasserstein Profile

Code for computing Wasserstein matching profiles for red and blue point sets on the line.

The repository contains:
- simple baseline implementations for correctness checks,
- faster heap-based profile algorithms,
- a C++ extension for the fast profile computation,
- scripts for synthetic experiments and runtime plots.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install numpy matplotlib numba kneed pybind11 scikit-build-core
```

The C++ extension uses FFTW. Install that as well.

```bash
sudo apt-get install libfftw3-dev
```

Then build and install the package from the repository root:

```bash
pip install -e wasserstein_profile
```

## Running the experiments

Run

```bash
python main.py
```

This produces the runtime-grid experiment, and shows a 3 x 4 panel, containing:
- **Row 1:** vary the variance of the second distribution,
- **Row 2:** vary the mean of the second distribution,
- **Row 3:** compare unimodal against bimodal.

Each panel benchmarks the algorithms for multiplicatively increasing `n` until each algorithm hits the timeout of $60$ seconds. The entire experiment takes about $6$ hours.

## Competitor implementation

The file `competition.py` contains the competitor implementation used for comparison experiments. It is copied from the [code](https://github.com/rtavenar/partial_ot_1d) provided in the ICLR paper [One for all and all for one: Efficient computation of partial Wasserstein distances on the line](https://openreview.net/forum?id=kzEPsHbJDv).

This code is included here only for benchmarking and experimental comparison.

## C++ extension

After installation, the fast solver is available from Python as:

```python
from wasserstein_profile import profile_squared, profile_squared_with_lifetimes
```

Example:

```python
import numpy as np
from wasserstein_profile import profile_squared

R = np.array([0.0, 1.0, 2.0], dtype=np.float64)
B = np.array([-2.0, -1.0, 0.0], dtype=np.float64)

costs = profile_squared(R, B)
print(costs)
```

`profile_squared_with_lifetimes` additionally returns interval lifetime data for reconstructing matchings later.