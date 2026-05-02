# Wasserstein Profile

Code for computing Wasserstein matching profiles for red and blue point sets on the line.
This code accompanies the paper 'Computing All Optimal Partial $p$-Wasserstein Matchings on the Line', currently in preparation, by Sebastian Angrick, Jacobus Conradi, Monika Csikos, Niko Hastrich, Danny Mittal, André Nusser, and Krzystof Onak.

The repository contains:
- simple baseline implementations for correctness checks,
- faster heap-based profile algorithms,
- a C++ extension for fast profile computation,
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

The C++ extension uses FFTW.

```bash
sudo apt-get install libfftw3-dev
```

Then build and install the extension from the repository root:

```bash
pip install -e wasserstein_profile
```

## Running the experiments

Run

```bash
python main.py
```

Most experiment scripts are configured by editing the constants at the top of the file.

The runtime-grid experiment produces a 3 x 4 panel layout:
- **Row 1:** vary the variance of the second distribution,
- **Row 2:** vary the mean of the second distribution,
- **Row 3:** compare unimodal against bimodal.

Each panel benchmarks the algorithms for multiplicatively increasing `n` until each algorithm hits the timeout.

## Competitor implementation

The file `competition.py` contains the competitor implementation used in the runtime comparisons. It is copied from the [code](https://github.com/rtavenar/partial_ot_1d) provided in the ICLR'25 paper [One for all and all for one: Efficient Computation of
Partial Wasserstein Distances on the Line](https://openreview.net/forum?id=kzEPsHbJDv) by Laetitia Chapel and Romain Tavenard.
This code is included here for benchmarking and experimental comparison.

## C++ extension

After installation, the fast solver is available from Python as:

```python
from wasserstein_profile import (
    profile_p,
    profile_p_with_lifetimes,
    profile_squared,
    profile_squared_with_lifetimes,
)
```

### Available functions

- `profile_p(R, B, p)`  
  Returns the full Wasserstein profile for integer `p >= 1`.

- `profile_p_with_lifetimes(R, B, p)`  
  Returns the profile together with interval lifetime data.

- `profile_squared(R, B)`  
  Convenience wrapper for `p = 2`.

- `profile_squared_with_lifetimes(R, B)`  
  Convenience wrapper for `p = 2` with interval lifetime output.

Inputs `R` and `B` must be sorted one-dimensional `float64` arrays of equal length.

Example:

```python
import numpy as np
from wasserstein_profile import profile_p

R = np.array([0.0, 1.0, 2.0], dtype=np.float64)
B = np.array([-2.0, -1.0, 0.0], dtype=np.float64)

costs = profile_p(R, B, 2)
print(costs)
```

The `_with_lifetimes` variants return:
- a NumPy array of profile costs,
- a NumPy array with one row per interval lifetime, storing

```text
[red_start, red_end, blue_start, blue_end, born_k, dead_k_exclusive]
```