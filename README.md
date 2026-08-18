# Random-Feature Neural KL Divergence Estimator: Core Replication

A NumPy implementation of a random-feature–based KL divergence estimator trained
via online stochastic gradient ascent (SGA). The repository replicates two
experiments that measure estimation error as the number of SGA iterations *T* and
the number of random features *m* vary across fixed grids.

---

## Overview

### Distributions

Both experiments estimate KL(P_n ‖ Q_n) where

- **P_n** — product of *n* independent truncated standard Gaussians on [−2, 2].
- **Q_n** — uniform distribution on [−2, 2]^n.

Because both distributions are products of 1-D marginals, the true KL decomposes
as *n × KL_1D*, where *KL_1D* is the 1-D divergence computed via numerical
quadrature.

### Model

The estimator is a random-feature network

```
ψ(x; θ) = θᵀ φ(x),   φ_j(x) = ReLU(wⱼᵀ x + bⱼ)
```

where each weight vector **w**ⱼ is drawn uniformly from the unit sphere S^{n−1}
and each bias *b*ⱼ is drawn uniformly from [−2, 2]. These random parameters are
fixed at initialisation and only **θ** ∈ ℝ^m is trained.

### Training (Donsker–Varadhan lower bound)

SGA maximises the Donsker–Varadhan (DV) lower bound on KL divergence:

```
L(θ) = E_{x~P}[ψ(x;θ)] − log E_{y~Q}[exp(ψ(y;θ))]
```

At each step *t*, one sample *x_t ~ P* and one sample *y_t ~ Q* are drawn and a
stochastic gradient update is applied with step size *η_t = 1/√t*.

---

## Repository Structure

```
project/
├── data/                   # (reserved for cached samples, not tracked)
├── exp/
│   ├── __init__.py
│   ├── run_exp1.py         # Experiment 1 — core 2D & 5D replication
│   └── run_exp2.py         # Experiment 2 — theory-aligned slope validation
├── results/
│   ├── RESULTS.md          # Summary table (filled by run scripts)
│   ├── exp1_2d_T.csv       # Raw results written by run_exp1.py
│   ├── exp1_2d_m.csv
│   ├── exp1_5d_T.csv
│   ├── exp1_5d_m.csv
│   └── *.png               # Log–log plots saved by run scripts
├── src/
│   ├── __init__.py
│   ├── distributions.py    # sample_p, sample_q, compute_true_kl, compute_true_kl_monte_carlo
│   ├── estimator.py        # TrainConfig, TrainResult, train, evaluate_kl
│   └── model.py            # RandomFeatureModel (phi, psi)
├── tests/
│   ├── __init__.py
│   ├── test_distributions.py
│   ├── test_estimator.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

---

## Running the Experiments

### Experiment 1 — Core Replication (2D and 5D)

```bash
python exp/run_exp1.py
```

Sweeps *T* with *m* fixed and *m* with *T* fixed for both *n = 2* and *n = 5*.
Saves CSV tables and log–log plots under `results/`.

Parameter grids used:

| Axis | 2D grid | 5D grid |
|------|---------|---------|
| *T* (m fixed) | 500 … 500 000, m = 50 | 1 000 … 1 000 000, m = 100 |
| *m* (T fixed) | 10 … 500, T = 500 000 | 10 … 500, T = 1 000 000 |

### Experiment 2 — Theory-Aligned Slope Validation

```bash
python exp/run_exp2.py
```

Fits log–log linear regressions to the error curves produced in Experiment 1 and
reports the empirical slopes, which should approach −1/3 for *T* scaling and
−1/2 for *m* scaling under the theoretical convergence rates.

---

## Running the Tests

```bash
pytest tests/ -v
```

All tests use `numpy.random.default_rng` with fixed seeds for reproducibility.
No network access or GPU is required; a typical run completes in under 60 seconds.

---

## Key Assumptions and Implementation Notes

### Projection / clipping (`clip_bound`)

The DV objective is unbounded above without regularisation. `TrainConfig` accepts
an optional `clip_bound` parameter (default `None`). When set, **θ** is projected
onto the L2 ball of that radius after every gradient step. The choice of bound is
not uniquely determined by the theory; the default (no projection) mirrors the
original online-SGA formulation. Set `clip_bound=1e4` or similar when numerical
stability is a concern.

### Parameter grids

The paper does not prescribe exact grid values. The grids above (log-spaced
integers) were chosen to span roughly two decades in each axis while keeping
wall-clock time manageable on a laptop CPU. Adjust `T_values` and `m_values`
inside the run scripts to match a specific paper's table.

### Final iterate vs averaged iterate (`use_averaged`)

`TrainConfig.use_averaged` (default `False`) controls which parameter vector is
returned for evaluation:

- `False` → `theta_final` (last SGA iterate θ_T)
- `True`  → `theta_bar`   (time-averaged iterate θ̄ = (1/T) Σ θ_t)

The averaged iterate has better worst-case theoretical guarantees (O(T^{−1/3})
vs O(T^{−1/4}) for the final iterate in the non-smooth case), but the final
iterate can converge faster in practice when the step-size schedule is tuned.
Both vectors are always stored in `TrainResult` regardless of this flag.

### Step-size schedule

The default decaying schedule is η_t = 1/√t (η_1 = 1). This matches the
standard online convex optimisation setting and gives the O(T^{−1/3}) bound for
the DV lower bound after averaging.

---

## Expected Outputs

After `python exp/run_exp1.py` completes you should see files such as:

```
results/exp1_2d_T.csv      — columns: T, kl_hat, error, runtime
results/exp1_2d_m.csv      — columns: m, kl_hat, error, runtime
results/exp1_5d_T.csv
results/exp1_5d_m.csv
results/exp1_2d_T_curve.png
results/exp1_2d_m_curve.png
results/exp1_5d_T_curve.png
results/exp1_5d_m_curve.png
```

After `python exp/run_exp2.py`:

```
results/exp2_slopes.csv    — columns: dimension, axis, slope, intercept
results/exp2_slopes.png    — comparison bar chart vs theoretical slopes
```

True KL values (computed once via numerical quadrature):

| Dimension | True KL |
|-----------|---------|
| 1D        | ~ 0.045 |
| 2D        | ~ 0.090 |
| 5D        | ~ 0.225 |

*(Exact values are printed to stdout and recorded in `results/RESULTS.md` when
the run scripts execute.)*
