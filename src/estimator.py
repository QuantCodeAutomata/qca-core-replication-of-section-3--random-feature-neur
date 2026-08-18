import numpy as np
from scipy.special import logsumexp
from dataclasses import dataclass, field
from typing import Optional
from src.model import RandomFeatureModel


@dataclass
class TrainConfig:
    T: int                    # number of iterations
    m: int                    # number of random features
    n: int                    # input dimension
    alpha: Optional[float] = None   # step size; if None: T^(-2/3)
    r: Optional[float] = None       # scaling; if None: 1/m
    clip_bound: Optional[float] = None  # C_theta for projection; if None: no projection
    use_averaged: bool = False       # use averaged iterate for evaluation
    n_eval: int = 5000              # number of evaluation samples per distribution
    seed: Optional[int] = None      # trial seed


@dataclass
class TrainResult:
    theta_final: np.ndarray          # final iterate theta_T
    theta_bar: np.ndarray            # time-averaged iterate
    kl_estimate: float               # D_hat evaluated on fresh samples
    projection_count: int            # how many times projection was active
    runtime: float                   # seconds


def train(
    model: RandomFeatureModel,
    config: TrainConfig,
    sample_p_fn,  # callable(size) -> np.ndarray (size, n)
    sample_q_fn,  # callable(size) -> np.ndarray (size, n)
) -> TrainResult:
    """
    Run the stochastic projected recursion for T iterations.
    Hyperparameters (if not provided):
      alpha = T^(-2/3)  (empirical from paper Section 3)
      r = 1/m           (empirical from paper Section 3)

    Initialize theta_0 ~ Uniform[-2e6/sqrt(m), 2e6/sqrt(m)]^m
    Initialize z_0 = 1.0

    For k = 0..T-1:
      Sample x_k ~ P_n, y_k ~ Q_n
      phi_x = phi(x_k), phi_y = phi(y_k)
      u_k = exp(phi_y.T @ theta_k)
      grad_theta = phi_x - (u_k / z_k) * phi_y
      theta_{k+1} = Pi_Theta(theta_k + alpha * r * grad_theta)
      z_{k+1} = z_k + alpha * (u_k - z_k)

    Projection Pi_Theta: clip each coordinate to [-clip_bound/m, clip_bound/m]
    if clip_bound is not None, else identity.

    Both updates use the OLD state (theta_k, z_k) from same iteration.
    Track running average: theta_bar = (1/k+1) * sum_{j=0}^{k} theta_j
    """
    import time

    T = config.T
    m = config.m
    alpha = config.alpha if config.alpha is not None else T**(-2/3)
    r = config.r if config.r is not None else 1.0 / m

    rng = np.random.default_rng(config.seed)

    # Initialize theta_0.
    # The paper states Uniform[-2e6/sqrt(m), 2e6/sqrt(m)], but this is within a
    # projection-set context (Pi_Theta clips coordinates to [-C_theta/m, C_theta/m]).
    # Without a known C_theta the large constant causes exp() overflow in float64.
    # Departure from paper: we initialize within [-1/m, 1/m]^m, which keeps
    # phi(y)^T theta_0 in a non-overflowing range (|phi_i| ≤ ~6.5, so the dot
    # product stays well below 500 in float64). When clip_bound is provided,
    # we initialize uniformly within the projection set instead.
    # Custom — Context7 found no library equivalent (paper Section 3 init)
    if config.clip_bound is not None:
        init_bound = config.clip_bound / m
    else:
        init_bound = 1.0 / m
    theta = rng.uniform(-init_bound, init_bound, size=m)

    # Initialize z_0
    z = 1.0

    # Running average
    theta_bar = np.zeros(m)

    projection_count = 0
    # Chunk size controls peak memory: CHUNK * m * 2 * 8 bytes.
    # At CHUNK=10_000 and m=200 this is ~32 MB, well within typical limits.
    CHUNK = 10_000
    cb = config.clip_bound / m if config.clip_bound is not None else None

    start = time.time()

    k = 0
    while k < T:
        # Draw samples in chunks to bound peak memory usage.
        chunk_size = min(CHUNK, T - k)
        x_chunk = sample_p_fn(chunk_size)       # (chunk_size, n)
        y_chunk = sample_q_fn(chunk_size)       # (chunk_size, n)
        phi_x_chunk = model.phi(x_chunk)        # (chunk_size, m)
        phi_y_chunk = model.phi(y_chunk)        # (chunk_size, m)

        for i in range(chunk_size):
            phi_x_k = phi_x_chunk[i]
            phi_y_k = phi_y_chunk[i]

            # u_k = exp(phi_y_k^T theta_k); clip inner product for float64 safety
            dot_y = np.clip(phi_y_k @ theta, -500.0, 500.0)
            u_k = np.exp(dot_y)

            # Stochastic projected gradient step on theta
            grad = phi_x_k - (u_k / z) * phi_y_k
            theta_new = theta + alpha * r * grad

            # Projection Pi_Theta (coordinate-wise clipping)
            if cb is not None:
                theta_clipped = np.clip(theta_new, -cb, cb)
                if not np.array_equal(theta_clipped, theta_new):
                    projection_count += 1
                theta_new = theta_clipped

            # z auxiliary update (uses OLD z)
            z = z + alpha * (u_k - z)
            theta = theta_new

            # Online running average (Welford-style)
            theta_bar = theta_bar + (theta - theta_bar) / (k + 1)
            k += 1

    runtime = time.time() - start

    return TrainResult(
        theta_final=theta,
        theta_bar=theta_bar,
        kl_estimate=0.0,  # to be filled by caller
        projection_count=projection_count,
        runtime=runtime,
    )


def evaluate_kl(
    model: RandomFeatureModel,
    theta: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    """
    Estimate KL divergence using Donsker-Varadhan formula:
    D_hat = mean_i psi(x_i, theta) - log(mean_i exp(psi(y_i, theta)))
    Use logsumexp for numerical stability.
    x_eval: (N, n), y_eval: (N, n)
    """
    scores_x = model.phi(x_eval) @ theta  # (N,)
    scores_y = model.phi(y_eval) @ theta  # (N,)
    N = len(scores_x)
    mean_p = np.mean(scores_x)
    log_mean_exp_q = logsumexp(scores_y) - np.log(N)
    return float(mean_p - log_mean_exp_q)
