import numpy as np
import pytest
from src.distributions import sample_p, sample_q, compute_true_kl, compute_true_kl_monte_carlo


def test_sample_p_shape():
    rng = np.random.default_rng(0)
    x = sample_p(2, 100, rng)
    assert x.shape == (100, 2)


def test_sample_p_5d_shape():
    rng = np.random.default_rng(1)
    x = sample_p(5, 50, rng)
    assert x.shape == (50, 5)


def test_sample_p_within_bounds():
    """All samples from P_n must lie within [-2,2]^n."""
    rng = np.random.default_rng(2)
    x = sample_p(5, 10000, rng)
    assert np.all(x >= -2.0) and np.all(x <= 2.0)


def test_sample_q_shape():
    rng = np.random.default_rng(3)
    y = sample_q(2, 100, rng)
    assert y.shape == (100, 2)


def test_sample_q_within_bounds():
    """All samples from Q_n must lie within [-2,2]^n."""
    rng = np.random.default_rng(4)
    y = sample_q(5, 10000, rng)
    assert np.all(y >= -2.0) and np.all(y <= 2.0)


def test_true_kl_2d_positive():
    kl = compute_true_kl(2)
    assert kl > 0, 'KL divergence must be positive'


def test_true_kl_5d_positive():
    kl = compute_true_kl(5)
    assert kl > 0


def test_true_kl_5d_greater_than_2d():
    """5D KL should be greater than 2D KL (KL is additive for product distributions)."""
    kl_2d = compute_true_kl(2)
    kl_5d = compute_true_kl(5)
    assert kl_5d > kl_2d


def test_true_kl_additivity():
    """D_KL(P_n || Q_n) = n * KL_1D, so 5D = (5/2) * 2D."""
    kl_2d = compute_true_kl(2)
    kl_5d = compute_true_kl(5)
    assert abs(kl_5d - (5 / 2) * kl_2d) < 1e-6


def test_true_kl_1d_reasonable_range():
    """KL_1D should be a small positive number (truncated vs uniform)."""
    kl_1d = compute_true_kl(1)
    assert 0 < kl_1d < 1.0


def test_true_kl_monte_carlo_close():
    """Monte Carlo estimate should be close to numerical integral."""
    rng = np.random.default_rng(42)
    kl_exact = compute_true_kl(2)
    kl_mc = compute_true_kl_monte_carlo(2, rng, n_samples=1000000)
    assert abs(kl_exact - kl_mc) < 0.01, f'MC={kl_mc:.4f} vs exact={kl_exact:.4f}'


def test_single_sample_p():
    rng = np.random.default_rng(5)
    x = sample_p(2, 1, rng)
    assert x.shape == (1, 2)
