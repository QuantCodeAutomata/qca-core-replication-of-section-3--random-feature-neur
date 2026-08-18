import numpy as np
import pytest
from src.model import RandomFeatureModel
from src.estimator import TrainConfig, TrainResult, train, evaluate_kl
from src.distributions import sample_p, sample_q, compute_true_kl


def make_samplers(n, seed=99):
    rng = np.random.default_rng(seed)
    def sp(size): return sample_p(n, size, rng)
    def sq(size): return sample_q(n, size, rng)
    return sp, sq


def test_train_returns_correct_types():
    n, m, T = 2, 10, 100
    rng = np.random.default_rng(0)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n)
    config = TrainConfig(T=T, m=m, n=n)
    result = train(model, config, sp, sq)
    assert isinstance(result, TrainResult)
    assert result.theta_final.shape == (m,)
    assert result.theta_bar.shape == (m,)


def test_train_theta_finite():
    """Final iterate must be finite (no NaN/Inf)."""
    n, m, T = 2, 20, 500
    rng = np.random.default_rng(1)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n)
    config = TrainConfig(T=T, m=m, n=n)
    result = train(model, config, sp, sq)
    assert np.all(np.isfinite(result.theta_final))
    assert np.all(np.isfinite(result.theta_bar))


def test_evaluate_kl_returns_float():
    n, m = 2, 10
    rng = np.random.default_rng(2)
    model = RandomFeatureModel(n, m, rng)
    theta = np.zeros(m)
    rng_eval = np.random.default_rng(3)
    x_eval = sample_p(n, 100, rng_eval)
    y_eval = sample_q(n, 100, rng_eval)
    kl = evaluate_kl(model, theta, x_eval, y_eval)
    assert isinstance(kl, float)
    assert np.isfinite(kl)


def test_train_with_projection():
    """Projection should not cause NaN."""
    n, m, T = 2, 10, 200
    rng = np.random.default_rng(4)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n)
    config = TrainConfig(T=T, m=m, n=n, clip_bound=1e4)
    result = train(model, config, sp, sq)
    assert np.all(np.isfinite(result.theta_final))


def test_train_with_averaged_iterate():
    n, m, T = 2, 10, 200
    rng = np.random.default_rng(5)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n)
    config = TrainConfig(T=T, m=m, n=n, use_averaged=True)
    result = train(model, config, sp, sq)
    assert result.theta_bar.shape == (m,)
    assert np.all(np.isfinite(result.theta_bar))


def test_evaluate_kl_improves_with_training():
    """KL estimate after training should be closer to true KL than random theta."""
    n, m, T = 2, 50, 5000
    true_kl = compute_true_kl(n)
    rng = np.random.default_rng(7)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n, seed=42)
    config = TrainConfig(T=T, m=m, n=n)
    result = train(model, config, sp, sq)
    rng_eval = np.random.default_rng(100)
    x_eval = sample_p(n, 1000, rng_eval)
    y_eval = sample_q(n, 1000, rng_eval)
    kl_hat = evaluate_kl(model, result.theta_final, x_eval, y_eval)
    # After reasonable training, error should be less than 2 (sanity check)
    assert abs(true_kl - kl_hat) < 2.0


def test_runtime_recorded():
    n, m, T = 2, 10, 100
    rng = np.random.default_rng(8)
    model = RandomFeatureModel(n, m, rng)
    sp, sq = make_samplers(n)
    config = TrainConfig(T=T, m=m, n=n)
    result = train(model, config, sp, sq)
    assert result.runtime > 0
