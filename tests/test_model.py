import numpy as np
import pytest
from src.model import RandomFeatureModel


def test_phi_single_output_shape():
    rng = np.random.default_rng(0)
    model = RandomFeatureModel(n=2, m=10, rng=rng)
    x = np.zeros(2)
    phi = model.phi(x)
    assert phi.shape == (10,)


def test_phi_batch_output_shape():
    rng = np.random.default_rng(1)
    model = RandomFeatureModel(n=5, m=20, rng=rng)
    X = np.ones((100, 5))
    phi = model.phi(X)
    assert phi.shape == (100, 20)


def test_phi_nonnegative():
    """ReLU features must be non-negative."""
    rng = np.random.default_rng(2)
    model = RandomFeatureModel(n=3, m=50, rng=rng)
    X = np.random.randn(200, 3)
    phi = model.phi(X)
    assert np.all(phi >= 0)


def test_weights_on_unit_sphere():
    """Each row of W must have unit L2 norm."""
    rng = np.random.default_rng(3)
    model = RandomFeatureModel(n=5, m=100, rng=rng)
    norms = np.linalg.norm(model.W, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_biases_in_range():
    """Biases must lie in [-2, 2]."""
    rng = np.random.default_rng(4)
    model = RandomFeatureModel(n=2, m=200, rng=rng)
    assert np.all(model.b >= -2.0) and np.all(model.b <= 2.0)


def test_psi_scalar_output():
    rng = np.random.default_rng(5)
    model = RandomFeatureModel(n=2, m=10, rng=rng)
    theta = np.ones(10)
    x = np.zeros(2)
    val = model.psi(x, theta)
    assert np.isscalar(val) or val.shape == ()


def test_psi_batch():
    rng = np.random.default_rng(6)
    model = RandomFeatureModel(n=2, m=10, rng=rng)
    theta = np.ones(10)
    X = np.zeros((50, 2))
    vals = model.psi(X, theta)
    assert vals.shape == (50,)


def test_different_seeds_different_weights():
    """Two models with different seeds should have different weights."""
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)
    m1 = RandomFeatureModel(n=2, m=10, rng=rng1)
    m2 = RandomFeatureModel(n=2, m=10, rng=rng2)
    assert not np.allclose(m1.W, m2.W)
