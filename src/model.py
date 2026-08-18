import numpy as np


class RandomFeatureModel:
    """
    Shallow random-feature ReLU network for KL divergence estimation.
    Hidden weights W (m x n) and biases b (m,) are fixed at init.
    Only output coefficients theta (m,) are trained.
    phi(x) = max(W @ x + b, 0)  (ReLU applied element-wise)
    psi(x, theta) = phi(x).T @ theta
    """

    def __init__(self, n: int, m: int, rng: np.random.Generator):
        # Sample W rows: each row is a unit-sphere vector in R^n
        # Draw standard normal vectors and normalize each to unit norm
        W_raw = rng.standard_normal((m, n))
        norms = np.linalg.norm(W_raw, axis=1, keepdims=True)
        self.W = W_raw / norms  # shape (m, n)
        # Sample biases uniformly from [-2, 2]
        self.b = rng.uniform(-2.0, 2.0, size=m)  # shape (m,)
        self.m = m
        self.n = n

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Compute feature map phi(x). x shape: (n,) or (N, n). Returns (m,) or (N, m)."""
        if x.ndim == 1:
            return np.maximum(self.W @ x + self.b, 0.0)
        else:  # (N, n)
            return np.maximum(x @ self.W.T + self.b, 0.0)  # (N, m)

    def psi(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute psi(x, theta) = phi(x).T @ theta."""
        return self.phi(x) @ theta
