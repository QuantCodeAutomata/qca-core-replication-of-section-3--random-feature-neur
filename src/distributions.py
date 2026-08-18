import numpy as np
from scipy import stats, integrate
from scipy.special import logsumexp


def sample_p(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from P_n: truncated standard Gaussian on [-2, 2]^n.

    Each coordinate is sampled independently from a 1-D standard normal
    truncated to [-2, 2] using ``scipy.stats.truncnorm(a=-2, b=2)``.

    Parameters
    ----------
    n:
        Dimensionality of the distribution.
    size:
        Number of samples to draw.
    rng:
        NumPy random generator used for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(size, n)`` containing the samples.
    """
    dist = stats.truncnorm(a=-2, b=2)
    return dist.rvs(size=(size, n), random_state=rng)


def sample_q(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from Q_n: uniform distribution on [-2, 2]^n.

    Parameters
    ----------
    n:
        Dimensionality of the distribution.
    size:
        Number of samples to draw.
    rng:
        NumPy random generator used for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(size, n)`` containing the samples.
    """
    return rng.uniform(low=-2.0, high=2.0, size=(size, n))


def compute_true_kl(n: int) -> float:
    """Compute D_KL(P_n || Q_n) analytically as ``n`` times the 1-D KL divergence.

    The 1-D KL divergence between the truncated Gaussian ``p_1`` and the
    uniform ``q_1 = 1/4`` on ``[-2, 2]`` is:

    .. math::

        KL_{1D} = \\int_{-2}^{2} p_1(x)\\,[\\log p_1(x) - \\log q_1(x)]\\,dx

    with ``p_1(x) = exp(-x^2/2) / Z_1`` and
    ``Z_1 = \\int_{-2}^{2} exp(-x^2/2)\\,dx``.  This simplifies to:

    .. math::

        KL_{1D} = E_{p_1}[-X^2/2] - \\log Z_1 + \\log 4

    Both integrals are evaluated with ``scipy.integrate.quad``.  The n-D KL
    factorises as ``D_KL(P_n || Q_n) = n * KL_{1D}`` because the coordinates
    are independent.

    Parameters
    ----------
    n:
        Dimensionality, giving ``D_KL(P_n || Q_n) = n * KL_1D``.

    Returns
    -------
    float
        The KL divergence ``D_KL(P_n || Q_n)``.
    """
    Z1, _ = integrate.quad(lambda x: np.exp(-x**2 / 2), -2.0, 2.0)

    # E_{p_1}[-X^2/2] = integral_{-2}^{2} (-x^2/2) * p_1(x) dx
    E_neg_half_x2, _ = integrate.quad(
        lambda x: (-x**2 / 2) * np.exp(-x**2 / 2) / Z1, -2.0, 2.0
    )

    kl_1d: float = E_neg_half_x2 - np.log(Z1) + np.log(4.0)
    return float(n * kl_1d)


def compute_true_kl_monte_carlo(
    n: int,
    rng: np.random.Generator,
    n_samples: int = 500_000,
) -> float:
    """Monte Carlo estimate of D_KL(P_n || Q_n) using samples from P_n.

    Estimates the expectation:

    .. math::

        E_P\\!\\left[\\log\\frac{p(x)}{q(x)}\\right]
        = E_P\\!\\left[-\\frac{\\|x\\|^2}{2} - \\log Z_n + n\\log 4\\right]

    where ``Z_n = Z_1^n`` and ``Z_1 = \\int_{-2}^{2} exp(-x^2/2)\\,dx``.

    Parameters
    ----------
    n:
        Dimensionality of the distribution.
    rng:
        NumPy random generator used for reproducibility.
    n_samples:
        Number of Monte Carlo samples. Default is ``500_000``.

    Returns
    -------
    float
        The Monte Carlo estimate of ``D_KL(P_n || Q_n)``.
    """
    Z1, _ = integrate.quad(lambda x: np.exp(-x**2 / 2), -2.0, 2.0)
    log_Zn: float = n * np.log(Z1)

    samples = sample_p(n=n, size=n_samples, rng=rng)  # shape (n_samples, n)

    # log(p(x) / q(x)) = -||x||^2/2 - log(Z_n) + n*log(4)
    log_ratios: np.ndarray = (
        -0.5 * np.sum(samples**2, axis=1) - log_Zn + n * np.log(4.0)
    )
    return float(np.mean(log_ratios))
