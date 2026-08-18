import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from src.distributions import sample_p, sample_q, compute_true_kl
from src.model import RandomFeatureModel
from src.estimator import TrainConfig, train, evaluate_kl


def run_single_trial(
    n: int,
    m: int,
    T: int,
    true_kl: float,
    n_eval: int = 5000,
    seed: int = 0,
    use_averaged: bool = False,
    clip_bound: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one trial for given (n, m, T). Returns dict with all metrics."""
    rng_model = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 100000)
    rng_eval = np.random.default_rng(seed + 200000)

    model = RandomFeatureModel(n, m, rng=rng_model)

    def sp(size): return sample_p(n, size, rng_data)
    def sq(size): return sample_q(n, size, rng_data)

    config = TrainConfig(
        T=T, m=m, n=n,
        use_averaged=use_averaged,
        n_eval=n_eval,
        seed=seed,
        clip_bound=clip_bound,
    )

    result = train(model, config, sp, sq)

    # Use final or averaged iterate
    theta_eval = result.theta_bar if use_averaged else result.theta_final

    # Fresh eval samples
    x_eval = sample_p(n, n_eval, rng_eval)
    y_eval = sample_q(n, n_eval, rng_eval)

    kl_hat = evaluate_kl(model, theta_eval, x_eval, y_eval)
    abs_err = abs(true_kl - kl_hat)

    return {
        'n': n, 'm': m, 'T': T, 'seed': seed,
        'use_averaged': use_averaged,
        'clip_bound': clip_bound,
        'true_kl': true_kl,
        'kl_hat': kl_hat,
        'abs_error': abs_err,
        'projection_count': result.projection_count,
        'runtime': result.runtime,
    }


def run_scaling_study(
    n: int,
    m_fixed: Optional[int],
    T_fixed: Optional[int],
    vary_values: List[int],
    n_trials: int = 10,
    n_eval: int = 5000,
    use_averaged: bool = False,
    clip_bound: Optional[float] = None,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run scaling study over either T or m. Returns DataFrame with aggregated results."""
    # One of m_fixed or T_fixed must be None (the varying axis)
    assert (m_fixed is None) != (T_fixed is None), 'Exactly one of m_fixed, T_fixed must be None'

    true_kl = compute_true_kl(n)
    vary_col = 'm' if m_fixed is None else 'T'
    rows = []

    total = len(vary_values) * n_trials
    with tqdm(total=total, desc=f'n={n} vary_{vary_col}') as pbar:
        for val in vary_values:
            m = val if m_fixed is None else m_fixed
            T = val if T_fixed is None else T_fixed

            for trial in range(n_trials):
                seed = base_seed + trial * 1000 + val
                row = run_single_trial(
                    n=n, m=m, T=T, true_kl=true_kl,
                    n_eval=n_eval, seed=seed,
                    use_averaged=use_averaged, clip_bound=clip_bound,
                )
                rows.append(row)
                pbar.update(1)

    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame, vary_col: str) -> pd.DataFrame:
    """Aggregate over trials: mean, std, stderr for abs_error."""
    grouped = df.groupby(vary_col)['abs_error']
    agg = grouped.agg(['mean', 'std', 'count']).reset_index()
    agg.columns = [vary_col, 'mean_abs_error', 'std_abs_error', 'n_trials']
    agg['stderr'] = agg['std_abs_error'] / np.sqrt(agg['n_trials'])
    agg['ci_3se_low'] = agg['mean_abs_error'] - 3 * agg['stderr']
    agg['ci_3se_high'] = agg['mean_abs_error'] + 3 * agg['stderr']
    return agg
