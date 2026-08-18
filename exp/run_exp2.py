"""
Experiment 2: Theory-aligned validation using the Polyak-Ruppert averaged iterate.

Theoretical predictions (from convergence analysis):
  - KL error ~ T^{-1/3}  →  log-log slope ≈ -1/3 when varying T
  - KL error ~ m^{-1/2}  →  log-log slope ≈ -1/2 when varying m

Key differences from Experiment 1:
  - use_averaged=True  : Polyak-Ruppert averaged iterate theta_bar_T is evaluated
  - n_eval=50,000       : larger evaluation set for tighter Monte-Carlo estimates
  - Slope fitting       : log-log regression over the full decreasing regime

Run as:
    python exp/run_exp2.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.distributions import compute_true_kl
from src.experiment import run_scaling_study, aggregate_results

np.seterr(over='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8')

# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------
N_TRIALS = 10
N_EVAL = 20_000   # larger than exp1 for tighter MC estimates (paper: 100,000)

# Grids — reduced from paper for practical runtime (~27µs/iter on n=5, m=100)
T_GRID = [500, 2_000, 5_000, 20_000, 50_000, 100_000]
M_GRID = [10, 20, 50, 100, 200]

# Fixed hyperparameters per dimension
# n=2
N_2D = 2
M_FIXED_2D = 50      # held fixed when varying T
T_FIXED_2D = 50_000  # held fixed when varying m (paper: 500,000)

# n=5
N_5D = 5
M_FIXED_5D = 100      # held fixed when varying T
T_FIXED_5D = 100_000  # held fixed when varying m (paper: 1,000,000)

# Expected theoretical rates
EXPECTED_SLOPE_T = -1 / 3
EXPECTED_SLOPE_M = -1 / 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_loglog_slope(x: np.ndarray, y: np.ndarray):
    """
    Fit a power-law slope on the decreasing regime of a log-log curve.

    Uses all points with positive, finite errors; the decreasing regime is
    identified as the suffix starting from the index where the minimum error
    first appears (scanning left-to-right) to capture the consistent decay.
    Returns (slope, intercept) in natural-log space.
    """
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan

    x_m, y_m = x[mask], y[mask]

    # Identify the decreasing regime: drop leading points where errors
    # increase, keeping a contiguous suffix of monotone-ish decrease.
    # Simple heuristic: start from the first local peak, or from index 0.
    start = 0
    for i in range(1, len(y_m)):
        if y_m[i] < y_m[i - 1]:
            start = i - 1
            break

    x_fit = x_m[start:]
    y_fit = y_m[start:]
    if len(x_fit) < 2:
        x_fit, y_fit = x_m, y_m  # fall back to all points

    slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    return slope, intercept


def plot_loglog_with_slope(agg_df, vary_col, title, xlabel, slope, intercept,
                           expected_slope, save_path, extra_series=None):
    """
    Log-log error-bar plot with a fitted power-law line and slope annotation.

    Parameters
    ----------
    extra_series : list of dict, optional
        Additional series to overlay, each with keys
        {'agg_df', 'label', 'color'}.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x = agg_df[vary_col].values.astype(float)
    y = agg_df['mean_abs_error'].values
    err_low = np.clip(y - agg_df['ci_3se_low'].values, 0, None)
    err_high = np.clip(agg_df['ci_3se_high'].values - y, 0, None)

    ax.errorbar(x, y, yerr=[err_low, err_high],
                fmt='o-', capsize=5, linewidth=1.8, markersize=6,
                label='Averaged iterate ± 3 SE')

    # Fitted slope line
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        y_line = np.exp(intercept) * x_line ** slope
        ax.plot(x_line, y_line, '--', linewidth=1.5,
                label=f'Fitted slope: {slope:.3f}  (expected {expected_slope:.2f})')

    # Optional extra series (e.g. non-averaged iterate for comparison)
    if extra_series:
        colors = ['C2', 'C3', 'C4']
        for i, series in enumerate(extra_series):
            s_df = series['agg_df']
            sx = s_df[vary_col].values.astype(float)
            sy = s_df['mean_abs_error'].values
            sl = np.clip(sy - s_df['ci_3se_low'].values, 0, None)
            sh = np.clip(s_df['ci_3se_high'].values - sy, 0, None)
            ax.errorbar(sx, sy, yerr=[sl, sh],
                        fmt='s--', capsize=5, linewidth=1.4, markersize=5,
                        color=colors[i % len(colors)],
                        label=series['label'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean absolute KL error', fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved plot → {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=== Experiment 2: Theory-Aligned Validation (Averaged Iterate) ===\n')

    # -----------------------------------------------------------------------
    # Data collection
    # -----------------------------------------------------------------------
    studies = [
        dict(
            label='2D vary-T (averaged)',
            n=N_2D, m_fixed=M_FIXED_2D, T_fixed=None, vary_values=T_GRID,
            vary_col='T', use_averaged=True,
            csv_name='exp2_2d_vary_T.csv',
            png_name='exp2_2d_vary_T.png',
            title=f'2D KL Error vs. T — Averaged Iterate (m={M_FIXED_2D})',
            xlabel='T (SGD iterations)',
            expected_slope=EXPECTED_SLOPE_T,
        ),
        dict(
            label='2D vary-m (averaged)',
            n=N_2D, m_fixed=None, T_fixed=T_FIXED_2D, vary_values=M_GRID,
            vary_col='m', use_averaged=True,
            csv_name='exp2_2d_vary_m.csv',
            png_name='exp2_2d_vary_m.png',
            title=f'2D KL Error vs. m — Averaged Iterate (T={T_FIXED_2D:,})',
            xlabel='m (random features)',
            expected_slope=EXPECTED_SLOPE_M,
        ),
        dict(
            label='5D vary-T (averaged)',
            n=N_5D, m_fixed=M_FIXED_5D, T_fixed=None, vary_values=T_GRID,
            vary_col='T', use_averaged=True,
            csv_name='exp2_5d_vary_T.csv',
            png_name='exp2_5d_vary_T.png',
            title=f'5D KL Error vs. T — Averaged Iterate (m={M_FIXED_5D})',
            xlabel='T (SGD iterations)',
            expected_slope=EXPECTED_SLOPE_T,
        ),
        dict(
            label='5D vary-m (averaged)',
            n=N_5D, m_fixed=None, T_fixed=T_FIXED_5D, vary_values=M_GRID,
            vary_col='m', use_averaged=True,
            csv_name='exp2_5d_vary_m.csv',
            png_name='exp2_5d_vary_m.png',
            title=f'5D KL Error vs. m — Averaged Iterate (T={T_FIXED_5D:,})',
            xlabel='m (random features)',
            expected_slope=EXPECTED_SLOPE_M,
        ),
    ]

    results = {}

    for study in tqdm(studies, desc='Studies', position=0):
        print(f"\n[{study['label']}]")
        df = run_scaling_study(
            n=study['n'],
            m_fixed=study['m_fixed'],
            T_fixed=study['T_fixed'],
            vary_values=study['vary_values'],
            n_trials=N_TRIALS,
            n_eval=N_EVAL,
            use_averaged=study['use_averaged'],
            clip_bound=None,
        )
        agg = aggregate_results(df, study['vary_col'])
        csv_path = os.path.join(RESULTS_DIR, study['csv_name'])
        agg.to_csv(csv_path, index=False)
        print(f'  Saved data  → {csv_path}')
        results[study['label']] = {'agg': agg, 'study': study}

    # -----------------------------------------------------------------------
    # Comparison study: non-averaged iterate for n=2, vary T
    # -----------------------------------------------------------------------
    print('\n[2D vary-T (non-averaged) — for comparison]')
    df_nonavg = run_scaling_study(
        n=N_2D, m_fixed=M_FIXED_2D, T_fixed=None, vary_values=T_GRID,
        n_trials=N_TRIALS, n_eval=N_EVAL,
        use_averaged=False, clip_bound=None,
    )
    agg_nonavg = aggregate_results(df_nonavg, 'T')
    agg_nonavg.to_csv(os.path.join(RESULTS_DIR, 'exp2_2d_vary_T_nonavg.csv'), index=False)

    # -----------------------------------------------------------------------
    # Slope fitting and individual plots
    # -----------------------------------------------------------------------
    print('\n\n=== Fitted Log-Log Slopes ===')
    fitted_slopes = {}

    for label, res in results.items():
        agg = res['agg']
        study = res['study']
        vc = study['vary_col']

        x = agg[vc].values.astype(float)
        y = agg['mean_abs_error'].values
        slope, intercept = fit_loglog_slope(x, y)
        fitted_slopes[label] = slope

        print(f"  {label:<35s}  fitted={slope:+.3f}  expected={study['expected_slope']:+.2f}")

        plot_loglog_with_slope(
            agg_df=agg,
            vary_col=vc,
            title=study['title'],
            xlabel=study['xlabel'],
            slope=slope,
            intercept=intercept,
            expected_slope=study['expected_slope'],
            save_path=os.path.join(RESULTS_DIR, study['png_name']),
        )

    # -----------------------------------------------------------------------
    # Comparison plot: averaged vs non-averaged for n=2, vary T
    # -----------------------------------------------------------------------
    print('\n[Comparison plot: n=2, vary T]')
    agg_avg_2d_T = results['2D vary-T (averaged)']['agg']
    slope_avg, intercept_avg = fit_loglog_slope(
        agg_avg_2d_T['T'].values.astype(float),
        agg_avg_2d_T['mean_abs_error'].values,
    )
    slope_nonavg, intercept_nonavg = fit_loglog_slope(
        agg_nonavg['T'].values.astype(float),
        agg_nonavg['mean_abs_error'].values,
    )

    # Build comparison plot manually so both iterates appear with slope lines
    fig, ax = plt.subplots(figsize=(8, 5))

    for agg_df, label, color, slope, intercept in [
        (agg_avg_2d_T, f'Averaged (slope={slope_avg:.3f})', 'C0', slope_avg, intercept_avg),
        (agg_nonavg, f'Non-averaged (slope={slope_nonavg:.3f})', 'C1', slope_nonavg, intercept_nonavg),
    ]:
        x = agg_df['T'].values.astype(float)
        y = agg_df['mean_abs_error'].values
        err_low = np.clip(y - agg_df['ci_3se_low'].values, 0, None)
        err_high = np.clip(agg_df['ci_3se_high'].values - y, 0, None)
        ax.errorbar(x, y, yerr=[err_low, err_high],
                    fmt='o-', capsize=4, linewidth=1.8, markersize=6,
                    color=color, label=f'{label} ± 3 SE')
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
            y_line = np.exp(intercept) * x_line ** slope
            ax.plot(x_line, y_line, '--', linewidth=1.3, color=color, alpha=0.7)

    ax.axhline(y=np.nan, color='grey', linestyle='--', linewidth=1.0,
               label=f'Expected slope: {EXPECTED_SLOPE_T:.2f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'2D KL Error vs. T: Averaged vs. Non-Averaged (m={M_FIXED_2D})', fontsize=13)
    ax.set_xlabel('T (SGD iterations)', fontsize=12)
    ax.set_ylabel('Mean absolute KL error', fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    cmp_path = os.path.join(RESULTS_DIR, 'exp2_comparison.png')
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f'  Saved plot → {cmp_path}')

    # -----------------------------------------------------------------------
    # Summary of slopes
    # -----------------------------------------------------------------------
    print('\n\n=== Rate Comparison Summary ===')
    print(f"{'Study':<35s}  {'Fitted':>8s}  {'Expected':>10s}  {'Δ':>8s}")
    print('-' * 70)
    for label, study_dict in [
        ('2D vary-T (averaged)', studies[0]),
        ('2D vary-m (averaged)', studies[1]),
        ('5D vary-T (averaged)', studies[2]),
        ('5D vary-m (averaged)', studies[3]),
    ]:
        fs = fitted_slopes.get(label, np.nan)
        es = study_dict['expected_slope']
        delta = fs - es if np.isfinite(fs) else np.nan
        print(f"  {label:<33s}  {fs:>+8.3f}  {es:>+10.2f}  {delta:>+8.3f}")

    slope_nonavg_val = slope_nonavg if np.isfinite(slope_nonavg) else np.nan
    print(f"\n  Comparison (n=2, vary T):")
    print(f"    Averaged iterate slope:     {slope_avg:+.3f}  (expected {EXPECTED_SLOPE_T:+.2f})")
    print(f"    Non-averaged iterate slope: {slope_nonavg_val:+.3f}")

    print('\nDone.')


if __name__ == '__main__':
    main()
