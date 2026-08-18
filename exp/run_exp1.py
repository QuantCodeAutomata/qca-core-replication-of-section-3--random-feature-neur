"""
Experiment 1: Core replication of KL divergence estimation scaling study.

Key replication assumptions:
- use_averaged=False: final iterate theta_T is evaluated (not Polyak-Ruppert average),
  matching the default reported in Section 3 of the paper.
- alpha = T^{-2/3}, r = 1/m: empirical step-size and regularisation hyperparameters
  taken from Section 3; these are set inside TrainConfig by the estimator module.
- clip_bound=None: no projection step applied, following the sensitivity assumption
  that unconstrained SGD suffices in practice.
- T and m grids: reconstructed from reasonable log-spaced ranges; the exact grids
  are not available from the paper repository.

Run as:
    python exp/run_exp1.py
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

# Suppress overflow warnings from exp() in the KL estimator; values are
# clipped downstream, so these are benign numerical artefacts.
np.seterr(over='ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8')

# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------
N_TRIALS = 10
N_EVAL = 5000

# 2D (n=2) grids  — reduced from paper to be computationally feasible at ~27µs/iter
N_2D = 2
T_GRID_2D = [500, 2_000, 5_000, 20_000, 50_000]
M_GRID_2D = [10, 20, 30, 50, 70, 100]
M_FIXED_2D = 50        # held fixed when varying T
T_FIXED_2D = 50_000    # held fixed when varying m (paper: 500,000)

# 5D (n=5) grids  — reduced from paper to be computationally feasible
N_5D = 5
T_GRID_5D = [500, 2_000, 5_000, 20_000, 50_000, 100_000]
M_GRID_5D = [10, 20, 50, 100, 150]
M_FIXED_5D = 100       # held fixed when varying T
T_FIXED_5D = 100_000   # held fixed when varying m (paper: 1,000,000)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_scaling(agg_df, vary_col, title, xlabel, save_path):
    """Log-log error-bar plot of mean absolute KL error vs. a scaling parameter."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = agg_df[vary_col].values.astype(float)
    y = agg_df['mean_abs_error'].values
    err_low = np.clip(y - agg_df['ci_3se_low'].values, 0, None)
    err_high = np.clip(agg_df['ci_3se_high'].values - y, 0, None)

    ax.errorbar(x, y, yerr=[err_low, err_high],
                fmt='o-', capsize=5, linewidth=1.8, markersize=6,
                label='Mean |KL error| ± 3 SE')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Mean absolute KL error', fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved plot → {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    studies = [
        dict(
            label='2D vary-T',
            n=N_2D,
            m_fixed=M_FIXED_2D, T_fixed=None,
            vary_values=T_GRID_2D,
            vary_col='T',
            csv_name='exp1_2d_vary_T.csv',
            png_name='exp1_2d_vary_T.png',
            title=f'2D KL Estimation Error vs. T (m={M_FIXED_2D})',
            xlabel='T (SGD iterations)',
        ),
        dict(
            label='2D vary-m',
            n=N_2D,
            m_fixed=None, T_fixed=T_FIXED_2D,
            vary_values=M_GRID_2D,
            vary_col='m',
            csv_name='exp1_2d_vary_m.csv',
            png_name='exp1_2d_vary_m.png',
            title=f'2D KL Estimation Error vs. m (T={T_FIXED_2D:,})',
            xlabel='m (random features)',
        ),
        dict(
            label='5D vary-T',
            n=N_5D,
            m_fixed=M_FIXED_5D, T_fixed=None,
            vary_values=T_GRID_5D,
            vary_col='T',
            csv_name='exp1_5d_vary_T.csv',
            png_name='exp1_5d_vary_T.png',
            title=f'5D KL Estimation Error vs. T (m={M_FIXED_5D})',
            xlabel='T (SGD iterations)',
        ),
        dict(
            label='5D vary-m',
            n=N_5D,
            m_fixed=None, T_fixed=T_FIXED_5D,
            vary_values=M_GRID_5D,
            vary_col='m',
            csv_name='exp1_5d_vary_m.csv',
            png_name='exp1_5d_vary_m.png',
            title=f'5D KL Estimation Error vs. m (T={T_FIXED_5D:,})',
            xlabel='m (random features)',
        ),
    ]

    summary_rows = []

    print('=== Experiment 1: Core Replication ===\n')
    for study in tqdm(studies, desc='Studies', position=0):
        print(f"\n[{study['label']}]")
        df = run_scaling_study(
            n=study['n'],
            m_fixed=study['m_fixed'],
            T_fixed=study['T_fixed'],
            vary_values=study['vary_values'],
            n_trials=N_TRIALS,
            n_eval=N_EVAL,
            use_averaged=False,   # final iterate theta_T (see module docstring)
            clip_bound=None,
        )
        agg = aggregate_results(df, study['vary_col'])

        csv_path = os.path.join(RESULTS_DIR, study['csv_name'])
        agg.to_csv(csv_path, index=False)
        print(f'  Saved data  → {csv_path}')

        plot_scaling(
            agg_df=agg,
            vary_col=study['vary_col'],
            title=study['title'],
            xlabel=study['xlabel'],
            save_path=os.path.join(RESULTS_DIR, study['png_name']),
        )

        for _, row in agg.iterrows():
            summary_rows.append({
                'study': study['label'],
                study['vary_col']: row[study['vary_col']],
                'mean_abs_error': row['mean_abs_error'],
                'stderr': row['stderr'],
            })

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print('\n\n=== Summary Table ===')
    for study in studies:
        vc = study['vary_col']
        rows = [r for r in summary_rows if r['study'] == study['label']]
        print(f"\n{study['label']}  (n={study['n']}):")
        print(f"  {'':>10s}  {'mean |err|':>14s}  {'stderr':>10s}")
        for r in rows:
            print(f"  {vc}={int(r[vc]):<8d}  {r['mean_abs_error']:>14.6f}  {r['stderr']:>10.6f}")

    print('\nDone.')


if __name__ == '__main__':
    main()
