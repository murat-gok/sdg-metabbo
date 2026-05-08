"""
Publication-quality figures for SDG-MetaBBO paper.

Generates:
  1. Convergence curves (median + IQR) — SDG vs L-SHADE per function
  2. Operator selection heatmap across functions
  3. Disagreement profile on Lunacek (basin boundary validation)
  4. Sensitivity plots (α and M)

Output: figures/ directory with PDF files ready for LaTeX \\includegraphics

Run: python experiments/plot_figures.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.asgd_full import SDGOptimizer
from src.core.de_engine import DEEngine
from src.core.operators import OPERATOR_NAMES, NUM_OPERATORS
from src.surrogate.ensemble import SurrogateEnsemble
from benchmarks.functions import get_simple_problem, SIMPLE_FUNCTIONS

# Output directory
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# SDG settings (light surrogate for speed)
SK = dict(surr_hidden=(32, 16), surr_epochs=8, surr_retrain_interval=15,
          surr_data_window=150, surr_min_data=50)

# Plot style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'SDG': '#2166ac',
    'L-SHADE': '#b2182b',
    'GD-only': '#999999',
}


def collect_convergence(fname, dim=10, evals=5000, n_seeds=15):
    """Run SDG and L-SHADE, collect convergence curves."""
    sdg_curves = []
    lsh_curves = []

    for s in range(n_seeds):
        func, lb, ub, opt = get_simple_problem(fname, dim)

        # SDG
        o = SDGOptimizer(func, dim, (lb, ub), evals, seed=s, **SK)
        r = o.run()
        curve = np.array(r['convergence_curve'])
        sdg_curves.append(np.abs(curve - opt))

        # L-SHADE
        e = DEEngine(func, dim, (lb, ub), evals, seed=s, fixed_operator=2)
        r2 = e.run()
        curve2 = np.array(r2['convergence_curve'])
        lsh_curves.append(np.abs(curve2 - opt))

    # Align lengths (pad with last value)
    max_len = max(max(len(c) for c in sdg_curves), max(len(c) for c in lsh_curves))
    for curves in [sdg_curves, lsh_curves]:
        for i in range(len(curves)):
            if len(curves[i]) < max_len:
                pad = np.full(max_len - len(curves[i]), curves[i][-1])
                curves[i] = np.concatenate([curves[i], pad])

    return np.array(sdg_curves), np.array(lsh_curves)


def plot_convergence_single(ax, sdg_curves, lsh_curves, fname, show_ylabel=True):
    """Plot convergence for one function on given axes."""
    gens = np.arange(sdg_curves.shape[1])

    for label, curves, color in [
        ('SDG', sdg_curves, COLORS['SDG']),
        ('L-SHADE', lsh_curves, COLORS['L-SHADE']),
    ]:
        median = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25, axis=0)
        q75 = np.percentile(curves, 75, axis=0)

        ax.semilogy(gens, median, color=color, linewidth=1.5, label=label)
        ax.fill_between(gens, q25, q75, color=color, alpha=0.15)

    ax.set_title(fname.capitalize())
    ax.set_xlabel('Generation')
    if show_ylabel:
        ax.set_ylabel('Error (|f - f*|)')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, len(gens) - 1)


def figure1_convergence_grid():
    """Figure 1: 2×3 grid of convergence curves."""
    print("Figure 1: Convergence curves (15 seeds)...")

    functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank', 'lunacek']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, fname in enumerate(functions):
        print(f"  Running {fname}...")
        sdg_c, lsh_c = collect_convergence(fname, n_seeds=15)
        plot_convergence_single(axes[i], sdg_c, lsh_c, fname,
                                show_ylabel=(i % 3 == 0))

    fig.suptitle('SDG-MetaBBO vs L-SHADE Convergence (D=10, 5000 FEs, 15 seeds)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'convergence_grid.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {path}")


def figure2_operator_heatmap():
    """Figure 2: Operator selection frequency heatmap across functions."""
    print("Figure 2: Operator heatmap...")

    functions = ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank', 'lunacek']
    dim = 10; evals = 5000
    n_seeds = 5

    # Collect operator usage
    usage_matrix = np.zeros((len(functions), NUM_OPERATORS))

    for fi, fname in enumerate(functions):
        total_usage = np.zeros(NUM_OPERATORS)
        for s in range(n_seeds):
            func, lb, ub, opt = get_simple_problem(fname, dim)
            o = SDGOptimizer(func, dim, (lb, ub), evals, seed=s, **SK)
            r = o.run()
            u = r['operator_usage']
            for oi in range(NUM_OPERATORS):
                total_usage[oi] += u.get(OPERATOR_NAMES[oi], 0)
        # Normalize to percentages
        total = total_usage.sum()
        usage_matrix[fi] = 100.0 * total_usage / total if total > 0 else 0

    # Short operator names for display
    short_names = ['rand/1', 'best/1', 'pbest/1', 'rand/2', 'pBX', 'GD']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(usage_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=30)

    ax.set_xticks(range(NUM_OPERATORS))
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_yticks(range(len(functions)))
    ax.set_yticklabels([f.capitalize() for f in functions])
    ax.set_xlabel('Operator')
    ax.set_ylabel('Function')
    ax.set_title('Operator Selection Frequency (%, 5 seeds)')

    # Add text annotations
    for i in range(len(functions)):
        for j in range(NUM_OPERATORS):
            val = usage_matrix[i, j]
            color = 'white' if val > 20 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label='Selection %', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'operator_heatmap.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {path}")


def figure3_disagreement_profile():
    """Figure 3: Surrogate disagreement profile on Lunacek D=2."""
    print("Figure 3: Disagreement profile...")

    dim = 2
    func, lb, ub, _ = get_simple_problem('lunacek', dim)
    rng = np.random.default_rng(42)

    # Training data in two basins + sparse boundary
    X_b1 = rng.normal(2.5, 0.8, (100, dim))
    X_b2 = rng.normal(-2.5, 0.8, (100, dim))
    X_bd = rng.uniform(-1, 1, (30, dim))
    X_train = np.clip(np.vstack([X_b1, X_b2, X_bd]), lb[0], ub[0])
    y_train = np.array([func(x) for x in X_train])

    ens = SurrogateEnsemble(input_dim=dim, n_members=5, base_seed=42)
    ens.train(X_train, y_train, epochs=100, bootstrap=True)

    # Evaluate along x-axis (x2 = 0)
    x_range = np.linspace(-5, 5, 200)
    X_eval = np.column_stack([x_range, np.zeros(200)])
    mu, sigma = ens.predict_with_uncertainty(X_eval)
    f_true = np.array([func(x) for x in X_eval])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # Top: true function + surrogate mean
    ax1.plot(x_range, f_true, 'k-', linewidth=1.5, label='True $f(x)$')
    ax1.plot(x_range, mu, '--', color=COLORS['SDG'], linewidth=1.5, label='Surrogate $\\mu(x)$')
    ax1.fill_between(x_range, mu - sigma, mu + sigma, color=COLORS['SDG'], alpha=0.15,
                     label='$\\mu \\pm \\sigma$')
    ax1.set_ylabel('$f(x_1, 0)$')
    ax1.legend(loc='upper left', framealpha=0.8)
    ax1.set_title('Lunacek bi-Rastrigin: True vs Surrogate')
    ax1.grid(True, alpha=0.3)

    # Bottom: disagreement
    ax2.fill_between(x_range, 0, sigma, color='#e66101', alpha=0.4)
    ax2.plot(x_range, sigma, color='#e66101', linewidth=1.5)
    ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='Basin boundary')
    ax2.axvline(x=2.5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Global opt')
    ax2.axvline(x=-2.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Local opt')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('Disagreement $\\sigma_{ens}$')
    ax2.legend(loc='upper right', framealpha=0.8)
    ax2.set_title('Ensemble Disagreement Profile')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'disagreement_profile.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {path}")


def figure4_sensitivity_alpha():
    """Figure 4: Sensitivity of α (disagree weight)."""
    print("Figure 4: Alpha sensitivity...")

    dim = 10; evals = 5000; n_seeds = 10
    alphas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    results = {}

    for alpha in alphas:
        errors = []
        for s in range(n_seeds):
            func, lb, ub, opt = get_simple_problem('rastrigin', dim)
            o = SDGOptimizer(func, dim, (lb, ub), evals, seed=s,
                             disagree_weight=alpha, **SK)
            r = o.run()
            errors.append(abs(r['best_fitness'] - opt))
        results[alpha] = errors

    fig, ax = plt.subplots(figsize=(7, 4))

    positions = range(len(alphas))
    bp = ax.boxplot([results[a] for a in alphas], positions=positions,
                    widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))

    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['SDG'])
        patch.set_alpha(0.3)

    ax.set_yscale('log')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{a:.1f}' for a in alphas])
    ax.set_xlabel('Disagreement weight $\\alpha$')
    ax.set_ylabel('Error (Rastrigin, D=10)')
    ax.set_title('Sensitivity of $\\alpha$ (10 seeds, 5000 FEs)')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'sensitivity_alpha.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {path}")


def figure5_changepoints():
    """Figure 5: Change-point histogram."""
    print("Figure 5: Change-point distribution...")

    dim = 10; evals = 5000; n_seeds = 20
    functions = ['sphere', 'rastrigin', 'lunacek']

    fig, ax = plt.subplots(figsize=(7, 4))
    data = []
    labels = []

    for fname in functions:
        cps = []
        for s in range(n_seeds):
            func, lb, ub, opt = get_simple_problem(fname, dim)
            o = SDGOptimizer(func, dim, (lb, ub), evals, seed=s, **SK)
            r = o.run()
            cps.append(r['bandit']['n_change_points'])
        data.append(cps)
        labels.append(fname.capitalize())

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
    colors_bp = ['#4daf4a', '#e66101', '#984ea3']
    for patch, color in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax.set_ylabel('Change points per run')
    ax.set_title('Page-Hinkley Detections (20 seeds, D=10, 5000 FEs)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation
    ax.annotate('Target: 2-10 per run\n(was 80-100 before R1 fix)',
                xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'changepoints.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("="*60)
    print("SDG-MetaBBO: Generating Publication Figures")
    print("="*60)
    print()

    figure3_disagreement_profile()    # Fast (~5s)
    figure5_changepoints()            # Moderate (~3min)
    figure4_sensitivity_alpha()       # Moderate (~3min)
    figure2_operator_heatmap()        # Moderate (~3min)
    figure1_convergence_grid()        # Slow (~10min)

    print()
    print("="*60)
    print(f"All figures saved to {FIG_DIR}/")
    print()
    print("Files generated:")
    for f in sorted(os.listdir(FIG_DIR)):
        size = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {f:<35} {size/1024:.1f} KB")
    print()
    print("Add to LaTeX with:")
    print('  \\includegraphics[width=\\textwidth]{figures/convergence_grid.pdf}')
    print("="*60)


if __name__ == "__main__":
    main()
