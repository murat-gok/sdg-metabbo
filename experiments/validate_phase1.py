"""
Phase 1 Validation: Base DE Engine Test
========================================

Validates that the DE engine with current-to-pbest/1 operator
achieves L-SHADE-level performance on standard test functions.

Run: python -m experiments.validate_phase1
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.de_engine import DEEngine
from src.core.operators import OPERATOR_NAMES
from benchmarks.functions import get_simple_problem, SIMPLE_FUNCTIONS


def run_single(func_name: str, dim: int, max_evals: int, operator_idx: int,
               seed: int) -> dict:
    """Run a single optimization trial."""
    func, lb, ub, optimum = get_simple_problem(func_name, dim)

    engine = DEEngine(
        func=func,
        dim=dim,
        bounds=(lb, ub),
        max_evals=max_evals,
        seed=seed,
        fixed_operator=operator_idx,
        use_lpsr=True,
    )

    result = engine.run()
    result["error"] = abs(result["best_fitness"] - optimum)
    return result


def run_experiment(func_name: str, dim: int = 10, max_evals: int = 100_000,
                   n_runs: int = 30, operator_idx: int = 2):
    """Run n_runs trials and report statistics."""
    errors = []
    best_fits = []
    times = []

    for run_id in range(n_runs):
        t0 = time.time()
        result = run_single(func_name, dim, max_evals, operator_idx, seed=run_id * 42)
        elapsed = time.time() - t0

        errors.append(result["error"])
        best_fits.append(result["best_fitness"])
        times.append(elapsed)

    errors = np.array(errors)
    best_fits = np.array(best_fits)

    return {
        "func": func_name,
        "dim": dim,
        "operator": OPERATOR_NAMES[operator_idx],
        "n_runs": n_runs,
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "error_median": float(np.median(errors)),
        "error_min": float(np.min(errors)),
        "error_max": float(np.max(errors)),
        "best_mean": float(np.mean(best_fits)),
        "best_std": float(np.std(best_fits)),
        "time_mean": float(np.mean(times)),
    }


def main():
    print("=" * 72)
    print("ASGD-MetaBBO Phase 1 Validation")
    print("DE Engine with SHADE-style Adaptive Parameters + LPSR")
    print("=" * 72)

    dim = 10
    max_evals = 10_000 * dim  # 100,000 FEs
    n_runs = 15  # Quick validation; use 30 for final

    # Test with the main operator (current-to-pbest/1 = L-SHADE core)
    operator_idx = 2  # DE/current-to-pbest/1/bin

    print(f"\nSettings: D={dim}, MaxFEs={max_evals}, Runs={n_runs}")
    print(f"Operator: {OPERATOR_NAMES[operator_idx]}")
    print("-" * 72)

    # Header
    print(f"{'Function':<16} {'Error Mean':>14} {'Error Std':>14} "
          f"{'Error Med':>14} {'Time(s)':>8}")
    print("-" * 72)

    results = []
    for func_name in SIMPLE_FUNCTIONS:
        r = run_experiment(func_name, dim, max_evals, n_runs, operator_idx)
        results.append(r)

        print(f"{r['func']:<16} {r['error_mean']:>14.6e} {r['error_std']:>14.6e} "
              f"{r['error_median']:>14.6e} {r['time_mean']:>8.2f}")

    print("-" * 72)

    # Validation criteria
    print("\n--- Phase 1 Validation Checks ---")
    checks = []

    for r in results:
        if r["func"] == "sphere":
            passed = r["error_mean"] < 1e-20
            checks.append(("Sphere error < 1e-20", passed))
            print(f"  [{'PASS' if passed else 'FAIL'}] Sphere mean error: {r['error_mean']:.2e}")

        elif r["func"] == "rosenbrock":
            passed = r["error_mean"] < 1e-5
            checks.append(("Rosenbrock error < 1e-5", passed))
            print(f"  [{'PASS' if passed else 'FAIL'}] Rosenbrock mean error: {r['error_mean']:.2e}")

        elif r["func"] == "rastrigin":
            passed = r["error_mean"] < 10.0
            checks.append(("Rastrigin error < 10", passed))
            print(f"  [{'PASS' if passed else 'FAIL'}] Rastrigin mean error: {r['error_mean']:.2e}")

        elif r["func"] == "ackley":
            passed = r["error_mean"] < 1e-10
            checks.append(("Ackley error < 1e-10", passed))
            print(f"  [{'PASS' if passed else 'FAIL'}] Ackley mean error: {r['error_mean']:.2e}")

    all_passed = all(c[1] for c in checks)
    print(f"\n{'='*72}")
    print(f"Phase 1 Validation: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"{'='*72}")

    # Also test each operator standalone on sphere to verify they all work
    print("\n--- Operator Sanity Check (Sphere D=10, 50k FEs, 5 runs) ---")
    for op_idx in range(6):
        try:
            r = run_experiment("sphere", dim=10, max_evals=50_000,
                             n_runs=5, operator_idx=op_idx)
            print(f"  [{op_idx}] {OPERATOR_NAMES[op_idx]:<35} error={r['error_mean']:.2e}")
        except Exception as e:
            print(f"  [{op_idx}] {OPERATOR_NAMES[op_idx]:<35} ERROR: {e}")

    print("\nPhase 1 complete. Ready for Phase 2 (Surrogate Ensemble).")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
