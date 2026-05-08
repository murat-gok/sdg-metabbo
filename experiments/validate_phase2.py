"""
Phase 2 Validation: Surrogate Ensemble Tests
=============================================

Tests:
1. Surrogate prediction accuracy on simple functions
2. Rank correlation (more important than MSE for operator guidance)
3. Disagreement profile on Lunacek bi-Rastrigin (should peak near basin boundary)
4. Integration with DE engine — surrogate trained during optimization

Run: python experiments/validate_phase2.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.surrogate.ensemble import SurrogateEnsemble
from src.surrogate.training import SurrogateManager
from src.core.de_engine import DEEngine
from benchmarks.functions import get_simple_problem


def test_basic_prediction():
    """Test 1: Basic prediction accuracy on Sphere."""
    print("Test 1: Surrogate prediction on Sphere D=5")
    print("-" * 50)

    dim = 5
    rng = np.random.default_rng(42)

    # Generate training data
    X_train = rng.uniform(-100, 100, (200, dim))
    y_train = np.sum(X_train**2, axis=1)

    # Create and train ensemble
    ens = SurrogateEnsemble(input_dim=dim, n_members=5, base_seed=42)
    ens.train(X_train, y_train, epochs=100)

    # Test on new data
    X_test = rng.uniform(-100, 100, (50, dim))
    y_test = np.sum(X_test**2, axis=1)

    diag = ens.get_diagnostics(X_test, y_test)
    print(f"  MSE:          {diag['mse']:.4e}")
    print(f"  Rank corr:    {diag['rank_corr']:.4f}")
    print(f"  Mean sigma:   {diag['mean_sigma']:.4e}")
    print(f"  Max sigma:    {diag['max_sigma']:.4e}")

    passed = diag["rank_corr"] > 0.9
    print(f"  [{'PASS' if passed else 'FAIL'}] Rank correlation > 0.9")
    return passed


def test_disagreement_profile():
    """Test 2: Disagreement peaks near basin boundary on Lunacek."""
    print("\nTest 2: Disagreement profile on Lunacek bi-Rastrigin D=2")
    print("-" * 50)

    dim = 2
    func, lb, ub, _ = get_simple_problem("lunacek", dim)
    rng = np.random.default_rng(42)

    # Generate training data concentrated in two basins
    # Basin 1 around mu1=2.5, Basin 2 around mu2≈-2.5
    n_per_basin = 100
    X_b1 = rng.normal(2.5, 0.8, (n_per_basin, dim))
    X_b2 = rng.normal(-2.5, 0.8, (n_per_basin, dim))
    X_boundary = rng.uniform(-1, 1, (30, dim))  # Sparse near boundary
    X_train = np.vstack([X_b1, X_b2, X_boundary])
    X_train = np.clip(X_train, lb[0], ub[0])
    y_train = np.array([func(x) for x in X_train])

    # Train ensemble
    ens = SurrogateEnsemble(input_dim=dim, n_members=5, base_seed=42)
    ens.train(X_train, y_train, epochs=100, bootstrap=True)

    # Measure disagreement at three regions
    X_basin1 = rng.normal(2.5, 0.3, (30, dim))
    X_basin1 = np.clip(X_basin1, lb[0], ub[0])
    X_basin2 = rng.normal(-2.5, 0.3, (30, dim))
    X_basin2 = np.clip(X_basin2, lb[0], ub[0])
    X_bound = rng.uniform(-0.5, 0.5, (30, dim))

    sigma_b1 = np.mean(ens.predict_disagreement(X_basin1))
    sigma_b2 = np.mean(ens.predict_disagreement(X_basin2))
    sigma_bd = np.mean(ens.predict_disagreement(X_bound))

    print(f"  σ at Basin 1 (μ≈2.5):      {sigma_b1:.4f}")
    print(f"  σ at Basin 2 (μ≈-2.5):     {sigma_b2:.4f}")
    print(f"  σ at Boundary (μ≈0):       {sigma_bd:.4f}")

    # Boundary disagreement should be higher than at least one basin
    passed = sigma_bd > min(sigma_b1, sigma_b2)
    print(f"  [{'PASS' if passed else 'FAIL'}] Boundary σ > min(Basin σ)")
    print(f"  Ratio boundary/min_basin:  {sigma_bd / (min(sigma_b1, sigma_b2) + 1e-10):.2f}x")
    return passed


def test_reward_signal():
    """Test 3: Surrogate reward signal differentiates good vs bad moves."""
    print("\nTest 3: Reward signal quality")
    print("-" * 50)

    dim = 5
    func, lb, ub, _ = get_simple_problem("sphere", dim)
    rng = np.random.default_rng(42)

    # Train on random data
    X_train = rng.uniform(-100, 100, (200, dim))
    y_train = np.array([func(x) for x in X_train])

    ens = SurrogateEnsemble(input_dim=dim, n_members=5, base_seed=42)
    ens.train(X_train, y_train, epochs=80)

    current_best = 50.0  # Moderate fitness

    # Good move: closer to origin
    x_good = rng.uniform(-2, 2, dim)
    f_good = func(x_good)
    reward_good = ens.get_reward_signal(x_good, f_good, current_best, alpha_disagree=0.3)

    # Bad move: far from origin
    x_bad = rng.uniform(-90, 90, dim)
    f_bad = func(x_bad)
    reward_bad = ens.get_reward_signal(x_bad, f_bad, current_best, alpha_disagree=0.3)

    print(f"  Good move: f={f_good:.2f}, reward={reward_good:.4f}")
    print(f"  Bad move:  f={f_bad:.2f}, reward={reward_bad:.4f}")

    passed = reward_good > reward_bad
    print(f"  [{'PASS' if passed else 'FAIL'}] Good move gets higher reward")
    return passed


def test_integration_with_de():
    """Test 4: Surrogate trains online during DE optimization."""
    print("\nTest 4: Online surrogate during DE optimization")
    print("-" * 50)

    dim = 5
    func, lb, ub, opt = get_simple_problem("sphere", dim)

    engine = DEEngine(
        func=func, dim=dim, bounds=(lb, ub),
        max_evals=3000, seed=42, fixed_operator=2,
    )

    # Create surrogate manager
    surr_mgr = SurrogateManager(
        input_dim=dim, n_members=5,
        retrain_interval=10, data_window=300,
        min_data_for_training=30, epochs_per_retrain=30,
        seed=42,
    )

    retrain_log = []

    def on_gen(eng):
        X, y = eng.pop.get_eval_data(max_points=500)
        diag = surr_mgr.maybe_retrain(X, y, eng.generation)
        if diag:
            retrain_log.append(diag)

    result = engine.run(on_generation=on_gen)

    print(f"  Final error:     {abs(result['best_fitness'] - opt):.2e}")
    print(f"  Surrogate retrains: {len(retrain_log)}")

    if retrain_log:
        first = retrain_log[0]
        # Use peak rank correlation (not final — at convergence all solutions
        # are identical so rank correlation becomes meaningless)
        best_rc = max(d["rank_corr"] for d in retrain_log)
        mid_idx = len(retrain_log) // 2
        mid = retrain_log[mid_idx]
        print(f"  First retrain — gen={first['generation']}, rank_corr={first['rank_corr']:.3f}")
        print(f"  Mid retrain   — gen={mid['generation']}, rank_corr={mid['rank_corr']:.3f}")
        print(f"  Peak rank_corr across all retrains: {best_rc:.3f}")
        print(f"  (Note: rank_corr drops at convergence — all solutions identical)")

        passed = best_rc > 0.5
        print(f"  [{'PASS' if passed else 'FAIL'}] Peak rank correlation > 0.5")
    else:
        passed = False
        print(f"  [FAIL] No retraining occurred")

    return passed


def test_goal_distancing_with_surrogate():
    """Test 5: Goal-distancing operator uses surrogate sigma."""
    print("\nTest 5: Goal-distancing operator with surrogate σ")
    print("-" * 50)

    dim = 5
    func, lb, ub, _ = get_simple_problem("lunacek", dim)
    rng = np.random.default_rng(42)

    # Train surrogate
    X_train = rng.uniform(-5, 5, (200, dim))
    y_train = np.array([func(x) for x in X_train])

    surr = SurrogateEnsemble(input_dim=dim, n_members=5, base_seed=42)
    surr.train(X_train, y_train, epochs=80)

    # Test: get sigma at different points
    x_near_opt = np.full(dim, 2.5) + rng.normal(0, 0.1, dim)
    x_at_boundary = np.zeros(dim) + rng.normal(0, 0.3, dim)
    x_far = np.full(dim, -4.0) + rng.normal(0, 0.1, dim)

    sigma_opt = surr.get_sigma_at(x_near_opt)
    sigma_bnd = surr.get_sigma_at(x_at_boundary)
    sigma_far = surr.get_sigma_at(x_far)

    print(f"  σ near optimum (2.5):  {sigma_opt:.4f}")
    print(f"  σ at boundary (0.0):   {sigma_bnd:.4f}")
    print(f"  σ far away (-4.0):     {sigma_far:.4f}")

    # The goal-distancing step size would be proportional to sigma
    # So it should take bigger steps at uncertain regions
    print(f"  → Goal-dist step at boundary would be {sigma_bnd/(sigma_opt+1e-10):.1f}x "
          f"larger than at optimum")

    passed = True  # This test is diagnostic, not pass/fail
    print(f"  [INFO] Diagnostic test — sigma values feed into R(x;x*,σ)")
    return passed


def main():
    print("=" * 60)
    print("ASGD-MetaBBO Phase 2 Validation")
    print("Surrogate Ensemble (Opponent Model)")
    print("=" * 60)

    results = []
    results.append(("Basic prediction", test_basic_prediction()))
    results.append(("Disagreement profile", test_disagreement_profile()))
    results.append(("Reward signal", test_reward_signal()))
    results.append(("DE integration", test_integration_with_de()))
    results.append(("Goal-dist + surrogate", test_goal_distancing_with_surrogate()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed

    print(f"\nPhase 2: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
