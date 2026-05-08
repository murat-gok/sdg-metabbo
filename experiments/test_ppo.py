"""Test trained PPO vs fixed defaults."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.core.asgd_full import SDGOptimizer
from benchmarks.functions import get_simple_problem

dim = 10; evals = 5000; seeds = 5
SK = dict(surr_hidden=(32,16), surr_epochs=8, surr_retrain_interval=15,
          surr_data_window=150, surr_min_data=50)

print("Trained PPO vs Fixed Policy (5 seeds, D=10, 5k FEs)")
print("-"*60)
for fname in ['sphere', 'rastrigin', 'ackley', 'lunacek']:
    fixed_err, ppo_err = [], []
    for s in range(seeds):
        func, lb, ub, opt = get_simple_problem(fname, dim)
        # Fixed policy
        o1 = SDGOptimizer(func, dim, (lb,ub), evals, seed=s, **SK)
        r1 = o1.run()
        fixed_err.append(abs(r1['best_fitness'] - opt))
        # Trained PPO
        o2 = SDGOptimizer(func, dim, (lb,ub), evals, seed=s,
                          checkpoint_path="checkpoints/meta_rl.pt", **SK)
        r2 = o2.run()
        ppo_err.append(abs(r2['best_fitness'] - opt))
    fm, pm = np.median(fixed_err), np.median(ppo_err)
    w = "PPO" if pm < fm else "Fixed" if fm < pm else "Tie"
    print(f"  {fname:<14} Fixed={fm:.2e}  PPO={pm:.2e}  -> {w}")
print("-"*60)
print("Done.")
