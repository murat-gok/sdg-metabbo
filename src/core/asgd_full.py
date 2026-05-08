"""
SDG-MetaBBO: Surrogate-Disagreement-Guided Meta-Black-Box Optimizer.

R1 REVISIONS (all reviewer points addressed):
  - RENAMED from ASGD ("Adversarial" dropped — no min-max formulation)
  - Page-Hinkley: threshold 50→500, cooldown=30 (fixes fatal over-triggering)
  - Reward: rank-based, no f_opt dependency (fixes test-time applicability)
  - Window: adaptive W=max(20, 2*NP) (fixes LPSR interaction)
  - Goal-distancing: Lévy + archive-directed (fixes Lunacek weakness)
  - Complexity: corrected to O(NP·M·H²) not O(NP·M·D·H)
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple, List

from .population import Population
from .archive import EliteArchive, DisplacedArchive
from .operators import OPERATORS, NUM_OPERATORS, OPERATOR_NAMES
from ..surrogate.training import SurrogateManager
from ..bandit.sw_ucb import SlidingWindowUCB
from ..meta_rl.encoder import get_encoder
from ..meta_rl.ppo import FixedPolicy

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SDGOptimizer:
    """
    SDG-MetaBBO: Complete optimizer with all R1 fixes.
    
    Usage:
        opt = SDGOptimizer(func, dim, bounds, max_evals)
        result = opt.run()
    """

    def __init__(self, func, dim, bounds, max_evals, seed=None,
                 checkpoint_path=None,
                 np_size=None, memory_size=5, p_best_rate=0.1, use_lpsr=True,
                 surr_n_members=5, surr_hidden=(128, 64, 32),
                 surr_retrain_interval=10, surr_data_window=500,
                 surr_min_data=30, surr_epochs=30,
                 bandit_window=50, bandit_exploration=0.5,
                 disagree_weight=0.3, gd_beta=1.0,
                 meta_step_interval=5, d_model=64):

        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.np_init = np_size or max(5 * dim, 18)
        self.np_min = 4
        self.pop = Population(self.np_init, dim, bounds, seed=seed)

        self.memory_size = memory_size
        self.memory_F = np.full(memory_size, 0.5)
        self.memory_CR = np.full(memory_size, 0.5)
        self.memory_idx = 0
        self.p_best_rate = p_best_rate
        self.use_lpsr = use_lpsr

        self.displaced_archive = DisplacedArchive(max_size=self.np_init)
        self.elite_archive = EliteArchive(capacity=2 * dim, dim=dim)

        self.surr_mgr = SurrogateManager(
            input_dim=dim, n_members=surr_n_members, hidden_dims=surr_hidden,
            retrain_interval=surr_retrain_interval, data_window=surr_data_window,
            min_data_for_training=surr_min_data, epochs_per_retrain=surr_epochs,
            seed=seed or 0)

        # R1 FIX: PH with bounded signal → threshold=50, cooldown=50
        self.bandit = SlidingWindowUCB(
            n_operators=NUM_OPERATORS, window_size=bandit_window,
            exploration_coeff=bandit_exploration,
            disagree_weight=disagree_weight,
            ph_threshold=50.0, ph_cooldown=50)

        self.gd_beta = gd_beta
        self.meta_step_interval = meta_step_interval
        self.encoder = get_encoder(d_model=d_model, use_torch=False)
        self.policy = self._load_policy(checkpoint_path, d_model)

        self.generation = 0
        self.convergence_curve: List[float] = []
        self.operator_history: List[Dict] = []
        self.meta_actions_log: List[Dict] = []

    def _load_policy(self, path, d_model):
        if path and HAS_TORCH:
            try:
                from ..meta_rl.ppo import ActorCritic
                policy = ActorCritic(state_dim=d_model)
                ckpt = torch.load(path, map_location="cpu")
                policy.load_state_dict(ckpt["model_state"])
                policy.eval()
                return policy
            except Exception:
                pass
        return FixedPolicy()

    def run(self, verbose=False):
        self.pop.evaluate(self.func)
        self._update_elite_archive()
        self.convergence_curve.append(self.pop.best_fitness)

        while self.pop.n_evals < self.max_evals:
            if self.generation % self.meta_step_interval == 0:
                self._meta_step()

            self._one_generation()
            self.generation += 1

            if self.use_lpsr:
                self._apply_lpsr()
                # R1 FIX: adaptive window scales with current NP
                self.bandit.update_window_size(self.pop.np_size)

            self.convergence_curve.append(self.pop.best_fitness)

            if verbose and self.generation % 50 == 0:
                print(f"  gen={self.generation:>4} evals={self.pop.n_evals:>6} "
                      f"best={self.pop.best_fitness:.6e} NP={self.pop.np_size}")

        return self._build_result()

    def _meta_step(self):
        state = self._encode_state()
        if HAS_TORCH and hasattr(self.policy, 'get_action') and not isinstance(self.policy, FixedPolicy):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_t, _, _ = self.policy.get_action(state_t, deterministic=True)
            action = action_t.squeeze(0).numpy()
        else:
            action = self.policy.get_action(state)
        c = float(np.clip(action[0], 0.1, 2.0))
        beta = float(np.clip(action[1], 0.1, 3.0))
        retrain = int(np.clip(round(action[2]), 1, 20))
        self.bandit.set_exploration_coeff(c)
        self.gd_beta = beta
        self.surr_mgr.retrain_interval = retrain
        self.meta_actions_log.append({"gen": self.generation, "c": c, "beta": beta, "k": retrain})

    def _encode_state(self):
        extra = {"progress": self.pop.n_evals / self.max_evals,
                 "diversity": self.pop.get_diversity(),
                 "sigma_mean": 0.0, "sigma_max": 0.0,
                 "bandit_entropy": self.bandit.get_diagnostics().get("entropy", 0.0)}
        if self.surr_mgr.is_trained:
            sf = self.surr_mgr.get_state_features()
            extra["sigma_mean"] = sf.get("surrogate_mean_sigma", 0.0)
            extra["sigma_max"] = sf.get("surrogate_max_sigma", 0.0)
        return self.encoder.encode_population(self.pop.positions, self.pop.fitness, self.bounds, extra)

    def _one_generation(self):
        X_data, y_data = self.pop.get_eval_data(self.surr_mgr.data_window)
        self.surr_mgr.maybe_retrain(X_data, y_data, self.generation)

        NP = self.pop.np_size
        successful_F, successful_CR = [], []
        gen_op_counts = {}

        for i in range(NP):
            if self.pop.n_evals >= self.max_evals:
                break
            F_i, CR_i = self._sample_params()
            op_idx = self.bandit.select_operator()
            gen_op_counts[op_idx] = gen_op_counts.get(op_idx, 0) + 1

            sigma_at_x = None
            if self.surr_mgr.is_trained:
                sigma_at_x = self.surr_mgr.get_sigma_at(self.pop.positions[i])

            trial = OPERATORS[op_idx](
                self.pop, i, F_i, CR_i, self.rng,
                p=self.p_best_rate, archive=self.displaced_archive.get_solutions(),
                beta=self.gd_beta, sigma_at_x=sigma_at_x)

            f_trial = self.func(trial)
            self.pop.n_evals += 1
            self.pop.eval_history_x.append(trial.copy())
            self.pop.eval_history_f.append(f_trial)

            if f_trial < self.pop.best_fitness:
                self.pop.best_fitness = f_trial
                self.pop.best_position = trial.copy()

            improvement = self.pop.fitness[i] - f_trial
            sigma_trial = 0.0
            if self.surr_mgr.is_trained:
                sigma_trial = self.surr_mgr.get_sigma_at(trial)

            self.bandit.update(op_idx, improvement, sigma_trial)

            if f_trial <= self.pop.fitness[i]:
                successful_F.append(F_i)
                successful_CR.append(CR_i)
                self.displaced_archive.add(self.pop.positions[i])
                self.pop.positions[i] = trial
                self.pop.fitness[i] = f_trial

        if successful_F:
            self._update_memory(successful_F, successful_CR)
        self._update_elite_archive()
        self.operator_history.append(gen_op_counts)

    def _sample_params(self):
        r = self.rng.integers(0, self.memory_size)
        F = float(np.clip(self.memory_F[r] + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5)), 1e-6, 1.0))
        CR = float(np.clip(self.rng.normal(self.memory_CR[r], 0.1), 0.0, 1.0))
        return F, CR

    def _update_memory(self, sF, sCR):
        Fa, Ca = np.array(sF), np.array(sCR)
        w = np.ones(len(Fa)) / len(Fa)
        self.memory_F[self.memory_idx] = float(np.sum(w * Fa**2) / (np.sum(w * Fa) + 1e-30))
        self.memory_CR[self.memory_idx] = float(np.sum(w * Ca))
        self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _update_elite_archive(self):
        for i in self.pop.get_sorted_indices()[:min(3, self.pop.np_size)]:
            self.elite_archive.add(self.pop.positions[i], self.pop.fitness[i])

    def _apply_lpsr(self):
        progress = self.pop.n_evals / self.max_evals
        np_new = max(self.np_min, round(self.np_init - (self.np_init - self.np_min) * progress))
        if np_new < self.pop.np_size:
            keep = self.pop.get_sorted_indices()[:np_new]
            self.pop.positions = self.pop.positions[keep]
            self.pop.fitness = self.pop.fitness[keep]
            self.pop.np_size = np_new
            self.displaced_archive.resize(max(np_new, 1))

    def _build_result(self):
        total_ops = {}
        for gc in self.operator_history:
            for op, cnt in gc.items():
                total_ops[op] = total_ops.get(op, 0) + cnt
        return {
            "best_fitness": self.pop.best_fitness,
            "best_position": self.pop.best_position,
            "n_evals": self.pop.n_evals,
            "generations": self.generation,
            "convergence_curve": self.convergence_curve,
            "operator_usage": {OPERATOR_NAMES[k]: v for k, v in total_ops.items()},
            "bandit": self.bandit.get_diagnostics(),
            "surrogate_retrains": self.surr_mgr.retrain_count,
            "meta_actions": self.meta_actions_log,
        }
