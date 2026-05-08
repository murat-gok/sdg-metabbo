"""
MetaBBO Environment for SDG-MetaBBO (R1 Revised).

R1 FIX: Reward uses rank-based normalization, not f_opt.
  - Training time: f_opt available → use for curriculum shaping
  - Test time: f_opt unknown → use Δrank or relative improvement
  - Both modes supported via use_known_opt flag
"""

import numpy as np
from typing import Callable, Optional, Dict, Tuple, List

from ..core.population import Population
from ..core.archive import EliteArchive, DisplacedArchive
from ..core.operators import OPERATORS, NUM_OPERATORS
from ..surrogate.training import SurrogateManager
from ..bandit.sw_ucb import SlidingWindowUCB
from .encoder import get_encoder


class MetaBBOEnv:

    def __init__(self, func, dim, bounds, max_evals, known_optimum=None,
                 seed=None, d_model=64, np_size=None, memory_size=5,
                 p_best_rate=0.1, surr_hidden=(64, 32), surr_epochs=15,
                 surr_data_window=200, surr_min_data=30):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.known_optimum = known_optimum  # None at test time
        self.seed = seed
        self.np_init = np_size or max(5 * dim, 18)
        self.np_min = 4
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.surr_hidden = surr_hidden
        self.surr_epochs = surr_epochs
        self.surr_data_window = surr_data_window
        self.surr_min_data = surr_min_data
        self.encoder = get_encoder(d_model=d_model, use_torch=False)
        self.state_dim = d_model
        self.action_dim = 3
        self.pop = None

    def reset(self, seed=None):
        s = seed if seed is not None else self.seed
        self.rng = np.random.default_rng(s)
        self.pop = Population(self.np_init, self.dim, self.bounds, seed=s)
        self.pop.evaluate(self.func)
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.displaced_archive = DisplacedArchive(max_size=self.np_init)
        self.elite_archive = EliteArchive(capacity=2 * self.dim, dim=self.dim)
        self._update_elite()
        self.surr_mgr = SurrogateManager(
            input_dim=self.dim, n_members=5, hidden_dims=self.surr_hidden,
            retrain_interval=10, data_window=self.surr_data_window,
            min_data_for_training=self.surr_min_data,
            epochs_per_retrain=self.surr_epochs, seed=s or 0)
        self.bandit = SlidingWindowUCB(n_operators=NUM_OPERATORS,
                                       ph_threshold=50.0, ph_cooldown=50)
        self.generation = 0
        self.prev_best = self.pop.best_fitness
        self.initial_best = self.pop.best_fitness
        # R1 FIX: track recent improvements for rank-based reward
        self.recent_improvements: List[float] = []
        self.done = False
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {}

        c = float(np.clip(action[0], 0.1, 2.0))
        gd_beta = float(np.clip(action[1], 0.1, 3.0))
        retrain_interval = int(np.clip(round(action[2]), 1, 20))

        self.bandit.set_exploration_coeff(c)
        self.surr_mgr.retrain_interval = retrain_interval

        X_data, y_data = self.pop.get_eval_data(self.surr_data_window)
        self.surr_mgr.maybe_retrain(X_data, y_data, self.generation)

        NP = self.pop.np_size
        sF, sCR = [], []
        gen_improvement = 0.0

        for i in range(NP):
            if self.pop.n_evals >= self.max_evals:
                break
            F_i, CR_i = self._sample_params()
            op_idx = self.bandit.select_operator()
            sigma = self.surr_mgr.get_sigma_at(self.pop.positions[i]) if self.surr_mgr.is_trained else None
            trial = OPERATORS[op_idx](self.pop, i, F_i, CR_i, self.rng,
                                      p=self.p_best_rate,
                                      archive=self.displaced_archive.get_solutions(),
                                      beta=gd_beta, sigma_at_x=sigma)
            f_trial = self.func(trial)
            self.pop.n_evals += 1
            self.pop.eval_history_x.append(trial.copy())
            self.pop.eval_history_f.append(f_trial)
            if f_trial < self.pop.best_fitness:
                self.pop.best_fitness = f_trial
                self.pop.best_position = trial.copy()
            imp = self.pop.fitness[i] - f_trial
            gen_improvement += max(0, imp)
            st = self.surr_mgr.get_sigma_at(trial) if self.surr_mgr.is_trained else 0.0
            self.bandit.update(op_idx, imp, st)
            if f_trial <= self.pop.fitness[i]:
                sF.append(F_i); sCR.append(CR_i)
                self.displaced_archive.add(self.pop.positions[i])
                self.pop.positions[i] = trial
                self.pop.fitness[i] = f_trial

        if sF:
            self._update_memory(sF, sCR)
        self._update_elite()
        self._apply_lpsr()
        self.bandit.update_window_size(self.pop.np_size)
        self.generation += 1

        # R1 FIX: Rank-based reward (no f_opt needed at test time)
        self.recent_improvements.append(gen_improvement)
        if len(self.recent_improvements) >= 2:
            # Reward = relative rank of this gen's improvement among recent
            recent = np.array(self.recent_improvements[-20:])
            rank = np.sum(recent <= gen_improvement) / len(recent)
            reward = rank
        elif self.known_optimum is not None:
            # Training time with known opt: normalized improvement
            denom = abs(self.initial_best - self.known_optimum) + 1e-30
            reward = (self.prev_best - self.pop.best_fitness) / denom
        else:
            reward = float(gen_improvement > 0)

        self.prev_best = self.pop.best_fitness
        self.done = self.pop.n_evals >= self.max_evals
        info = {"best_fitness": self.pop.best_fitness, "n_evals": self.pop.n_evals,
                "generation": self.generation,
                "error": abs(self.pop.best_fitness - (self.known_optimum or 0))}
        return self._get_state(), float(reward), self.done, info

    def _get_state(self):
        extra = {"progress": self.pop.n_evals / self.max_evals,
                 "sigma_mean": 0.0, "sigma_max": 0.0,
                 "bandit_entropy": self.bandit.get_diagnostics().get("entropy", 0.0) if hasattr(self, 'bandit') else 0.0,
                 "diversity": self.pop.get_diversity()}
        if hasattr(self, 'surr_mgr') and self.surr_mgr.is_trained:
            sf = self.surr_mgr.get_state_features()
            extra["sigma_mean"] = sf.get("surrogate_mean_sigma", 0.0)
            extra["sigma_max"] = sf.get("surrogate_max_sigma", 0.0)
        return self.encoder.encode_population(self.pop.positions, self.pop.fitness, self.bounds, extra)

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

    def _update_elite(self):
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
