"""
Sliding-Window UCB Bandit for Adaptive Operator Selection.

R1 REVISIONS:
  1. Adaptive window: W = max(20, 2*NP) scales with LPSR population reduction
  2. Single global Page-Hinkley detector (not per-operator) with cooldown
  3. Soft reset: halves window counts instead of zeroing
  4. Expected ~5-10 change points per run (was 80-100, fatal flaw)

Theoretical anchor: Garivier-Moulines SW-UCB regret O(√(Υ_T·T·log T))
Now valid because Υ_T is controlled by cooldown.
"""

import numpy as np
from typing import Dict, List
from .page_hinkley import PageHinkley
from .credit import CreditAssigner


class SlidingWindowUCB:

    def __init__(self, n_operators: int = 6, window_size: int = 50,
                 exploration_coeff: float = 0.5, credit_decay: float = 0.5,
                 disagree_weight: float = 0.3, ph_delta: float = 0.01,
                 ph_threshold: float = 50.0, ph_cooldown: int = 50,
                 min_applications: int = 2):
        self.n_operators = n_operators
        self.base_window = window_size
        self.c = exploration_coeff
        self.min_applications = min_applications

        self.credit = CreditAssigner(n_operators, window_size, credit_decay, disagree_weight)

        # R1 FIX: single global detector with cooldown (was per-operator, no cooldown)
        self.detector = PageHinkley(delta=ph_delta, threshold=ph_threshold, cooldown=ph_cooldown)

        self.op_counts = np.zeros(n_operators, dtype=int)
        self.op_window_counts = np.zeros(n_operators, dtype=int)
        self.total_steps = 0
        self.selection_history: List[int] = []
        self.change_points: List[int] = []

    def select_operator(self) -> int:
        self.total_steps += 1

        for k in range(self.n_operators):
            if self.op_counts[k] < self.min_applications:
                self._record_selection(k)
                return k

        credits = self.credit.compute_credits()
        total_window = max(1, int(self.op_window_counts.sum()))
        ucb_scores = np.zeros(self.n_operators)
        for k in range(self.n_operators):
            n_k = max(1, self.op_window_counts[k])
            ucb_scores[k] = credits[k] + self.c * np.sqrt(np.log(total_window) / n_k)

        max_score = np.max(ucb_scores)
        best_ops = np.where(np.abs(ucb_scores - max_score) < 1e-10)[0]
        selected = np.random.choice(best_ops)
        self._record_selection(selected)
        return int(selected)

    def update(self, operator_idx: int, fitness_improvement: float, sigma: float = 0.0):
        self.credit.record(operator_idx, fitness_improvement, sigma)
        # R1 FIX: normalize reward for PH to prevent scale-dependent triggering
        # Use sign(improvement) instead of raw value — PH detects trend changes
        normalized_reward = np.sign(fitness_improvement) * min(1.0, abs(fitness_improvement) / (abs(fitness_improvement) + 1.0))
        if self.detector.update(normalized_reward):
            self._handle_change()

    def update_window_size(self, current_np: int):
        """R1 FIX: Adaptive window scales with LPSR. W = max(20, 2*NP)."""
        new_w = max(20, 2 * current_np)
        self.credit.window_size = new_w

    def _record_selection(self, k: int):
        self.op_counts[k] += 1
        self.op_window_counts[k] += 1
        self.selection_history.append(k)
        w = self.credit.window_size
        if len(self.selection_history) > w:
            oldest = self.selection_history[-w - 1]
            self.op_window_counts[oldest] = max(0, self.op_window_counts[oldest] - 1)

    def _handle_change(self):
        self.change_points.append(self.total_steps)
        # R1 FIX: soft reset — halve counts instead of zeroing
        self.op_window_counts = np.maximum(self.op_window_counts // 2, 1)
        self.credit.reset()

    def set_exploration_coeff(self, c: float):
        self.c = max(0.01, c)

    def get_operator_probabilities(self) -> np.ndarray:
        if self.total_steps < self.n_operators * self.min_applications:
            return np.ones(self.n_operators) / self.n_operators
        credits = self.credit.compute_credits()
        total_window = max(1, int(self.op_window_counts.sum()))
        ucb_scores = np.zeros(self.n_operators)
        for k in range(self.n_operators):
            n_k = max(1, self.op_window_counts[k])
            ucb_scores[k] = credits[k] + self.c * np.sqrt(np.log(total_window) / n_k)
        ucb_scores -= np.max(ucb_scores)
        exp_scores = np.exp(ucb_scores)
        return exp_scores / (exp_scores.sum() + 1e-30)

    def get_diagnostics(self) -> Dict:
        credits = self.credit.compute_credits()
        probs = self.get_operator_probabilities()
        return {
            "credits": credits.tolist(),
            "success_rates": self.credit.get_success_rates().tolist(),
            "probabilities": probs.tolist(),
            "window_counts": self.op_window_counts.tolist(),
            "total_counts": self.op_counts.tolist(),
            "n_change_points": len(self.change_points),
            "entropy": float(-np.sum(probs * np.log(probs + 1e-30))),
        }
