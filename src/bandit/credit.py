"""
Rank-Based Credit Assignment for Adaptive Operator Selection.

R1 REVISION: 
  - Reward no longer requires f_opt (reviewer: "test-time uygulanabilirlik sorunu")
  - Uses pure rank-based credit (Δrank) following RLDE-AFL convention
  - α (disagree weight) exposed as tunable for sensitivity analysis
  
Reference: Li, Fialho, Kwong, Zhang (IEEE TEVC 18(1), 2014) — FRRMAB
"""

import numpy as np
from typing import List, Tuple


class CreditAssigner:
    """Rank-based credit assignment for operator selection."""

    def __init__(self, n_operators: int, window_size: int = 50,
                 decay: float = 0.5, disagree_weight: float = 0.3):
        self.n_operators = n_operators
        self.window_size = window_size
        self.decay = decay
        self.disagree_weight = disagree_weight
        self.history: List[Tuple[int, float, float]] = []

    def record(self, operator_idx: int, fitness_improvement: float, sigma: float = 0.0):
        """Record operator application. improvement = f(parent) - f(trial), positive = better."""
        self.history.append((operator_idx, fitness_improvement, sigma))
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def compute_credits(self) -> np.ndarray:
        if len(self.history) == 0:
            return np.ones(self.n_operators) / self.n_operators

        n = len(self.history)
        ops = np.array([h[0] for h in self.history])
        improvements = np.array([h[1] for h in self.history])
        sigmas = np.array([h[2] for h in self.history])

        # R1 FIX: rank-based score — no f_opt needed
        # Rank improvements within window (scale-invariant)
        imp_ranks = np.argsort(np.argsort(improvements)).astype(float) / max(n - 1, 1)
        sig_ranks = np.argsort(np.argsort(sigmas)).astype(float) / max(n - 1, 1)

        scores = (1 - self.disagree_weight) * imp_ranks + self.disagree_weight * sig_ranks

        # Time-decay
        time_weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        time_weights /= time_weights.sum() + 1e-30

        credits = np.zeros(self.n_operators)
        counts = np.zeros(self.n_operators)
        for i in range(n):
            credits[ops[i]] += scores[i] * time_weights[i]
            counts[ops[i]] += time_weights[i]

        for k in range(self.n_operators):
            if counts[k] > 0:
                credits[k] /= counts[k]
            else:
                credits[k] = 0.5

        return credits

    def get_success_rates(self) -> np.ndarray:
        if not self.history:
            return np.zeros(self.n_operators)
        successes = np.zeros(self.n_operators)
        counts = np.zeros(self.n_operators)
        for op, imp, _ in self.history:
            counts[op] += 1
            if imp > 0:
                successes[op] += 1
        rates = np.zeros(self.n_operators)
        for k in range(self.n_operators):
            if counts[k] > 0:
                rates[k] = successes[k] / counts[k]
        return rates

    def reset(self):
        self.history.clear()
