"""
Population management for ASGD-MetaBBO.
Handles initialization, bounds enforcement, fitness evaluation, and population state.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class Population:
    """Population of candidate solutions for DE-based optimization."""

    def __init__(
        self,
        np_size: int,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        seed: Optional[int] = None,
    ):
        """
        Args:
            np_size: Population size (number of individuals).
            dim: Problem dimensionality.
            bounds: Tuple of (lower_bounds, upper_bounds), each shape (dim,).
            seed: Random seed for reproducibility.
        """
        self.np_size = np_size
        self.dim = dim
        self.lb = np.asarray(bounds[0], dtype=np.float64)
        self.ub = np.asarray(bounds[1], dtype=np.float64)
        self.rng = np.random.default_rng(seed)

        # Population matrix: (np_size, dim)
        self.positions = self._initialize()
        # Fitness vector: (np_size,)  — initialized to +inf (minimization)
        self.fitness = np.full(np_size, np.inf)
        # Generation counter
        self.generation = 0
        # Function evaluation counter
        self.n_evals = 0
        # Best-so-far tracking
        self.best_fitness = np.inf
        self.best_position = None
        # History for surrogate training
        self.eval_history_x = []
        self.eval_history_f = []

    def _initialize(self) -> np.ndarray:
        """Latin Hypercube Sampling initialization for better coverage."""
        positions = np.zeros((self.np_size, self.dim))
        for j in range(self.dim):
            # Create evenly spaced intervals, sample one point per interval
            intervals = np.linspace(0, 1, self.np_size + 1)
            points = self.rng.uniform(intervals[:-1], intervals[1:])
            self.rng.shuffle(points)
            positions[:, j] = self.lb[j] + points * (self.ub[j] - self.lb[j])
        return positions

    def evaluate(self, func: Callable, indices: Optional[np.ndarray] = None):
        """
        Evaluate fitness for specified individuals (or all if indices=None).

        Args:
            func: Objective function f(x) → scalar.
            indices: Which individuals to evaluate. None = all.
        """
        if indices is None:
            indices = np.arange(self.np_size)

        for i in indices:
            f_val = func(self.positions[i])
            self.fitness[i] = f_val
            self.n_evals += 1

            # Record for surrogate training
            self.eval_history_x.append(self.positions[i].copy())
            self.eval_history_f.append(f_val)

            # Update best-so-far
            if f_val < self.best_fitness:
                self.best_fitness = f_val
                self.best_position = self.positions[i].copy()

    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Clip solution vector to feasible bounds."""
        return np.clip(x, self.lb, self.ub)

    def bounce_back(self, x: np.ndarray, base: np.ndarray) -> np.ndarray:
        """
        Bounce-back boundary handling: if x_j is out of bounds,
        set x_j = (base_j + bound_j) / 2.
        More principled than simple clipping for DE.
        """
        result = x.copy()
        for j in range(self.dim):
            if result[j] < self.lb[j]:
                result[j] = (base[j] + self.lb[j]) / 2.0
            elif result[j] > self.ub[j]:
                result[j] = (base[j] + self.ub[j]) / 2.0
        return result

    def get_pbest_indices(self, p: float) -> np.ndarray:
        """
        Get indices of top-p fraction individuals (for current-to-pbest).

        Args:
            p: Fraction in (0, 1]. E.g., p=0.1 → top 10%.

        Returns:
            Array of indices of the best ceil(p * NP) individuals.
        """
        k = max(1, int(np.ceil(p * self.np_size)))
        return np.argsort(self.fitness)[:k]

    def get_sorted_indices(self) -> np.ndarray:
        """Return indices sorted by fitness (best first)."""
        return np.argsort(self.fitness)

    def get_fitness_stats(self) -> dict:
        """Return population fitness statistics for state encoding."""
        valid = self.fitness[self.fitness < np.inf]
        if len(valid) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        return {
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median": float(np.median(valid)),
        }

    def get_diversity(self) -> float:
        """Population diversity: mean pairwise Euclidean distance (normalized)."""
        if self.np_size < 2:
            return 0.0
        # Efficient: use variance across dimensions
        var_per_dim = np.var(self.positions, axis=0)
        range_per_dim = self.ub - self.lb
        # Normalized diversity: mean of (std / range) across dimensions
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.sqrt(var_per_dim) / range_per_dim
        return float(np.mean(np.nan_to_num(normalized)))

    def get_stagnation_counter(self, window: int = 10) -> int:
        """
        Count how many of the last `window` generations had no improvement.
        (Requires external tracking — simplified version based on eval history.)
        """
        if len(self.eval_history_f) < 2:
            return 0
        recent = self.eval_history_f[-min(window * self.np_size, len(self.eval_history_f)):]
        if min(recent) < self.best_fitness:
            return 0
        return window  # Simplified: stagnated for full window

    def get_eval_data(self, max_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get recent evaluation data for surrogate training.

        Args:
            max_points: Maximum number of recent points to return.

        Returns:
            (X, y) where X is (n, dim) and y is (n,).
        """
        n = min(max_points, len(self.eval_history_x))
        if n == 0:
            return np.empty((0, self.dim)), np.empty(0)
        X = np.array(self.eval_history_x[-n:])
        y = np.array(self.eval_history_f[-n:])
        return X, y

    def copy(self) -> "Population":
        """Create a deep copy of the population."""
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"Population(NP={self.np_size}, D={self.dim}, "
            f"gen={self.generation}, evals={self.n_evals}, "
            f"best={self.best_fitness:.6e})"
        )
