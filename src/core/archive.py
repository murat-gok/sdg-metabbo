"""
Elite Archive for ASGD-MetaBBO.

Capacity-limited archive of best-ever solutions.
Biological motivation: Portia's ~2-item parallel-individuation working memory.
Implementation: fixed-size buffer with worst-fitness eviction.

Also maintains a displaced-parent archive for DE/current-to-pbest/1 (L-SHADE style).
"""

import numpy as np
from typing import Optional, Tuple, List


class EliteArchive:
    """Capacity-limited archive of elite solutions."""

    def __init__(self, capacity: int, dim: int):
        """
        Args:
            capacity: Maximum number of solutions to store.
            dim: Problem dimensionality.
        """
        self.capacity = capacity
        self.dim = dim
        self.positions: List[np.ndarray] = []
        self.fitnesses: List[float] = []

    def add(self, x: np.ndarray, f: float):
        """
        Add a solution to the archive.
        If at capacity, evict the worst solution (if new is better).
        """
        if len(self.positions) < self.capacity:
            self.positions.append(x.copy())
            self.fitnesses.append(f)
        else:
            # Find worst in archive
            worst_idx = int(np.argmax(self.fitnesses))
            if f < self.fitnesses[worst_idx]:
                self.positions[worst_idx] = x.copy()
                self.fitnesses[worst_idx] = f

    def get_best(self) -> Tuple[Optional[np.ndarray], float]:
        """Return the best solution in archive."""
        if len(self.positions) == 0:
            return None, np.inf
        best_idx = int(np.argmin(self.fitnesses))
        return self.positions[best_idx].copy(), self.fitnesses[best_idx]

    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all archived solutions and their fitnesses."""
        if len(self.positions) == 0:
            return np.empty((0, self.dim)), np.empty(0)
        return np.array(self.positions), np.array(self.fitnesses)

    def get_diversity(self) -> float:
        """Mean pairwise distance among archived solutions (normalized)."""
        if len(self.positions) < 2:
            return 0.0
        X = np.array(self.positions)
        var = np.mean(np.var(X, axis=0))
        return float(np.sqrt(var))

    @property
    def size(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        best_f = min(self.fitnesses) if self.fitnesses else np.inf
        return f"EliteArchive(size={self.size}/{self.capacity}, best={best_f:.6e})"


class DisplacedArchive:
    """
    Archive of displaced parents for DE/current-to-pbest/1 (L-SHADE style).
    
    When a parent is replaced by its trial vector, the parent is stored here.
    Used as the extended donor pool in current-to-pbest mutation.
    Size is bounded by r_arc * NP.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.solutions: List[np.ndarray] = []

    def add(self, x: np.ndarray):
        """Add a displaced parent. If full, randomly replace one."""
        if len(self.solutions) < self.max_size:
            self.solutions.append(x.copy())
        else:
            # Random replacement (JADE/L-SHADE convention)
            idx = np.random.randint(0, self.max_size)
            self.solutions[idx] = x.copy()

    def get_solutions(self) -> List[np.ndarray]:
        """Return list of archived solutions."""
        return self.solutions

    def resize(self, new_max: int):
        """Resize archive (for LPSR). Remove random entries if shrinking."""
        self.max_size = new_max
        while len(self.solutions) > self.max_size:
            idx = np.random.randint(0, len(self.solutions))
            self.solutions.pop(idx)

    @property
    def size(self) -> int:
        return len(self.solutions)

    def __repr__(self) -> str:
        return f"DisplacedArchive(size={self.size}/{self.max_size})"
