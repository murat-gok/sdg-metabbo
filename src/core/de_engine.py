"""
DE Engine for ASGD-MetaBBO.

Core optimization loop with:
- Modular operator selection (supports both fixed and bandit-controlled modes)
- SHADE-style adaptive F and CR via historical memory (Tanabe & Fukunaga 2014)
- Linear Population Size Reduction (LPSR) from L-SHADE
- Greedy selection preserving elitism (Rudolph 1994 Theorem 2)
- Integration points for surrogate, bandit, and meta-RL components

Phase 1 mode: Fixed operator (current-to-pbest/1) for validation.
Full mode: Bandit-selected operators with surrogate feedback.
"""

import numpy as np
from typing import Callable, Optional, Dict, List, Tuple
from .population import Population
from .archive import EliteArchive, DisplacedArchive
from .operators import OPERATORS, NUM_OPERATORS, get_operator_name


class DEEngine:
    """Differential Evolution engine with modular operator interface."""

    def __init__(
        self,
        func: Callable,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        max_evals: int,
        np_size: Optional[int] = None,
        seed: Optional[int] = None,
        # SHADE memory
        memory_size: int = 5,
        p_best_rate: float = 0.1,
        # Archive
        archive_rate: float = 1.0,
        elite_capacity: Optional[int] = None,
        # LPSR
        use_lpsr: bool = True,
        np_min: Optional[int] = None,
        # Operator selection mode
        fixed_operator: Optional[int] = None,  # None = use external selector
    ):
        """
        Args:
            func: Objective function f(x) → scalar (minimization).
            dim: Problem dimensionality.
            bounds: (lower, upper) bounds arrays.
            max_evals: Maximum function evaluations.
            np_size: Population size. Default: 5*D (L-SHADE convention).
            seed: Random seed.
            memory_size: SHADE historical memory size (H).
            p_best_rate: Fraction for pbest selection.
            archive_rate: Displaced archive size = archive_rate * NP_init.
            elite_capacity: Elite archive capacity. Default: 2*D.
            use_lpsr: Whether to use Linear Population Size Reduction.
            np_min: Minimum population size for LPSR. Default: 4.
            fixed_operator: If set, always use this operator index. If None, requires external selection.
        """
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Population
        self.np_init = np_size or max(5 * dim, 18)
        self.np_min = np_min or 4
        self.pop = Population(self.np_init, dim, bounds, seed=seed)

        # SHADE memory for F and CR
        self.memory_size = memory_size
        self.memory_F = np.full(memory_size, 0.5)
        self.memory_CR = np.full(memory_size, 0.5)
        self.memory_idx = 0

        # Parameters
        self.p_best_rate = p_best_rate
        self.use_lpsr = use_lpsr

        # Archives
        self.displaced_archive = DisplacedArchive(
            max_size=int(archive_rate * self.np_init)
        )
        self.elite_archive = EliteArchive(
            capacity=elite_capacity or 2 * dim, dim=dim
        )

        # Operator mode
        self.fixed_operator = fixed_operator

        # Tracking
        self.generation = 0
        self.convergence_curve: List[float] = []
        self.operator_usage: List[Dict[int, int]] = []

    def run(
        self,
        operator_selector: Optional[Callable] = None,
        on_generation: Optional[Callable] = None,
    ) -> Dict:
        """
        Run the optimization loop.

        Args:
            operator_selector: Function(pop, target_idx, generation, **info) → operator_index.
                Only used if fixed_operator is None.
            on_generation: Callback(engine) called after each generation.

        Returns:
            Dictionary with optimization results.
        """
        # Initial evaluation
        self.pop.evaluate(self.func)
        self._update_elite_archive()
        self.convergence_curve.append(self.pop.best_fitness)

        while self.pop.n_evals < self.max_evals:
            self._one_generation(operator_selector)
            self.generation += 1

            # LPSR: reduce population size
            if self.use_lpsr:
                self._linear_pop_reduction()

            # Track convergence
            self.convergence_curve.append(self.pop.best_fitness)

            # Callback
            if on_generation is not None:
                on_generation(self)

            # Early stop if budget exhausted
            if self.pop.n_evals >= self.max_evals:
                break

        return {
            "best_fitness": self.pop.best_fitness,
            "best_position": self.pop.best_position,
            "n_evals": self.pop.n_evals,
            "generations": self.generation,
            "convergence_curve": self.convergence_curve,
            "operator_usage": self.operator_usage,
        }

    def _one_generation(self, operator_selector: Optional[Callable] = None):
        """Execute one DE generation: mutate, crossover, select for all individuals."""
        NP = self.pop.np_size
        successful_F = []
        successful_CR = []
        gen_operator_counts = {i: 0 for i in range(NUM_OPERATORS)}

        trial_vectors = []
        trial_indices = []

        for i in range(NP):
            if self.pop.n_evals >= self.max_evals:
                break

            # Generate F_i and CR_i from SHADE memory
            F_i, CR_i = self._sample_parameters()

            # Select operator
            if self.fixed_operator is not None:
                op_idx = self.fixed_operator
            elif operator_selector is not None:
                op_idx = operator_selector(
                    pop=self.pop,
                    target_idx=i,
                    generation=self.generation,
                    F=F_i,
                    CR=CR_i,
                )
            else:
                op_idx = 2  # Default: current-to-pbest/1

            gen_operator_counts[op_idx] = gen_operator_counts.get(op_idx, 0) + 1

            # Get operator function
            operator_fn = OPERATORS[op_idx]

            # Build operator kwargs
            op_kwargs = {
                "p": self.p_best_rate,
                "archive": self.displaced_archive.get_solutions(),
            }

            # Generate trial vector
            trial = operator_fn(self.pop, i, F_i, CR_i, self.rng, **op_kwargs)

            trial_vectors.append(trial)
            trial_indices.append(i)

        # Evaluate all trial vectors and perform greedy selection
        for idx, (i, trial) in enumerate(zip(trial_indices, trial_vectors)):
            if self.pop.n_evals >= self.max_evals:
                break

            f_trial = self.func(trial)
            self.pop.n_evals += 1
            self.pop.eval_history_x.append(trial.copy())
            self.pop.eval_history_f.append(f_trial)

            # Update best-so-far
            if f_trial < self.pop.best_fitness:
                self.pop.best_fitness = f_trial
                self.pop.best_position = trial.copy()

            # Greedy selection (preserves elitism)
            if f_trial <= self.pop.fitness[i]:
                # Success: record parameters
                F_i, CR_i = self._sample_parameters()  # Re-sample for recording
                successful_F.append(F_i)
                successful_CR.append(CR_i)

                # Archive displaced parent
                self.displaced_archive.add(self.pop.positions[i])

                # Replace
                self.pop.positions[i] = trial
                self.pop.fitness[i] = f_trial

        # Update SHADE memory
        if successful_F:
            self._update_memory(successful_F, successful_CR)

        # Update elite archive
        self._update_elite_archive()

        # Track operator usage
        self.operator_usage.append(gen_operator_counts)
        self.pop.generation = self.generation

    def _sample_parameters(self) -> Tuple[float, float]:
        """Sample F and CR from SHADE historical memory."""
        # Select random memory cell
        r = self.rng.integers(0, self.memory_size)

        # Sample F from Cauchy(memory_F[r], 0.1), truncated to (0, 1]
        F = self._cauchy_sample(self.memory_F[r], 0.1)
        F = np.clip(F, 0.0, 1.0)
        if F <= 0:
            F = self._cauchy_sample(self.memory_F[r], 0.1)
            F = np.clip(F, 1e-6, 1.0)

        # Sample CR from N(memory_CR[r], 0.1), clipped to [0, 1]
        CR = self.rng.normal(self.memory_CR[r], 0.1)
        CR = np.clip(CR, 0.0, 1.0)

        return float(F), float(CR)

    def _cauchy_sample(self, loc: float, scale: float) -> float:
        """Sample from Cauchy distribution."""
        return loc + scale * np.tan(np.pi * (self.rng.random() - 0.5))

    def _update_memory(self, successful_F: List[float], successful_CR: List[float]):
        """Update SHADE historical memory with weighted Lehmer mean."""
        if not successful_F:
            return

        F_arr = np.array(successful_F)
        CR_arr = np.array(successful_CR)

        # Weighted Lehmer mean (SHADE convention)
        # Weights: fitness improvements (simplified to uniform here)
        weights = np.ones(len(F_arr)) / len(F_arr)

        # Lehmer mean of F
        self.memory_F[self.memory_idx] = float(
            np.sum(weights * F_arr**2) / (np.sum(weights * F_arr) + 1e-30)
        )

        # Weighted arithmetic mean of CR
        self.memory_CR[self.memory_idx] = float(np.sum(weights * CR_arr))

        self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _linear_pop_reduction(self):
        """
        Linear Population Size Reduction (LPSR) from L-SHADE.
        NP_new = round(NP_init - (NP_init - NP_min) * evals / max_evals)
        """
        progress = self.pop.n_evals / self.max_evals
        np_new = max(
            self.np_min,
            round(self.np_init - (self.np_init - self.np_min) * progress),
        )

        if np_new < self.pop.np_size:
            # Remove worst individuals
            sorted_idx = self.pop.get_sorted_indices()
            keep = sorted_idx[:np_new]

            self.pop.positions = self.pop.positions[keep]
            self.pop.fitness = self.pop.fitness[keep]
            self.pop.np_size = np_new

            # Resize displaced archive proportionally
            self.displaced_archive.resize(max(np_new, 1))

    def _update_elite_archive(self):
        """Add current population's best solutions to elite archive."""
        sorted_idx = self.pop.get_sorted_indices()
        # Add top 3 to elite archive (or fewer if NP < 3)
        for i in sorted_idx[: min(3, self.pop.np_size)]:
            self.elite_archive.add(
                self.pop.positions[i], self.pop.fitness[i]
            )

    def get_state(self) -> Dict:
        """Get current optimization state (for meta-RL / logging)."""
        stats = self.pop.get_fitness_stats()
        return {
            "generation": self.generation,
            "n_evals": self.pop.n_evals,
            "progress": self.pop.n_evals / self.max_evals,
            "best_fitness": self.pop.best_fitness,
            "pop_size": self.pop.np_size,
            "diversity": self.pop.get_diversity(),
            "fitness_mean": stats["mean"],
            "fitness_std": stats["std"],
            "elite_archive_size": self.elite_archive.size,
            "displaced_archive_size": self.displaced_archive.size,
        }
