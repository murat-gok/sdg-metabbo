"""
Online Surrogate Training Manager for ASGD-MetaBBO.

Manages:
  - When to retrain the surrogate (every k generations or on-demand)
  - Data windowing (last N evaluated points)
  - Integration with the DE engine's evaluation history
  - Surrogate quality monitoring

In Phase 6 (meta-RL), the retrain interval becomes a learned action.
For now, it uses a fixed schedule.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .ensemble import SurrogateEnsemble


class SurrogateManager:
    """Manages surrogate lifecycle during optimization."""

    def __init__(
        self,
        input_dim: int,
        n_members: int = 5,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        lr: float = 1e-3,
        retrain_interval: int = 5,
        data_window: int = 500,
        min_data_for_training: int = 30,
        epochs_per_retrain: int = 50,
        batch_size: int = 64,
        seed: int = 0,
    ):
        """
        Args:
            input_dim: Problem dimensionality.
            n_members: Number of ensemble members.
            hidden_dims: MLP architecture per member.
            lr: Learning rate.
            retrain_interval: Retrain every k generations.
            data_window: Use last N evaluated points for training.
            min_data_for_training: Don't train until we have this many points.
            epochs_per_retrain: Epochs per retraining.
            batch_size: Mini-batch size for training.
            seed: Random seed.
        """
        self.ensemble = SurrogateEnsemble(
            input_dim=input_dim,
            n_members=n_members,
            hidden_dims=hidden_dims,
            lr=lr,
            base_seed=seed,
        )
        self.retrain_interval = retrain_interval
        self.data_window = data_window
        self.min_data = min_data_for_training
        self.epochs = epochs_per_retrain
        self.batch_size = batch_size
        self.last_retrain_gen = -1
        self.retrain_count = 0

        # Track surrogate quality over time
        self.quality_history = []

    def should_retrain(self, generation: int, n_data: int) -> bool:
        """Check if surrogate should be retrained this generation."""
        if n_data < self.min_data:
            return False
        if self.last_retrain_gen < 0:
            return True  # First training
        return (generation - self.last_retrain_gen) >= self.retrain_interval

    def retrain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> Dict:
        """
        Retrain the surrogate ensemble on recent data.

        Args:
            X: (n, dim) all evaluated solutions.
            y: (n,) their fitness values.
            generation: Current generation number.

        Returns:
            Diagnostics dict.
        """
        # Windowed data: use only last `data_window` points
        n = len(X)
        if n > self.data_window:
            X_train = X[-self.data_window:]
            y_train = y[-self.data_window:]
        else:
            X_train = X
            y_train = y

        # Train ensemble
        self.ensemble.train(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            bootstrap=True,
        )

        self.last_retrain_gen = generation
        self.retrain_count += 1

        # Compute quality diagnostics on training data
        diag = self.ensemble.get_diagnostics(X_train, y_train)
        diag["generation"] = generation
        diag["n_data"] = len(X_train)
        diag["retrain_count"] = self.retrain_count
        self.quality_history.append(diag)

        return diag

    def maybe_retrain(
        self,
        X: np.ndarray,
        y: np.ndarray,
        generation: int,
    ) -> Optional[Dict]:
        """
        Retrain if the schedule says so.

        Args:
            X, y: Full evaluation history.
            generation: Current generation.

        Returns:
            Diagnostics if retrained, None otherwise.
        """
        if self.should_retrain(generation, len(X)):
            return self.retrain(X, y, generation)
        return None

    def get_sigma_at(self, x: np.ndarray) -> float:
        """Get disagreement at a single point (for goal-distancing operator)."""
        return self.ensemble.get_sigma_at(x)

    def get_sigma_batch(self, X: np.ndarray) -> np.ndarray:
        """Get disagreement for a batch of points."""
        return self.ensemble.predict_disagreement(X)

    def get_reward(
        self,
        x_new: np.ndarray,
        f_new: float,
        current_best: float,
        alpha: float = 0.3,
    ) -> float:
        """Get surrogate-informed reward for the bandit."""
        return self.ensemble.get_reward_signal(x_new, f_new, current_best, alpha)

    def get_predicted_improvement(
        self, candidates: np.ndarray, current_best: float
    ) -> np.ndarray:
        """Predict improvement for candidate solutions."""
        return self.ensemble.predict_improvement(candidates, current_best)

    @property
    def is_trained(self) -> bool:
        return self.ensemble.is_trained

    def get_state_features(self) -> Dict:
        """
        Get surrogate state features for the meta-RL encoder.

        Returns dict with features that describe the surrogate's
        current quality and the landscape's uncertainty profile.
        """
        if not self.quality_history:
            return {
                "surrogate_mse": 0.0,
                "surrogate_rank_corr": 0.0,
                "surrogate_mean_sigma": 0.0,
                "surrogate_max_sigma": 0.0,
                "retrain_count": 0,
            }

        last = self.quality_history[-1]
        return {
            "surrogate_mse": last.get("mse", 0.0),
            "surrogate_rank_corr": last.get("rank_corr", 0.0),
            "surrogate_mean_sigma": last.get("mean_sigma", 0.0),
            "surrogate_max_sigma": last.get("max_sigma", 0.0),
            "retrain_count": self.retrain_count,
        }

    def __repr__(self) -> str:
        return (f"SurrogateManager(ensemble={self.ensemble}, "
                f"retrains={self.retrain_count})")
