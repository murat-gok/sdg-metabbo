"""
Deep Ensemble Surrogate (Opponent Model) for ASGD-MetaBBO.

Maintains M independently initialized MLPs trained on the same data.
Provides:
  - Mean prediction: μ(x) = (1/M) Σ f_m(x)
  - Disagreement: σ(x) = std({f_1(x), ..., f_M(x)})
  - Predicted improvement over current best

Biological motivation: Portia's aggressive mimicry as opponent modelling —
the ensemble models the landscape's "response" to candidate moves,
and disagreement signals where the model is uncertain (analogous to
Portia's trial-and-error when the prey's response is unpredictable).

Prior art positioning:
  - Disagreement-based exploration is mature (QBC 1992, CAL-SAPSO 2017, AutoSAEA 2023)
  - Our novelty: disagreement integrated as bandit reward signal + goal-distancing scale
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .mlp import NumpyMLP


class SurrogateEnsemble:
    """Ensemble of M independently trained MLPs."""

    def __init__(
        self,
        input_dim: int,
        n_members: int = 5,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        lr: float = 1e-3,
        base_seed: int = 0,
    ):
        """
        Args:
            input_dim: Problem dimensionality.
            n_members: Number of ensemble members (M).
            hidden_dims: MLP architecture.
            lr: Learning rate for each member.
            base_seed: Base random seed; each member gets base_seed + i.
        """
        self.input_dim = input_dim
        self.n_members = n_members
        self.is_trained = False
        self.n_training_points = 0

        # Create M independent MLPs with different seeds
        self.members = [
            NumpyMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                lr=lr,
                seed=base_seed + i * 137,  # Well-separated seeds
            )
            for i in range(n_members)
        ]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        bootstrap: bool = True,
    ):
        """
        Train all ensemble members on data.

        Args:
            X: (n_samples, input_dim) training inputs.
            y: (n_samples,) training targets.
            epochs: Training epochs per member.
            batch_size: Mini-batch size.
            bootstrap: If True, each member trains on a bootstrap sample
                       (adds diversity beyond weight initialization).
        """
        n = len(X)
        if n < 5:
            return  # Not enough data

        for i, member in enumerate(self.members):
            if bootstrap:
                # Bootstrap sample (with replacement)
                rng = np.random.default_rng(i * 31 + n)
                idx = rng.choice(n, size=n, replace=True)
                X_boot, y_boot = X[idx], y[idx]
            else:
                X_boot, y_boot = X, y

            member.reset_adam()
            member.fit(X_boot, y_boot, epochs=epochs, batch_size=batch_size)

        self.is_trained = True
        self.n_training_points = n

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all ensemble members.

        Args:
            X: (n_samples, input_dim)

        Returns:
            predictions: (n_members, n_samples) — each row is one member's predictions.
        """
        if not self.is_trained:
            return np.zeros((self.n_members, len(X)))

        preds = np.array([m.predict(X) for m in self.members])
        return preds

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble mean prediction.

        Returns:
            μ(x): (n_samples,)
        """
        preds = self.predict_all(X)
        return np.mean(preds, axis=0)

    def predict_disagreement(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble disagreement (standard deviation across members).

        This is the core "opponent uncertainty" signal:
        - High σ → the model doesn't agree → unexplored region or basin boundary
        - Low σ → the model agrees → well-explored region

        Returns:
            σ(x): (n_samples,)
        """
        preds = self.predict_all(X)
        return np.std(preds, axis=0)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both mean prediction and disagreement.

        Returns:
            (μ, σ): each (n_samples,)
        """
        preds = self.predict_all(X)
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    def predict_improvement(
        self, X: np.ndarray, current_best: float
    ) -> np.ndarray:
        """
        Predicted improvement over current best.

        EI_surrogate(x) = max(0, current_best - μ(x))

        This is a simplified Expected Improvement without the variance
        integral — the disagreement is used separately as exploration bonus.

        Returns:
            improvement: (n_samples,) non-negative predicted improvement.
        """
        mu = self.predict_mean(X)
        return np.maximum(0, current_best - mu)

    def get_reward_signal(
        self,
        x_new: np.ndarray,
        f_new: float,
        current_best: float,
        alpha_disagree: float = 0.3,
    ) -> float:
        """
        Compute the surrogate-informed reward for the bandit.

        reward = (1 - α) * normalized_improvement + α * normalized_disagreement

        This is where the surrogate acts as "opponent":
        - The improvement term rewards operators that found better solutions
        - The disagreement term rewards operators that explored uncertain regions

        Args:
            x_new: (input_dim,) the new solution evaluated.
            f_new: Its true fitness.
            current_best: Current best fitness.
            alpha_disagree: Weight on disagreement bonus [0, 1].

        Returns:
            Scalar reward signal for the bandit.
        """
        if not self.is_trained:
            # No surrogate yet — use raw improvement only
            improvement = max(0, current_best - f_new)
            return improvement

        X = x_new.reshape(1, -1)
        mu, sigma = self.predict_with_uncertainty(X)

        # Normalized improvement (surrogate-predicted)
        improvement = max(0, current_best - f_new)

        # Normalize disagreement relative to training data range
        sigma_val = float(sigma[0])

        # Combined reward
        reward = (1.0 - alpha_disagree) * improvement + alpha_disagree * sigma_val

        return reward

    def get_sigma_at(self, x: np.ndarray) -> float:
        """
        Get disagreement at a single point.
        Used by the goal-distancing operator.

        Args:
            x: (input_dim,) single solution vector.

        Returns:
            σ(x): scalar disagreement.
        """
        if not self.is_trained:
            return 0.0
        X = x.reshape(1, -1)
        return float(self.predict_disagreement(X)[0])

    def get_diagnostics(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Compute diagnostic metrics for the ensemble.

        Returns dict with:
            - mse: Mean squared error of ensemble mean
            - rank_corr: Spearman rank correlation
            - mean_sigma: Average disagreement
            - max_sigma: Maximum disagreement
        """
        if not self.is_trained or len(X) == 0:
            return {"mse": float("inf"), "rank_corr": 0.0,
                    "mean_sigma": 0.0, "max_sigma": 0.0}

        mu, sigma = self.predict_with_uncertainty(X)

        mse = float(np.mean((mu - y) ** 2))

        # Spearman rank correlation (more important than MSE for ranking)
        from scipy.stats import spearmanr
        if len(y) > 2:
            corr, _ = spearmanr(mu, y)
            rank_corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            rank_corr = 0.0

        return {
            "mse": mse,
            "rank_corr": rank_corr,
            "mean_sigma": float(np.mean(sigma)),
            "max_sigma": float(np.max(sigma)),
        }

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (f"SurrogateEnsemble(M={self.n_members}, D={self.input_dim}, "
                f"{status}, n_data={self.n_training_points})")
