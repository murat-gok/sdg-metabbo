"""
Lightweight MLP in pure NumPy.

Used as the surrogate model building block. Each ensemble member is one MLP.
Architecture: Input(D) → 128 → ReLU → 64 → ReLU → 32 → ReLU → 1

Includes:
- Xavier/He initialization (independent per instance for ensemble diversity)
- Adam optimizer with mini-batch SGD
- Forward pass and backpropagation
- Standardization of inputs and outputs

For GPU training on your machine, swap to the PyTorch backend (same API).
"""

import numpy as np
from typing import List, Tuple, Optional


class NumpyMLP:
    """Simple feedforward MLP with Adam optimizer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.rng = np.random.default_rng(seed)

        # Build layers
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        dims = [input_dim] + list(hidden_dims) + [1]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            # Xavier uniform initialization
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            W = self.rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        # Adam state
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # Input/output standardization
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass.

        Args:
            X: Input (batch_size, input_dim), assumed already standardized.

        Returns:
            (output, activations) where activations[i] is pre-activation of layer i.
        """
        activations = [X]
        h = X
        for i in range(len(self.weights)):
            z = h @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                h = self._relu(z)
            else:
                h = z  # Linear output
            activations.append(z)
        return h, activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for input X.

        Args:
            X: (n_samples, input_dim) raw (unstandardized) input.

        Returns:
            y_pred: (n_samples,) predicted values in original scale.
        """
        X_std = self._standardize_x(X)
        out, _ = self.forward(X_std)
        # De-standardize output
        return (out.ravel() * self.y_std) + self.y_mean

    def _standardize_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean is not None:
            return (X - self.x_mean) / (self.x_std + 1e-8)
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        """
        Train the MLP on data.

        Args:
            X: (n_samples, input_dim)
            y: (n_samples,)
            epochs: Training epochs.
            batch_size: Mini-batch size.
        """
        n = len(X)
        if n == 0:
            return

        # Compute standardization parameters
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 1e-10 else 1.0

        # Standardize
        X_std = (X - self.x_mean) / (self.x_std + 1e-8)
        y_std = (y - self.y_mean) / self.y_std

        for epoch in range(epochs):
            # Shuffle
            perm = self.rng.permutation(n)
            X_shuf = X_std[perm]
            y_shuf = y_std[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end].reshape(-1, 1)

                self._train_step(X_batch, y_batch)

    def _train_step(self, X: np.ndarray, y: np.ndarray):
        """One gradient step with Adam."""
        batch_size = len(X)

        # Forward pass — collect all activations
        layer_inputs = [X]
        h = X
        pre_activations = []

        for i in range(len(self.weights)):
            z = h @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                h = self._relu(z)
            else:
                h = z  # Output layer: linear
            if i < len(self.weights) - 1:
                layer_inputs.append(h)

        # Loss: MSE
        output = h  # (batch, 1)
        loss_grad = 2.0 * (output - y) / batch_size  # dL/dout

        # Backward pass
        grad_weights = []
        grad_biases = []
        delta = loss_grad

        for i in range(len(self.weights) - 1, -1, -1):
            if i < len(self.weights) - 1:
                # ReLU gradient
                delta = delta * self._relu_grad(pre_activations[i])

            # Gradient w.r.t. weights and biases
            inp = layer_inputs[i] if i < len(layer_inputs) else X
            dW = inp.T @ delta
            db = np.sum(delta, axis=0)
            grad_weights.insert(0, dW)
            grad_biases.insert(0, db)

            # Propagate to previous layer
            if i > 0:
                delta = delta @ self.weights[i].T

        # Adam update
        self.adam_t += 1
        for i in range(len(self.weights)):
            # Weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_weights[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_weights[i] ** 2
            m_hat = self.m_w[i] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.v_w[i] / (1 - self.beta2 ** self.adam_t)
            self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_biases[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_biases[i] ** 2
            m_hat = self.m_b[i] / (1 - self.beta1 ** self.adam_t)
            v_hat = self.v_b[i] / (1 - self.beta2 ** self.adam_t)
            self.biases[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss on given data (raw, unstandardized)."""
        y_pred = self.predict(X)
        return float(np.mean((y_pred - y) ** 2))

    def reset_adam(self):
        """Reset Adam optimizer state (useful when retraining on new data)."""
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.adam_t = 0
