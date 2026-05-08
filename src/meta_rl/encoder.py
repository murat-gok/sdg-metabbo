"""
Two-Stage Attention Encoder for ASGD-MetaBBO (Phase 5).

Encodes the current population state into a fixed-size vector for the PPO policy.

Architecture (from NeurELA / RLDE-AFL):
  Stage 1: Multi-head self-attention over NP individuals (permutation invariant)
  Stage 2: Multi-head self-attention over D dimensions
  + Mantissa-exponent fitness embedding to handle wide fitness scales

Input:  Population matrix (NP, D) + fitness vector (NP,)
Output: State vector s_t ∈ ℝ^{d_model + n_extra}

References:
  - NeurELA (Ma et al., ICLR 2025, arXiv:2408.10672)
  - RLDE-AFL (Guo et al., GECCO 2025, arXiv:2503.18061)

Requires PyTorch. Run on your machine with GPU.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from typing import Tuple, Optional


if HAS_TORCH:

    class MantissaExponentEmbedding(nn.Module):
        """
        Encode fitness values as (mantissa, exponent) pairs.
        Handles the extreme dynamic range of fitness values across
        different optimization stages (1e+10 → 1e-30).

        f → (sign(f) * mantissa, exponent) where f = mantissa * 10^exponent
        Then project through a linear layer to d_model dimensions.
        """

        def __init__(self, d_model: int):
            super().__init__()
            self.proj = nn.Linear(2, d_model)

        def forward(self, fitness: torch.Tensor) -> torch.Tensor:
            """
            Args:
                fitness: (batch, NP) raw fitness values.

            Returns:
                embedding: (batch, NP, d_model)
            """
            # Decompose into mantissa and exponent
            abs_f = torch.abs(fitness) + 1e-30
            exponent = torch.floor(torch.log10(abs_f))
            mantissa = torch.sign(fitness) * abs_f / (10.0 ** exponent)

            # Stack and project
            me = torch.stack([mantissa, exponent], dim=-1)  # (batch, NP, 2)
            return self.proj(me)  # (batch, NP, d_model)


    class AttentionBlock(nn.Module):
        """Multi-head self-attention block with residual connection and LayerNorm."""

        def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=n_heads,
                dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, d_model)
            Returns:
                (batch, seq_len, d_model)
            """
            # Self-attention with residual
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)

            # FFN with residual
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x


    class PopulationEncoder(nn.Module):
        """
        Two-stage attention encoder for population state.

        Stage 1: Attention over individuals (NP dimension)
            Input: (batch, NP, d_model)  →  aggregated to (batch, D, d_model)
        Stage 2: Attention over dimensions (D dimension)
            Input: (batch, D, d_model)   →  aggregated to (batch, d_model)

        The two-stage design gives permutation invariance over individuals
        AND allows transfer across different population sizes and dimensions.
        """

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers_stage1: int = 2,
            n_layers_stage2: int = 2,
            n_extra_features: int = 5,
            dropout: float = 0.1,
        ):
            """
            Args:
                d_model: Hidden dimension throughout the encoder.
                n_heads: Number of attention heads per block.
                n_layers_stage1: Attention blocks for inter-individual stage.
                n_layers_stage2: Attention blocks for inter-dimension stage.
                n_extra_features: Number of scalar features appended to output
                    (generation_progress, sigma_mean, sigma_max, bandit_entropy, diversity).
                dropout: Dropout rate.
            """
            super().__init__()
            self.d_model = d_model
            self.n_extra = n_extra_features

            # Position embedding (per-dimension, learned)
            # We'll initialize with max_dim=100; actual D can be smaller
            self.dim_embed = nn.Embedding(200, d_model)

            # Fitness embedding
            self.fitness_embed = MantissaExponentEmbedding(d_model)

            # Input projection: each individual's D coordinates → d_model
            # We project per-coordinate, then add dimension embedding
            self.coord_proj = nn.Linear(1, d_model)

            # Stage 1: attention over individuals (NP axis)
            self.stage1_blocks = nn.ModuleList([
                AttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers_stage1)
            ])

            # Stage 2: attention over dimensions (D axis)
            self.stage2_blocks = nn.ModuleList([
                AttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers_stage2)
            ])

            # Final projection: d_model + n_extra → output
            self.output_proj = nn.Linear(d_model + n_extra_features, d_model)

        def forward(
            self,
            positions: torch.Tensor,
            fitness: torch.Tensor,
            extra_features: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                positions: (batch, NP, D) population positions, normalized to [0,1].
                fitness: (batch, NP) fitness values (raw scale).
                extra_features: (batch, n_extra) scalar features
                    [progress, sigma_mean, sigma_max, bandit_entropy, diversity].

            Returns:
                state: (batch, d_model) population state embedding.
            """
            batch, NP, D = positions.shape

            # --- Stage 1: Inter-individual attention ---
            # For each dimension d, we have NP coordinate values + fitness
            # Reshape to process each dimension separately

            # Project coordinates: (batch, NP, D, 1) → (batch, NP, D, d_model)
            coords = positions.unsqueeze(-1)  # (batch, NP, D, 1)
            coord_emb = self.coord_proj(coords)  # (batch, NP, D, d_model)

            # Add dimension positional embedding
            dim_ids = torch.arange(D, device=positions.device)
            dim_emb = self.dim_embed(dim_ids)  # (D, d_model)
            coord_emb = coord_emb + dim_emb.unsqueeze(0).unsqueeze(0)

            # Add fitness embedding: (batch, NP, d_model)
            fit_emb = self.fitness_embed(fitness)  # (batch, NP, d_model)
            # Broadcast and add to each dimension
            coord_emb = coord_emb + fit_emb.unsqueeze(2)  # (batch, NP, D, d_model)

            # Process each dimension with attention over individuals
            # Reshape: (batch * D, NP, d_model)
            x = coord_emb.permute(0, 2, 1, 3).reshape(batch * D, NP, self.d_model)

            for block in self.stage1_blocks:
                x = block(x)

            # Aggregate over individuals: mean pooling → (batch * D, d_model)
            x = x.mean(dim=1)

            # Reshape back: (batch, D, d_model)
            x = x.reshape(batch, D, self.d_model)

            # --- Stage 2: Inter-dimension attention ---
            for block in self.stage2_blocks:
                x = block(x)

            # Aggregate over dimensions: mean pooling → (batch, d_model)
            state = x.mean(dim=1)

            # --- Append extra features and project ---
            state = torch.cat([state, extra_features], dim=-1)
            state = self.output_proj(state)

            return state  # (batch, d_model)

        def encode_population(
            self,
            positions_np: np.ndarray,
            fitness_np: np.ndarray,
            bounds: Tuple[np.ndarray, np.ndarray],
            extra_dict: dict,
        ) -> np.ndarray:
            """
            Convenience method: encode from numpy arrays.

            Args:
                positions_np: (NP, D) population positions.
                fitness_np: (NP,) fitness values.
                bounds: (lb, ub) for normalization.
                extra_dict: dict with keys matching n_extra_features.

            Returns:
                state: (d_model,) numpy array.
            """
            self.eval()

            lb, ub = bounds
            # Normalize positions to [0, 1]
            pos_norm = (positions_np - lb) / (ub - lb + 1e-30)

            # Build tensors (batch=1)
            pos_t = torch.FloatTensor(pos_norm).unsqueeze(0)
            fit_t = torch.FloatTensor(fitness_np).unsqueeze(0)

            extra = [
                extra_dict.get("progress", 0.0),
                extra_dict.get("sigma_mean", 0.0),
                extra_dict.get("sigma_max", 0.0),
                extra_dict.get("bandit_entropy", 0.0),
                extra_dict.get("diversity", 0.0),
            ]
            extra_t = torch.FloatTensor([extra])

            with torch.no_grad():
                state = self.forward(pos_t, fit_t, extra_t)

            return state.squeeze(0).numpy()


# ─────────────────────────────────────────────────
# NumPy fallback for environments without PyTorch
# ─────────────────────────────────────────────────

class PopulationEncoderNumpy:
    """
    Lightweight NumPy encoder for testing without PyTorch.
    Uses simple statistics instead of attention.
    Same output shape as the PyTorch version.
    """

    def __init__(self, d_model: int = 64, n_extra: int = 5):
        self.d_model = d_model
        self.n_extra = n_extra
        self.output_dim = d_model

    def encode_population(
        self,
        positions_np: np.ndarray,
        fitness_np: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        extra_dict: dict,
    ) -> np.ndarray:
        """Encode population state using hand-crafted features."""
        lb, ub = bounds
        NP, D = positions_np.shape

        # Normalize
        pos_norm = (positions_np - lb) / (ub - lb + 1e-30)

        # Per-dimension statistics (D features each)
        dim_means = np.mean(pos_norm, axis=0)     # (D,)
        dim_stds = np.std(pos_norm, axis=0)        # (D,)

        # Fitness statistics
        f_sorted = np.sort(fitness_np)
        f_stats = np.array([
            np.mean(fitness_np),
            np.std(fitness_np),
            np.min(fitness_np),
            np.median(fitness_np),
            f_sorted[max(0, len(f_sorted)//10)] if len(f_sorted) > 0 else 0,  # 10th percentile
        ])

        # Mantissa-exponent of best
        best_f = np.min(np.abs(fitness_np)) + 1e-30
        me = np.array([np.log10(best_f), best_f / (10 ** np.floor(np.log10(best_f)))])

        # Extra features
        extras = np.array([
            extra_dict.get("progress", 0.0),
            extra_dict.get("sigma_mean", 0.0),
            extra_dict.get("sigma_max", 0.0),
            extra_dict.get("bandit_entropy", 0.0),
            extra_dict.get("diversity", 0.0),
        ])

        # Concatenate and pad/truncate to d_model
        raw = np.concatenate([
            dim_means[:self.d_model//4],
            dim_stds[:self.d_model//4],
            f_stats,
            me,
            extras,
        ])

        # Pad or truncate to d_model
        if len(raw) < self.d_model:
            state = np.zeros(self.d_model)
            state[:len(raw)] = raw
        else:
            state = raw[:self.d_model]

        return state


def get_encoder(d_model: int = 64, use_torch: bool = True):
    """Factory: get the best available encoder."""
    if use_torch and HAS_TORCH:
        return PopulationEncoder(d_model=d_model)
    else:
        return PopulationEncoderNumpy(d_model=d_model)
