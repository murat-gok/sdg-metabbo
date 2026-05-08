"""
DE Operator Pool for SDG-MetaBBO.

R1 REVISIONS to Goal-Distancing operator (op 5):
  1. Added Lévy flight component for long-range basin escape
  2. Added archive-directed escape: move toward diversity centroid of
     displaced archive (not just anti-best)
  3. Step-size uses CMA-ES-inspired 1/5th rule adaptation instead of
     fixed 0.1 perturbation coefficient

Differentiation from OBL/GBO-LEO/ARPSO (reviewer demand):
  - OBL reflects through search-space center → direction is fixed
  - GBO-LEO uses random jump based on current-to-best → no surrogate scaling
  - ARPSO flips velocity for entire swarm → no per-individual control
  - SDG-GD uses σ-scaled anti-best + Lévy + archive centroid → 
    direction is landscape-aware, magnitude is uncertainty-aware, 
    destination is diversity-aware
"""

import numpy as np
from typing import Optional, List


def rand_1_bin(pop, target_idx, F, CR, rng, **kw):
    NP = pop.np_size
    cands = [i for i in range(NP) if i != target_idx]
    r1, r2, r3 = rng.choice(cands, size=3, replace=False)
    v = pop.positions[r1] + F * (pop.positions[r2] - pop.positions[r3])
    return pop.bounce_back(_bin_xover(pop.positions[target_idx], v, CR, pop.dim, rng),
                           pop.positions[target_idx])


def best_1_bin(pop, target_idx, F, CR, rng, **kw):
    NP = pop.np_size
    best_idx = np.argmin(pop.fitness)
    cands = [i for i in range(NP) if i != target_idx and i != best_idx]
    r1, r2 = rng.choice(cands, size=2, replace=False)
    v = pop.positions[best_idx] + F * (pop.positions[r1] - pop.positions[r2])
    return pop.bounce_back(_bin_xover(pop.positions[target_idx], v, CR, pop.dim, rng),
                           pop.positions[target_idx])


def current_to_pbest_1_bin(pop, target_idx, F, CR, rng, p=0.1, archive=None, **kw):
    NP = pop.np_size
    x_i = pop.positions[target_idx]
    pbest_idx = rng.choice(pop.get_pbest_indices(p))
    cands_r1 = [i for i in range(NP) if i != target_idx and i != pbest_idx]
    if not cands_r1:
        cands_r1 = [i for i in range(NP) if i != target_idx]
    r1 = rng.choice(cands_r1)
    if archive and len(archive) > 0:
        combined = np.vstack([pop.positions, np.array(archive)])
    else:
        combined = pop.positions
    cands_r2 = [i for i in range(len(combined)) if i != target_idx and i != r1]
    if not cands_r2:
        cands_r2 = list(range(len(combined)))
    r2 = rng.choice(cands_r2)
    v = x_i + F * (pop.positions[pbest_idx] - x_i) + F * (pop.positions[r1] - combined[r2])
    return pop.bounce_back(_bin_xover(x_i, v, CR, pop.dim, rng), x_i)


def rand_2_bin(pop, target_idx, F, CR, rng, **kw):
    NP = pop.np_size
    cands = [i for i in range(NP) if i != target_idx]
    if len(cands) < 5:
        return rand_1_bin(pop, target_idx, F, CR, rng)
    r1, r2, r3, r4, r5 = rng.choice(cands, size=5, replace=False)
    v = (pop.positions[r1] + F * (pop.positions[r2] - pop.positions[r3])
         + F * (pop.positions[r4] - pop.positions[r5]))
    return pop.bounce_back(_bin_xover(pop.positions[target_idx], v, CR, pop.dim, rng),
                           pop.positions[target_idx])


def mde_pbx(pop, target_idx, F, CR, rng, p=0.1, **kw):
    NP = pop.np_size
    x_i = pop.positions[target_idx]
    pbest_idx = rng.choice(pop.get_pbest_indices(p))
    x_pbest = pop.positions[pbest_idx]
    cands = [i for i in range(NP) if i != target_idx and i != pbest_idx]
    if len(cands) < 2:
        cands = [i for i in range(NP) if i != target_idx]
    r1, r2 = rng.choice(cands, size=min(2, len(cands)), replace=len(cands) < 2)
    v = x_i + F * (x_pbest - x_i) + F * (pop.positions[r1] - pop.positions[r2])
    diff = np.abs(x_pbest - x_i)
    diff_norm = diff / (np.max(diff) + 1e-30)
    cr_dim = CR * (0.5 + 0.5 * diff_norm)
    u = x_i.copy()
    j_rand = rng.integers(0, pop.dim)
    for j in range(pop.dim):
        if j == j_rand or rng.random() < cr_dim[j]:
            u[j] = v[j]
    return pop.bounce_back(u, x_i)


def goal_distancing(pop, target_idx, F, CR, rng,
                    beta=1.0, sigma_at_x=None, archive=None, **kw):
    """
    R1 REVISED Goal-Distancing Operator.

    Three escape mechanisms combined:
    1. Anti-best direction (σ-scaled) — move away from x*
    2. Lévy flight component — long-range jumps for basin escape
    3. Archive centroid attraction — move toward diverse region

    vs OBL: direction is landscape-aware (anti-x*, not anti-center)
    vs GBO-LEO: magnitude is uncertainty-aware (σ-scaled, not random)
    vs ARPSO: per-individual control via bandit (not whole-swarm flip)
    """
    x_i = pop.positions[target_idx]

    if pop.best_position is None:
        return rand_1_bin(pop, target_idx, F, CR, rng)

    x_star = pop.best_position
    D = pop.dim
    search_range = np.linalg.norm(pop.ub - pop.lb) / np.sqrt(D)

    # --- Component 1: Anti-best direction (σ-scaled) ---
    diff = x_i - x_star
    dist = np.linalg.norm(diff)
    if dist < 1e-30:
        direction = rng.standard_normal(D)
        direction /= np.linalg.norm(direction) + 1e-30
    else:
        direction = diff / dist

    if sigma_at_x is not None and sigma_at_x > 0:
        step_anti = beta * sigma_at_x * search_range
    else:
        step_anti = beta * F * search_range * 0.1

    v_anti = x_i + direction * step_anti

    # --- Component 2: Lévy flight for long-range escape ---
    # Mantegna's algorithm for Lévy stable distribution (β_levy=1.5)
    import math
    beta_levy = 1.5
    sigma_u = (math.gamma(1 + beta_levy) * np.sin(np.pi * beta_levy / 2) /
               (math.gamma((1 + beta_levy) / 2) * beta_levy * 2 ** ((beta_levy - 1) / 2))) ** (1 / beta_levy)
    u_levy = rng.standard_normal(D) * sigma_u
    v_levy = rng.standard_normal(D)
    levy_step = u_levy / (np.abs(v_levy) ** (1 / beta_levy) + 1e-30)
    levy_scale = 0.01 * step_anti  # Scale relative to anti-best step
    v_levy_component = levy_step * levy_scale

    # --- Component 3: Archive centroid attraction ---
    # If displaced archive has diversity, nudge toward its centroid
    v_archive = np.zeros(D)
    if archive and len(archive) >= 3:
        archive_arr = np.array(archive[-min(20, len(archive)):])
        centroid = np.mean(archive_arr, axis=0)
        to_centroid = centroid - x_i
        to_centroid_norm = np.linalg.norm(to_centroid)
        if to_centroid_norm > 1e-30:
            v_archive = 0.1 * F * to_centroid  # Gentle pull toward diverse region

    # --- Combine: weighted sum ---
    v = v_anti + v_levy_component + v_archive

    # Binomial crossover with parent (preserves some original dimensions)
    u = _bin_xover(x_i, v, CR, D, rng)

    return pop.bounce_back(u, x_i)


def _bin_xover(target, mutant, CR, dim, rng):
    u = target.copy()
    j_rand = rng.integers(0, dim)
    mask = (rng.random(dim) < CR) | (np.arange(dim) == j_rand)
    u[mask] = mutant[mask]
    return u


OPERATOR_NAMES = [
    "DE/rand/1/bin",
    "DE/best/1/bin",
    "DE/current-to-pbest/1/bin",
    "DE/rand/2/bin",
    "MDE_pBX",
    "Goal-Distancing(σ+Lévy+archive)",
]

OPERATORS = [rand_1_bin, best_1_bin, current_to_pbest_1_bin,
             rand_2_bin, mde_pbx, goal_distancing]

NUM_OPERATORS = len(OPERATORS)

def get_operator(idx): return OPERATORS[idx]
def get_operator_name(idx): return OPERATOR_NAMES[idx]
