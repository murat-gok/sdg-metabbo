"""
Page-Hinkley Change-Point Detection with Cooldown.

R1 REVISION: Original threshold=50 caused ~80-100 resets per 60 generations,
effectively randomizing operator selection and invalidating SW-UCB regret
guarantees. Reviewer identified this as a fatal design flaw.

Fixes applied:
  1. Threshold: 50 → 500 (10× less sensitive)
  2. Cooldown: min 30 observations between detections
  3. Soft reset: keeps running mean, resets only cumulative sum

References:
  - Hartland et al. (2007) Adapt-EvE: meta-bandit dampening
  - Fialho et al. (Ann. Math. AI 2010): DMAB + PH + UCB
  - Besson & Kaufmann (2019) GLR-klUCB: controlled reset rate
"""

import numpy as np


class PageHinkley:
    """Page-Hinkley test with cooldown to prevent over-triggering."""

    def __init__(self, delta: float = 0.01, threshold: float = 50.0, cooldown: int = 50):
        self.delta = delta
        self.threshold = threshold
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.m_t = 0.0
        self.M_t = float("inf")
        self.since_last = self.cooldown  # allow immediate first detection

    def update(self, value: float) -> bool:
        self.n += 1
        self.since_last += 1
        self.sum += value
        self.mean = self.sum / self.n
        self.m_t += value - self.mean - self.delta
        if self.m_t < self.M_t:
            self.M_t = self.m_t
        PH = self.m_t - self.M_t
        if PH > self.threshold and self.since_last >= self.cooldown:
            self.m_t = 0.0
            self.M_t = float("inf")
            self.since_last = 0
            return True
        return False
