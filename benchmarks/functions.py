"""
Benchmark interface for ASGD-MetaBBO.

Provides unified access to:
- CEC 2022 (12 functions, D ∈ {10, 20}) via opfunu
- BBOB/COCO (24 noiseless functions) via custom wrappers
- Simple test functions for debugging

All functions are MINIMIZATION problems.
"""

import numpy as np
from typing import Callable, Tuple, Optional

# ─────────────────────────────────────────────────
# Simple test functions (always available, no deps)
# ─────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    """Sphere function. Global minimum: f(0,...,0) = 0."""
    return float(np.sum(x**2))

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Global minimum: f(1,...,1) = 0."""
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function. Global minimum: f(0,...,0) = 0. Highly multimodal."""
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    """Ackley function. Global minimum: f(0,...,0) = 0."""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20 * np.exp(-0.2 * np.sqrt(sum1 / n))
                 - np.exp(sum2 / n) + 20 + np.e)

def griewank(x: np.ndarray) -> float:
    """Griewank function. Global minimum: f(0,...,0) = 0."""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(sum_part - prod_part + 1)

def lunacek_bi_rastrigin(x: np.ndarray) -> float:
    """
    Lunacek bi-Rastrigin (simplified, unrotated).
    Two-funnel deceptive function — the main target for goal-distancing.
    Global minimum at x = μ₁ = 2.5.
    """
    n = len(x)
    mu1 = 2.5
    s = 1.0 - 1.0 / (2.0 * np.sqrt(n + 20) - 8.2)
    mu2 = -np.sqrt((mu1**2 - 1.0) / s)
    d = 1.0

    sum1 = np.sum((x - mu1)**2)
    sum2 = n * d + s * np.sum((x - mu2)**2)
    sum3 = 10 * (n - np.sum(np.cos(2 * np.pi * (x - mu1))))

    return float(min(sum1, sum2) + sum3)


# ─────────────────────────────────────────────────
# Test function registry
# ─────────────────────────────────────────────────

SIMPLE_FUNCTIONS = {
    "sphere": {
        "func": sphere,
        "bounds": (-100, 100),
        "optimum": 0.0,
        "type": "unimodal",
    },
    "rosenbrock": {
        "func": rosenbrock,
        "bounds": (-30, 30),
        "optimum": 0.0,
        "type": "unimodal",
    },
    "rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimum": 0.0,
        "type": "multimodal",
    },
    "ackley": {
        "func": ackley,
        "bounds": (-32, 32),
        "optimum": 0.0,
        "type": "multimodal",
    },
    "griewank": {
        "func": griewank,
        "bounds": (-600, 600),
        "optimum": 0.0,
        "type": "multimodal",
    },
    "lunacek": {
        "func": lunacek_bi_rastrigin,
        "bounds": (-5, 5),
        "optimum": 0.0,
        "type": "deceptive",
    },
}


def get_simple_problem(name: str, dim: int) -> Tuple[Callable, np.ndarray, np.ndarray, float]:
    """
    Get a simple test problem.
    
    Returns: (func, lb, ub, known_optimum)
    """
    info = SIMPLE_FUNCTIONS[name]
    lb = np.full(dim, info["bounds"][0])
    ub = np.full(dim, info["bounds"][1])
    return info["func"], lb, ub, info["optimum"]


# ─────────────────────────────────────────────────
# CEC 2022 Interface (requires opfunu)
# ─────────────────────────────────────────────────

def get_cec2022_problem(func_num: int, dim: int = 10):
    """
    Get CEC 2022 benchmark function.
    
    Args:
        func_num: Function number (1-12).
        dim: Dimensionality (10 or 20).
    
    Returns: (func, lb, ub, known_optimum)
    
    Requires: pip install opfunu
    """
    try:
        import opfunu
    except ImportError:
        raise ImportError(
            "opfunu is required for CEC 2022 benchmarks. "
            "Install with: pip install opfunu"
        )

    # CEC 2022 function class names in opfunu
    func_class = getattr(opfunu.cec_based, f"F{func_num}2022")(ndim=dim)
    
    def wrapped_func(x):
        return float(func_class.evaluate(x))
    
    lb = func_class.lb
    ub = func_class.ub
    optimum = func_class.f_global
    
    return wrapped_func, lb, ub, optimum


def get_cec2022_suite(dim: int = 10):
    """Get all 12 CEC 2022 functions for a given dimensionality."""
    suite = {}
    for i in range(1, 13):
        try:
            func, lb, ub, opt = get_cec2022_problem(i, dim)
            suite[f"CEC2022_f{i}"] = {
                "func": func, "lb": lb, "ub": ub, "optimum": opt
            }
        except Exception as e:
            print(f"Warning: CEC2022 f{i} D={dim} failed to load: {e}")
    return suite
