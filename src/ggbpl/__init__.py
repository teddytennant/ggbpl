"""GGBPL and iGGBPL for l0-constrained sparse nonnegative matrix factorization."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

from ggbpl.snmf import (
    EPS,
    GAMMA,
    RHO1,
    ggbpl_sweep,
    grad_U,
    grad_V,
    iggbpl_step,
    inner_U,
    inner_V,
    metric_U,
    metric_V,
    next_inertia,
    objective,
    project_weighted_nonneg_l0,
    reconstruction,
    run_experiment,
    run_ggbpl,
    run_iggbpl,
    synthetic_instance,
)

__all__ = [
    "EPS",
    "GAMMA",
    "RHO1",
    "ggbpl_sweep",
    "grad_U",
    "grad_V",
    "iggbpl_step",
    "inner_U",
    "inner_V",
    "metric_U",
    "metric_V",
    "next_inertia",
    "objective",
    "project_weighted_nonneg_l0",
    "reconstruction",
    "run_experiment",
    "run_ggbpl",
    "run_iggbpl",
    "synthetic_instance",
]
