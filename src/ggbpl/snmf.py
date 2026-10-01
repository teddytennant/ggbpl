"""l0-constrained sparse NMF by GGBPL and iGGBPL.

The smooth objective, finite on the nonnegative l0 ball, is

    J(U, V) = 0.5 * ||X - U V||_F^2 + 0.5 * alpha1 * ||U||_F^2
              + 0.5 * alpha2 * ||V||_F^2

with U >= 0, V >= 0, ||U||_0 <= s1, ||V||_0 <= s2.

Block metrics follow the paper: diagonal row sums of V V^T and of U^T U,
shifted by alpha + eps. The V metric is r by r, so it is built from U^T U
(not U U^T). Step length is sigma = 1/gamma with gamma = 2. The inertial
restart uses rho1 = 1e-4. SNCP is not implemented.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

GAMMA = 2.0
RHO1 = 1e-4
EPS = 1e-8

SYNTH_M = 20
SYNTH_N = 16
SYNTH_R = 3
SYNTH_S1 = 18
SYNTH_S2 = 14
SYNTH_ALPHA = 1e-2
SYNTH_SEED = 0
SYNTH_ITERS = 40


def _as_f64(array):
    return jnp.asarray(array, dtype=jnp.float64)


def objective(X, U, V, alpha1, alpha2):
    """Smooth objective H. Meaningful on the feasible set, where J = H."""
    residual = _as_f64(X) - _as_f64(U) @ _as_f64(V)
    return 0.5 * jnp.sum(residual * residual) + 0.5 * alpha1 * jnp.sum(
        _as_f64(U) * _as_f64(U)
    ) + 0.5 * alpha2 * jnp.sum(_as_f64(V) * _as_f64(V))


def reconstruction(X, U, V):
    residual = _as_f64(X) - _as_f64(U) @ _as_f64(V)
    return jnp.sqrt(jnp.sum(residual * residual))


def grad_U(X, U, V, alpha1):
    """Gradient of H with respect to U at (U, V)."""
    U = _as_f64(U)
    V = _as_f64(V)
    return (U @ V - _as_f64(X)) @ V.T + alpha1 * U


def grad_V(X, U, V, alpha2):
    """Gradient of H with respect to V at (U, V)."""
    U = _as_f64(U)
    V = _as_f64(V)
    return U.T @ (U @ V - _as_f64(X)) + alpha2 * V


def metric_diag_U(V, alpha1, eps=EPS):
    """Diagonal of M_U: row sums of V V^T, plus alpha1 + eps."""
    V = _as_f64(V)
    r = V.shape[0]
    row_sums = (V @ V.T) @ jnp.ones((r,), dtype=V.dtype)
    return row_sums + (alpha1 + eps)


def metric_diag_V(U, alpha2, eps=EPS):
    """Diagonal of M_V: row sums of U^T U, plus alpha2 + eps."""
    U = _as_f64(U)
    r = U.shape[1]
    row_sums = (U.T @ U) @ jnp.ones((r,), dtype=U.dtype)
    return row_sums + (alpha2 + eps)


def metric_U(V, alpha1, eps=EPS):
    """r by r metric for the U block, built from the current V."""
    return jnp.diag(metric_diag_U(V, alpha1, eps))


def metric_V(U, alpha2, eps=EPS):
    """r by r metric for the V block, built from the current U."""
    return jnp.diag(metric_diag_V(U, alpha2, eps))


def inner_U(A, B, M):
    """<A, B>_U = tr(A M B^T). M may be r by r or its diagonal."""
    A = _as_f64(A)
    B = _as_f64(B)
    M = _as_f64(M)
    if M.ndim == 1:
        return jnp.sum(M * jnp.sum(A * B, axis=0))
    return jnp.trace(A @ M @ B.T)


def inner_V(A, B, M):
    """<A, B>_V = tr(A^T M B). M may be r by r or its diagonal."""
    A = _as_f64(A)
    B = _as_f64(B)
    M = _as_f64(M)
    if M.ndim == 1:
        return jnp.sum(M * jnp.sum(A * B, axis=1))
    return jnp.trace(A.T @ M @ B)


def project_weighted_nonneg_l0(B, s, entry_weights=None):
    """Weighted nonnegative l0 projection.

    Keep an entry only if it is positive. Among positive entries, keep the s
    with the largest weight * value^2. Ties break toward the smaller C-order
    index. Kept entries stay at their positive value; the rest are zero.
    If fewer than s entries are positive, keep all of them.

    entry_weights broadcasts to B. The default weight is 1, which keeps the
    s largest relu(B)^2 entries.
    """
    B = _as_f64(B)
    s = int(s)
    if s < 0:
        raise ValueError("s must be nonnegative")
    if entry_weights is None:
        weights = jnp.ones((), dtype=B.dtype)
    else:
        weights = _as_f64(entry_weights)
    weights = jnp.broadcast_to(weights, B.shape)
    if s == 0 or B.size == 0:
        return jnp.zeros_like(B)

    positive = B > 0
    scores = jnp.where(positive, weights * jnp.square(B), -jnp.inf)
    flat_scores = scores.reshape(-1)
    n = int(flat_scores.shape[0])
    k = min(s, n)
    index = jnp.arange(n, dtype=jnp.int32)
    # Primary key is the last: larger score first, then smaller index.
    order = jnp.lexsort((index, -flat_scores))
    chosen = order[:k]
    chosen_scores = flat_scores[chosen]
    valid = jnp.isfinite(chosen_scores)
    keep = jnp.zeros((n,), dtype=bool).at[chosen].set(valid)
    return jnp.where(keep.reshape(B.shape), B, 0.0)


def _check_gamma(gamma):
    if not float(gamma) > 1.0:
        raise ValueError("gamma must be > 1")


def _factors(X, U, V):
    X = _as_f64(X)
    U = _as_f64(U)
    V = _as_f64(V)
    if X.ndim != 2 or U.ndim != 2 or V.ndim != 2:
        raise ValueError("X, U, and V must be matrices")
    if U.shape[0] != X.shape[0] or V.shape[1] != X.shape[1] or U.shape[1] != V.shape[0]:
        raise ValueError(
            f"shape mismatch: X {tuple(X.shape)}, U {tuple(U.shape)}, V {tuple(V.shape)}"
        )
    return X, U, V


def ggbpl_sweep(
    X,
    V,
    Y_U,
    Y_V,
    s1,
    s2,
    alpha1,
    alpha2,
    gamma=GAMMA,
    eps=EPS,
):
    """One Gauss-Seidel prox-linear sweep.

    M_U and the U gradient use the accepted V (not Y_V). M_V and the V
    gradient use the newly accepted U and the linearization point Y_V.
    """
    _check_gamma(gamma)
    X = _as_f64(X)
    V = _as_f64(V)
    Y_U = _as_f64(Y_U)
    Y_V = _as_f64(Y_V)
    sigma = 1.0 / float(gamma)

    diag_U = metric_diag_U(V, alpha1, eps)
    g_U = grad_U(X, Y_U, V, alpha1)
    B_U = Y_U - sigma * g_U / diag_U[None, :]
    U_new = project_weighted_nonneg_l0(B_U, s1, diag_U[None, :])

    diag_V = metric_diag_V(U_new, alpha2, eps)
    g_V = grad_V(X, U_new, Y_V, alpha2)
    B_V = Y_V - sigma * g_V / diag_V[:, None]
    V_new = project_weighted_nonneg_l0(B_V, s2, diag_V[:, None])
    return {
        "U": U_new,
        "V": V_new,
        "B_U": B_U,
        "B_V": B_V,
        "M_U": jnp.diag(diag_U),
        "M_V": jnp.diag(diag_V),
        "diag_U": diag_U,
        "diag_V": diag_V,
    }


def next_inertia(t):
    """Map t_k to (t_{k+1}, beta_{k+1}) with beta_{k+1} = t_k / t_{k+1}."""
    t = float(t)
    t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
    beta_next = t / t_next
    return t_next, beta_next


def iggbpl_step(
    X,
    U,
    V,
    U_prev,
    V_prev,
    beta,
    s1,
    s2,
    alpha1,
    alpha2,
    gamma=GAMMA,
    eps=EPS,
    rho1=RHO1,
):
    """One iGGBPL iteration, including the restart fallback.

    Extrapolation uses the previous accepted iterate, not the previous Y.
    On restart, beta is zeroed and the sweep is recomputed from (U, V).
    """
    X, U, V = _factors(X, U, V)
    U_prev = _as_f64(U_prev)
    V_prev = _as_f64(V_prev)
    beta = float(beta)
    Y_U = U + beta * (U - U_prev)
    Y_V = V + beta * (V - V_prev)
    candidate = ggbpl_sweep(
        X, V, Y_U, Y_V, s1, s2, alpha1, alpha2, gamma=gamma, eps=eps
    )
    d2 = inner_U(candidate["U"] - U, candidate["U"] - U, candidate["diag_U"])
    d2 = d2 + inner_V(candidate["V"] - V, candidate["V"] - V, candidate["diag_V"])
    J_old = objective(X, U, V, alpha1, alpha2)
    J_new = objective(X, candidate["U"], candidate["V"], alpha1, alpha2)
    restarted = bool(J_new > J_old - float(rho1) * d2)
    if restarted:
        candidate = ggbpl_sweep(
            X, V, U, V, s1, s2, alpha1, alpha2, gamma=gamma, eps=eps
        )
        J_new = objective(X, candidate["U"], candidate["V"], alpha1, alpha2)
    candidate = dict(candidate)
    candidate["restarted"] = restarted
    candidate["objective"] = J_new
    candidate["Y_U"] = Y_U
    candidate["Y_V"] = Y_V
    return candidate


def _history_from(U, V, objective0):
    return {
        "U_hist": [U],
        "V_hist": [V],
        "objectives": [objective0],
    }


def _pack(history, U, V, extra=None):
    result = {
        "U": U,
        "V": V,
        "objectives": jnp.stack(history["objectives"]),
        "U_hist": jnp.stack(history["U_hist"]),
        "V_hist": jnp.stack(history["V_hist"]),
    }
    if extra:
        result.update(extra)
    return result


def run_ggbpl(
    X,
    U,
    V,
    s1,
    s2,
    alpha1,
    alpha2,
    n_iter,
    gamma=GAMMA,
    eps=EPS,
):
    """GGBPL: no inertia. Y is the current accepted block.

    objectives[0] is the initial value. objectives[k] is the value after k
    full U-then-V sweeps.
    """
    X, U, V = _factors(X, U, V)
    n_iter = int(n_iter)
    history = _history_from(U, V, objective(X, U, V, alpha1, alpha2))
    for _ in range(n_iter):
        step = ggbpl_sweep(
            X, V, U, V, s1, s2, alpha1, alpha2, gamma=gamma, eps=eps
        )
        U, V = step["U"], step["V"]
        history["U_hist"].append(U)
        history["V_hist"].append(V)
        history["objectives"].append(objective(X, U, V, alpha1, alpha2))
    return _pack(history, U, V)


def run_iggbpl(
    X,
    U,
    V,
    s1,
    s2,
    alpha1,
    alpha2,
    n_iter,
    gamma=GAMMA,
    eps=EPS,
    rho1=RHO1,
):
    """iGGBPL with Nesterov inertia and the objective restart test.

    t_1 = 1, beta_1 = 0. After every accepted step, including a restarted
    step, t and beta advance as in Algorithm 2.
    """
    X, U, V = _factors(X, U, V)
    n_iter = int(n_iter)
    history = _history_from(U, V, objective(X, U, V, alpha1, alpha2))
    U_prev, V_prev = U, V
    t = 1.0
    beta = 0.0
    ts = [t]
    betas = [beta]
    n_restarts = 0
    for _ in range(n_iter):
        step = iggbpl_step(
            X,
            U,
            V,
            U_prev,
            V_prev,
            beta,
            s1,
            s2,
            alpha1,
            alpha2,
            gamma=gamma,
            eps=eps,
            rho1=rho1,
        )
        n_restarts += int(step["restarted"])
        U_prev, V_prev = U, V
        U, V = step["U"], step["V"]
        t, beta = next_inertia(t)
        ts.append(t)
        betas.append(beta)
        history["U_hist"].append(U)
        history["V_hist"].append(V)
        history["objectives"].append(objective(X, U, V, alpha1, alpha2))
    return _pack(
        history,
        U,
        V,
        extra={
            "t": jnp.asarray(ts, dtype=jnp.float64),
            "beta": jnp.asarray(betas, dtype=jnp.float64),
            "n_restarts": n_restarts,
        },
    )


def synthetic_instance(seed=SYNTH_SEED):
    """Sparse nonnegative factors, exact X = U_true V_true, projected init."""
    rng = np.random.default_rng(seed)
    m, n, r = SYNTH_M, SYNTH_N, SYNTH_R
    s1, s2 = SYNTH_S1, SYNTH_S2
    U_true = np.zeros((m, r), dtype=np.float64)
    V_true = np.zeros((r, n), dtype=np.float64)
    u_support = rng.choice(m * r, size=s1, replace=False)
    v_support = rng.choice(r * n, size=s2, replace=False)
    U_true.ravel()[u_support] = rng.uniform(0.2, 1.0, size=s1)
    V_true.ravel()[v_support] = rng.uniform(0.2, 1.0, size=s2)
    X = U_true @ V_true
    U0 = np.asarray(
        project_weighted_nonneg_l0(rng.uniform(0.05, 1.0, size=(m, r)), s1),
        dtype=np.float64,
    )
    V0 = np.asarray(
        project_weighted_nonneg_l0(rng.uniform(0.05, 1.0, size=(r, n)), s2),
        dtype=np.float64,
    )
    return {
        "X": X,
        "U0": U0,
        "V0": V0,
        "U_true": U_true,
        "V_true": V_true,
        "s1": s1,
        "s2": s2,
        "alpha1": SYNTH_ALPHA,
        "alpha2": SYNTH_ALPHA,
        "m": m,
        "n": n,
        "r": r,
    }


def run_experiment(
    n_iter=SYNTH_ITERS,
    seed=SYNTH_SEED,
    gamma=GAMMA,
    eps=EPS,
    rho1=RHO1,
):
    """Run iGGBPL and GGBPL from the same synthetic initialization."""
    instance = synthetic_instance(seed)
    common = dict(
        s1=instance["s1"],
        s2=instance["s2"],
        alpha1=instance["alpha1"],
        alpha2=instance["alpha2"],
        n_iter=n_iter,
        gamma=gamma,
        eps=eps,
    )
    inertial = run_iggbpl(
        instance["X"], instance["U0"], instance["V0"], rho1=rho1, **common
    )
    plain = run_ggbpl(instance["X"], instance["U0"], instance["V0"], **common)
    return {"instance": instance, "iggbpl": inertial, "ggbpl": plain}


def nnz(array):
    return int(jnp.sum(_as_f64(array) != 0))


def is_feasible(U, V, s1, s2):
    U = _as_f64(U)
    V = _as_f64(V)
    return bool(
        jnp.all(U >= 0)
        and jnp.all(V >= 0)
        and nnz(U) <= int(s1)
        and nnz(V) <= int(s2)
    )
