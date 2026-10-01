"""Tests for the l0-SNMF GGBPL kernel."""

import math
from itertools import combinations

import numpy as np
import pytest

from ggbpl.snmf import (
    GAMMA,
    RHO1,
    ggbpl_sweep,
    grad_U,
    grad_V,
    iggbpl_step,
    inner_U,
    inner_V,
    is_feasible,
    metric_U,
    metric_V,
    next_inertia,
    objective,
    project_weighted_nonneg_l0,
    reconstruction,
    run_experiment,
    run_ggbpl,
    run_iggbpl,
)


def _brute_project(B, weights, s):
    B = np.asarray(B, dtype=float)
    W = np.broadcast_to(np.asarray(weights, dtype=float), B.shape).copy()
    positions = [p for p in np.ndindex(*B.shape) if B[p] > 0]
    best = None
    best_cost = np.inf
    kmax = min(s, len(positions))
    base = np.sum(W * B * B)
    for k in range(kmax + 1):
        for subset in combinations(positions, k):
            cost = base
            for p in subset:
                cost -= W[p] * B[p] * B[p]
            if cost < best_cost - 1e-12:
                best_cost = cost
                out = np.zeros_like(B)
                for p in subset:
                    out[p] = B[p]
                best = out
    return best, best_cost


def test_identity_metric_keeps_largest_relu_squares():
    B = np.array([[1.0, -2.0, 0.4], [0.3, 0.9, -0.1]])
    out = np.asarray(project_weighted_nonneg_l0(B, 3))
    expected = np.array([[1.0, 0.0, 0.4], [0.0, 0.9, 0.0]])
    np.testing.assert_allclose(out, expected)


def test_keeps_all_positives_when_fewer_than_s():
    B = np.array([[1.0, -1.0], [0.0, 0.5]])
    out = np.asarray(project_weighted_nonneg_l0(B, 10))
    np.testing.assert_allclose(out, np.array([[1.0, 0.0], [0.0, 0.5]]))


def test_tie_breaks_toward_smaller_index():
    B = np.array([[1.0, 1.0], [1.0, 0.0]])
    out = np.asarray(project_weighted_nonneg_l0(B, 1))
    np.testing.assert_allclose(out, np.array([[1.0, 0.0], [0.0, 0.0]]))


def test_weighted_projection_matches_brute_force():
    B = np.array([[0.5, -1.0, 0.2], [0.1, 0.4, 0.3]])
    column_weights = np.array([1.0, 2.0, 0.5])
    entry_w = np.broadcast_to(column_weights, B.shape)
    out = np.asarray(project_weighted_nonneg_l0(B, 2, entry_w))
    brute, _ = _brute_project(B, entry_w, 2)
    np.testing.assert_allclose(out, brute)
    np.testing.assert_allclose(out, np.array([[0.5, 0.0, 0.0], [0.0, 0.4, 0.0]]))

    row_weights = np.array([0.3, 4.0])
    entry_rows = np.broadcast_to(row_weights[:, None], B.shape)
    out_rows = np.asarray(project_weighted_nonneg_l0(B, 2, entry_rows))
    brute_rows, _ = _brute_project(B, entry_rows, 2)
    np.testing.assert_allclose(out_rows, brute_rows)

    rng = np.random.default_rng(4)
    for _ in range(6):
        trial = rng.normal(size=(2, 3))
        weights = rng.uniform(0.2, 3.0, size=(2, 3))
        got = np.asarray(project_weighted_nonneg_l0(trial, 2, weights))
        ref, ref_cost = _brute_project(trial, weights, 2)
        got_cost = np.sum(weights * (got - trial) ** 2)
        np.testing.assert_allclose(got_cost, ref_cost, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(got, ref)


def test_metrics_are_row_sum_diagonals():
    V = np.array([[0.2, 0.0, 0.5], [0.1, 0.4, 0.0]])
    alpha1 = 0.01
    eps = 1e-8
    M_U = np.asarray(metric_U(V, alpha1, eps))
    expected_U = np.diag((V @ V.T) @ np.ones(2)) + (alpha1 + eps) * np.eye(2)
    np.testing.assert_allclose(M_U, expected_U)

    U = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    alpha2 = 0.3
    M_V = np.asarray(metric_V(U, alpha2, eps))
    gram = U.T @ U
    expected_V = np.diag(gram @ np.ones(2)) + (alpha2 + eps) * np.eye(2)
    np.testing.assert_allclose(M_V, expected_V)
    assert M_V.shape == (2, 2)
    assert np.all(np.diag(M_U) > 0)
    assert np.all(np.diag(M_V) > 0)


def test_inner_products_match_trace_form():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(4, 3))
    M = np.diag(rng.uniform(0.5, 2.0, size=3))
    got = float(inner_U(A, B, M))
    np.testing.assert_allclose(got, np.trace(A @ M @ B.T))
    C = rng.normal(size=(3, 5))
    D = rng.normal(size=(3, 5))
    got_V = float(inner_V(C, D, M))
    np.testing.assert_allclose(got_V, np.trace(C.T @ M @ D))


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 5))
    U = rng.random((4, 2))
    V = rng.random((2, 5))
    alpha1, alpha2 = 0.01, 0.02
    step = 1e-6
    num_U = np.zeros_like(U)
    for idx in np.ndindex(U.shape):
        up = U.copy()
        um = U.copy()
        up[idx] += step
        um[idx] -= step
        num_U[idx] = (
            float(objective(X, up, V, alpha1, alpha2))
            - float(objective(X, um, V, alpha1, alpha2))
        ) / (2 * step)
    np.testing.assert_allclose(np.asarray(grad_U(X, U, V, alpha1)), num_U, rtol=1e-5, atol=1e-6)

    num_V = np.zeros_like(V)
    for idx in np.ndindex(V.shape):
        vp = V.copy()
        vm = V.copy()
        vp[idx] += step
        vm[idx] -= step
        num_V[idx] = (
            float(objective(X, U, vp, alpha1, alpha2))
            - float(objective(X, U, vm, alpha1, alpha2))
        ) / (2 * step)
    np.testing.assert_allclose(np.asarray(grad_V(X, U, V, alpha2)), num_V, rtol=1e-5, atol=1e-6)


def test_sweep_matches_closed_form_B():
    X = np.array([[1.0, 0.5, 0.2], [0.2, 1.0, 0.0], [0.0, 0.3, 0.7]])
    U = np.array([[0.4, 0.1], [0.0, 0.5], [0.2, 0.0]])
    V = np.array([[0.3, 0.0, 0.2], [0.1, 0.4, 0.0]])
    alpha1, alpha2 = 0.01, 0.02
    eps = 1e-8
    gamma = 2.0
    sigma = 1.0 / gamma
    M_U = np.diag((V @ V.T) @ np.ones(2)) + (alpha1 + eps) * np.eye(2)
    g_U = (U @ V - X) @ V.T + alpha1 * U
    B_U = U - sigma * g_U @ np.linalg.inv(M_U)
    step = ggbpl_sweep(X, V, U, V, 4, 4, alpha1, alpha2, gamma=gamma, eps=eps)
    np.testing.assert_allclose(np.asarray(step["B_U"]), B_U, rtol=1e-12, atol=1e-12)
    U_new = np.asarray(step["U"])
    M_V = np.diag((U_new.T @ U_new) @ np.ones(2)) + (alpha2 + eps) * np.eye(2)
    g_V = U_new.T @ (U_new @ V - X) + alpha2 * V
    B_V = V - sigma * np.linalg.inv(M_V) @ g_V
    np.testing.assert_allclose(np.asarray(step["B_V"]), B_V, rtol=1e-12, atol=1e-12)
    assert is_feasible(step["U"], step["V"], 4, 4)


def test_inertia_schedule_golden_ratio():
    t1 = 1.0
    beta1 = 0.0
    t2, beta2 = next_inertia(t1)
    assert beta1 == 0.0
    assert t2 == pytest.approx((1.0 + math.sqrt(5.0)) / 2.0)
    assert beta2 == pytest.approx(t1 / t2)
    assert t2 * t2 == pytest.approx(t2 + 1.0)
    t3, beta3 = next_inertia(t2)
    assert t3 == pytest.approx((1.0 + math.sqrt(1.0 + 4.0 * t2 * t2)) / 2.0)
    assert beta3 == pytest.approx(t2 / t3)


def test_iggbpl_records_schedule_and_matches_ggbpl_at_beta0():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(5, 4))
    U = np.asarray(project_weighted_nonneg_l0(rng.random((5, 2)), 6))
    V = np.asarray(project_weighted_nonneg_l0(rng.random((2, 4)), 5))
    inertial = run_iggbpl(X, U, V, 6, 5, 1e-2, 1e-2, n_iter=1)
    plain = run_ggbpl(X, U, V, 6, 5, 1e-2, 1e-2, n_iter=1)
    t2 = (1.0 + math.sqrt(5.0)) / 2.0
    assert float(inertial["t"][0]) == pytest.approx(1.0)
    assert float(inertial["beta"][0]) == pytest.approx(0.0)
    assert float(inertial["t"][1]) == pytest.approx(t2)
    assert float(inertial["beta"][1]) == pytest.approx(1.0 / t2)
    np.testing.assert_allclose(np.asarray(inertial["U"]), np.asarray(plain["U"]))
    np.testing.assert_allclose(np.asarray(inertial["V"]), np.asarray(plain["V"]))
    assert inertial["objectives"].shape == (2,)


def test_restart_recomputes_from_old_point():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5, 4))
    U = np.asarray(project_weighted_nonneg_l0(rng.random((5, 2)), 6))
    V = np.asarray(project_weighted_nonneg_l0(rng.random((2, 4)), 5))
    fallback = ggbpl_sweep(X, V, U, V, 6, 5, 1e-2, 1e-2)
    saw_restart = False
    for beta in (1.0, 2.0, 5.0, 10.0, 20.0):
        step = iggbpl_step(
            X,
            U,
            V,
            np.zeros_like(U),
            np.zeros_like(V),
            beta,
            6,
            5,
            1e-2,
            1e-2,
            rho1=RHO1,
        )
        if step["restarted"]:
            saw_restart = True
            np.testing.assert_allclose(np.asarray(step["U"]), np.asarray(fallback["U"]))
            np.testing.assert_allclose(np.asarray(step["V"]), np.asarray(fallback["V"]))
    assert saw_restart


def test_random_feasible_objectives_do_not_increase():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6, 5))
    U = np.asarray(project_weighted_nonneg_l0(rng.random((6, 3)), 8))
    V = np.asarray(project_weighted_nonneg_l0(rng.random((3, 5)), 7))
    plain = run_ggbpl(X, U, V, 8, 7, 1e-2, 1e-2, n_iter=12, gamma=GAMMA)
    inertial = run_iggbpl(X, U, V, 8, 7, 1e-2, 1e-2, n_iter=12, gamma=GAMMA)
    plain_obj = np.asarray(plain["objectives"])
    inertial_obj = np.asarray(inertial["objectives"])
    assert np.all(np.diff(plain_obj) <= 1e-9)
    assert np.all(np.diff(inertial_obj) <= 1e-9)
    for k in range(plain["U_hist"].shape[0]):
        assert is_feasible(plain["U_hist"][k], plain["V_hist"][k], 8, 7)
        assert is_feasible(inertial["U_hist"][k], inertial["V_hist"][k], 8, 7)


def test_synthetic_experiment_decreases():
    result = run_experiment()
    instance = result["instance"]
    inertial = result["iggbpl"]
    plain = result["ggbpl"]
    X = instance["X"]
    s1, s2 = instance["s1"], instance["s2"]
    alpha1, alpha2 = instance["alpha1"], instance["alpha2"]
    j0 = float(inertial["objectives"][0])
    j_i = float(inertial["objectives"][-1])
    j_g = float(plain["objectives"][-1])
    assert j_i < j0
    assert j_g < j0
    recon0 = float(reconstruction(X, instance["U0"], instance["V0"]))
    recon_i = float(reconstruction(X, inertial["U"], inertial["V"]))
    assert recon_i < recon0
    assert inertial["objectives"].shape[0] == 41
    for k in range(inertial["U_hist"].shape[0]):
        assert is_feasible(inertial["U_hist"][k], inertial["V_hist"][k], s1, s2)
        assert is_feasible(plain["U_hist"][k], plain["V_hist"][k], s1, s2)
    assert float(objective(X, inertial["U"], inertial["V"], alpha1, alpha2)) == pytest.approx(j_i)
