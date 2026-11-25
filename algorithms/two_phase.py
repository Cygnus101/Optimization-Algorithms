import numpy as np

from common import pivot
from .simplex import simplex


def two_phase(A, b, c, tol=1e-8, max_iter=1000):
    """Minimize c @ x subject to A @ x = b, x >= 0, without a given basis.

    Inputs are not modified. Convert inequalities to equalities beforehand.
    Rows are normalized by their largest coefficient before Phase I.
    Return x, z, basis, history, status; x and z are None if Phase I fails.
    Status is 'optimal', 'infeasible', 'unbounded' or 'iteration_limit'.
    max_iter is the pivot limit for each simplex phase separately.
    History entries have phase (1 or 2) and stage ('simplex' or 'cleanup').
    Phase I x and r include artificial variables; its z is their sum.
    Phase II uses only original variables and the original objective.
    Terminal Phase II history stores duals for every original row, including
    redundant rows. On Phase I failure, basis refers to augmented columns.
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    c = np.asarray(c, dtype=float)
    m, n = A.shape
    original_A, original_b = A.copy(), b.copy()
    scale = np.max(np.abs(A), axis=1)
    scale[scale == 0] = 1
    A = A / scale[:, None]
    b = b / scale
    rows = b < 0
    row_map = np.where(rows, -1.0, 1.0) / scale
    A[rows] *= -1
    b[rows] *= -1

    A1 = np.column_stack((A, np.eye(m)))
    c1 = np.r_[np.zeros(n), np.ones(m)]
    basis = np.arange(n, n + m)
    _, w, basis, history, status = simplex(A1, b, c1, basis, tol, max_iter)
    for h in history:
        h.update(phase=1, stage="simplex")
    if status != "optimal":
        return None, None, basis, history, status
    if w > tol:
        return None, None, basis, history, "infeasible"

    T = history[-1]["T"].copy()
    i = 0
    while i < len(basis):
        if basis[i] < n:
            i += 1
            continue
        threshold = 32 * np.finfo(float).eps * np.max(np.abs(T[i, :n]), initial=0)
        cols = [j for j in range(n) if j not in basis and abs(T[i, j]) > threshold]
        T[i, -1] = 0
        if cols:
            q = cols[0]
            T = pivot(T, i, q)
            basis[i] = q
            i += 1
        else:
            T = np.delete(T, i, axis=0)
            basis = np.delete(basis, i)

        x = np.zeros(n + m)
        x[basis] = T[:, -1]
        r = c1 - c1[basis] @ T[:, :-1]
        history.append({"T": T.copy(), "r": r.copy(),
                        "basis": basis.copy(), "x": x.copy(),
                        "z": float(c1 @ x), "phase": 1, "stage": "cleanup"})

    A2 = T[:, :n]
    b2 = T[:, -1]
    x, z, basis, h2, status = simplex(A2, b2, c, basis, tol, max_iter)
    for h in h2:
        h.update(phase=2, stage="simplex")
    transform = T[:, n:n + m] * row_map[None, :]
    duals = transform.T @ h2[-1]["duals"]
    h2[-1]["duals"] = duals
    if status == "optimal":
        residual = (original_A @ x - original_b) / scale
        allowance = tol * (1 + np.abs(original_b / scale))
        allowance += 32 * np.finfo(float).eps * (np.abs(original_A / scale[:, None]) @ np.abs(x))
        reduced = c - original_A.T @ duals
        dual_allowance = tol * np.maximum(np.abs(c), np.abs(original_A).T @ np.abs(duals))
        dual_allowance += 32 * np.finfo(float).eps * (
            np.abs(c) + np.linalg.norm(original_A, axis=0) * np.linalg.norm(duals))
        gap = abs(c @ x - original_b @ duals)
        gap_allowance = tol * (1 + abs(c @ x) + abs(original_b @ duals))
        if (np.any(np.abs(residual) > allowance) or np.any(x < -tol)
                or np.any(reduced < -dual_allowance) or gap > gap_allowance):
            status = "numerical_failure"
    history.extend(h2)
    return x, z, basis, history, status
