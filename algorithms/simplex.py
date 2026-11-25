import numpy as np

from common import pivot


def simplex(A, b, c, basis, tol=1e-8, max_iter=1000):
    """Minimize c@x with Ax=b, x>=0 from a feasible basis.

    Return x, z, basis, history, status. Use Dantzig pivots, switching
    permanently to Bland's rule on a repeated basis. Terminal history
    includes original-row dual multipliers. Inputs are not modified.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    indices = np.asarray(basis)
    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    m, n = A.shape
    if (b.shape != (m,) or c.shape != (n,) or indices.shape != (m,)
            or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b))
            or not np.all(np.isfinite(c))):
        raise ValueError("Require finite A, b, c of matching dimensions")
    if (not np.isfinite(tol) or tol <= 0 or not isinstance(max_iter, (int, np.integer))
            or max_iter < 0):
        raise ValueError("Require positive tol and nonnegative integer max_iter")
    basis = np.array(indices, dtype=int, copy=True)
    if (not np.array_equal(indices, basis) or len(set(basis)) != m
            or np.any(basis < 0) or np.any(basis >= n)):
        raise ValueError("Require m distinct valid basis indices")
    scale = np.max(np.abs(A), axis=1) if n else np.ones(m)
    scale[scale == 0] = 1
    As, bs = A / scale[:, None], b / scale
    try:
        T = np.linalg.solve(As[:, basis], np.column_stack((As, bs)))
    except np.linalg.LinAlgError:
        raise ValueError("Basis must be nonsingular") from None
    eps = np.finfo(float).eps
    if np.any(T[:, -1] < -tol * (1 + np.abs(T[:, -1]))):
        raise ValueError("Initial basis must be feasible; use two_phase")
    history, seen = [], set()
    bland = False
    k = 0
    while True:
        key = tuple(basis)
        bland = bland or key in seen
        seen.add(key)
        terms = c[basis] @ T[:, :-1]
        r = c - terms
        r[basis] = 0
        rtol = tol * np.maximum(np.abs(c), np.abs(c[basis]) @ np.abs(T[:, :-1]))
        x = np.zeros(n)
        x[basis] = T[:, -1]
        z = float(c @ x)
        history.append({"T": T.copy(), "r": r.copy(), "basis": basis.copy(),
                        "x": x.copy(), "z": z, "bland": bland})
        if not np.all(np.isfinite(T)) or not np.all(np.isfinite(r)):
            status = "numerical_failure"
            break
        residual = As @ x - bs
        allowance = tol * (1 + np.abs(bs)) + 32 * eps * (np.abs(As) @ np.abs(x))
        if np.any(np.abs(residual) > allowance) or np.any(x < -tol):
            status = "numerical_failure"
            break
        candidates = np.flatnonzero(r < -rtol)
        if not len(candidates):
            status = "optimal"
            break
        q = candidates[0] if bland else candidates[np.argmin(r[candidates])]
        column = T[:, q]
        threshold = 32 * eps * np.max(np.abs(column), initial=0)
        rows = np.flatnonzero(column > threshold)
        if not len(rows):
            status = "unbounded" if np.all(column <= 0) else "numerical_failure"
            break
        if k >= max_iter:
            status = "iteration_limit"
            break
        ratios = np.maximum(T[rows, -1], 0) / column[rows]
        minimum = np.min(ratios)
        tied = rows[np.abs(ratios - minimum) <= 8 * eps * abs(minimum)]
        p = tied[np.argmin(basis[tied])]
        T = pivot(T, p, q)
        basis[p] = q
        k += 1
    y = np.linalg.solve(As[:, basis].T, c[basis]) / scale
    history[-1]["duals"] = y
    return x, z, basis, history, status
