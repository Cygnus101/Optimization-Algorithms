import numpy as np

from .two_phase import two_phase


def dual(A, b, c, tol=1e-8, max_iter=1000, *, senses=None, maximize=False):
    """Form and solve the dual for a primal with nonnegative variables.

    Default primal: min c@x, Ax=b. Optional senses contains '=', '<=',
    or '>=' per row; maximize selects a maximizing primal. Dual row signs
    follow primal row senses, with free variables split into two parts.
    Return y, value, basis, history, dual_status, as before. The terminal
    history also records primal_status and dual_status separately; an
    infeasible dual does not by itself imply an unbounded primal.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    m, n = A.shape
    senses = ['='] * m if senses is None else list(senses)
    if (len(senses) != m or any(s not in ('=', '<=', '>=') for s in senses)
            or b.shape != (m,) or c.shape != (n,)):
        raise ValueError("Require matching dimensions and senses '=', '<=', '>='")
    scale = np.max(np.abs(A), axis=1)
    scale[scale == 0] = 1
    As, bs = A / scale[:, None], b / scale
    columns, free_columns = [], []
    primal_slacks = []
    identity = np.eye(m)
    for i, sense in enumerate(senses):
        e = identity[:, i]
        if sense == '=':
            columns.append(e)
            free_columns.append(-e)
        else:
            sign = 1 if sense == '>=' else -1
            columns.append((-sign if maximize else sign) * e)
            primal_slacks.append(-sign * e)
    columns.extend(free_columns)
    M = np.column_stack(columns) if columns else np.empty((m, 0))
    direction = 1 if maximize else -1
    D = np.column_stack((As.T @ M, -direction * np.eye(n)))
    cost = np.r_[direction * (bs @ M), np.zeros(n)]
    u, _, basis, history, status = two_phase(D, c, cost, tol, max_iter)
    primal_A = np.column_stack((As, *primal_slacks))
    primal_c = np.r_[(-c if maximize else c), np.zeros(len(primal_slacks))]
    primal = two_phase(primal_A, bs, primal_c, tol, max_iter)
    history[-1].update(primal_status=primal[-1], dual_status=status)
    if status != 'optimal':
        return None, None, basis, history, status
    y = (M @ u[:M.shape[1]]) / scale
    return y, float(b @ y), basis, history, status
