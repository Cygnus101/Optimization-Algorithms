import numpy as np

from common import line_search, quadratic, quadratic_step, state


def quasi_newton(f, grad, x0, update=None, H0=None, tol=1e-8, max_iter=1000, Q=None):
    """Shared inverse-Hessian iteration, defaulting to the BFGS update.

    update(H,s,y) returns (new_H, applied). Unsafe updates are skipped and
    recorded. Non-descent directions reset H to the identity.
    Q selects exact quadratic line steps; otherwise use strong Wolfe.
    Return x, f(x), history, status, with independent H snapshots.
    """
    if update is None:
        from .bfgs import bfgs_update
        update = bfgs_update
    x = np.array(x0, dtype=float, copy=True)
    H = np.eye(len(x)) if H0 is None else np.array(H0, dtype=float, copy=True)
    quadratic(H)
    if H.shape != (len(x), len(x)):
        raise ValueError("H0 must match x0")
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        raise ValueError("H0 must be positive definite") from None
    if Q is not None:
        Q = np.asarray(Q, dtype=float)
        quadratic(Q)
        if Q.shape != H.shape:
            raise ValueError("Q must match x0")
    history = []
    for k in range(max_iter + 1):
        fx, g = float(f(x)), np.asarray(grad(x), dtype=float)
        history.append(state(x, fx, g, H=H.copy(), alpha=None, update=None))
        if not np.isfinite(fx) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(H)):
            status = "nonfinite"
            break
        if np.linalg.norm(g) <= tol:
            status = "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        d = -H @ g
        if g @ d >= -np.sqrt(np.finfo(float).eps) * np.linalg.norm(g) * np.linalg.norm(d):
            H = np.eye(len(x))
            d = -g
            history[-1]["direction"] = "steepest_fallback"
        a = quadratic_step(g, d, Q) if Q is not None else line_search(f, grad, x, d)
        if a is None and Q is None:
            H = np.eye(len(x))
            d = -g
            a = line_search(f, grad, x, d)
            history[-1]["direction"] = "steepest_fallback"
        if a is None:
            status = "nonpositive_curvature" if Q is not None else "line_search_failed"
            break
        history[-1]["alpha"] = a
        s = a * d
        xn = x + s
        if np.array_equal(xn, x):
            status = "stagnation"
            break
        gn = np.asarray(grad(xn), dtype=float)
        H, applied = update(H, s, gn - g)
        history[-1]["update"] = "applied" if applied else "skipped"
        x = xn
    return x, fx, history, status
