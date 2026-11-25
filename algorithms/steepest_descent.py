import numpy as np

from common import line_search, quadratic, quadratic_step, state


def steepest_descent(f, grad, x0, tol=1e-8, max_iter=1000, Q=None):
    """Exact line steps when Q is supplied for a quadratic; otherwise strong Wolfe.

    Return x, f(x), history, status. 'stationary' only certifies a small gradient.
    Q must be the symmetric Hessian of the supplied objective.
    """
    x = np.array(x0, dtype=float, copy=True)
    if Q is not None:
        Q = np.asarray(Q, dtype=float)
        quadratic(Q)
        if Q.shape != (len(x), len(x)):
            raise ValueError("Q must match x0")
    history = []
    for k in range(max_iter + 1):
        fx, g = float(f(x)), np.asarray(grad(x), dtype=float)
        history.append(state(x, fx, g, alpha=None))
        if not np.isfinite(fx) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)):
            status = "nonfinite"
            break
        if np.linalg.norm(g) <= tol:
            status = "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        d = -g
        a = quadratic_step(g, d, Q) if Q is not None else line_search(f, grad, x, d)
        if a is None:
            status = "nonpositive_curvature" if Q is not None else "line_search_failed"
            break
        history[-1]["alpha"] = a
        xn = x + a * d
        if np.array_equal(xn, x):
            status = "stagnation"
            break
        x = xn
    return x, fx, history, status
