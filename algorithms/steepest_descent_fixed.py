import numpy as np

from common import state


def steepest_descent_fixed(f, grad, x0, alpha, tol=1e-8, max_iter=1000):
    """Use the supplied constant step without adapting it; return x, fx, history, status."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    x = np.array(x0, dtype=float, copy=True)
    history = []
    for k in range(max_iter + 1):
        fx, g = float(f(x)), np.asarray(grad(x), dtype=float)
        history.append(state(x, fx, g, alpha=alpha))
        if not np.isfinite(fx) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)):
            status = "nonfinite"
            break
        if np.linalg.norm(g) <= tol:
            status = "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        xn = x - alpha * g
        if np.array_equal(xn, x):
            status = "stagnation"
            break
        x = xn
    return x, fx, history, status
