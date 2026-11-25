import numpy as np

from common import state


def iterative(f, df, h, x0, tol=1e-8, max_iter=1000):
    """Iterate x=h(x) to solve df=0; the caller chooses a suitable map h."""
    x = float(x0)
    history = []
    for k in range(max_iter + 1):
        fx, g = float(f(x)), float(df(x))
        history.append(state(x, fx, g))
        if not np.all(np.isfinite([x, fx, g])):
            status = "nonfinite"
            break
        if abs(g) <= tol:
            status = "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        xn = float(h(x))
        residual = abs(float(df(xn)))
        if abs(xn - x) <= tol * max(1.0, abs(x)) and residual > tol and residual >= 0.99 * abs(g):
            status = "stagnation"
            break
        x = xn
    return x, fx, history, status
