import numpy as np

from common import state


def newton(f, df, ddf, x0, tol=1e-8, max_iter=1000):
    """Full scalar Newton steps for df=0; stationary need not mean minimum."""
    x = float(x0)
    history = []
    for k in range(max_iter + 1):
        fx, g, h = float(f(x)), float(df(x)), float(ddf(x))
        history.append(state(x, fx, g, h=h))
        if not np.all(np.isfinite([fx, g, h])):
            status = "nonfinite"
            break
        if abs(g) <= tol:
            history[-1]["classification"] = "maximum" if h < 0 else "minimum" if h > 0 else "undetermined"
            status = "not_minimum" if h < 0 else "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        if h == 0:
            status = "zero_curvature"
            break
        xn = x - g / h
        if xn == x:
            status = "stagnation"
            break
        if any(xn == float(previous["x"]) for previous in history[:-1]):
            status = "cycle_detected"
            break
        x = xn
    return x, fx, history, status
