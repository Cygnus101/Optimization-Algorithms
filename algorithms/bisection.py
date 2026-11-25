import numpy as np

from common import state


def bisection(f, df, a, b, tol=1e-8, max_iter=1000):
    """Bisect df=0 on [a,b]; return x, f(x), history, status.

    Require continuous df and df(a)<=0<=df(b); convexity makes the root
    a minimum. 'bracket_converged' means interval accuracy, not small df.
    """
    a, b = float(a), float(b)
    ga, gb = float(df(a)), float(df(b))
    if not a < b or not np.all(np.isfinite([a, b, ga, gb])):
        raise ValueError("Require finite a < b and finite endpoint derivatives")
    if ga > 0 or gb < 0:
        raise ValueError("Require df(a) <= 0 <= df(b)")
    history = []
    for k in range(max_iter + 1):
        x = a if ga == 0 else b if gb == 0 else a + (b - a) / 2
        fx, g = float(f(x)), float(df(x))
        history.append(state(x, fx, g, a=a, b=b))
        if not np.all(np.isfinite([fx, g])):
            status = "nonfinite"
            break
        if abs(g) <= tol:
            status = "stationary"
            break
        if (b - a) / 2 <= tol:
            status = "bracket_converged"
            break
        if x == a or x == b:
            status = "stagnation"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        if g < 0:
            a, ga = x, g
        else:
            b, gb = x, g
    return x, fx, history, status
