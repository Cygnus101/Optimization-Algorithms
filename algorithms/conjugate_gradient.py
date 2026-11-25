import numpy as np

from common import quadratic, state


def conjugate_gradient(Q, b, x0, tol=1e-8, max_iter=1000):
    """Minimize 0.5*x@Q@x-b@x by linear CG; require symmetric positive definite Q."""
    Q = np.asarray(Q, dtype=float)
    b = np.asarray(b, dtype=float)
    f, grad, _ = quadratic(Q, b)
    try:
        np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        raise ValueError("Q must be positive definite") from None
    x = np.array(x0, dtype=float, copy=True)
    g = grad(x)
    d = -g
    history = []
    for k in range(max_iter + 1):
        fx = float(f(x))
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
        q = d @ Q @ d
        relative = np.finfo(float).eps * np.linalg.norm(d) * np.linalg.norm(Q @ d)
        if not np.isfinite(q) or q <= relative:
            status = "nonpositive_curvature"
            break
        a = float((g @ g) / q)
        history[-1]["alpha"] = a
        xn = x + a * d
        if np.array_equal(xn, x):
            status = "stagnation"
            break
        gn = grad(xn)
        beta = (gn @ gn) / (g @ g)
        d = -gn + beta * d
        x, g = xn, gn
    return x, fx, history, status
