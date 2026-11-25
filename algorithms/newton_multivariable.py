import numpy as np

from common import line_search, state


def newton_multivariable(f, grad, hess, x0, tol=1e-8, max_iter=1000, damped=False):
    """Full Newton by default; damped=True explicitly selects strong Wolfe steps.

    Solve H d = -g instead of inverting H. No Hessian regularization or
    fallback direction is used. A non-descent direction is reported.
    """
    x = np.array(x0, dtype=float, copy=True)
    history = []
    for k in range(max_iter + 1):
        fx, g = float(f(x)), np.asarray(grad(x), dtype=float)
        history.append(state(x, fx, g, alpha=None))
        if not np.isfinite(fx) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(x)):
            status = "nonfinite"
            break
        if np.linalg.norm(g) <= tol:
            H = np.asarray(hess(x), dtype=float)
            if not np.all(np.isfinite(H)):
                status = "nonfinite"
                break
            if np.linalg.norm(H - H.T) > 1e-10 * np.linalg.norm(H):
                raise ValueError("Hessian must be symmetric")
            eig = np.linalg.eigvalsh(H)
            margin = 32 * np.finfo(float).eps * np.max(np.abs(eig), initial=0)
            negative = np.min(eig) < -margin
            history[-1]["classification"] = "not_minimum" if negative else "minimum" if np.min(eig) > margin else "undetermined"
            status = "not_minimum" if negative else "stationary"
            break
        if k == max_iter:
            status = "iteration_limit"
            break
        H = np.asarray(hess(x), dtype=float)
        if not np.all(np.isfinite(H)):
            status = "nonfinite"
            break
        if np.linalg.norm(H - H.T) > 1e-10 * np.linalg.norm(H):
            raise ValueError("Hessian must be symmetric")
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            status = "singular_hessian"
            break
        if not np.all(np.isfinite(d)):
            status = "nonfinite"
            break
        if g @ d >= 0:
            status = "non_descent"
            break
        a = line_search(f, grad, x, d) if damped else 1.0
        if a is None:
            status = "line_search_failed"
            break
        history[-1]["alpha"] = a
        xn = x + a * d
        if np.array_equal(xn, x):
            status = "stagnation"
            break
        x = xn
    return x, fx, history, status
