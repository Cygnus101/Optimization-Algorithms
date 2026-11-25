import numpy as np

from .quasi_newton import quasi_newton


def bfgs_update(H, s, y, tol=1e-8):
    """BFGS inverse update; skip nonpositive or numerically weak curvature."""
    q = s @ y
    if q <= tol * np.linalg.norm(s) * np.linalg.norm(y):
        return H.copy(), False
    V = np.eye(len(s)) - np.outer(s, y) / q
    Hn = V @ H @ V.T + np.outer(s, s) / q
    if not np.all(np.isfinite(Hn)):
        return H.copy(), False
    return (Hn + Hn.T) / 2, True


def bfgs(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None):
    """BFGS inverse-Hessian correction using the shared quasi-Newton loop."""
    return quasi_newton(f, grad, x0, bfgs_update, H0, tol, max_iter, Q)
