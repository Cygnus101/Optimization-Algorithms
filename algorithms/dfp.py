import numpy as np

from .quasi_newton import quasi_newton


def dfp_update(H, s, y, tol=1e-8):
    """DFP inverse update; skip unsafe curvature or a small y.T@H@y."""
    v = H @ y
    p, q = s @ y, y @ v
    if p <= tol * np.linalg.norm(s) * np.linalg.norm(y):
        return H.copy(), False
    if q <= tol * np.linalg.norm(y) * np.linalg.norm(v):
        return H.copy(), False
    Hn = H + np.outer(s, s) / p - np.outer(v, v) / q
    if not np.all(np.isfinite(Hn)):
        return H.copy(), False
    return (Hn + Hn.T) / 2, True


def dfp(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None):
    """DFP inverse-Hessian correction using the shared quasi-Newton loop."""
    return quasi_newton(f, grad, x0, dfp_update, H0, tol, max_iter, Q)
