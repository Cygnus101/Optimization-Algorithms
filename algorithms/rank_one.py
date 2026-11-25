import numpy as np

from .quasi_newton import quasi_newton


def sr1_update(H, s, y, tol=1e-8):
    """Inverse SR1 update; skip a small denominator relative to ||v|| ||y||."""
    v = s - H @ y
    q = v @ y
    if abs(q) <= tol * np.linalg.norm(v) * np.linalg.norm(y):
        return H.copy(), False
    Hn = H + np.outer(v, v) / q
    if not np.all(np.isfinite(Hn)):
        return H.copy(), False
    return (Hn + Hn.T) / 2, True


def rank_one(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None):
    """SR1 inverse-Hessian correction using the shared quasi-Newton loop."""
    return quasi_newton(f, grad, x0, sr1_update, H0, tol, max_iter, Q)
