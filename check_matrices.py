"""Run python check_matrices.py; add (name, Q, b) entries to cases below.

All objectives are 0.5*x@Q@x-b@x, from x0=ones. Fixed descent uses alpha=0.1
for every case, deliberately exposing step-size sensitivity. Other methods
use exact quadratic steps where applicable. 'stationary' is not a minimum
certificate on nonconvex cases. Rejections and iteration limits are expected.
"""
import numpy as np

from algorithms import (
    steepest_descent, steepest_descent_fixed, newton_multivariable,
    conjugate_gradient, quasi_newton, rank_one, dfp, bfgs,
)
from common import quadratic


cases = [
    ("SPD", np.array([[4., 1], [1, 3]]), np.array([1., 2])),
    ("Ill-conditioned SPD", np.diag([1., 1e4]), np.array([0., 9999.])),
    ("Singular PSD, bounded", np.diag([1., 0]), np.zeros(2)),
    ("Singular PSD, unbounded", np.diag([1., 0]), np.array([0., 1.])),
    ("Indefinite, unbounded", np.diag([1., -1]), np.zeros(2)),
    ("Negative definite", -np.eye(2), np.zeros(2)),
    ("Zero Q, constant", np.zeros((2, 2)), np.zeros(2)),
    ("Zero Q, unbounded", np.zeros((2, 2)), np.ones(2)),
    ("Nonsymmetric", np.array([[2., 3], [0, 2]]), np.ones(2)),
    ("Rectangular", np.ones((2, 3)), np.ones(2)),
]


def check_matrices(cases, tol=1e-6, max_iter=300):
    rows = []
    names = ["Steepest", "Fixed", "Newton", "CG", "Quasi-Newton", "SR1", "DFP", "BFGS"]
    for label, Q, b in cases:
        try:
            f, g, h = quadratic(Q, b)
        except ValueError as error:
            for name in names:
                rows.append(dict(case=label, method=name, status="invalid_input",
                                 reason=str(error), updates=None, gnorm=None, fx=None))
            continue
        x0 = np.ones(len(b))
        runs = [
            lambda: steepest_descent(f, g, x0, tol=tol, max_iter=max_iter, Q=Q),
            lambda: steepest_descent_fixed(f, g, x0, .1, tol=tol, max_iter=max_iter),
            lambda: newton_multivariable(f, g, h, x0, tol=tol, max_iter=max_iter),
            lambda: conjugate_gradient(Q, b, x0, tol=tol, max_iter=max_iter),
            lambda: quasi_newton(f, g, x0, tol=tol, max_iter=max_iter, Q=Q),
            lambda: rank_one(f, g, x0, tol=tol, max_iter=max_iter, Q=Q),
            lambda: dfp(f, g, x0, tol=tol, max_iter=max_iter, Q=Q),
            lambda: bfgs(f, g, x0, tol=tol, max_iter=max_iter, Q=Q),
        ]
        for name, run in zip(names, runs):
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    x, fx, history, status = run()
                rows.append(dict(case=label, method=name, status=status, reason="",
                                 updates=len(history)-1, gnorm=history[-1]["gnorm"], fx=fx))
            except ValueError as error:
                rows.append(dict(case=label, method=name, status="invalid_input",
                                 reason=str(error), updates=None, gnorm=None, fx=None))
    return rows


if __name__ == "__main__":
    print(f"{'Case':26} {'Method':13} {'Status':23} {'Steps':>5} {'||g||':>12} {'f(x)':>12}")
    for row in check_matrices(cases):
        steps = "-" if row["updates"] is None else str(row["updates"])
        norm = "-" if row["gnorm"] is None else f"{row['gnorm']:.4g}"
        value = "-" if row["fx"] is None else f"{row['fx']:.4g}"
        print(f"{row['case']:26} {row['method']:13} {row['status']:23} {steps:>5} {norm:>12} {value:>12}")
