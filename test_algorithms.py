"""Run with python test_algorithms.py. Only NumPy is needed."""
import numpy as np

from algorithms import (
    simplex, two_phase, dual, bisection, newton, iterative,
    steepest_descent, steepest_descent_fixed, newton_multivariable,
    conjugate_gradient, quasi_newton, rank_one, dfp, bfgs,
)
from algorithms.rank_one import sr1_update
from algorithms.dfp import dfp_update
from algorithms.bfgs import bfgs_update
from common import line_search, pivot, quadratic


def rejects(run):
    try:
        run()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def check_history(result, f, grad):
    x, fx, history, status = result
    np.testing.assert_allclose(fx, f(x))
    for h in history:
        np.testing.assert_allclose(h["fx"], f(h["x"]), atol=1e-10)
        np.testing.assert_allclose(h["g"], grad(h["x"]), atol=1e-10)
    for key in ("x", "g"):
        for i, h in enumerate(history):
            assert all(not np.shares_memory(h[key], prev[key]) for prev in history[:i])
    if "H" in history[0]:
        for left, right in zip(history[:-1], history[1:]):
            assert not np.shares_memory(left["H"], right["H"])
    if isinstance(x, np.ndarray):
        assert not np.shares_memory(x, history[-1]["x"])
    return status


def test_lp():
    A = np.array([[1., 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]])
    b = np.array([4., 6, 8])
    c = np.array([-2., -5, 0, 0, 0])
    before = [v.copy() for v in (A, b, c)]
    x, z, basis, H, status = simplex(A, b, c, [2, 3, 4])
    assert status == "optimal"
    np.testing.assert_allclose(x, [2, 6, 2, 0, 0])
    np.testing.assert_allclose([h["z"] for h in H], [0, -30, -34])
    for scale in (1e-8, 1., 1e8):
        p = two_phase(A * scale, b * scale, c)
        d = dual(A * scale, b * scale, c)
        assert p[-1] == d[-1] == "optimal"
        np.testing.assert_allclose(p[1], -34)
        np.testing.assert_allclose(p[1], d[1])
        np.testing.assert_allclose(A @ p[0], b, atol=1e-7)
        assert np.all((A * scale).T @ d[0] <= c + 1e-7)
        np.testing.assert_allclose(p[0] * (c - (A * scale).T @ d[0]), 0, atol=1e-7)
    for v, old in zip((A, b, c), before):
        np.testing.assert_array_equal(v, old)
    assert two_phase([[1], [1]], [1, 2], [1])[-1] == "infeasible"
    r = two_phase([[1, 1], [2, 2]], [1, 2], [1, 2])
    assert r[-1] == "optimal" and len(r[2]) == 1
    assert two_phase([[-1]], [0], [1])[-1] == "optimal"
    assert two_phase([[0, 0]], [0], [1, 2])[-1] == "optimal"
    assert two_phase([[0]], [1], [1])[-1] == "infeasible"
    assert two_phase([[1, -1]], [0], [-1, 0])[-1] == "unbounded"
    assert two_phase([[2, 1]], [2], [3, 1], max_iter=0)[-1] == "iteration_limit"
    assert dual([[1], [1]], [1, 2], [1])[-1] == "unbounded"
    assert dual([[1, -1]], [0], [-1, 0])[-1] == "infeasible"
    T = np.array([[2., 1, 4], [1, 1, 3]])
    old = T.copy()
    np.testing.assert_allclose(pivot(T, 0, 0)[:, 0], [1, 0])
    np.testing.assert_array_equal(T, old)


def test_scalar():
    f = lambda x: x**4 + 2 * x**2 - 8 * x + 3
    df = lambda x: 4 * x**3 + 4 * x - 8
    ddf = lambda x: 12 * x**2 + 4
    h = lambda x: np.cbrt(2 - x)
    runs = [bisection(f, df, 0, 3), newton(f, df, ddf, 3), iterative(f, df, h, 0)]
    for r in runs:
        assert check_history(r, f, df) in ("stationary", "bracket_converged")
        np.testing.assert_allclose(r[0], 1, atol=1e-7)
        np.testing.assert_allclose(r[1], -2)
    rejects(lambda: bisection(f, df, 2, 3))
    rejects(lambda: bisection(f, df, 3, 0))
    assert bisection(f, df, 1, 3)[0] == 1
    assert newton(lambda x: x**4 + x, lambda x: 4*x**3 + 1,
                  lambda x: 12*x**2, 0)[-1] == "zero_curvature"
    assert newton(f, df, ddf, 1, max_iter=0)[-1] == "stationary"
    assert newton(f, df, ddf, 3, max_iter=0)[-1] == "iteration_limit"
    assert iterative(lambda x: x*x/2, lambda x: x, lambda x: -2*x, 1,
                     max_iter=10)[-1] == "iteration_limit"
    assert iterative(f, df, lambda x: x, 0)[-1] == "stagnation"
    assert newton(lambda x: -x*x, lambda x: -2*x, lambda x: -2, 1)[-1] == "not_minimum"


def test_quadratics():
    rng = np.random.default_rng(42)
    count = 0
    for n in (1, 2, 4, 8):
        for cond in (1., 10., 100.):
            U, _ = np.linalg.qr(rng.normal(size=(n, n)))
            Q = U @ np.diag(np.geomspace(1., cond, n)) @ U.T
            b = rng.normal(size=n)
            x0 = rng.normal(size=n)
            H0 = np.eye(n)
            before = [v.copy() for v in (Q, b, x0, H0)]
            f, g, h = quadratic(Q, b)
            target = np.linalg.solve(Q, b)
            runs = [
                steepest_descent(f, g, x0, tol=1e-6, max_iter=3000, Q=Q),
                steepest_descent_fixed(f, g, x0, 1/cond, tol=1e-6, max_iter=3000),
                newton_multivariable(f, g, h, x0, tol=1e-6),
                conjugate_gradient(Q, b, x0, tol=1e-6),
                quasi_newton(f, g, x0, H0=H0, tol=1e-6, Q=Q),
                rank_one(f, g, x0, H0=H0, tol=1e-6, Q=Q),
                dfp(f, g, x0, H0=H0, tol=1e-6, Q=Q),
                bfgs(f, g, x0, H0=H0, tol=1e-6, Q=Q),
            ]
            for r in runs:
                assert check_history(r, f, g) == "stationary", (n, cond, r[-1])
                np.testing.assert_allclose(r[0], target, atol=2e-6, rtol=2e-6)
                count += 1
            for v, old in zip((Q, b, x0, H0), before):
                np.testing.assert_array_equal(v, old)
    print(f"Passed {count} solves on rotated positive-definite quadratics.")
    for scale in (1e-8, 1e8):
        Q = scale * np.array([[4., 1], [1, 3]])
        b = scale * np.array([1., 2.])
        f, g, h = quadratic(Q, b)
        tol = scale * 1e-7
        runs = [steepest_descent(f, g, [3, -2], tol=tol, Q=Q),
                newton_multivariable(f, g, h, [3, -2], tol=tol),
                conjugate_gradient(Q, b, [3, -2], tol=tol),
                rank_one(f, g, [3, -2], tol=tol, Q=Q),
                dfp(f, g, [3, -2], tol=tol, Q=Q),
                bfgs(f, g, [3, -2], tol=tol, Q=Q)]
        for r in runs:
            assert r[-1] == "stationary"
            np.testing.assert_allclose(r[0], np.linalg.solve(Q, b), atol=1e-6)


def test_breakdowns():
    x0 = np.array([1., 1.])
    for Q in (np.diag([1., 0]), np.diag([1., -1]), -np.eye(2), np.zeros((2, 2))):
        rejects(lambda: conjugate_gradient(Q, np.zeros(2), x0))
    rejects(lambda: quadratic([[2, 3], [0, 2]]))
    rejects(lambda: conjugate_gradient(np.eye(2), [1], x0))
    f, g, h = quadratic(np.diag([1., 0]))
    assert newton_multivariable(f, g, h, x0)[-1] == "singular_hessian"
    assert steepest_descent(f, g, x0, Q=np.diag([1., 0]))[-1] == "stationary"
    f, g, h = quadratic(-np.eye(2))
    assert newton_multivariable(f, g, h, x0)[-1] == "non_descent"
    assert steepest_descent(f, g, x0, Q=-np.eye(2))[-1] == "nonpositive_curvature"
    f, g, h = quadratic(np.eye(2))
    assert steepest_descent_fixed(f, g, x0, 3, max_iter=10)[-1] == "iteration_limit"
    rejects(lambda: steepest_descent_fixed(f, g, x0, -1))
    rejects(lambda: bfgs(f, g, x0, H0=-np.eye(2)))
    for run in [lambda: steepest_descent(f, g, x0, max_iter=0),
                lambda: steepest_descent_fixed(f, g, x0, .1, max_iter=0),
                lambda: newton_multivariable(f, g, h, x0, max_iter=0),
                lambda: conjugate_gradient(np.eye(2), np.zeros(2), x0, max_iter=0),
                lambda: bfgs(f, g, x0, max_iter=0)]:
        assert run()[-1] == "iteration_limit"
    assert bfgs(f, g, [0, 0], max_iter=0)[-1] == "stationary"
    assert steepest_descent(lambda x: np.nan, g, x0)[-1] == "nonfinite"
    assert line_search(lambda x: -x[0], lambda x: np.array([-1.]), np.zeros(1), np.ones(1)) is None
    H = np.eye(2)
    for update in (sr1_update, dfp_update, bfgs_update):
        old = H.copy()
        Hn, applied = update(H, np.zeros(2), np.zeros(2))
        assert not applied
        np.testing.assert_array_equal(H, old)
        assert not np.shares_memory(Hn, H)
    s, y = np.array([1., 2]), np.array([2., 3])
    for update in (sr1_update, dfp_update, bfgs_update):
        Hn, applied = update(H, s, y)
        assert applied
        np.testing.assert_allclose(Hn @ y, s, atol=1e-12)
        np.testing.assert_allclose(Hn, Hn.T)


def test_general_functions():
    f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    g = lambda x: np.array([2*(x[0]-1)-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
    h = lambda x: np.array([[2-400*x[1]+1200*x[0]**2, -400*x[0]], [-400*x[0], 200]])
    x0 = np.array([-1.2, 1.])
    for r in (bfgs(f, g, x0, tol=1e-6), dfp(f, g, x0, tol=1e-6),
              newton_multivariable(f, g, h, x0, damped=True),
              steepest_descent(f, g, x0, tol=1e-4, max_iter=10000)):
        assert check_history(r, f, g) == "stationary"
        np.testing.assert_allclose(r[0], [1, 1], atol=3e-4)
    sr1 = rank_one(f, g, x0)
    assert check_history(sr1, f, g) == "stationary"
    np.testing.assert_allclose(sr1[0], [1, 1], atol=3e-4)
    f, g, _ = quadratic([[4, 1], [1, 3]], [1, 2])
    x = np.array([3., -2.])
    d = -g(x)
    a = line_search(f, g, x, d)
    assert a is not None
    assert f(x+a*d) <= f(x) + 1e-4*a*(g(x)@d)
    assert abs(g(x+a*d)@d) <= .9*abs(g(x)@d)


def test_matrix_cases():
    from check_matrices import cases, check_matrices

    rows = check_matrices(cases)
    assert len(rows) == 80
    table = {(r["case"], r["method"]): r for r in rows}
    for method in ("Steepest", "Fixed", "Newton", "CG", "Quasi-Newton", "SR1", "DFP", "BFGS"):
        assert table["SPD", method]["status"] == "stationary"
        assert table["Rectangular", method]["status"] == "invalid_input"
        assert table["Nonsymmetric", method]["status"] == "invalid_input"
    assert table["Ill-conditioned SPD", "Steepest"]["status"] == "iteration_limit"
    assert table["Ill-conditioned SPD", "CG"]["status"] == "stationary"
    assert table["Singular PSD, bounded", "Newton"]["status"] == "singular_hessian"
    assert table["Singular PSD, bounded", "Steepest"]["status"] == "stationary"
    assert table["Indefinite, unbounded", "CG"]["status"] == "invalid_input"
    assert table["Negative definite", "Newton"]["status"] == "non_descent"
    print("Passed 80 matrix-case checks, including expected failures.")


if __name__ == "__main__":
    test_lp()
    test_scalar()
    test_quadratics()
    test_breakdowns()
    test_general_functions()
    test_matrix_cases()
    print("All algorithm tests passed, including expected failures.")
