"""Targeted numerical regressions; run with python test_regressions.py."""
import numpy as np

from algorithms import (
    simplex, two_phase, dual, bisection, newton, iterative,
    steepest_descent, steepest_descent_fixed, newton_multivariable,
    conjugate_gradient, quasi_newton, rank_one, dfp, bfgs,
)
from algorithms.rank_one import sr1_update
from algorithms.dfp import dfp_update
from algorithms.bfgs import bfgs_update
from common import quadratic, line_search


def test_simplex():
    G = np.array([[.5, -5.5, -2.5, 9], [.5, -1.5, -.5, 1], [1, 0, 0, 0]])
    A = np.column_stack((G, np.eye(3)))
    c = np.r_[[-10., 57, 9, 24], np.zeros(3)]
    r = simplex(A, [0, 0, 1], c, [4, 5, 6], max_iter=100)
    assert r[-1] == 'optimal'
    np.testing.assert_allclose(r[1], -1)
    assert any(h['bland'] for h in r[3])
    for coefficient in (1e-10, 1e-16):
        r = simplex([[coefficient, 1]], [1], [-1, 0], [1])
        assert r[-1] == 'optimal'
        np.testing.assert_allclose(r[0], [1 / coefficient, 0])
    r = simplex([[1e8, 1, 0], [1e8, 0, 1]], [1e8 + .05, 1e8], [-1, 0, 0], [1, 2])
    assert r[-1] == 'optimal'
    assert r[0][0] <= 1
    assert np.min(r[0]) >= 0
    for factor in (1e-12, 1e12):
        r = simplex([[factor, factor]], [factor], [-factor, 0], [1])
        assert r[-1] == 'optimal'
        np.testing.assert_allclose(r[0], [1, 0])
    try:
        simplex([[1]], [-1], [1], [0])
    except ValueError:
        pass
    else:
        raise AssertionError('Infeasible initial basis accepted')


def test_two_phase():
    r = two_phase([[1, 1, 0], [1, 0, -1]], [1, 1.00000005], [1, 0, 0], tol=1e-9)
    assert r[-1] == 'infeasible'
    A = np.array([[1., 1], [2, 2], [-3, -3]])
    b = np.array([1., 2, -3])
    c = np.array([1., 2])
    for factors in (np.ones(3), np.array([1e-10, 1e8, 1.])):
        As, bs = A * factors[:, None], b * factors
        r = two_phase(As, bs, c)
        assert r[-1] == 'optimal'
        y = r[3][-1]['duals']
        assert y.shape == (3,)
        np.testing.assert_allclose(As @ r[0], bs)
        assert np.min(c - As.T @ y) >= -1e-8
        np.testing.assert_allclose(bs @ y, r[1])
        np.testing.assert_allclose(r[0] * (c - As.T @ y), 0, atol=1e-8)
    assert two_phase([[0]], [1], [1])[-1] == 'infeasible'
    r = two_phase([[0]], [0], [1])
    assert r[-1] == 'optimal'
    np.testing.assert_allclose(r[3][-1]['duals'], [0])
    r = two_phase([[-1, -1]], [-2], [1, 2])
    assert r[-1] == 'optimal'
    np.testing.assert_allclose(r[0], [2, 0])


def test_dual():
    r = dual([[-1], [1]], [-1, 2], [1], senses=['<=', '<='], maximize=True)
    assert r[-1] == 'optimal'
    np.testing.assert_allclose(r[1], 2)
    assert np.all(r[0] >= -1e-8)
    for maximize, senses, target in (
            (False, ['>=', '<='], 1), (True, ['>=', '<='], 2),
            (False, ['=', '<='], 1), (True, ['=', '<='], 1)):
        r = dual([[1], [1]], [1, 2], [1], senses=senses, maximize=maximize)
        assert r[-1] == r[3][-1]['primal_status'] == 'optimal'
        np.testing.assert_allclose(r[1], target)
    r = dual([[1], [1]], [1, 2], [1])
    assert r[-1] == 'unbounded' and r[3][-1]['primal_status'] == 'infeasible'
    r = dual([[1, -1]], [0], [-1, 0])
    assert r[-1] == 'infeasible' and r[3][-1]['primal_status'] == 'unbounded'
    r = dual([[0]], [1], [-1])
    assert r[-1] == r[3][-1]['primal_status'] == 'infeasible'


def test_scalar():
    try:
        bisection(lambda x: -x*x, lambda x: -2*x, -1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Maximum bracket accepted')
    r = newton(lambda x: -x*x, lambda x: -2*x, lambda x: -2, 1)
    assert r[-1] == 'not_minimum' and r[2][-1]['classification'] == 'maximum'
    r = newton(lambda x: x*x, lambda x: 2*x, lambda x: 2, 1)
    assert r[-1] == 'stationary' and r[2][-1]['classification'] == 'minimum'
    r = newton(lambda x: x**4/4-x*x+2*x, lambda x: x**3-2*x+2,
               lambda x: 3*x*x-2, 0)
    assert r[-1] == 'cycle_detected' and len(r[2]) == 2
    r = iterative(lambda x: x*x, lambda x: 2*x, lambda x: x-1e-10*2*x, 1)
    assert r[-1] == 'stagnation' and r[2][-1]['gnorm'] > 1
    r = iterative(lambda x: x*x, lambda x: 2*x, lambda x: .5*x, 1)
    assert r[-1] == 'stationary' and abs(r[0]) <= 1e-8


def test_offset_and_stationarity():
    g = lambda x: 2*x
    for method in (steepest_descent, quasi_newton, rank_one, dfp, bfgs):
        for offset in (0., 1e10, 1e20):
            r = method(lambda x: offset + x@x, g, [1e-3, 0])
            assert r[-1] == 'stationary', (method.__name__, offset, r[-1])
            np.testing.assert_allclose(r[0], [0, 0], atol=1e-8)
    f = lambda x: x[0]**2-x[1]**2
    g = lambda x: np.array([2*x[0], -2*x[1]])
    for method in (steepest_descent, quasi_newton, rank_one, dfp, bfgs):
        assert method(f, g, [0, 0])[-1] == 'stationary'
    assert steepest_descent_fixed(f, g, [0, 0], .1)[-1] == 'stationary'
    r = newton_multivariable(f, g, lambda x: np.diag([2., -2]), [0, 0])
    assert r[-1] == 'not_minimum'
    assert r[2][-1]['classification'] == 'not_minimum'
    assert line_search(lambda x: -x[0], lambda x: np.array([-1.]), np.zeros(1), np.ones(1)) is None


def test_cg():
    for scale, b, tol in ((1e-11, [1e-11, 2e-11], 1e-24), (1., [1e-8, 0], 1e-20)):
        Q = scale * np.eye(2)
        r = conjugate_gradient(Q, b, [0, 0], tol=tol)
        assert r[-1] == 'stationary'
        np.testing.assert_allclose(r[0], np.linalg.solve(Q, b), atol=1e-20)
        assert np.linalg.norm(Q @ r[0] - b) <= tol
    rng = np.random.default_rng(24)
    U, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    Q = U @ np.diag(np.geomspace(1., 1e4, 8)) @ U.T
    b = rng.normal(size=8)
    r = conjugate_gradient(Q, b, np.zeros(8), tol=1e-9)
    assert r[-1] == 'stationary'
    assert len(r[2]) - 1 > 8
    assert np.linalg.norm(Q @ r[0] - b) <= 1e-9
    for Q in (np.diag([1., -1]), np.diag([1., 0]), [[1, 2], [0, 1]]):
        try:
            conjugate_gradient(Q, [0, 0], [1, 1])
        except ValueError:
            pass
        else:
            raise AssertionError('Non-SPD input accepted')


def test_updates_and_rosenbrock():
    for update in (sr1_update, dfp_update, bfgs_update):
        H = np.eye(2)
        for s, y in ((np.zeros(2), np.zeros(2)), (np.ones(2), np.zeros(2))):
            Hn, applied = update(H, s, y)
            assert not applied and np.all(np.isfinite(Hn))
    for update in (dfp_update, bfgs_update):
        assert not update(np.eye(2), np.ones(2), -np.ones(2))[1]
    f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    g = lambda x: np.array([2*(x[0]-1)-400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
    for method in (rank_one, dfp, bfgs, quasi_newton):
        r = method(f, g, [-1.2, 1.], tol=1e-6, max_iter=3000)
        assert r[-1] == 'stationary', (method.__name__, r[-1])
        assert np.linalg.norm(g(r[0])) <= 1e-6
        np.testing.assert_allclose(r[0], [1, 1], atol=1e-5)


def test_lp_vertices():
    from itertools import combinations

    rng = np.random.default_rng(73)
    for _ in range(20):
        G = np.vstack((rng.normal(size=(3, 3)), np.eye(3)))
        b = rng.uniform(.5, 3, size=6)
        c = rng.normal(size=3)
        inequalities = np.vstack((G, -np.eye(3)))
        rhs = np.r_[b, np.zeros(3)]
        vertices = []
        for rows in combinations(range(9), 3):
            try:
                x = np.linalg.solve(inequalities[list(rows)], rhs[list(rows)])
            except np.linalg.LinAlgError:
                continue
            if np.all(inequalities @ x <= rhs + 1e-9):
                vertices.append(x)
        values = np.array(vertices) @ c
        scale = 10. ** rng.uniform(-6, 6, size=6)
        signs = np.array([1., -1, 1, -1, 1, -1])
        A = G * (scale * signs)[:, None]
        bs = b * scale * signs
        senses = ['<=' if sign > 0 else '>=' for sign in signs]
        for maximize in (False, True):
            target = np.max(values) if maximize else np.min(values)
            r = dual(A, bs, c, senses=senses, maximize=maximize)
            assert r[-1] == r[3][-1]['primal_status'] == 'optimal'
            np.testing.assert_allclose(r[1], target, atol=1e-7, rtol=1e-7)
            sign = -1 if maximize else 1
            assert np.all(sign * (c - A.T @ r[0]) >= -1e-7)
    print('Verified 40 mixed-sense LPs against enumerated vertices.')


if __name__ == '__main__':
    tests = [test_simplex, test_two_phase, test_dual, test_scalar,
             test_offset_and_stationarity, test_cg, test_updates_and_rosenbrock, test_lp_vertices]
    for test in tests:
        test()
        print('Passed', test.__name__)
    print('All 8 targeted regression groups passed.')
