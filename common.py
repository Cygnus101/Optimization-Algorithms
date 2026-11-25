import numpy as np


def pivot(T, p, q):
    """Return a copy of T after a Gauss-Jordan pivot at (p, q)."""
    T = np.array(T, dtype=float, copy=True)
    T[p] = T[p] / T[p, q]
    for i in range(T.shape[0]):
        if i != p:
            T[i] = T[i] - T[i, q] * T[p]
    return T


def state(x, fx, g, **details):
    """Independent history entry; g is the derivative or gradient."""
    return {"x": np.array(x, dtype=float, copy=True), "fx": float(fx),
            "g": np.array(g, dtype=float, copy=True),
            "gnorm": float(np.linalg.norm(g)), **details}


def quadratic(Q, b=None):
    """Return f, grad, hess for 0.5*x@Q@x - b@x; Q must be symmetric."""
    Q = np.array(Q, dtype=float, copy=True)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not np.all(np.isfinite(Q)):
        raise ValueError("Q must be a finite square matrix")
    if np.linalg.norm(Q - Q.T) > 1e-10 * np.linalg.norm(Q):
        raise ValueError("Q must be symmetric")
    b = np.zeros(len(Q)) if b is None else np.array(b, dtype=float, copy=True)
    if b.shape != (len(Q),) or not np.all(np.isfinite(b)):
        raise ValueError("b must be a finite vector matching Q")

    def f(x):
        return 0.5 * x @ Q @ x - b @ x

    def grad(x):
        return Q @ x - b

    def hess(x):
        return Q.copy()

    return f, grad, hess


def quadratic_step(g, d, Q):
    """Exact positive step for a quadratic, or None for nonpositive curvature."""
    q = d @ Q @ d
    if not np.isfinite(q) or q <= 0:
        return None
    return float(-(g @ d) / q)


def line_search(f, grad, x, d, alpha0=1.0, c1=1e-4, c2=0.9, max_iter=40):
    """Strong Wolfe search by expansion and bisection; return alpha or None."""
    f0 = float(f(x))
    slope = float(grad(x) @ d)
    if not np.isfinite(f0) or not np.isfinite(slope) or slope >= 0:
        return None

    rounding = 8 * np.finfo(float).eps * max(abs(f0), np.finfo(float).tiny)

    def precision_step(a, fa):
        if abs(fa - f0) > rounding or abs(a * slope) > rounding:
            return False
        sa = float(grad(x + a * d) @ d)
        return np.isfinite(sa) and abs(sa) <= -c2 * slope

    def zoom(lo, hi):
        for _ in range(max_iter):
            a = (lo + hi) / 2
            if a == lo or a == hi:
                return None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                fa = float(f(x + a * d))
                flo = float(f(x + lo * d))
            if np.isfinite(fa) and precision_step(a, fa):
                return a
            if not np.isfinite(fa) or fa > f0 + c1 * a * slope or fa >= flo:
                hi = a
            else:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    sa = float(grad(x + a * d) @ d)
                if not np.isfinite(sa):
                    hi = a
                    continue
                if abs(sa) <= -c2 * slope:
                    return a
                if sa * (hi - lo) >= 0:
                    hi = lo
                lo = a
        return None

    prev, fp, a = 0.0, f0, float(alpha0)
    for k in range(max_iter):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fa = float(f(x + a * d))
        if np.isfinite(fa) and precision_step(a, fa):
            return a
        if not np.isfinite(fa) or fa > f0 + c1 * a * slope or (k > 0 and fa >= fp):
            return zoom(prev, a)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sa = float(grad(x + a * d) @ d)
        if not np.isfinite(sa):
            return zoom(prev, a)
        if abs(sa) <= -c2 * slope:
            return a
        if sa >= 0:
            return zoom(a, prev)
        prev, fp, a = a, fa, 2 * a
    return None


def show_result(result):
    """Print the solution, objective and status from either return convention."""
    x, fx = result[:2]
    print("Status:", result[-1])
    print("Solution:", x)
    print("Objective:", fx)
    if len(result) == 5:
        print("Basis (zero-based):", result[2])
    else:
        print("Updates:", len(result[2]) - 1)
        print("Derivative / gradient norm:", result[2][-1]["gnorm"])


def plot_scalar(f, history, a, b):
    """Plot scalar iterates on f and their objective history."""
    import matplotlib.pyplot as plt

    t = np.linspace(a, b, 300)
    points = np.array([h["x"] for h in history])
    values = [h["fx"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(t, [f(v) for v in t])
    axes[0].plot(points, values, "o-", color="crimson", markersize=4)
    axes[0].set(xlabel="x", ylabel="f(x)", title="Iterates on objective")
    axes[1].plot(values, "o-", color="crimson", markersize=4)
    axes[1].set(xlabel="State", ylabel="f(x)", title="Objective history")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_quadratic(Q, b, history):
    """Plot a two-variable quadratic trajectory and objective history."""
    import matplotlib.pyplot as plt

    points = np.array([h["x"] for h in history])
    lo = np.min(points, axis=0) - 1
    hi = np.max(points, axis=0) + 1
    X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 150), np.linspace(lo[1], hi[1], 150))
    P = np.stack((X, Y), axis=-1)
    Z = .5 * np.einsum("...i,ij,...j->...", P, Q, P) - P @ b
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].contour(X, Y, Z, levels=18)
    axes[0].plot(points[:, 0], points[:, 1], "o-", color="crimson", markersize=3)
    axes[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Optimization trajectory")
    axes[1].plot([h["fx"] for h in history], "o-", color="crimson", markersize=3)
    axes[1].set(xlabel="State", ylabel="f(x)", title="Objective history")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_phases(history):
    """Show auxiliary and transformed original LP objectives separately."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for phase, ax in enumerate(axes, start=1):
        values = [h["z"] for h in history if h["phase"] == phase]
        ax.plot(values, "o-", color="crimson")
        ax.set(title=f"Phase {phase}", xlabel="State",
               ylabel="Artificial sum w" if phase == 1 else "Solver objective")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


def plot_lp(G, b, history):
    """Bounded two-variable region Gx<=b, x>=0, with feasible iterates."""
    from itertools import combinations
    import matplotlib.pyplot as plt

    G = np.vstack((G, -np.eye(2)))
    b = np.r_[b, 0, 0]
    vertices = []
    for i, j in combinations(range(len(b)), 2):
        try:
            v = np.linalg.solve(G[[i, j]], b[[i, j]])
        except np.linalg.LinAlgError:
            continue
        if np.all(G @ v <= b + 1e-8):
            vertices.append(v)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    if vertices:
        v = np.unique(np.round(vertices, 10), axis=0)
        center = v.mean(axis=0)
        order = np.argsort(np.arctan2(v[:, 1]-center[1], v[:, 0]-center[0]))
        v = v[order]
        axes[0].fill(v[:, 0], v[:, 1], color="lightblue", alpha=0.6)
        closed = np.vstack((v, v[0]))
        axes[0].plot(closed[:, 0], closed[:, 1], color="steelblue", label="Feasible set")
    points = np.array([h["x"][:2] for h in history])
    axes[0].plot(points[:, 0], points[:, 1], "o-", color="crimson", label="Simplex path")
    for k, point in enumerate(points):
        axes[0].annotate(str(k), point, xytext=(6, 6), textcoords="offset points")
    for start, end in zip(points[:-1], points[1:]):
        axes[0].annotate("", xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color="crimson"))
    axes[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="Feasible trajectory")
    axes[0].legend()
    axes[1].plot([h["z"] for h in history], "o-", color="crimson")
    axes[1].set(xlabel="State", ylabel="z", title="Objective history")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.margins(0.15)
    fig.tight_layout()
    plt.show()
