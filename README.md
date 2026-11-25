# Optimization programming assignment

Fourteen algorithm sections, implemented as importable Python functions and demonstrated in **`Prog_assign.ipynb`**. This README describes the current local Python implementations. Quasi-Newton is a shared framework defaulting to BFGS, so the 14 sections are not 14 distinct update formulas.

## Files

| File | Role |
| --- | --- |
| `Prog_assign.ipynb` | Main notebook: imports, then separate problem-input, solve, and plot cells for each of the 14 methods. |
| `algorithms/__init__.py` | Exports all 14 solver functions; supports `from algorithms import simplex`, etc. |
| `algorithms/simplex.py` | Feasible-basis primal simplex for equality-form minimization. |
| `algorithms/two_phase.py` | Artificial-variable initialization, cleanup, then simplex on the original objective. |
| `algorithms/dual.py` | Constructs the dual of the supplied primal and solves it with two-phase; also solves the primal to record its status. |
| `algorithms/bisection.py` | Bisects a derivative bracket to find a scalar stationary point. |
| `algorithms/newton.py` | Full scalar Newton steps on the derivative. |
| `algorithms/iterative.py` | Caller-defined scalar fixed-point iteration. |
| `algorithms/steepest_descent.py` | Negative-gradient descent with quadratic-exact or strong Wolfe steps. |
| `algorithms/steepest_descent_fixed.py` | Negative-gradient descent with a constant positive step. |
| `algorithms/newton_multivariable.py` | Solves the Hessian system; optional strong Wolfe damping. |
| `algorithms/conjugate_gradient.py` | Linear CG for a symmetric positive definite quadratic. |
| `algorithms/quasi_newton.py` | Shared inverse-Hessian iteration and custom-update interface; defaults to BFGS. |
| `algorithms/rank_one.py` | Inverse SR1 update and wrapper around the shared iteration. |
| `algorithms/dfp.py` | Inverse DFP update and wrapper. |
| `algorithms/bfgs.py` | Inverse BFGS update and wrapper. |
| `common.py` | `pivot`: copied Gauss–Jordan tableau pivot; `state`: independent history snapshots; `quadratic`: objective/gradient/Hessian factory; `quadratic_step`: exact quadratic step; `line_search`: strong Wolfe search; `show_result`: tuple display; plotting helpers below. |
| `check_matrices.py` | Runs eight vector methods on ten matrix cases, printing expected successes and failures. Edit its `cases` list to try other matrices. Its `invalid_input` label is produced by this script when it catches `ValueError`, not returned by the solvers. |
| `test_algorithms.py` | Standalone assertions for LPs, scalar methods, histories, 96 rotated-SPD solves, general functions, breakdowns, and 80 matrix-case results. |
| `test_regressions.py` | Eight targeted regression groups: cycling/scaling, two-phase cleanup, dual formulations, scalar failures, stationarity/offsets, CG, inverse updates, and 40 mixed-sense LP comparisons against enumerated vertices. |
| `requirements.txt` | NumPy, Matplotlib, Notebook, and ipykernel; versions are not pinned. |

`__pycache__/` contains generated Python caches, not assignment source. `.git/` holds repository metadata.

## Installation and running

Open this directory itself in VS Code. From a terminal in the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook Prog_assign.ipynb
```

Alternatively, open `Prog_assign.ipynb` in VS Code with Python/Jupyter support and select the `.venv` interpreter as the notebook kernel. Run the first import cell, then a method's input, solve, and plot cells in that order, or run all cells. Sections deliberately reuse names such as `A`, `b`, `f`, and `result`: rerun the matching input cell before its solver. A vector method needs its own quadratic setup cell, not the preceding scalar function definitions.

Run verification scripts from the same directory:

```bash
python test_algorithms.py
python test_regressions.py
python check_matrices.py
```

The algorithm modules define functions; running a module file directly is not a command-line problem solver. Put the examples below in notebook cells or a script in the project root and run `python your_script.py`. Keep the project root on Python's import path because modules import `common` from there.

## Inputs, outputs, and shared conventions

Use finite floating-point arrays, compatible dimensions, correct analytic derivatives, positive `tol`, and a nonnegative integer `max_iter`. Defaults are **`tol=1e-8`, `max_iter=1000`** throughout. Validation is uneven: simplex validates these controls explicitly; other routines can raise ordinary Python/NumPy errors for malformed inputs. There is no symbolic differentiation or automatic constraint parser.

### Linear programs

`simplex` and `two_phase` solve **minimize `c @ x`, subject to `A @ x == b`, `x >= 0`**. For `m` constraints and `n` variables, shapes are `A: (m,n)`, `b: (m,)`, `c: (n,)`. Columns follow exactly the order of variables you supply, including slacks/surpluses. They have no `maximize` or `senses` parameter.

- To maximize `p @ x`, pass `c=-p`; the original maximum is `-z`.
- Convert `a @ x <= b` into `a @ x + s = b`, with `s>=0` and zero objective coefficient.
- Convert `a @ x >= b` into `a @ x - s = b`, with `s>=0` and zero objective coefficient. This surplus column need not provide a feasible starting basis; use two-phase.
- Equalities need no slack. Multiplying a row by `-1` reverses its inequality sense before conversion. Two-phase automatically flips negative right-hand sides after receiving equalities; simplex requires a feasible supplied basis.
- Variables are nonnegative. Represent a free variable manually as `u-v`, `u,v>=0`, and transform both its matrix column and cost. Bounds and other transformations are manual.

A basis is a sequence of **zero-based column indices**, one distinct column per row, not row indices or mathematical variable labels. Thus `[2,3,4]` selects `x3,x4,x5` in the notebook. `B=A[:,basis]` must be square, nonsingular, and satisfy `solve(B,b)>=0` within tolerance.

LP returns:

```python
x, z, basis, history, status = simplex(A, b, c, basis)
# Same five-item convention for two_phase.
y, value, basis, history, dual_status = dual(A, b, c)
```

`x` includes all supplied columns; `z=c@x`. Each simplex history entry holds `T` (constraint tableau, RHS last), `r` (reduced costs), `basis`, `x`, `z`, and `bland`. Terminal history includes original-row `duals`. Two-phase adds `phase` and `stage`; cleanup entries do not have `bland`. Phase I entries include artificial columns and their sum as `z`; Phase II uses the supplied columns and objective. Following redundant-row removal, the returned basis can be shorter than the original row count. If Phase I fails, two-phase returns `x=z=None` and an augmented-column basis. On Phase II failure it returns the last iterate, which is not an optimum.

For `dual`, the first result is a vector of **dual multipliers**, not primal decision variables; the basis and history describe the internally transformed dual. Inspect `history[-1]['primal_status']` and `history[-1]['dual_status']` separately. A nonoptimal dual returns `y=value=None`.

### Scalar and vector minimization

Scalar callables accept a scalar; vector callables accept a one-dimensional NumPy array. `f` returns a scalar, `df` a scalar derivative, `grad` a matching vector, and `hess` a matching symmetric matrix. Results are:

```python
x, fx, history, status = solver(...)
```

`fx=f(x)`. History includes the initial and terminal recorded states, each with independent `x`, `fx`, `g`, and `gnorm`. Vector `gnorm` is the Euclidean gradient norm; scalar `gnorm` is the absolute derivative. Extra fields include `a,b` for bisection, `h` for scalar Newton, `alpha` for vector steps, and `H,update` for quasi-Newton. `alpha` describes the step attempted from that state; terminal entries usually have `None`. Quasi-Newton `update` is `applied` or `skipped` on a step, with optional `direction='steepest_fallback'`.

`stationary` means `abs(df)<=tol` or `norm(grad)<=tol`, not a global minimum certificate. Most methods do not classify stationary points. For maximization with these unconstrained methods, minimize `-f` and negate its derivatives/Hessian too, then negate the returned value.

### Line searches and common quadratic input

For sections 7–14 below, run this setup first (it matches their notebook input cells):

```python
import numpy as np
from common import quadratic, show_result

Q = np.array([[4., 1.], [1., 3.]])
b = np.array([1., 2.])
x0 = np.array([3., -2.])
f, grad, hess = quadratic(Q, b)
```

This defines `f(x)=0.5*x@Q@x-b@x`, `grad(x)=Q@x-b`, `hess(x)=Q`. `quadratic` checks finite square symmetric `Q` and a matching finite `b` (default zero); it does **not** require positive definiteness. All eight examples below reach approximately **`x=[1/11,7/11]`, `fx=-15/22`**, status **`stationary`**. Compare numerically, not by exact printed digits.

For steepest descent and the quasi-Newton family, passing `Q=Q` selects

`alpha = -(g @ d)/(d @ Q @ d)`.

This is an exact line minimizer only when `Q` really is the constant Hessian of your supplied quadratic. The code validates symmetry/shape, not consistency with `f,grad`. Nonpositive/nonfinite directional curvature stops with `nonpositive_curvature`.

Omit `Q` for a general smooth objective. `common.line_search` uses expansion and bisection/zoom to approximately satisfy strong Wolfe conditions:

`f(x+alpha*d) <= f(x)+1e-4*alpha*(g@d)` and
`abs(grad(x+alpha*d)@d) <= 0.9*abs(g@d)`.

It starts at `alpha=1`, has bounded search loops (40 iterations each), and returns `None` on failure. It includes a rounding-aware derivative check for tiny objective changes. This is not exact minimization or a guaranteed search on every function. The solver interfaces do not expose its tuning parameters. Damped Newton uses this search; full Newton and fixed-step descent do not.

## 1. Simplex

```python
simplex(A, b, c, basis, tol=1e-8, max_iter=1000)
```

**Purpose/update:** minimize an equality-form LP from a basic feasible solution. Form `T=B^-1[A|b]`, compute `r=c-c_B@T[:,:-1]`, and stop if no reduced cost is sufficiently negative. Normally enter the most negative reduced-cost column. Leave by the minimum nonnegative RHS/pivot-column ratio among positive pivots, then Gauss–Jordan pivot and replace that basis index. Ratio ties select the smallest basic-variable index. On a repeated basis, permanently switch to Bland's smallest-index entering rule.

Equality-form example (also used in the Python tests):

```python
import numpy as np
from algorithms.simplex import simplex

A = np.array([[1.,0,1,0,0], [0,1,0,1,0], [1,1,0,0,1]])
b = np.array([4.,6,8])
c = np.array([-2.,-5,0,0,0])
x, z, basis, history, status = simplex(A, b, c, [2,3,4], tol=1e-8, max_iter=1000)
# x = [2,6,2,0,0], z = -34, status = 'optimal'
# This is also max 2*x1+5*x2 = 34 after negating the objective.
```

**Mathematics:** degeneracy allows zero-length pivots and cycling under ordinary pivot rules. In exact arithmetic Bland's rule avoids cycling. If an improving column has no positive entry, the LP is unbounded.

**Implementation:** a singular initial `B` raises `ValueError('Basis must be nonsingular')`; it does not try another basis or construct artificials. Duplicate, nonintegral, out-of-range, or wrong-length indices raise `ValueError`. A nonsingular but infeasible initial basis raises `ValueError('Initial basis must be feasible; use two_phase')`. Rows are scaled by their largest absolute coefficient; reduced-cost, ratio, residual, and pivot tests account for tolerance/roundoff. A repeated basis triggers fallback, not a `cycle_detected` status. Remaining outcomes are `optimal`, `unbounded`, `iteration_limit`, and `numerical_failure`. Ill-conditioning and extreme column/cost scaling can still lose accuracy; row scaling is not a guarantee. A final basis solve may itself raise a NumPy linear-algebra error if numerical singularity develops.

## 2. Two-phase

```python
two_phase(A, b, c, tol=1e-8, max_iter=1000)
```

**Purpose/update:** obtain a feasible basis automatically. Normalize rows, flip negative RHS rows, append an identity artificial matrix, then minimize `w=sum(artificials)` with artificials initially basic. An optimal `w>tol` means infeasible. If `w<=tol`, pivot remaining basic artificials out using original columns; delete rows with no available pivot as redundant, remove artificial columns, then minimize `c@x` with simplex. Reconstruct original-row dual multipliers and check residuals, reduced costs, and duality gap.

```python
from algorithms.two_phase import two_phase

x, z, basis, history, status = two_phase([[2.,1.]], [2.], [3.,1.], tol=1e-8)
# x = [0,2], z = 2, status = 'optimal'
```

**Mathematics:** inconsistent equalities/nonnegativity make Phase I positive; redundant constraints need not make a problem infeasible. Phase II may be unbounded.

**Implementation:** accepts equalities only; add slack/surplus columns yourself. Handles negative RHS and redundant rows, including reconstruction of multipliers for deleted original rows. The artificial-sum test uses normalized rows and an absolute `tol`, so borderline feasibility depends on scaling and tolerance. No exact-arithmetic certificate is produced. Statuses include propagated simplex `numerical_failure`, as well as `infeasible`, `optimal`, `unbounded`, and `iteration_limit`. `max_iter` applies separately to each simplex phase; artificial cleanup is outside those pivot counters. Malformed/empty array cases are not comprehensively validated and may raise NumPy errors.

## 3. Dual formulation and solve (not dual simplex)

```python
dual(A, b, c, tol=1e-8, max_iter=1000, *, senses=None, maximize=False)
```

**Purpose/update:** form the mathematical dual of the supplied primal, represent signed/free multipliers using nonnegative internal variables, introduce dual slacks, and call two-phase. It separately calls two-phase on the converted primal for its status. It does not perform dual-simplex pivots or return the primal solution.

Default primal: `min c@x`, `Ax=b`, `x>=0`; dual: `max b@y`, `A.T@y<=c`, free `y`. With a minimizing primal, `>=` rows give `y>=0`, `<=` rows give `y<=0`. With a maximizing primal these signs reverse and dual constraints are `A.T@y>=c`, minimizing `b@y`. Equality-row multipliers remain free. Rows are scaled internally and multipliers mapped back.

```python
from algorithms.dual import dual

# Primal: max x, with -x <= -1, x <= 2, x >= 0.
y, value, basis, history, status = dual(
    [[-1.],[1.]], [-1.,2.], [1.],
    senses=['<=','<='], maximize=True, tol=1e-8)
# y = [0,1], value = 2, status = 'optimal'
# history[-1]['primal_status'] == 'optimal'
```

`senses` must have one of `'='`, `'<='`, `'>='` for each row; default is all equalities. Unlike simplex/two-phase, this interface directly supports mixed senses and `maximize=True`. Variables in the **primal** still must be nonnegative. Return `value=b@y` is already in the original objective direction; do not negate it for `maximize=True`.

**Mathematics:** an infeasible dual alone does not prove an unbounded primal; both can be infeasible. Strong duality applies when finite optima exist.

**Implementation:** the returned status is the dual's status, not the primal's. Inspect both terminal history fields rather than translating one into the other. Internal history objective signs and variable coordinates differ from returned `y,value`. This performs two solves and inherits two-phase tolerances/failures; it does not return a separate numerical strong-duality-gap comparison between the two solves.

## 4. Derivative bisection

```python
bisection(f, df, a, b, tol=1e-8, max_iter=1000)
```

**Purpose/update:** solve `df(x)=0` on `[a,b]`. Use midpoint `x=(a+b)/2`; if `df(x)<0`, replace `a`, otherwise replace `b`. An endpoint with exactly zero derivative is used directly. Stop at small derivative or half-width.

```python
from algorithms.bisection import bisection

f = lambda x: (x-1)**2
x, fx, history, status = bisection(f, lambda x: 2*(x-1), 0., 3., tol=1e-8)
# x approximately 1, fx approximately 0, status = 'stationary'
```

**Mathematics:** continuous derivative and `df(a)<=0<=df(b)` give a root bracket; convexity makes the stationary point a global minimum. A nonconvex bracket need not isolate the desired minimum. Discontinuity invalidates the root guarantee. A constrained boundary minimum can have nonzero derivative and no such bracket.

**Implementation:** rejects invalid endpoints or the wrong derivative orientation with `ValueError`, including a typical maximum bracket. It does not minimize over arbitrary closed intervals or compare boundary objective values. `bracket_converged` certifies only half-width `<=tol`; other statuses are `stationary`, `nonfinite`, `stagnation`, `iteration_limit`. It does not test continuity or convexity.

## 5. Scalar Newton

```python
newton(f, df, ddf, x0, tol=1e-8, max_iter=1000)
```

**Purpose/update:** solve stationarity using `x_next=x-df(x)/ddf(x)` with full steps.

```python
from algorithms.newton import newton

x, fx, history, status = newton(
    lambda x: (x-1)**2, lambda x: 2*(x-1), lambda x: 2., 3., tol=1e-8)
# x = 1, fx = 0, status = 'stationary'; classification = 'minimum'
```

**Mathematics:** fast local convergence needs a suitable initial point and nonzero second derivative near the root. It can diverge, oscillate, or converge to a maximum; tiny curvature creates huge steps. Zero `df` is the target, whereas zero `ddf` prevents a Newton step.

**Implementation:** at small `df`, classifies by the sign of `ddf`: negative gives `not_minimum`/`maximum`; positive gives `stationary`/`minimum`; zero gives `stationary`/`undetermined` (higher-order behavior is unchecked). Nonstationary exactly zero `ddf` gives `zero_curvature`; no near-zero threshold, damping, or bracket safeguard is used. Exact repeated iterates give `cycle_detected`; exact unchanged steps give `stagnation`. Other statuses: `nonfinite`, `iteration_limit`. Approximate cycles may simply reach the limit.

## 6. Fixed-point iterative method

```python
iterative(f, df, h, x0, tol=1e-8, max_iter=1000)
```

**Purpose/update:** choose a rearrangement whose fixed point solves `df=0`, then iterate `x_next=h(x)`.

```python
from algorithms.iterative import iterative

x, fx, history, status = iterative(
    lambda x: x*x, lambda x: 2*x, lambda x: .5*x, 1., tol=1e-8)
# abs(x) <= 5e-9, fx approximately 0, status = 'stationary'
```

**Mathematics:** contraction on an invariant neighborhood (e.g. `abs(h')<1`) is a sufficient convergence condition. A bad rearrangement may diverge or cycle, and a fixed point of an unrelated `h` need not solve the optimization problem.

**Implementation:** caller supplies `h`; the code does not construct it or verify contractivity. It checks `df`, not just step size. A relative-small step with residual above `tol` and at least 99% of the old residual gives `stagnation`, avoiding false convergence for an almost-identity map. Other statuses: `stationary`, `nonfinite`, `iteration_limit`. No minimum classification or explicit cycle detector.

## 7. Steepest descent with line search

```python
steepest_descent(f, grad, x0, tol=1e-8, max_iter=1000, Q=None)
```

**Purpose/update:** `g=grad(x)`, `d=-g`, choose the shared exact-quadratic or strong Wolfe step, then `x_next=x+alpha*d`.

```python
from algorithms.steepest_descent import steepest_descent
# Run the common quadratic setup above first.
x, fx, history, status = steepest_descent(f, grad, x0, Q=Q, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** ill-conditioned valleys cause zigzagging and slow convergence. Smoothness/convexity assumptions matter; a zero gradient can be a saddle or maximum.

**Implementation:** no Hessian classification, preconditioning, or fallback if the selected step search fails. Statuses: `stationary`, `nonpositive_curvature` (Q branch), `line_search_failed` (general branch), `nonfinite`, `stagnation`, `iteration_limit`.

## 8. Fixed-step steepest descent

```python
steepest_descent_fixed(f, grad, x0, alpha, tol=1e-8, max_iter=1000)
```

**Purpose/update:** `x_next=x-alpha*grad(x)` using the same `alpha` every time.

```python
from algorithms.steepest_descent_fixed import steepest_descent_fixed
# Run the common quadratic setup above first.
x, fx, history, status = steepest_descent_fixed(f, grad, x0, alpha=.1, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** for an SPD quadratic, convergence from arbitrary starts requires `0<alpha<2/lambda_max(Q)`. Large steps oscillate/diverge; very small steps converge slowly and can round to no movement. For this Q, `lambda_max` is about 4.618, so `.1` is suitable.

**Implementation:** rejects nonfinite/nonpositive `alpha` with `ValueError` but does not check the stability bound or adapt the step. No line search or minimum classification. Statuses: `stationary`, `nonfinite`, `stagnation`, `iteration_limit`.

## 9. Multivariable Newton

```python
newton_multivariable(f, grad, hess, x0, tol=1e-8, max_iter=1000, damped=False)
```

**Purpose/update:** solve `hess(x) @ d = -grad(x)`; use `x_next=x+d` by default, or `x_next=x+alpha*d` with strong Wolfe search when `damped=True`.

```python
from algorithms.newton_multivariable import newton_multivariable
# Run the common quadratic setup above first.
x, fx, history, status = newton_multivariable(f, grad, hess, x0, tol=1e-8)
# One full update: x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** a singular Hessian prevents a unique Newton direction; an indefinite one can give ascent. Full steps can overshoot away from a minimum, and an ill-conditioned Hessian amplifies errors. Small gradient alone does not exclude a saddle/maximum.

**Implementation:** no regularization, pseudoinverse, or steepest-descent fallback. Nonsymmetric Hessian raises `ValueError`; failed solve gives `singular_hessian`; `g@d>=0` gives `non_descent`. At stationarity, eigenvalues classify negative curvature as `not_minimum`, strictly positive curvature as `minimum`, otherwise `undetermined`; the latter still returns `stationary`. Classification uses a machine-precision relative eigenvalue margin. Other statuses: `line_search_failed` (damped branch), `nonfinite`, `stagnation`, `iteration_limit`. Damping cannot repair a rejected Newton direction.

## 10. Linear conjugate gradient

```python
conjugate_gradient(Q, b, x0, tol=1e-8, max_iter=1000)
```

**Purpose/update:** solve `Qx=b`, equivalently minimize the SPD quadratic. Start `g=Qx-b`, `d=-g`; repeat `alpha=(g@g)/(d@Q@d)`, `x_next=x+alpha*d`, `g_next=Q@x_next-b`, `beta=(g_next@g_next)/(g@g)`, `d_next=-g_next+beta*d`.

```python
from algorithms.conjugate_gradient import conjugate_gradient
# Run the common quadratic setup above first.
x, fx, history, status = conjugate_gradient(Q, b, x0, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** linear CG requires symmetric positive definite Q, not a general nonlinear objective or merely semidefinite matrix. In exact arithmetic at most n steps suffice; finite precision loses conjugacy and can require more.

**Implementation:** checks symmetry via `quadratic` and positive definiteness via Cholesky, raising `ValueError` on rejection. Uses dense arrays, recomputed gradient, no preconditioner, and no artificial n-step cap. If `d@Q@d <= eps*norm(d)*norm(Q@d)` or is nonfinite, returns `nonpositive_curvature`: even accepted SPD input can encounter numerical denominator breakdown. Other statuses: `stationary`, `nonfinite`, `stagnation`, `iteration_limit`.

## 11. Shared quasi-Newton method

```python
quasi_newton(f, grad, x0, update=None, H0=None, tol=1e-8, max_iter=1000, Q=None)
```

**Purpose/update:** approximate the inverse Hessian H. Start `H=I` unless supplied; set `d=-H@g`, choose a line step, set `s=alpha*d`, `y=grad(x+s)-g`, then `(H,applied)=update(H,s,y)`. The secant condition is `H_next@y=s`. Default `update=None` selects `bfgs_update`, so this default has the same algorithm as section 14.

```python
from algorithms.quasi_newton import quasi_newton
# Run the common quadratic setup above first.
x, fx, history, status = quasi_newton(f, grad, x0, H0=np.eye(2), Q=Q, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** inverse approximations can be poorly conditioned or yield non-descent directions; local convergence requires more than a secant condition. No global minimum guarantee for nonconvex objectives.

**Implementation shared by SR1/DFP/BFGS:** requires finite symmetric SPD `H0` of matching shape, otherwise `ValueError`. A non-descent or nearly orthogonal direction resets H to identity and uses `-g`. A failed general line search retries with identity/steepest descent; a failed Q step stops. Custom updates must return `(new_H, applied)` and preserve appropriate shape/meaning; this contract is not comprehensively validated. Safe skipped updates are recorded, not solver failures. No minimum classification or general condition-number control. Statuses: `stationary`, `line_search_failed`, `nonpositive_curvature`, `nonfinite`, `stagnation`, `iteration_limit`.

## 12. Rank-one correction (inverse SR1)

```python
rank_one(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None)
```

**Purpose/update:** use the shared loop with `v=s-H@y`, `H_next=H+outer(v,v)/(v@y)`.

```python
from algorithms.rank_one import rank_one
# Run the common quadratic setup above first.
x, fx, history, status = rank_one(f, grad, x0, Q=Q, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** a small `v@y` makes the correction unstable; SR1 can make H indefinite even from SPD H. It does not inherently maintain descent.

**Implementation:** `sr1_update(H,s,y,tol=1e-8)` skips when `abs(v@y)<=tol*norm(v)*norm(y)` or the candidate is nonfinite; otherwise symmetrizes the result. Shared direction fallback handles unusable directions but this is not a trust-region SR1 method. Initial H must still be SPD. Shared statuses and line-search limitations apply; repeated skips can slow progress.

## 13. DFP

```python
dfp(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None)
```

**Purpose/update:** with `v=H@y`, use `H_next=H+outer(s,s)/(s@y)-outer(v,v)/(y@v)` in the shared loop.

```python
from algorithms.dfp import dfp
# Run the common quadratic setup above first.
x, fx, history, status = dfp(f, grad, x0, Q=Q, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** positive curvature `s@y>0` and a suitable positive definite approximation support SPD preservation; small denominators or poor conditioning make finite-precision updates unreliable.

**Implementation:** `dfp_update(H,s,y,tol=1e-8)` skips if `s@y<=tol*norm(s)*norm(y)` or `y@v<=tol*norm(y)*norm(v)`, or if the candidate is nonfinite. Accepted results are symmetrized. There is no damping or explicit condition-number repair; shared identity resets and statuses apply. A skipped update is not a curvature-error status.

## 14. BFGS

```python
bfgs(f, grad, x0, H0=None, tol=1e-8, max_iter=1000, Q=None)
```

**Purpose/update:** `rho=1/(s@y)`, `V=I-rho*outer(s,y)`, `H_next=V@H@V.T+rho*outer(s,s)` in the shared loop.

```python
from algorithms.bfgs import bfgs
# Run the common quadratic setup above first.
x, fx, history, status = bfgs(f, grad, x0, Q=Q, tol=1e-8)
# x approximately [1/11,7/11], fx approximately -15/22, 'stationary'
```

**Mathematics:** with SPD H and `s@y>0`, the update preserves positive definiteness in exact arithmetic. Nonconvexity, inaccurate gradients, weak curvature, and ill-conditioning can defeat convergence or numerical stability.

**Implementation:** `bfgs_update(H,s,y,tol=1e-8)` skips nonpositive/weak relative curvature `s@y<=tol*norm(s)*norm(y)` and nonfinite candidates, and symmetrizes accepted results. There is no damped BFGS formula or limited-memory storage; H is dense. Shared fallbacks/statuses apply. For all three update helpers, their default curvature tolerance is separate from the wrapper's gradient `tol`: the wrapper does not forward that parameter to the update helper.

## Notebook examples and plots

The notebook's scalar sections use the following alternative to the small scalar examples above:

```python
import numpy as np
from algorithms import bisection, newton, iterative
from common import show_result, plot_scalar

f = lambda x: x**4 + 2*x**2 - 8*x + 3
df = lambda x: 4*x**3 + 4*x - 8
ddf = lambda x: 12*x**2 + 4
h = lambda x: np.cbrt(2-x)

r1 = bisection(f, df, 0, 3)
r2 = newton(f, df, ddf, 3)
r3 = iterative(f, df, h, 0)
# Each reaches x approximately 1, f(x) approximately -2, status 'stationary'.
show_result(r2)
plot_scalar(f, r2[2], -.3, 3.2)
```

Section 2 uses the two-phase example above. At the final inspection, section 1 had been edited during this documentation task: its third matrix row is `[0,1,1,0,1]`, unlike `[1,1,0,0,1]` in the simplex example above and its unchanged mathematical heading. That current input solves to `x=[4,6,0,0,2]`, `z=-38`, rather than `-34`. The row now contains the existing slack variable `x3`; `plot_lp(A[:,:2],b,...)` does not represent that full coupled constraint. Check the intended formulation before using this plot for submission. The notebook is preserved as found. Section 3 independently sets the original equality-form `A,b,c` shown in the README simplex example and gets `y=[0,-3,-2]`, value `-34`, status `optimal`. Sections 7–14 use the common quadratic setup and their individual calls above. To change a notebook problem, edit its input cell and rerun that section; there is no interactive input prompt.

Plot helpers are demonstrations with dimension/form restrictions:

- `plot_scalar(f,history,a,b)`: scalar iterates on the objective and objective history; plotting limits do not constrain the solver.
- `plot_quadratic(Q,b,history)`: two-variable quadratic contours/trajectory; not an n-dimensional or general-objective plot.
- `plot_lp(G,b,history)`: bounded two-variable region `G@x<=b, x>=0`; uses the first two coordinates of history. For equality plots, the notebook stacks `A,-A` and `b,-b`. It is not a general mixed-constraint visualization.
- `plot_phases(history)`: separates artificial-sum and transformed Phase II objectives. The dual's internal Phase II objective may have the opposite sign from its returned value.
- `show_result(result)`: prints status, solution, objective, and basis for five-item LP returns; prints history-based update count and gradient norm for four-item returns. Do not treat printed values as optimal when the status reports failure.

## Reading statuses and numerical results

| Status | Meaning and action |
| --- | --- |
| `optimal` | LP termination tests passed to tolerance; check residuals when changing scale. |
| `stationary` | Small derivative/gradient; check convexity or second-order classification before claiming a minimum. |
| `bracket_converged` | Bisection interval accuracy reached; derivative need not be small. |
| `infeasible` | Two-phase auxiliary objective remains positive; in `dual`, describes the dual. |
| `unbounded` | Simplex found an improving direction without a positive limiting pivot; in `dual`, describes the dual. Returned iterate is not an unbounded optimum. |
| `iteration_limit` | Update/pivot budget exhausted; no convergence claim. |
| `numerical_failure` | LP finite-value, feasibility, or final optimality verification failed. |
| `nonfinite` | Nonfinite objective, derivative, iterate, or relevant matrix encountered; inspect domain and scaling. |
| `stagnation` | No representable movement, or the fixed-point residual stopped improving with a tiny step. |
| `zero_curvature` | Scalar Newton has nonzero derivative but exactly zero second derivative. |
| `cycle_detected` | Scalar Newton proposed an exact previously recorded iterate. |
| `singular_hessian` | Multivariable Newton could not solve its Hessian system. |
| `non_descent` | Multivariable Newton direction has `g@d>=0`. |
| `not_minimum` | Newton reached a stationary point with negative second-order curvature. |
| `line_search_failed` | Bounded strong Wolfe search failed (including retry where implemented). |
| `nonpositive_curvature` | Exact-quadratic step has invalid directional curvature, or CG's denominator fails its relative test. |

`ValueError` indicates rejected inputs rather than a returned status; callback/domain exceptions are not universally caught. Most non-LP stopping tolerances are absolute gradient tolerances: rescaling an objective changes their meaning. Row normalization helps LP scaling but cannot cure nearly dependent bases, extreme columns, overflow, or all floating-point cancellation. A failure status need not mean the mathematical problem has no solution. Conversely, a small gradient on a nonconvex problem does not establish a minimum.

## Verification

The documentation examples were checked against the local Python source and run with its current interfaces. Both standalone test scripts passed on 25 November 2026, including 96 rotated-SPD solves, 80 matrix-case checks, eight targeted regression groups, and 40 mixed-sense LP vertex comparisons. All notebook code cells were also executed in order with a noninteractive plotting backend without saving the notebook. This checks code execution, not visual plot appearance or VS Code kernel selection. Dependencies are unpinned, so another environment can produce different final digits or iteration counts.
