# optimization.py
# Optimization wrappers and parameter sweep utilities
# Originally used in: Economics Research in conjuction with my work at Ellington Management Group (transition matrix estimation with lambda sweeps),
#                     Yale University ECON 412: International Environmental Economics (constrained cost minimization)

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def parameter_sweep_1d(objective_fn, param_range, x0, args=(),
                       bounds=None, constraints=None, method=None):
    """Run a 1D parameter sweep over an objective function.

    Minimizes the objective for each value in param_range, collecting
    results into a DataFrame. The parameter value is prepended to the
    args tuple passed to the objective.

    Args:
        objective_fn: callable with signature f(x, param, *args)
        param_range: array-like of parameter values to sweep
        x0: initial guess for the decision variables
        args: additional arguments passed to objective_fn after the parameter
        bounds: variable bounds for scipy.optimize.minimize
        constraints: constraint dicts for scipy.optimize.minimize
        method: optimization method (default: let scipy choose)

    Returns:
        results_df: DataFrame with columns ['param', 'objective', 'success']
        solutions: list of solution arrays (one per parameter value)
    """
    results = []
    solutions = []

    for idx, param in enumerate(param_range):
        full_args = (param,) + tuple(args)
        res = minimize(objective_fn, x0=x0, args=full_args,
                       bounds=bounds, constraints=constraints, method=method)

        results.append({
            'param': param,
            'objective': res.fun,
            'success': res.success
        })
        solutions.append(res.x)

        print(f'Iter {idx}: param={param:.4g}, objective={res.fun:.6g}, success={res.success}')

    results_df = pd.DataFrame(results)
    return results_df, solutions


def parameter_sweep_2d(objective_fn, param1_range, param2_range, x0, args=(),
                       bounds=None, method=None):
    """Run a 2D parameter grid sweep over an objective function.

    Minimizes the objective for each (param1, param2) combination.
    The two parameter values are prepended to the args tuple.

    Args:
        objective_fn: callable with signature f(x, param1, param2, *args)
        param1_range: array-like of first parameter values
        param2_range: array-like of second parameter values
        x0: initial guess for the decision variables
        args: additional arguments after the two parameters
        bounds: variable bounds for scipy.optimize.minimize
        method: optimization method

    Returns:
        results_df: DataFrame with columns ['param1', 'param2', 'objective', 'success']
        solutions: 2D list of solution arrays indexed [i][j]
    """
    results = []
    solutions = [[None]*len(param2_range) for _ in range(len(param1_range))]
    iters = 0

    for idx, p1 in enumerate(param1_range):
        for idy, p2 in enumerate(param2_range):
            full_args = (p1, p2) + tuple(args)
            res = minimize(objective_fn, x0=x0, args=full_args,
                           bounds=bounds, method=method)

            results.append({
                'param1': p1,
                'param2': p2,
                'objective': res.fun,
                'success': res.success
            })
            solutions[idx][idy] = res.x
            iters += 1

    print(f'Completed {iters} optimizations over {len(param1_range)}x{len(param2_range)} grid')
    results_df = pd.DataFrame(results)
    return results_df, solutions


def constrained_minimize(objective_fn, x0, args=(), bounds=None,
                         constraints=None, method=None):
    """Thin wrapper around scipy.optimize.minimize with standardized output.

    Runs a single constrained minimization and returns results in a
    consistent dict format.

    Args:
        objective_fn: callable to minimize
        x0: initial guess (array-like)
        args: extra arguments to objective_fn
        bounds: sequence of (min, max) pairs for each variable
        constraints: dict or list of constraint dicts
        method: solver method string

    Returns:
        dict with keys: 'x' (solution), 'objective' (function value),
        'success' (bool), 'message' (solver message)
    """
    res = minimize(objective_fn, x0=x0, args=args, bounds=bounds,
                   constraints=constraints, method=method)

    return {
        'x': res.x,
        'objective': res.fun,
        'success': res.success,
        'message': res.message
    }
