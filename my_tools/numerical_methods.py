# numerical_methods.py
# Numerical differentiation, linear algebra, and error metrics
# Originally used in: Optimization research with Ellington Management Group (transition probability matrix estimation),
#                     Fifth Year (spline fitting for volatility surfaces)

import numpy as np
from scipy.optimize import minimize


def numderiv_grid(grid, h):
    """Compute numerical derivatives on a 2D grid using finite differences.

    Uses forward difference at left boundary, backward difference at right boundary,
    and central difference for interior points.

    Args:
        grid: 2D numpy array (rows x cols)
        h: step size between grid points

    Returns:
        2D numpy array of same shape with derivative values
    """
    T = grid.shape[0]
    K = grid.shape[1]
    deriv = np.zeros((T, K))
    for idx in range(T):
        for idy in range(K):
            if idy == 0:
                deriv[idx, idy] = abs(grid[idx, idy] - grid[idx, idy+1]) / h
            elif idy == K-1:
                deriv[idx, idy] = abs(grid[idx, idy] - grid[idx, idy-1]) / h
            else:
                deriv[idx, idy] = abs(grid[idx, idy+1] - grid[idx, idy-1]) / (2*h)
    return deriv


def generate_tridiagonal_system(num_levels):
    """Generate a symmetric banded matrix with [1, -4, 6, -4, 1] stencil.

    Useful for natural cubic spline systems and finite difference discretizations.
    Boundary rows are modified for natural boundary conditions.

    Args:
        num_levels: size of the square matrix

    Returns:
        numpy array of shape (num_levels, num_levels)
    """
    main_diag = np.ones(num_levels) * 6
    X = np.diag(main_diag)
    first_diag = np.ones(num_levels-1) * -4
    X = X + np.diag(first_diag, k=1) + np.diag(first_diag, k=-1)
    second_diag = np.ones(num_levels-2)
    X = X + np.diag(second_diag, k=2) + np.diag(second_diag, k=-2)

    X[0, 0] = 3
    X[1, 0] = -3
    X[num_levels-1, num_levels-1] = 3
    X[num_levels-2, num_levels-1] = -3

    return X


def compute_error(actual, predicted, error_type='SSQ'):
    """Compute reconstruction error between two matrices.

    Args:
        actual: numpy array (ground truth)
        predicted: numpy array (model output)
        error_type: 'SSQ' for sum of squared errors, 'SA' for sum of absolute errors

    Returns:
        scalar error value
    """
    diff = actual - predicted
    if error_type == 'SSQ':
        return ((diff)**2).sum()
    elif error_type == 'SA':
        return abs(diff).sum()


def rowsum_penalty(P, error_type='SA'):
    """Penalize rows of a matrix that don't sum to 1 (stochastic matrix constraint).

    Args:
        P: numpy array (matrix whose rows should sum to 1)
        error_type: 'SSQ' or 'SA'

    Returns:
        scalar penalty value
    """
    expected_rowsum = np.ones((P.shape[0],))
    actual_rowsum = np.sum(P, axis=1)
    if error_type == 'SSQ':
        return ((expected_rowsum - actual_rowsum)**2).sum()
    elif error_type == 'SA':
        return (abs(expected_rowsum - actual_rowsum)).sum()


def kl_divergence(P, Q):
    """Compute generalized KL divergence between two non-negative matrices.

    Uses the form: sum(P * log(P/Q) - P + Q)

    Args:
        P: numpy array (reference distribution)
        Q: numpy array (approximate distribution)

    Returns:
        scalar divergence value
    """
    left = P * np.log(P / Q)
    right = P - Q
    diff = left - right
    return diff.sum().sum()


def recover_eigendecomposition(P):
    """Eigenvalue decomposition with kernel/stationary distribution recovery.

    Computes the eigendecomposition of P and recovers the diagonal scaling
    matrix D = diag(1/|v_1|) and the similarity transform F = (1/lambda_1) * D @ P @ D^{-1}.

    Args:
        P: square numpy array

    Returns:
        F: transformed matrix
        psi: stationary kernel (1/|first_eigenvector|)
        eigenvals: array of eigenvalues
        eigenvecs: matrix of eigenvectors
    """
    eigenvals, eigenvecs = np.linalg.eig(P)
    first_eigenval = abs(eigenvals[0])
    first_eigenvec = abs(eigenvecs[:, 0])

    psi = 1 / first_eigenvec
    D = np.diag(psi)
    D_inv = np.linalg.inv(D)

    F = (1 / first_eigenval) * D @ P @ D_inv

    return F, psi, eigenvals, eigenvecs
