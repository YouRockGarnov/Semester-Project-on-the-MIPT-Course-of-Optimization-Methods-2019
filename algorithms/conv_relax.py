from cvxpy import *
import numpy as np


def floor_singular_values(X, decimals):
    U, sigma, V = np.linalg.svd(X)
    sigma = np.diag(sigma.round(decimals))
    sigma = np.resize(sigma, (U.shape[0], V.shape[0]))
    return np.linalg.multi_dot([U, sigma, V.conjugate()])


def nuclear_norm_minimization(A, mask, value_protection_coefficient=100, decimals=2):
    """
    Convex relaxation via Nuclear Norm Minimization [Candès and Recht approach]
    ---------------------------------------------------------------------
    |   A : m x n array,                                                |
    |       matrix to complete                                          |
    |   mask : m x n array,                                             |
    |       matrix with entries zero (if missing) or one (if present)   |
    |   value_protection_coefficient : significance of sum_squares of   |
    |       multiply(mask, X - A), shows how much constraints enforced  |
    |   decimals : to which decimal place to floor singular values      |
    ---------------------------------------------------------------------
    """
    X = Variable(shape=A.shape)
    objective = Minimize(norm(X, "nuc") + value_protection_coefficient * sum_squares(multiply(mask, X - A)))
    problem = Problem(objective, [])
    problem.solve(solver=SCS)
    return floor_singular_values(X.value, decimals)
