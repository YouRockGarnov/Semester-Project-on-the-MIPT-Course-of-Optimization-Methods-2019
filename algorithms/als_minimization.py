import numpy as np;


def als_minimization(A, mask, k=10, mu=1e-6, epsilon=1e-6):
    """
    alternating least squares minimization.
    ---------------------------------------------------------------------
    |   A : m x n array,                                                |
    |       matrix to complete                                          |
    |   mask : m x n array,                                             |
    |       matrix with entries zero (if missing) or one (if present)   |
    |   k : nubmer of left singular vectors to be used,                 |
    |       Rank on the resulting matrix can't be higher than k         |
    |   mu : coefficient of unit vector,                                |
    |       Unit vector is user to estimate minimum of function via     |
    |       solving of system of linear equations                       |
    |   epsilon : criteria of convergence,                              |
    |       We stop calculation if norm of difference of two neibouring |
    |       iteration in less than epsilon                              |
    ---------------------------------------------------------------------
    """

    m, n = A.shape
    max_iterations = 100

    U, _, _ = np.linalg.svd(np.multiply(A, mask))
    U = U[:, :k]
    V = np.zeros((n, k))

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    X = np.dot(U, V.T)

    for _ in range(max_iterations):
        for j in range(n):
            V[j] = np.linalg.solve(
                np.linalg.multi_dot([U.T, C_v[j], U]) + mu * np.eye(k),
                np.linalg.multi_dot([U.T, C_v[j], A[:,j]]), )

        for i in range(m):
            U[i] = np.linalg.solve(
                np.linalg.multi_dot([V.T, C_u[i], V]) + mu * np.eye(k),
                np.linalg.multi_dot([V.T, C_u[i], A[i,:]]), )

        new_X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - new_X) / m / n
        if mean_diff < epsilon:
            break
        X = new_X

    return new_X
