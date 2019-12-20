#! /usr/bin/env python3
from algorithms.als_minimization import als_minimization
from algorithms.conv_relax import nuclear_norm_minimization as nuclear_solve
import numpy as np
import scipy.stats as sps


def main():
    print_matrises = False
    cases = 10
    N, M, est_known = 100, 50, 1000
    threshold = 1e-3

    print('''

    running {} cases...
    '''.format(cases))

    for i in range(cases):
        A, mask = create_case(N=N, M=M, est=est_known)
        X1 = als_minimization(A, mask)
        X2 = nuclear_solve(A, mask)

        print("case={}     entries={}     filled-with-zeros={}     als-rank={}     nucl-rank={}".format(
            i + 1,
            mask.sum(),
            np.linalg.matrix_rank(np.multiply(A, mask)),
            np.linalg.matrix_rank(X1),
            np.linalg.matrix_rank(X2),
        ))

        print("als maximum deviation={0:.5f}   ".format(np.max(np.abs(np.multiply(X1, mask) - np.multiply(A, mask))))
            + "  nucl maximum deviation={0:.5f}".format(np.max(np.abs(np.multiply(X2, mask) - np.multiply(A, mask)))))


def create_case(N=100, M=100, est=200):
    A = sps.randint.rvs(low=-10, high=10, size=(N, M))
    mask = sps.bernoulli.rvs(est/(N*M), size=(N, M))
    return A, mask


if __name__ == "__main__":
    main()
