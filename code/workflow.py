import timeit
import numpy as np


class IterativeAlgorithm:
    def __init__(self, tol, max_iter, A, b, initial_guess, s):
        self.tol = tol  # convergence toleration
        self.max_iter = max_iter  # max iterations
        self.A = A
        self.b = b  # s.t. Ax = b
        self.initial_guess = np.zeros_like(b) if initial_guess is None else initial_guess  # initial guess x_0
        self.s = s  # number of times performing the linear iterations after convergence
        self.iterations = 0

    def input_stream(self):  # to input A, b
        pass

    def control_flow(self):
        results = []
        for _ in range(self.s):
            x, elapsed_time = self.jacobi()
            results.append({'solution': x, 'iterations': self.iterations, 'elapsed_time': elapsed_time})
            self.initial_guess = x
        print(results)

        self.iterations = 0
        self.initial_guess = np.zeros_like(b)

        results = []
        for _ in range(self.s):
            x, elapsed_time = self.gauss_seidel()
            results.append({'solution': x, 'iterations': self.iterations, 'elapsed_time': elapsed_time})
            self.initial_guess = x
        print(results)

    def jacobi(self):
        start_time = timeit.default_timer()
        x = np.array(self.initial_guess, dtype=np.double)
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x_old[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
                self.iterations += 1
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < self.tol:
                break

        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time  # calculate running time

        return x, elapsed_time

    def gauss_seidel(self):
        start_time = timeit.default_timer()
        x = np.array(self.initial_guess, dtype=np.double)
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
                self.iterations += 1
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < self.tol:
                break

        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time  # calculate running time

        return x, elapsed_time

    def cg(self):  # conjugate gradient method (which require A to be symmetric positive definite)
        pass

    def gmres(self):  # generalized minimal residual method
        pass


class ExtraLeast:
    def __init__(self, initial_u, s):
        self.initial_u = initial_u  # initial value (dim is not decided)
        self.s = s  # extrapolation parameter

    def workflow(self):  # calculate the approximate solution of components of each dimension separately
        pass

    def extra_least(self):
        pass


if __name__ == "__main__":
    np.set_printoptions(precision=16, suppress=True)
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)

    b = np.array([15, 10, 10, 10], dtype=float)

    iterative_algorithm = IterativeAlgorithm(tol=1e-6, max_iter=1000, A=A, b=b, initial_guess=None, s=20)
    iterative_algorithm.control_flow()
