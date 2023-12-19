import numpy as np


class ImprovedIterationScheme:
    def __init__(self, tol, max_iter, A, b, initial_value, s):
        self.tol = tol  # convergence toleration of iterative algorithm
        self.max_iter = max_iter  # max iterations
        self.A = A
        self.b = b  # s.t. Ax = b
        self.initial_value = np.zeros_like(b) if initial_value is None else initial_value  # initial guess x_0
        self.s = s  # extrapolation parameter
        self.iterations = 0  # count of iterations
        self.dim = 0  # dimension of solution vector
        self.solution = []  # initialize solution list
        self.init = None

    def linear_iteration(self):
        # perform linear iteration method until convergence
        self.iterations = 0
        temp = self.s
        self.s = 0
        x = self.jacobi()
        self.s = temp
        print(f'solution: {x}, iterations:{self.iterations}')

    def improved_iteration(self):
        self.iterations = 0
        while self.iterations <= self.max_iter:
            self.solution = []
            results = []
            mark = 0
            for _ in range(self.s):
                x = self.jacobi()
                results.append({'solution': x, 'iterations': self.iterations})
                self.initial_value = x
                if mark == 0:
                    self.init = self.initial_value
                mark += 1
            mark = 0
            self.initial_value = self.init
            print(results)
            for item in results:
                self.solution.append(item['solution'])
                self.dim = len(item['solution'])

            new_solution = []
            for j in range(self.dim):
                lst = []
                for i in range(len(self.solution)):
                    # print(self.solution[i][j])
                    lst.append(self.solution[i][j])
                # print(lst)

                w1 = self.w1_calculation(lst)
                # print(f'w1={w1}')
                w2 = self.w2_calculation(lst)
                # print(f'w2={w2}')
                # print(f'-w2/w1={-w2 / w1}')
                new_solution.append(-w2 / w1)
            self.iterations += 1
            print(f'new solution:{new_solution}, iterations={self.iterations}')

            if np.linalg.norm(new_solution - self.initial_value, ord=2) < self.tol:
                break

            self.initial_value = new_solution

    @staticmethod
    def w1_calculation(lst):
        total_sum = 0
        inner_sum = 0
        for i in range(len(lst) - 1):
            for j in range(len(lst)):
                inner_sum += lst[i] - lst[j]
            # print(inner_sum)
            # print(lst[i+1] - lst[i])
            total_sum += (lst[i + 1] - lst[i]) * inner_sum
        return total_sum

    @staticmethod
    def w2_calculation(lst):
        total_sum = 0
        inner_sum = 0
        for i in range(len(lst) - 1):
            for j in range(len(lst)):
                inner_sum += lst[j] ** 2 - lst[j] * lst[i]
            # print(inner_sum)
            # print(lst[i+1] - lst[i])
            total_sum += (lst[i + 1] - lst[i]) * inner_sum
        return total_sum

    def jacobi(self):
        x = np.array(self.initial_value, dtype=np.double)
        s = self.s
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x_old[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
            self.iterations += 1
            # when s is set to 0, go on until convergence
            if s != 0:
                s -= 1
                break
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=2) < self.tol:
                break

        return x

    def gauss_seidel(self):
        x = np.array(self.initial_value, dtype=np.double)
        s = self.s
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
            self.iterations += 1
            # when s is set to 0, go on until convergence
            if s != 0:
                s -= 1
                break
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=2) < self.tol:
                break

        return x


if __name__ == "__main__":
    np.set_printoptions(precision=16, suppress=True)
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 4]])

    b = np.array([15, 10, 10, 15])

    # n = 10
    # A = np.diag(4 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=-1) + np.diag(-1 * np.ones(n - 1), k=1)
    # b = np.full(n, 10)
    # b[0] = b[-1] = 15

    example = ImprovedIterationScheme(tol=1e-7, max_iter=200, A=A, b=b, initial_value=None, s=3)
    example.linear_iteration()
    input()
    example.improved_iteration()
