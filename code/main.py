import numpy as np
import matplotlib.pyplot as plt
import os


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
        self.init = None  # used when convergence condition is |u_k+s - u_k| < TOL
        self.exact_solution = np.full_like(b, 5)
        self.fig, self.ax = plt.subplots()

    def linear_iteration(self):
        # perform linear iteration method until convergence
        self.iterations = 0   # initialize number of iterations
        # when only performing linear iteration, set extrapolation parameter to 0
        temp = self.s
        self.s = 0
        x = self.jacobi()
        self.s = temp
        # print results
        print(f'solution: {x}, iterations:{self.iterations}')
        print(f'error={x - self.exact_solution}, norm={np.linalg.norm(x - self.exact_solution, ord=2)}')

    def improved_iteration(self):
        self.iterations = 0  # initialize number of iterations
        while self.iterations <= self.max_iter:
            self.solution = []
            results = []

            # do if convergence condition is |u_k+s - u_k| < TOL
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

            # do if convergence condition is |u_k+s - u_k+s-1| < TOL
            # for _ in range(self.s):
            #     x = self.jacobi()
            #     results.append({'solution': x, 'iterations': self.iterations})
            #     self.initial_value = x

            # print(results)
            for item in results:
                self.solution.append(item['solution'])
                self.dim = len(item['solution'])

            new_solution = []
            for j in range(self.dim):
                lst = []  # s vectors to calculate new solution in the next step
                mapping_vectors = []  # s+1 vectors, mapped from u_j to (u_j, u_exact - u_j) by each dimension for plotting

                # calculate new solution
                for i in range(len(self.solution)):
                    lst.append(self.solution[i][j])
                w1 = self.w1_calculation(lst)
                w2 = self.w2_calculation(lst)
                new_solution.append(-w2 / w1)

                # plotting
                for i in range(len(self.solution)):
                    mapping_vectors.append(
                        np.array([self.solution[i][j], self.exact_solution[j] - self.solution[i][j]]))  # add former s solutions to plotting
                mapping_vectors.append(np.array([-w2 / w1, self.exact_solution[j] + w2 / w1]))  # add new solution to plotting
                # print(mapping_vectors)
                self.plot_points(mapping_vectors, j+1)

            self.iterations += 1  # after calculating a new solution, iterations++
            print(f'new solution:{new_solution}, iterations={self.iterations}')
            print(f'error={new_solution-self.exact_solution}, norm={np.linalg.norm(new_solution - self.exact_solution, ord=2)}')

            # break when convergence
            if np.linalg.norm(new_solution - self.initial_value, ord=2) < self.tol:
                break
            # update initial value
            self.initial_value = new_solution

    @staticmethod
    def w1_calculation(lst):
        total_sum = 0
        inner_sum = 0
        for k in range(len(lst) - 1):
            for j in range(k, len(lst)):
                inner_sum += lst[k] - lst[j]
            total_sum += (lst[k + 1] - lst[k]) * inner_sum
        return total_sum

    @staticmethod
    def w2_calculation(lst):
        total_sum = 0
        inner_sum = 0
        for k in range(len(lst) - 1):
            for j in range(k, len(lst)):
                inner_sum += lst[j] ** 2 - lst[j] * lst[k]
            total_sum += (lst[k + 1] - lst[k]) * inner_sum
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

    def plot_points(self, lst, dim):
        self.ax.clear()

        x_values = [point[0] for point in lst]
        y_values = [point[1] for point in lst]

        plt.scatter(x_values, y_values, marker='o', label=f'iter={self.iterations+1}')
        plt.title(f'iterations={self.iterations+1}, dim={dim}')
        plt.legend()
        plt.grid(True)
        # plt.show()

        folder_name = f"iter_{self.iterations+1}"
        os.makedirs(folder_name, exist_ok=True)
        file_name = f"{folder_name}/dim_{dim}.png"
        plt.savefig(file_name)


if __name__ == "__main__":
    np.set_printoptions(precision=16, suppress=True)
    # A = np.array([[4, -1, 0, 0],
    #               [-1, 4, -1, 0],
    #               [0, -1, 4, -1],
    #               [0, 0, -1, 4]])
    #
    # b = np.array([15, 10, 10, 15])

    n = 6
    A = np.diag(4 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=-1) + np.diag(-1 * np.ones(n - 1), k=1)
    b = np.full(n, 10)
    b[0] = b[-1] = 15

    example = ImprovedIterationScheme(tol=1e-5, max_iter=40, A=A, b=b, initial_value=None, s=4)
    example.linear_iteration()
    input()
    example.improved_iteration()
