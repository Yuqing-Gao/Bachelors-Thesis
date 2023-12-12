import timeit
import numpy as np
import matplotlib.pyplot as plt


def input_stream():  # to input A, b
    pass


class IterativeAlgorithm:
    def __init__(self, tol, max_iter, A, b, initial_value, s):
        self.tol = tol  # convergence toleration
        self.max_iter = max_iter  # max iterations
        self.A = A
        self.b = b  # s.t. Ax = b
        self.initial_value = np.zeros_like(b) if initial_value is None else initial_value  # initial guess x_0
        self.s = s + 1  # number of times performing the linear iterations after convergence
        self.iterations = 0  # count of iterations
        self.output_vectors = []  # the 2d vectors which will be used in regression

    def control_flow(self):
        results1 = []
        for _ in range(self.s):
            x, elapsed_time = self.jacobi()
            results1.append({'solution': x, 'iterations': self.iterations, 'elapsed_time': elapsed_time})
            self.initial_value = x
        print(results1)
        self.mapping(results1)
        # self.scatter_plot(self.output_vectors)
        print(self.output_vectors)

        # self.iterations = 0
        # self.initial_value = np.zeros_like(b)
        # results2 = []
        # self.output_vectors=[]
        # for _ in range(self.s):
        #     x, elapsed_time = self.gauss_seidel()
        #     results2.append({'solution': x, 'iterations': self.iterations, 'elapsed_time': elapsed_time})
        #     self.initial_value = x
        # print(results2)
        # self.mapping(results2)
        # self.scatter_plot(self.output_vectors)
        # print(self.output_vectors)

    def mapping(self, lst):  # map u_j to (u_j, u_j+1 - u_j) by each dimension
        solution = []
        dim = 0
        for item in lst:
            solution.append(item['solution'])
            dim = len(item['solution'])  # record the dimension

        for j in range(dim):
            processed_vectors = []
            for i in range(len(solution) - 1):
                processed_vectors.append(
                    {f'dim={j + 1}': np.array([solution[i][j], solution[i + 1][j] - solution[i][j]])})
            self.output_vectors.append(processed_vectors)

    @staticmethod
    def scatter_plot(data):
        dimensions = [list(item[0].keys())[0] for item in data[0]]
        x_values = {dim: [] for dim in dimensions}
        y_values = {dim: [] for dim in dimensions}

        for dimension_data in data:
            for item in dimension_data:
                dim_key = list(item.keys())[0]
                x, y = item[dim_key]
                x_values[dim_key].append(x)
                y_values[dim_key].append(y)

        # 绘制图形
        for dim_key in dimensions:
            plt.scatter(x_values[dim_key], y_values[dim_key], label=dim_key)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Scatter Plot for Each Dimension')
        plt.show()

    def jacobi(self):
        start_time = timeit.default_timer()
        x = np.array(self.initial_value, dtype=np.double)
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x_old[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
            self.iterations += 1
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=2) < self.tol:
                break

        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time  # calculate running time

        return x, elapsed_time

    def gauss_seidel(self):
        start_time = timeit.default_timer()
        x = np.array(self.initial_value, dtype=np.double)
        # iterate
        for _ in range(self.max_iter):
            x_old = x.copy()
            # loop over rows
            for i in range(self.A.shape[0]):
                x[i] = (self.b[i] - np.dot(self.A[i, :i], x[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
            self.iterations += 1
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=2) < self.tol:
                break

        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time  # calculate running time

        return x, elapsed_time

    def cg(self):  # conjugate gradient method (which require A to be symmetric positive definite)
        pass

    def gmres(self):  # generalized minimal residual method
        pass


class Regressor:
    def __init__(self, lst, s, tol):
        self.initial_u = lst  # initial value (including vectors separated from several dimensions)
        self.s = s  # extrapolation parameter
        self.tol = tol  # tolerance for

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

    iterative_algorithm = IterativeAlgorithm(tol=1e-5, max_iter=1000, A=A, b=b, initial_value=None, s=5)
    iterative_algorithm.control_flow()
    vectors = iterative_algorithm.output_vectors
