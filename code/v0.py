import timeit
import numpy as np
import matplotlib.pyplot as plt


def input_stream():  # to input A, b
    pass


class IterativeAlgorithm:
    def __init__(self, tol, max_iter, A, b, initial_value, s):
        self.tol = tol  # convergence toleration of iterative algorithm
        self.max_iter = max_iter  # max iterations
        self.A = A
        self.b = b  # s.t. Ax = b
        self.initial_value = np.zeros_like(b) if initial_value is None else initial_value  # initial guess x_0
        self.s = s + 1  # number of times performing the linear iterations after convergence
        self.iterations = 0  # count of iterations
        self.output_vectors = []  # the 2d vectors which will be used in regression
        self.dim = 0  # dimension of solution vector
        self.solution = []  # initialize solution list
        self.u_dict = {}  # dict used to save u_k

    def linear_iteration(self):
        results1 = []
        for _ in range(self.s):
            x, elapsed_time = self.jacobi()
            results1.append({'solution': x, 'iterations': self.iterations, 'elapsed_time': elapsed_time})
            self.initial_value = x
        print(results1)
        self.mapping(results1)

    def improved_iteration(self):
        self.extra_least(1e-6)

    def mapping(self, lst):  # map u_j to (u_j, u_j+1 - u_j) by each dimension
        for item in lst:
            self.solution.append(item['solution'])
            self.dim = len(item['solution'])  # record the dimension

        for j in range(self.dim):
            processed_vectors = []
            for i in range(len(self.solution) - 1):
                processed_vectors.append(
                    {f'dim={j + 1}': np.array([self.solution[i][j], self.solution[i + 1][j] - self.solution[i][j]])})
            self.output_vectors.append(processed_vectors)

    def extra_least(self, tol2):
        """
        1. separate output_vectors into u_k to u_k+s, by each dimension
        2. calculate -|A^TA|w2/|A^TA|w1
        3. stop when convergence, else set k=k+1 and do 1-2 again
        """
        # print(self.output_vectors)
        dim_counter = 0
        for item in self.output_vectors:
            print(f"dim={dim_counter + 1}")
            u_counter = 0
            u_lst = []
            for sub_item in item:
                # print(sub_item)
                for value in sub_item.values():
                    array_name = f"u_{u_counter + 1}"
                    self.u_dict[array_name] = value
                    u_lst.append(value)
                    u_counter += 1
            for array_name, array_value in self.u_dict.items():
                print(f"{array_name}: {array_value}")
            print(u_lst)

            for k in range(len(u_lst)):
                # calculate |A^TA|w1 and |A^TA|w2
                w_1 = self.calculate_w1(k, u_lst)
                w_2 = self.calculate_w2(k, u_lst)
                print(w_2)
                u_new = -np.divide(w_2, w_1)
                u_lst.append(u_new)
                if np.linalg.norm(u_lst[k] - u_new, ord=2) < tol2:
                    print(u_lst)
                    break

            dim_counter += 1  # go to the next dimension

    @staticmethod
    def calculate_w1(k, u_arrays):
        total_sum = 0

        return total_sum

    @staticmethod
    def calculate_w2(k, u_arrays):
        total_sum = 0
        inner_sum = 0
        for i in range(k, len(u_arrays)):
            for j in range(k, len(u_arrays)):
                inner_sum += u_arrays[i]-u_arrays[j]
            total_sum += np.multiply((u_arrays[i+1]-u_arrays[i]), inner_sum)
        return total_sum

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


if __name__ == "__main__":
    np.set_printoptions(precision=16, suppress=True)
    A = np.array([[4, -1, 0, 0],
                  [-1, 4, -1, 0],
                  [0, -1, 4, -1],
                  [0, 0, -1, 3]], dtype=float)

    b = np.array([15, 10, 10, 10], dtype=float)

    iterative_algorithm = IterativeAlgorithm(tol=1e-5, max_iter=200, A=A, b=b, initial_value=None, s=5)
    iterative_algorithm.linear_iteration()
    iterative_algorithm.extra_least(1e-6)
