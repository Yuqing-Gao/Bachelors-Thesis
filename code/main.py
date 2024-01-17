import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import os


# 1. set k = 1 (use k = 0 instead in python)
# 2. attain u_k+1, ..., u_k+s-1 by linear iteration method
# 3. derive u_k+s by accelerating method using u_k to u_k+s-1
# 4. if convergence, stop. Else do 2 and 3 again where u_k+s is considered as initial value


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
        self.solution = []  # initialize solution list for every iteration
        self.all_solutions = []  # initialize solution list
        self.init_start = None  # u^bar_k
        self.exact_solution = np.full_like(b, 1)  # exact solution of the given question
        self.ax = plt.gca()

    def linear_iteration(self):
        # perform linear iteration method until convergence
        self.iterations = 0  # initialize number of iterations
        # when only performing linear iteration, set extrapolation parameter to 0
        temp = self.s
        self.s = 0
        x = self.jacobi()
        self.s = temp
        # print results
        # print(f'solution: {x}, iterations:{self.iterations}')
        # print(f'error={x - self.exact_solution}, norm={np.linalg.norm(x - self.exact_solution, ord=2)}')

    def improved_iteration(self, if_plot):
        self.iterations = 0  # initialize number of iterations
        while self.iterations <= self.max_iter:
            self.solution = []
            results = []

            # do if convergence condition is |u_k+s - u_k| < TOL
            # self.init_start = self.initial_value  # record u^bar_k
            # results.append({'solution': self.initial_value, 'iterations': self.iterations})  # add the first initial value to solution list
            # for _ in range(self.s):
            #     x = self.jacobi()
            #     results.append({'solution': x, 'iterations': self.iterations})
            #     self.initial_value = x

            # do if convergence condition is |u_k+s - u_k+s-1| < TOL
            for _ in range(self.s):
                x = self.jacobi()
                results.append({'solution': x, 'iterations': self.iterations})
                self.initial_value = x

            # print(results)
            # add the iterating solutions to solution list
            for item in results:
                self.solution.append(item['solution'])
                self.all_solutions.append(item['solution'])
                self.dim = len(item['solution'])

            new_solution = []  # list to restore new solution calculated by -w2/w1
            for j in range(self.dim):
                lst = []  # s+1 vectors to calculate new solution in the next step
                mapping_vectors = []  # s+1 vectors, mapped from u_j to (u_j, u_exact - u_j) by each dimension for plotting

                # calculate new solution
                for i in range(len(self.solution)):
                    lst.append(self.solution[i][j])

                w1 = self.w1_calculation(lst)
                if w1 == 0:  # if divided by 0, stop iterating
                    new_solution = self.init_start
                    break
                w2 = self.w2_calculation(lst)
                new_solution.append(-w2 / w1)

                # plotting
                for i in range(len(self.solution)):
                    mapping_vectors.append(
                        np.array([self.solution[i][j],
                                  self.exact_solution[j] - self.solution[i][j]]))  # add former s solutions to plotting
                mapping_vectors.append(
                    np.array([-w2 / w1, self.exact_solution[j] + w2 / w1]))  # add new solution to plotting
                # print(mapping_vectors)
                # if if_plot is True:
                #     self.plot_points(mapping_vectors, j+1)

            # new_solution = new_solution + (np.array(new_solution) - np.array(self.initial_value)) * np.exp(-self.iterations ** 2)
            if self.iterations <= 100:
                new_solution = self.initial_value + (np.array(new_solution) - self.initial_value) * 1
            else:
                new_solution = self.jacobi()

            # if if_plot is True:
            #     self.plot_points_norm()
            #     self.plot_points_value()

            self.iterations += 1  # after calculating a new solution, iterations++

            # print(f'new solution:{new_solution}, iterations={self.iterations}')
            # print(
            #     f'error={new_solution - self.exact_solution}, norm={np.linalg.norm(new_solution - self.exact_solution, ord=2)}')

            # break when convergence
            if np.linalg.norm(np.array(new_solution) - np.array(self.initial_value), ord=2) < self.tol:
                for i in range(len(self.all_solutions)):
                    print(self.all_solutions[i], i)
                break
            # update initial value
            self.initial_value = new_solution

    @staticmethod
    def w1_calculation(lst):
        total_sum = 0
        for k in range(len(lst) - 1):
            inner_sum = 0
            for j in range(len(lst) - 1):
                inner_sum += lst[k] - lst[j]
            total_sum += (lst[k + 1] - lst[k]) * inner_sum
        return total_sum

    @staticmethod
    def w2_calculation(lst):
        total_sum = 0
        for k in range(len(lst) - 1):
            inner_sum = 0
            for j in range(len(lst) - 1):
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
            print(x, self.iterations)
        return x

    def plot_points(self, lst, dim):
        self.ax.clear()

        y_values = [point[0] for point in lst]
        x_values = list(range(self.iterations - self.s, self.iterations - self.s + len(lst)))

        plt.scatter(x_values, y_values, marker='.', label=f'iter={self.iterations + 1}')
        plt.axhline(y=self.exact_solution[dim - 1])
        plt.xlabel('number of iterations')
        plt.ylabel('value')
        plt.title(f'dim={dim}')
        plt.legend()
        plt.grid(True)
        # plt.show()

        folder_name = f"results/iter_{self.iterations + 1}"
        os.makedirs(folder_name, exist_ok=True)
        file_name = f"{folder_name}/dim_{dim}.png"
        plt.savefig(file_name)

    def plot_points_norm(self):
        self.ax.clear()

        folder_name = f"results/iter_{self.iterations + 1}"
        os.makedirs(folder_name, exist_ok=True)
        self.ax.clear()
        x_values = list(range(self.iterations - self.s, self.iterations - self.s + len(self.solution)))
        y_values = [np.linalg.norm(self.exact_solution - point, ord=2) for point in self.solution]
        plt.scatter(x_values, y_values, marker='.', label='solutions')
        plt.xlabel(f'number of iterations')
        plt.ylabel('error in 2-norm')
        plt.legend()
        plt.grid(True)

        file_name = f"{folder_name}/error_norm.png"
        plt.savefig(file_name)

    def plot_points_value(self):
        self.ax.clear()

        folder_name = f"results/iter_{self.iterations}"
        os.makedirs(folder_name, exist_ok=True)
        self.ax.clear()
        x_values = list(range(self.iterations - self.s, self.iterations - self.s + len(self.solution)))
        y_values = [np.linalg.norm(point, ord=2) for point in self.solution]
        plt.scatter(x_values, y_values, marker='.', label='solutions')
        plt.axhline(y=np.linalg.norm(self.exact_solution, ord=2))
        plt.xlabel(f'number of iterations')
        plt.ylabel('value in 2-norm')
        plt.legend()
        plt.grid(True)

        file_name = f"{folder_name}/value_norm.png"
        plt.savefig(file_name)

    def plot_all_solutions(self):
        folder_name = "results/all_iterating_solutions"
        os.makedirs(folder_name, exist_ok=True)
        for i in range(self.dim):
            self.ax.clear()
            x_values = list(range(len(self.all_solutions)))
            y_values = [self.exact_solution[i] - point[i] for point in self.all_solutions]
            plt.scatter(x_values, y_values, marker='.', label='solutions')
            plt.xlabel('number of iterations')
            plt.ylabel('error')
            plt.title(f'dim={i + 1}')
            plt.legend()
            plt.grid(True)

            file_name = f"{folder_name}/dim_{i + 1}.png"
            plt.savefig(file_name)

    def plot_all_solutions_norm(self):
        folder_name = "results/all_iterating_solutions"
        os.makedirs(folder_name, exist_ok=True)
        self.ax.clear()
        x_values = list(range(len(self.all_solutions)))
        y_values = [np.linalg.norm(self.exact_solution - point, ord=2) for point in self.all_solutions]
        plt.scatter(x_values, y_values, marker='.', label='solutions')
        plt.xlabel('number of iterations')
        plt.ylabel('error in 2-norm')
        plt.legend()
        plt.grid(True)

        file_name = f"{folder_name}/error_norm.png"
        plt.savefig(file_name)

    def plot_all_solutions_value(self):
        folder_name = "results/all_iterating_solutions"
        os.makedirs(folder_name, exist_ok=True)
        self.ax.clear()
        x_values = list(range(len(self.all_solutions)))
        y_values = [np.linalg.norm(point, ord=2) for point in self.all_solutions]
        plt.scatter(x_values, y_values, marker='.', label='solutions')
        plt.axhline(y=np.linalg.norm(self.exact_solution, ord=2))
        plt.xlabel('number of iterations')
        plt.ylabel('value in 2-norm')
        plt.legend()
        plt.grid(True)

        file_name = f"{folder_name}/value_norm.png"
        plt.savefig(file_name)

    def plot_all_solutions_value_3d(self):
        # do if n = 3:
        folder_name = "results/all_iterating_solutions"
        os.makedirs(folder_name, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        num_points = len(self.all_solutions)
        colors = np.arange(num_points)
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        scalar_map = ScalarMappable(cmap=cmap, norm=normalize)

        for i in range(num_points):
            color_val = scalar_map.to_rgba(colors[i])
            ax.scatter(self.all_solutions[i][0], self.all_solutions[i][1], self.all_solutions[i][2], c=[color_val],
                       cmap='viridis', s=50)
            ax.text(self.all_solutions[i][0], self.all_solutions[i][1], self.all_solutions[i][2], str(i), color='black',
                    fontsize=8)

        scalar_map.set_array(colors)
        cbar = plt.colorbar(scalar_map)
        cbar.set_label('Index')

        file_name = f"{folder_name}/value.png"
        plt.savefig(file_name)
        plt.show()

    def plot_all_solutions_value_2d(self):
        plt.figure()
        # do if n = 2:
        folder_name = "results/all_iterating_solutions"
        os.makedirs(folder_name, exist_ok=True)

        num_points = len(self.all_solutions)
        colors = np.arange(num_points)
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        scalar_map = ScalarMappable(cmap=cmap, norm=normalize)

        for i in range(num_points):
            color_val = scalar_map.to_rgba(colors[i])
            plt.scatter(self.all_solutions[i][0], self.all_solutions[i][1], c=[color_val], cmap='viridis', s=50)

        scalar_map.set_array(colors)
        cbar = plt.colorbar(scalar_map)
        cbar.set_label('Index')

        file_name = f"{folder_name}/value.png"
        plt.savefig(file_name)
        plt.show()


def test_s(n_start, n_end, s_range):
    for n in range(n_start, n_end + 1):
        print(f'n={n}')
        np.set_printoptions(precision=16, suppress=True)
        A = np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=-1) + np.diag(-1 * np.ones(n - 1), k=1)
        b = np.full(n, 0)
        b[0] = b[-1] = 1

        for s in range(1, s_range):
            try:
                example = ImprovedIterationScheme(tol=1e-8, max_iter=1000, A=A, b=b, initial_value=None, s=s)
                example.linear_iteration()
                iter1 = example.iterations
                example.improved_iteration(if_plot=False)
                iter2 = example.iterations
                if iter1 > iter2:
                    print(f's={s}, {iter2}({iter1})')
            except Exception as e:
                print(f"Error in s = {s}: {e}")
                continue


def main(n, s):
    np.set_printoptions(precision=16, suppress=True)
    # A = np.array([[2, -1, 0, 0],
    #               [-1, 2, -1, 0],
    #               [0, -1, 2, -1],
    #               [0, 0, -1, 2]])
    #
    # b = np.array([1, 0, 0, 1])

    A = np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=-1) + np.diag(-1 * np.ones(n - 1), k=1)
    b = np.full(n, 0)
    b[0] = b[-1] = 1

    example = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, initial_value=None, s=s)
    example.linear_iteration()
    print(example.iterations)
    example.improved_iteration(if_plot=True)
    print(example.iterations)
    example.plot_all_solutions()
    example.plot_all_solutions_norm()
    example.plot_all_solutions_value()
    if n == 3:
        example.plot_all_solutions_value_3d()
    elif n == 2:
        example.plot_all_solutions_value_2d()


if __name__ == "__main__":
    # test_s(n_start=3, n_end=3, s_range=50)
    main(n=3, s=5)
