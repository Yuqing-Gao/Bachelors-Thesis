import numpy as np


class ImprovedIterationScheme:
    def __init__(self, tol, max_iter, A, b, exact_solution, initial_value, s):
        self.tol = tol  # convergence toleration of iterative algorithm
        self.max_iter = max_iter  # max iterations allowed
        self.A = A
        self.b = b  # s.t. Ax = b
        self.initial_value = np.zeros_like(b) if initial_value is None else initial_value  # initial guess x_0
        self.s = s  # extrapolation parameter
        self.iterations = 0  # count of linear iterations
        self.dim = len(self.b)  # dimension of solution vector
        self.all_solutions = []  # initialize solution list
        self.exact_solution = exact_solution  # exact solution of the given question
        self.lst1 = []  # lst to record the results of jacobi's method
        self.lst2 = []  # lst to record the results of improved method

    def linear_iteration(self):
        # perform linear iteration method until convergence
        self.iterations = 0  # initialize
        x = np.array(self.initial_value, dtype=np.double)
        self.lst1.append(x)

        # iterate
        while self.iterations <= self.max_iter:
            x_old = x.copy()
            x = self.jacobi(x_old)
            self.lst1.append(x)
            # stop when convergence
            if np.linalg.norm(x - x_old, ord=2) < self.tol:
                break
        # print(self.lst1, len(self.lst1)-1)

    def improved_iteration(self, if_theta):
        self.iterations = 0  # initialize
        pointer = 0  # decide where to start to calculate the extrapolation solution

        while self.iterations <= self.max_iter:
            self.initial_value = np.array(self.initial_value, dtype=np.double)
            self.lst2.append(self.initial_value)
            # do linear iterations for s times
            for _ in range(self.s):
                x_old = self.initial_value.copy()
                self.initial_value = self.jacobi(x_old)
                self.lst2.append(self.initial_value)
                if np.linalg.norm(np.array(x_old) - np.array(self.initial_value), ord=2) < self.tol:
                    self.iterations += 1
                    return  # stop iterating

            calculating_vectors = []  # s+1 vectors to calculate new solution in the next step
            for i in range(self.s + 1):
                calculating_vectors.append(self.lst2[pointer + i])
            # print(calculating_vectors)
            new_solution = self.extrapolation(calculating_vectors)  # calculate new solution

            if if_theta is True:
                theta = self.calculate_theta(new_solution)  # calculate theta
                if self.iterations <= 100:
                    new_solution = self.initial_value + (np.array(new_solution) - self.initial_value) * theta
                else:
                    new_solution = self.jacobi(new_solution)

            # break when convergence
            if np.linalg.norm(np.array(new_solution) - np.array(self.initial_value), ord=2) < self.tol:
                self.iterations += 1
                break
            self.iterations += 1
            pointer += self.s + 1
            self.initial_value = new_solution

        # print(self.lst2, len(self.lst2)-1)

    def calculate_theta(self, new_solution):
        """
        :param new_solution: solution that calculate by extrapolation
        :return: parameter to accelerate convergence
        """
        r = self.A @ self.initial_value - self.b
        delta_uk = np.array(new_solution) - np.array(self.initial_value)
        numerator = np.transpose(delta_uk) @ np.transpose(self.A) @ r
        denominator = np.transpose(delta_uk) @ np.transpose(self.A) @ self.A @ delta_uk
        # print(f'r={r}, delta_uk={delta_uk}, numerator={numerator}, denominator={denominator}r')
        if denominator == 0:
            theta = 0  # denominator = 0 indicates delta_uk is null vector (?)
        else:
            theta = - numerator / denominator
        return theta

    def extrapolation(self, calculating_vectors):
        """
        :param calculating_vectors: s+1 solutions
        :return: 1 extrapolation solution
        """
        new_solution = []  # list to restore new solution calculated by -w2/w1
        for j in range(self.dim):
            lst = []  # s+1 scalars to calculate new solution in the next step

            for i in range(len(calculating_vectors)):
                lst.append(calculating_vectors[i][j])

            # calculate w1
            w1 = 0
            for k in range(len(lst) - 1):
                inner_sum = 0
                for j in range(len(lst) - 1):
                    inner_sum += lst[k] - lst[j]
                w1 += (lst[k + 1] - lst[k]) * inner_sum
            # if divided by 0, stop iterating
            if w1 == 0:
                new_solution = self.initial_value
                break
            # calculate w2
            w2 = 0
            for k in range(len(lst) - 1):
                inner_sum = 0
                for j in range(len(lst) - 1):
                    inner_sum += lst[j] ** 2 - lst[j] * lst[k]
                w2 += (lst[k + 1] - lst[k]) * inner_sum

            # calculate extrapolation solution
            new_solution.append(-w2 / w1)
        return new_solution

    def jacobi(self, x_old):
        """
        :param x_old: current solution
        :return: next solution after iterate once
        """
        x_new = np.zeros_like(self.b, dtype=np.double)
        for i in range(self.dim):
            x_new[i] = (self.b[i] - np.dot(self.A[i, :i], x_old[:i]) - np.dot(self.A[i, (i + 1):], x_old[(i + 1):])) / \
                       self.A[i, i]
        self.iterations += 1
        return x_new


def initialization(n):
    """
    :param n: dimension of the given question
    :return: matrix A, vector b, exact solution
    """
    np.set_printoptions(precision=16, suppress=True)
    A = np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n - 1), k=-1) + np.diag(-1 * np.ones(n - 1), k=1)
    b = np.full(n, 0)
    b[0] = b[-1] = 1
    exact_solution = np.full_like(b, 1)
    return A, b, exact_solution


def test_s(n_start, n_end, s_range):
    for n in range(n_start, n_end + 1):
        print(f'n={n}')
        A, b, exact_solution = initialization(n)

        for s in range(3, s_range):
            try:
                test_case1 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                                     initial_value=None, s=s)
                test_case2 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                                     initial_value=None, s=s)
                test_case3 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                                     initial_value=None, s=s)
                test_case1.linear_iteration()
                iter1 = test_case1.iterations
                test_case2.improved_iteration(if_theta=False)
                iter2 = test_case2.iterations
                test_case3.improved_iteration(if_theta=True)
                iter3 = test_case3.iterations
                if iter1 > iter2 and iter3 >= iter2:
                    print(f's={s}, {iter2}({iter1}), if theta is added:{iter3}')
                elif iter1 > iter2 > iter3:
                    print(f's={s}, {iter2}({iter1}), if theta is added:{iter3} *')
            except Exception as e:
                print(f"Error in s = {s}: {e}")
                continue


def test(n, s, if_theta):
    A, b, exact_solution = initialization(n)
    test_case1 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                         initial_value=None, s=s)
    test_case1.linear_iteration()
    test_case2 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                         initial_value=None, s=s)
    test_case2.improved_iteration(if_theta=False)
    test_case3 = ImprovedIterationScheme(tol=1e-8, max_iter=200, A=A, b=b, exact_solution=exact_solution,
                                         initial_value=None, s=s)
    test_case3.improved_iteration(if_theta=True)
    # save all iterating results to .txt file
    np.savetxt(r'results/jacobi.txt', test_case1.lst1)
    np.savetxt(r'results/without_theta.txt', test_case2.lst2)
    np.savetxt(r'results/with_theta.txt', test_case3.lst2)
    print(f'jacobi:{test_case1.iterations}, without theta: {test_case2.iterations}, with theta: {test_case3.iterations}')


if __name__ == "__main__":
    # test_s(n_start=2, n_end=7, s_range=50)
    test(n=6, s=9, if_theta=True)
