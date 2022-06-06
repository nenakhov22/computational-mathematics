import math

import numpy as np
import sympy
from sympy import Symbol, diff
from sympy.solvers import solve

a = 1.8
b = 2.9
alpha = 0
beta = 4 / 7
right_solution = 57.48462064655285571820619434508191055583

speed = 0  # init


def f(x): return 4 * sympy.cos(2.5 * x) * sympy.exp(4 * x / 7) + 2.5 * sympy.sin(5.5 * x) * sympy.exp(
    -3 * x / 5) + 4.3 * x


def f_moments(x, j): return ((x - a) ** (-alpha) * (b - x) ** (-beta)) * (x ** j)


def omega(x, matrix_a):
    return sum([x ** i * (matrix_a[i] if i < len(matrix_a) else 1) for i in range(len(matrix_a), -1, -1)])


# Моменты Аналитическое инегрирование x^i/(b-x)^beta i=0..5
u_0 = lambda x: (b - x) ** (1 - beta) / (beta - 1)
u_1 = lambda x: -(b - x) ** (1 - beta) * (b - beta * x + x) / (beta - 1) / (beta - 2)
u_2 = lambda x: (b - x) ** (1 - beta) * (2 * b ** 2 - 2 * (beta - 1) * b * x + (beta ** 2 - 3 * beta + 2) * x ** 2) / (
        beta - 1) / (beta - 2) / (beta - 3)
u_3 = lambda x: ((b - x) ** (1 - beta) * (
        -6 * b ** 3 + 6 * (beta - 1) * b ** 2 * x - 3 * (beta ** 2 - 3 * beta + 2) * b * x ** 2 + (
        beta ** 3 - 6 * beta ** 2 + 11 * beta - 6) * x ** 3)) / ((beta - 4) * (beta - 3) * (beta - 2) * (beta - 1))
u_4 = lambda x: (b - x) ** (1 - beta) * (
        b ** 4 / (beta - 1) - (4 * b ** 3 * (b - x)) / (beta - 2) + (6 * b ** 2 * (b - x) ** 2) / (beta - 3) - (
        4 * b * (b - x) ** 3) / (beta - 4) + (b - x) ** 4 / (beta - 5))
u_5 = lambda x: (b - x) ** (1 - beta) * (
        b ** 5 / (beta - 1) - (5 * b ** 4 * (b - x)) / (beta - 2) + (10 * b ** 3 * (b - x) ** 2) / (beta - 3) - (
        10 * b ** 2 * (b - x) ** 3) / (beta - 4) + (5 * b * (b - x) ** 4) / (beta - 5) - (b - x) ** 5 / (
                beta - 6))


# Newton Cotes
def newton_cotes(lb, ub, logger=False):
    node = np.array([lb, (lb + ub) / 2, ub])
    moments = [u_0(ub) - u_0(lb), u_1(ub) - u_1(lb), u_2(ub) - u_2(lb)]

    if logger: print("moments: ", moments)

    matrix_b = np.transpose([np.array(moments, dtype="float")])
    # if logger: print("moments matrix: \n", matrix_b, '\n')
    xs = np.array([node ** 0, node, node ** 2], dtype="float")
    if logger: print("x_i matrix: \n", xs, '\n')

    A = np.linalg.solve(xs, matrix_b)
    if logger: print("A matrix :\n", A, '\n')

    solution = sum([f(node[i]) * A[i] for i in range(3)])
    if logger: print("Solution :", solution, '\n')

    M_n = 164.718  # max of |d3/dx3 f(x)|
    integ = 0.0842312  # аналитически интегрированный |p(x)w(x)| от a до b
    R_n = M_n / 6 * integ
    if logger: print("Оценка погрешности : ", R_n)

    x = Symbol('x')
    # if logger: print("Оценка погрешности :", ((diff(f(x), x, 4) / 2880) * (ub - lb) ** 5).subs({x: 2.5}))
    if logger: print("Точная Погрешность :", right_solution - solution)

    return solution


def gauss(lb, ub, logger=False):
    moments = [u_0(ub) - u_0(lb), u_1(ub) - u_1(lb), u_2(ub) - u_2(lb),
               u_3(ub) - u_3(lb), u_4(ub) - u_4(lb), u_5(ub) - u_5(lb)]
    if logger:
        print("moments : ", moments, '\n')

    matrix_moments = np.array(
        [[moments[0], moments[1], moments[2]],
         [moments[1], moments[2], moments[3]],
         [moments[2], moments[3], moments[4]]])
    matrix_b = -np.array([[moments[3]], [moments[4]], [moments[5]]])
    matrix_a = np.linalg.solve(matrix_moments, matrix_b)
    if logger:
        print("moments coefficients : \n", matrix_moments, '\n')
        print("b vector : \n", matrix_b, '\n')
        print("a coefficients : \n", matrix_a, '\n')

    x = Symbol('x')
    matrix_x = solve(omega(x, matrix_a))
    matrix_x = np.array([value.get(x).args[0] for value in matrix_x])
    if logger: print("matrix x_j:\n", matrix_x, '\n')

    matrix_j = np.array([matrix_x ** 0, matrix_x ** 1, matrix_x ** 2], dtype="float")
    matrix_b = np.array([[moments[0]], [moments[1]], [moments[2]]])
    A = np.linalg.solve(matrix_j, matrix_b)
    if logger: print("A matrix :\n", A, '\n')

    solution = sum([f(matrix_x[i]) * A[i] for i in range(3)])
    if logger: print("Solution :", solution, '\n')

    Mn = 26589.4  # max of |d6/dx6 f(x)| 1.8 < x < 2.9
    integ = 0.0015211  # аналитически интегрированный |p(x)w^2(x)| от a до b
    if logger: print("Оценка погрешности : ", Mn / math.factorial(6) * integ)
    if logger: print("Точная Погрешность : ", right_solution - solution)

    return solution


def SKF(a, b, iqf, k=2):  # Interpolation quadrature formulas
    solutions = []
    solution = 0
    h_r = []
    while abs(right_solution - solution) > 1e-6:
        h = (b - a) / k
        h_r.append(h)
        lb = a  # lower bound
        ub = a + h  # upper bound
        solution = 0
        step = 1
        while ub <= b:
            solution += iqf(lb, ub)
            lb = a + h * step
            ub = a + h * (step + 1)
            step += 1
        print("solution", step, " ", solution)
        solutions.append(solution)
        k *= 2

        global speed
        if len(solutions) >= 3:
            s = len(solutions)
            speed = -(math.log(abs(
                (solutions[s - 1] - solutions[s - 2]) / (solutions[s - 2] - solutions[s - 3]))) / math.log(2))
            print("speed ", speed, end='')
            print(" Оценка погрешности : ", (solutions[s - 1] - solutions[s - 2]) / (2 ** speed - 1))  # по Рунге
    solutions = np.array(solutions, dtype='float')

    steps_matrix = []  # формирую матрицу коэффициентов из длины шагов для Ричардсона
    for i in range(len(h_r)):
        steps_matrix.append([1])
        steps_matrix[i].extend([-(h_r[i]**j) for j in range(k, k+len(h_r)-1)])
    steps_matrix = np.array(steps_matrix, dtype='float')
    C_p = np.linalg.solve(steps_matrix, solutions)
    print(" Оценка погрещности по Ричардсону :", C_p[0]-solutions[-1])
    return solutions


print("12 вариант")
print("----------NewTon-Cotes----------")
newton_cotes(a, b, True)  # Ньютон-Котс
print("----------SKF Newton-Cotes----------")
solutions = SKF(a, b, newton_cotes)  # стартовое деление 1->2->4
print("----------SKF Newton-Cotes h-opt----------")
Rh_1 = (solutions[1] - solutions[0]) / (1 - 2 ** (-speed))
h_opt = (b - a) / 2 * ((1e-6 / abs(Rh_1)) ** (1 / speed)) * .95
print("h-opt ", h_opt)
SKF(a, b, newton_cotes, math.ceil((b - a) / h_opt))

print("----------Gauss----------")
gauss(a, b, True)  # Гаусс
print("----------SKF Gauss----------")
solutions = SKF(a, b, gauss)  # стартовое деление 1->2->4
print("----------SKF Gauss h-opt----------")
Rh_1 = (solutions[1] - solutions[0]) / (1 - 2 ** (-speed))
h_opt = (b - a) / 2 * ((1e-6 / abs(Rh_1)) ** (1 / speed)) * .95
print("h-opt ", h_opt)
SKF(a, b, gauss, math.ceil((b - a) / h_opt))
