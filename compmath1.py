import math

eps = 0.000001
eps_sqrt = eps / 3.75
eps_atan = eps / 1.44
eps_cos = eps / 6.25


def f_exact(x):
    return math.sqrt(1 + math.atan(16.7 * x + 0.1)) / math.cos(7 * x + 3)


def f_approx(x):
    return my_sqrt(1 + my_atan(16.7 * x + 0.1)) / my_cos(7 * x + 3)


def my_sqrt(x):
    x_0 = 1
    x_1 = (x_0 + x / x_0) / 2
    while (abs(x_0 - x_1) > eps_sqrt):
        x_0 = x_1
        x_1 = (x_0 + x / x_0) / 2
    return x_1


def my_cos(x):
    cos = 1
    factorial_num = 2
    R = x * x / -factorial_num
    while (abs(R) > eps_cos):
        cos += R
        R *= (x * x) / -(factorial_num * factorial_num + 3 * factorial_num + 2)
        factorial_num += 2
    return cos


def my_atan(x):
    atan = 0
    num = 1
    R = x
    while (abs(R / num) > eps_atan):
        atan += R / num
        R *= -(x * x)
        num += 2
    return atan


arglist = []
x= 10
while(x<=50):
    arglist.append(x/1000)
    x+=5
for i in arglist:
    print(i, f_exact(i), f_approx(i), abs(f_exact(i) - f_approx(i)))
