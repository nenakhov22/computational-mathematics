import numpy as np
import random as rand
def LUPQ(A, n):
    L = np.zeros((n, n))
    U = np.copy(A)
    P = np.eye(n)
    Q = np.eye(n)
    rank = n
    for i in range(n):
        max = U[i][i]
        for j in range(i, n):
            for k in range(i, n):
                if abs(U[j][k]) >= abs(max):
                    max = U[j][k]
                    index = j, k
        if (abs(U[index[0], index[1]]) < 1e-14):
            rank = i
            break
        if index[0] != i:
            for j in range(n):
                L[index[0]][j], L[i][j] = L[i][j], L[index[0]][j]
                U[index[0]][j], U[i][j] = U[i][j], U[index[0]][j]
                P[index[0]][j], P[i][j] = P[i][j], P[index[0]][j]
        if index[1] != i:
            for j in range(n):
                L[j][index[1]], L[j][i] = L[j][i], L[j][index[1]]
                U[j][index[1]], U[j][i] = U[j][i], U[j][index[1]]
                Q[j][index[1]], Q[j][i] = Q[j][i], Q[j][index[1]]

        for j in range(i + 1, n):
            L[j][i] = U[j][i] / U[i][i]
            U[j] -= U[i] * L[j][i]
    for j in range(n):
        L[j][j] = 1
    return L, U, P, Q, rank

def Determinant(U, n):
    det = 1
    for i in range(n):
        det *= U[i][i]
    return det

def Solution(L, U, P, Q, b, n, rank):
    #Ly = Pb, Uz = y, x = Qz
    y = np.matmul(P, b)
    for i in range(1, n):
        y[i] -= sum([L[i][k] * y[k] for k in range(0, i)])
    z = y
    for i in range(n):
        if (abs(U[i, i]) < 1e-12):
            if (abs(y[i]) > 1e-12):
                raise BaseException("Решения не существует")
            else:
                continue
    for i in range(rank-1, -1, -1):
        z[i] -= sum([U[i][k] * z[k] for k in range(n-1, i, -1)])
        z[i] /= U[i][i]
    return np.matmul(Q, z)

def inverse(L, U, P, Q, n):
    A_inverse = np.eye(n)
    E = np.eye(n)
    P1 = [i for i in range(n)]
    for i in range(n):
        A_inverse[i] = Solution(L, U, P, Q, E[P1[i]], n)
    A_inverse = [[A_inverse[j][i] for j in range(n)] for i in range(n)]
    return A_inverse

def norma(A):
    norma = 0
    for i in range(A.shape[0]):
        n = 0
        for j in range(A.shape[1]): n += abs(A[i, j])
        if (n > norma): norma = n
    return norma

def norma1(A, A1, n):
    t = norma(A) * norma(A1)
    return t

# b = np.transpose(np.matrix([rand.randint(0,10) for i in range(n)]))
# A = np.matrix([[rand.randint(0,10)+0.0 for j in range(n)] for i in range(n)])
# n = np.random.randint(2, 5)
# A = np.random.random((n, n))
# b = np.array(np.random.sample(n))
# A = np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]], dtype=float)
# n = 5
# A = np.array([[1., 2., -1., 2.,-5.],
#               [2., -3., 2, 4.,13.],
#               [1., 2., -1., 2.,-5.],
#               [-1., -6., 7, -3., 7.],
#               [4., 9., -43., 1., 12.]], dtype=float)
# b = [4.,1.,9.,2.,13.]
# b = np.array(np.random.sample(n))
# n = 4
# A = np.array([[1., 2., -1., 2.],
#               [2., -3., 2, 4.],
#               [3., 1., 1., 6.],
#               [-1., -6., 7, -3]], dtype=float)
n = 5
A = np.matrix([[rand.randint(0, 10) + 0.0 for j in range(n)] for i in range(n)])
A[:, 2] = A[:, 1] * 2 + A[:, 0]
A[:, 3] = A[:, 0] * 2 + A[:, 1]
b = np.transpose(np.matrix([rand.randint(0, 10) for i in range(n)]))
b = np.matmul(A, b)
# n = 3
# A = np.matrix([[65,22,31],[3,2,1],[3,2,1]], dtype=float)
# b = np.matrix([5,0,6], dtype=float)
# b = np.transpose(b)
print('A =')
print(A,'\n')
L, U, P, Q, rank = LUPQ(A, n)
print('LU =')
LU = np.matmul(L, U)
print(LU,'\n')
print('L =')
print(L,'\n')
print('U =')
print(U,'\n')
print('rank =', rank,'\n')
print('Перестановка строк P =')
print(P,'\n')
print('Перестановка столбцов Q =')
print(Q,'\n')
print('PAQ =')
print(np.matmul(np.matmul(P,A), Q),'\n')
print('LU - PAQ =')
LUPAQ = np.matrix(np.matmul(L, U)) - np.matrix(np.matmul(P, np.matmul(A, Q)))
if np.all(abs(LUPAQ)) < 10 ** (-14):
    LUPAQ = np.zeros((n, n))
print(LUPAQ,'\n')
print('Определитель:')
detA=Determinant(U, n)
print(detA,'\n')
try:
    x = Solution(L, U, P, Q, b, n, rank)
    print('Решение СЛАУ:')
    print(x, '\n')
    print("Ax - b =", np.subtract(np.matmul(A, x), b), '\n')
except BaseException as e:
    print(e)
if (detA < 10 ** (-14)):
    print("det A == 0 Обратной матрицы не существует")
else:
    A1 = np.array(inverse(LU, P, Q, n))
    print('Обратная матрица')
    print(A1, '\n')
    print('Python обратная матрица')
    print(np.linalg.inv(A), '\n')
    print("A*invA=\n", np.matmul(A, A1), '\n')
    print("invA*A=\n", np.matmul(A1, A), '\n')
    print('Число обусловленности')
    print(norma1(A, A1, n))
