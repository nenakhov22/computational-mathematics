import numpy as np
import numpy
import math
import random
from numpy import linalg as LA
import time

counter = 0

def F(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x.tolist()
    return np.transpose(numpy.mat([
    math.cos(x2 * x1) - math.exp(-3 * x3) + x4 * x5 ** 2 - x6 - math.sinh(2 * x8) * x9 + 2 * x10 + 2.0004339741653854440,
    math.sin(x2 * x1) + x3 * x9 * x7 - math.exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994,
    x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904,
    2 * math.cos(-x9 + x4) + x5 / (x3 + x1) - math.sin(x2 ** 2) + math.cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757,
    math.sin(x5) + 2 * x8 * (x3 + x1) - math.exp(-x7 * (-x10 + x6)) + 2 * math.cos(x2) - 1.0 / (-x9 + x4) - 0.3685896273101277862,
    math.exp(x1 - x4 - x9) + x5 ** 2 / x8 + math.cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115,
    x2 ** 3 * x7 - math.sin(x10 / x5 + x8) + (x1 - x6) * math.cos(x4) + x3 - 0.7380430076202798014,
    x5 * (x1 - 2 * x6) ** 2 - 2 * math.sin(-x9 + x3) + 0.15e1 * x4 - math.exp(x2 * x7 + x10) + 3.5668321989693809040,
    7 / x6 + math.exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499,
    x10 * x1 + x9 * x2 - x8 * x3 + math.sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]
    ))

def J(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x.tolist()
    return numpy.mat([[-x2 * math.sin(x2 * x1), -x1 * math.sin(x2 * x1), 3 * math.exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
                -1, 0, -2 * math.cosh(2 * x8) * x9, -math.sinh(2 * x8), 2],
               [x2 * math.cos(x2 * x1), x1 * math.cos(x2 * x1), x9 * x7, 0, 6 * x5,
                -math.exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, math.exp(-x10 + x6)],
               [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
               [-x5 / (x3 + x1) ** 2, -2 * x2 * math.cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * math.sin(-x9 + x4),
                1.0 / (x3 + x1), 0, -2 * math.cos(x7 * x10) * x10 * math.sin(x7 * x10), -1,
                2 * math.sin(-x9 + x4), -2 * math.cos(x7 * x10) * x7 * math.sin(x7 * x10)],
               [2 * x8, -2 * math.sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, math.cos(x5),
                x7 * math.exp(-x7 * (-x10 + x6)), -(x10 - x6) * math.exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
                -1.0 / (-x9 + x4) ** 2, -x7 * math.exp(-x7 * (-x10 + x6))],
               [math.exp(x1 - x4 - x9), -1.5 * x10 * math.sin(3 * x10 * x2), -x6,-math.exp(x1 - x4 - x9),
                2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -math.exp(x1 - x4 - x9), -1.5 * x2 * math.sin(3 * x10 * x2)],
               [math.cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * math.sin(x4), x10 / x5 ** 2 * math.cos(x10 / x5 + x8),
                -math.cos(x4), x2 ** 3, -math.cos(x10 / x5 + x8), 0, -1.0 / x5 * math.cos(x10 / x5 + x8)],
               [2 * x5 * (x1 - 2 * x6), -x7 * math.exp(x2 * x7 + x10), -2 * math.cos(-x9 + x3), 1.5,
               (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * math.exp(x2 * x7 + x10), 0, 2 * math.cos(-x9 + x3),
                -math.exp(x2 * x7 + x10)],
               [-3, -2 * x8 * x10 * x7, 0, math.exp(x5 + x4), math.exp(x5 + x4),
                -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
               [x10, x9, -x8, math.cos(x4 + x5 + x6) * x7, math.cos(x4 + x5 + x6) * x7,
                math.cos(x4 + x5 + x6) * x7, math.sin(x4 + x5 + x6), -x3, x2, x1]])
def maxPos(A, r, c):
    pos = [r,c]
    for i in range(r,len(A)):
        for j in range(c,len(A[i])):
            if(A[i][j]>=A[pos[0]][pos[1]]):
                pos=[i,j]
    return pos
def LUPQ(A):
    global counter
    
    A = A.tolist()
    U = [ i[:] for i in A]
    L = [[1 if (i==j) else 0 for j in range(len(A))] for i in range(len(A))]
    Q = [i[:] for i in L]
    P = [i[:] for i in Q]

    for i in range(len(U)):
        pos = maxPos(U,i,i)
        P[i],P[pos[0]] = P[pos[0]],P[i]
        Q[i],Q[pos[1]] = Q[pos[1]],Q[i]
        U = np.dot(np.dot(P,A),np.transpose(Q))
    
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            L[j][i] = U[j][i]/U[i][i]
            U[j] = [U[j][k] - U[i][k]*L[j][i] for k in range(len(U[j]))]
            #считаю операции
            counter += len(U[j]) + 1
            
    return L,U.tolist(),P,np.transpose(Q)

def solve_LU(L,U,b,P,Q):
    global counter

    b = np.matmul(P,b)
    
    y = [0 for i in range(len(L))]
    for i in range(len(y)):
        y[i] = b[i] - sum([L[i][k]*y[k] for k in range(0,i)])
        #считаю операции
        counter += i
        
    x = [0 for i in range(len(U))]
    for i in range(len(x)-1,-1,-1):
        x[i] = (y[i]-sum([ U[i][k]*x[k] for k in range(i+1,len(y))]))/U[i][i]
        #считаю операции
        counter += len(y)
    return np.dot(Q,np.squeeze(np.asarray(x)))

#hyb - режим в котором каждая k-ая операция mod
def Newton(eps,k,hyb = False):
    #начальное приближение
    #x = np.array([0.5,0.5,1.5,-1.0,-0.5,1.5,0.5,-0.5,1.5,-1.5])
    x = np.array([0.5,0.5,1.5,-1.0,-0.2,1.5,0.5,-0.5,1.5,-1.5])
    i = 0
    Ja = J(x)
    while(True):
        if(hyb and i%k==0 or not hyb and i <= k):
            Ja = J(x)
            L,U,P,Q = LUPQ(Ja)
        else: print("mod ",end='')
        print("x"+str(i)+" =",x)

        xk = np.add(x,solve_LU(L,U,-F(x),P,Q))
        delta = LA.norm(np.subtract(x,xk))
        x = np.squeeze(np.asarray(xk))
        i+=1
        
        if(delta<eps): return x
start = time.time() * 1000
x = Newton(1e-14,2,False)#точность, kая итерация, режим каждой kой
print("\n%%%%% answer found in ",time.time() * 1000 - start,"millis with ",counter,"operations %%%%%\n")
print("F(x) = ",F(x))
