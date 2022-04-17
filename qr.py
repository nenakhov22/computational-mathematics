import numpy as np
import random as r
import math
def transpose_matrix(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def QR(A):
    R = A.copy()
    Q = np.eye(R.shape[0])
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[1]):
            Q_temp = np.eye(A.shape[0])

            s = -R[j, i] / np.sqrt(R[i, i] ** 2 + R[j, i] ** 2)
            c = R[i, i] / np.sqrt(R[i, i] ** 2 + R[j, i] ** 2)

            Q_temp[i, i] = c
            Q_temp[j, i] = s
            Q_temp[i, j] = -s
            Q_temp[j, j] = c

            R = np.dot(Q_temp, R)
            Q_temp[i, j] = s
            Q_temp[j, i] = -s
            Q = np.dot(Q, Q_temp)
    return Q, R


size = r.randint(2, 3)
A = np.matrix([
    [r.randint(0, 100) for j in range(size)]
    for i in range(size)])
Q, R = QR(A)
print("A\n", A)
print("Q>\n", Q)
print("R>\n", R)
print("QR>\n", np.dot(Q, R))
print("QR>\n", np.dot(Q, np.transpose(Q)))
#QRx=b
#Rx=transpose(Q)*b
#замена y=transpose(Q)*b
#Rx=y
b=np.array([r.randint(0, 100) for j in range(R.shape[0])])
print('b=',b)
y=np.matmul(np.transpose(Q),b)
x=np.array([0.0 for j in range(R.shape[0])])
x[R.shape[0]-1]=(float(y[R.shape[0]-1]))/(float(R[size-1,size-1]))
for i in range(x.shape[0]-2,-1,-1):
    x[i]=y[i]
    print('i=',i)
    for j in range(i+1,x.shape[0]):
        print('j=',j)
        x[i]-=x[j]*R[i,j]
    x[i]/=R[i,i]
print('x=',x)
print(np.linalg.solve(R,y))
print(np.matmul(R,x)-y)
