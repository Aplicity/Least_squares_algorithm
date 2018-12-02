import numpy as np
import time
import matplotlib.pyplot as plt


def householder_reflection(A):
    """Householder变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    for cnt in range(r - 1):
        x = R[cnt:, cnt]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(r)
        Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
    return (Q, R)

def givens_rotation(A):
    """
    Givens变换
    matalb 默认[Q,R] = qr(A)
    """
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
            r_ = np.hypot(R[col, col], R[row, col])  # d
            c = R[col, col]/r_
            s = -R[row, col]/r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
    return (Q, R)


def gram_schmidt(A):
    """Gram-schmidt正交化"""
    Q=np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) # 减去待求向量在以求向量上的投影
        e = u / np.linalg.norm(u)  # 归一化
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return (Q, R)



np.random.seed(1)
n = 20
default_cost = []
for m in range(2000,21000, 1000):
    X = np.matrix( np.random.rand(m,n) * 100 )
    Y = np.matrix(np.sum(X,axis = 1) - 20 * np.random.rand(m,1) + 10)
    start = time.clock()
    W1 = np.linalg.inv(X.T *X) * X.T * Y
    end = time.clock()
    cost = end - start
    default_cost.append(cost)

SVD_cost = []
for m in range(2000,21000, 1000):
    X = np.matrix( np.random.rand(m,n) * 100 )
    Y = np.matrix(np.sum(X,axis = 1) - 20 * np.random.rand(m,1) + 10)
    start = time.clock()
    U,S,V = np.linalg.svd(X)
    W2 = V * np.matrix(np.linalg.inv(np.diag(S))) * U[:,:n].T * Y
    end = time.clock()
    cost = end - start
    SVD_cost.append(cost)


QR_cost = []
for m in range(2000,21000, 1000):
    if m == 2000:
        m = 2010
    X = np.matrix( np.random.rand(m,n) * 100 )
    Y = np.matrix(np.sum(X,axis = 1) - 20 * np.random.rand(m,1) + 10)
    start = time.clock()
    Q,R = np.linalg.qr(X)
    W3 = np.linalg.inv(R) * Q.T * Y
    end = time.clock()
    cost = end - start
    QR_cost.append(cost)

plt.figure(figsize = (16,6))
plt.subplot(1,2,1)
plt.plot(list(range(2000,21000, 1000)), default_cost, label = 'Normal matrix method')
plt.plot(list(range(2000,21000, 1000)), QR_cost, label = 'QR matrix decomposition')
plt.plot(list(range(2000,21000, 1000)), SVD_cost, label = 'SVD matrix decomposition')
plt.legend()
plt.xlabel('the numbel of samples')
plt.ylabel('CPU time consuming')

plt.subplot(1,2,2)
plt.plot(list(range(2000,21000, 1000)), default_cost, label = 'Normal matrix method')
plt.plot(list(range(2000,21000, 1000)), QR_cost, label = 'QR matrix decomposition')
plt.legend()
plt.xlabel('the numbel of samples')
plt.ylabel('CPU time consuming')

plt.show()



