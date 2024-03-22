import numpy as np
from typing import List
from max_rank_split import *


def show_matrix(A):
    A_shape = A.shape
    for i in range(A_shape[0]):
        for j in range(A_shape[1]):
            print('{:>10.2}'.format(A[i, j]), end='')
        print()


def show(Er, Y):
    for i in range(len(Y)):
        yi = 0
        if abs(Er[i, i]) > 1e-9:
            yi = Y[i][0]
            print(f"x[{i}] = {Y[i][0]}", end='')
        else:
            yi = 1.
            print(f"x[{i}] = ?", end='')
        for j in range(i + 1, len(Y)):
            yi -= Er[j, i]
            if abs(Er[i, j]) > 1e-9:
                if Er[i, j] > 0:
                    print(f" + {Er[i, j]} * x[{j}]", end='')
                else:
                    print(f" - {-Er[i, j]} * x[{j}]", end='')
                    # print(f"x[{i}] = {yi}", end='')
        print()


def col_sample(A):
    return [i.T for i in row_sample(A.T)]


def pinv_col_full_r(A):
    R1, _ = row_sample(A.T @ A)  # 初等行变换，化简成行最简，主元为1
    return R1 @ A.T


def pinv_row_full_r(A):
    return pinv_col_full_r(A.T).T


def pinv2(A):
    R, row = row_sample(A)  # 初等行变换，化简成行最简，主元为1 RA=G
    C, _ = row_sample(R)  # 初等行变换，化简成行最简，主元为1 AC=K
    rank = np.linalg.matrix_rank(row)  # 求秩
    row = row[:rank, :]  # 2, 4
    C = C[:, :rank]
    kernal_inv, col = col_sample(C)  # 初等行变换，化简成行最简，主元为1 RC=F,
    return pinv_row_full_r(row) @ pinv_col_full_r(C)


def b2x2(A, B, C, D):
    return np.vstack([np.hstack((A, B)),
                      np.hstack((C, D))])


def b2x1(A, B, ):
    return np.vstack((A, B))


def b1x2(A, B, ):
    return np.hstack((A, B))


# 基于满秩分解的伪逆分解
def pinv_split_add_col(A, a, R, F, G, FTF, S):
    Ra = R @ a
    rank, col = G.shape
    aTF = a.T @ F
    aTa = a.T @ a
    if rank == A.shape[0] or abs(Ra[rank:]).max() < 1e-5:  # 跟之前的向量线性相关
        R2, F2 = R, F
        Ra = Ra[:rank]
        G2 = b1x2(G, Ra)
        FTF2 = FTF
        S2 = S + (FTF @ Ra) @ Ra.T  # 先计算nx1的结果，再和1xn相乘，速度更快
    else:  # 跟之前的向量线性无关
        G_ext = b2x1(G, np.zeros((A.shape[0] - rank, col)))
        R2, G2, rank, _ = row_sample(b1x2(G_ext, Ra), R)  # O((k-r)*n)
        F2 = b1x2(F, a)
        FTF2 = b2x2(FTF, aTF.T, aTF, aTa)
        G2 = G2[:rank]
        S2 = b2x2(S, aTF.T,
                  (aTF @ G) @ G.T, aTa)  # 先计算1xn的结果，再和nxm相乘，速度更快
    S2_inv = inv(S2)  # O(r^3) 秩的三次方
    A_pinv2 = G2.T @ S2_inv @ F2.T  # O(n*r*n)
    return A_pinv2, R2, F2, G2, FTF2, S2


# 基于满秩分解的伪逆分解
def pinv_split_add_col_init(A):
    R, G, rank, cols = row_sample(A)  # O(n^3)
    G = G[:rank, ]
    F = A[:, cols]
    FTF = F.T @ F
    S = FTF @ (G @ G.T)  # 先计算1xn的结果，再和nxm相乘，速度更快
    S_inv = inv(S)  # O(r^3) 秩的三次方
    A_pinv = G.T @ S_inv @ F.T
    return A_pinv, R, F, G, FTF, S


if __name__ == '__main__':
    w1 = np.array([
        [1, 2, 3, 1],
        [1, 2, 3, 1],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ])
    w2 = np.array([
        [1, 2, 3, 1],
        [7, 4, 3, 1],
        [1, 2, 3, 4],
        [2, 1, 3, 4],
        [1, 4, 3, 4],
        [1, 8, 2, 4],
    ])
    w3 = np.array([
        [1, 2, 3, 1],
        [1, 2, 3, 1],
    ])
    Ak = w2
    Ak_1, ak = Ak[:, :-1], Ak[:, -1:]

    A_pinv, R, F, G, FTF, S = pinv_split_add_col_init(Ak_1)
    A_pinv3, _, _, _, _, _ = pinv_split_add_col_init(Ak)
    A_pinv2, R2, F2, G2, FTF2, S2 = pinv_split_add_col(Ak_1, ak, R, F, G, FTF, S)
    # print_matrix(A_pinv, name='pinv by add  1')
    print_matrix(A_pinv2, name='pinv by add   ')
    print_matrix(A_pinv3, name='pinv by once  ')
