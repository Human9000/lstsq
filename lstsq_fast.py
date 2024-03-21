import numpy as np


def swap_main_item(A, r, c, smooth):
    if abs(A[r, c]) > smooth:  # 当前位置的元素可以为主元
        return True
    # 寻找下方的主元
    for i in range(r + 1, A.shape[0]):
        if abs(A[i, c]) > smooth:  # 找到适合做主元的行
            temp = A[i, :].copy()
            A[i, :] = A[r, :]
            A[r, :] = temp
            return True
    return False


def row_sample2(A):
    """
    @param A: 矩阵
    @return R: 行变化矩阵
    @return Rs: 行阶梯最简矩阵，主元为1，零行全在下方
    @return r: 矩阵的秩
    """
    n, m = A.shape
    A_ext = np.hstack((A, np.identity(n)))  # 生成增广矩阵
    smooth = 1e-9
    r = 0  # 行指针, 标志当前处理的行, 执行完之后也代表矩阵的秩
    for c in range(m):  # c 是列指针，用于指向处理的类
        if not swap_main_item(A_ext, r, c, smooth):  # 当前列的没有元素能作为主元，则跳过
            continue
        A_ext[r, :] = (A_ext[r, :]) / (A_ext[r, c])  # 当前元素归1化
        for q in range(r):  # 上侧归0化
            A_ext[q, :] = A_ext[q, :] - (A_ext[r, :] * A_ext[q, c]) / (A_ext[r, c])
        for q in range(r + 1, n):  # 下侧归0化
            A_ext[q, :] = A_ext[q, :] - (A_ext[r, :] * A_ext[q, c]) / (A_ext[r, c])
        r += 1  # 向下移动一行
    R, Er = A_ext[:, m:], A_ext[:, :m]
    return R, Er, r


# 满秩分解
def full_rank_split(A):
    # 满秩分解 A === F @ G,
    R, G, r = row_sample2(A)  # 初等行变换，化简成行最简，主元为1
    F, E, _ = row_sample2(R)  # 初等行变换，化简成行最简，主元为1
    # 满秩压缩，将F和G转化为列满秩和行满秩矩阵
    F = F[:, :r]
    G = G[:r, :]
    return F, G


# 基于满秩分解的伪逆分解
def pinv_split(A):
    F, G = full_rank_split(A)
    # 满秩分解 A === F @ G
    FT_F = F.T @ F  # rxr的矩阵
    G_GT = G @ G.T  # rxr的矩阵
    FT_F_G_GT = FT_F @ G_GT  # rxr的矩阵
    inv, _, _ = row_sample2(FT_F_G_GT)  # 初等行变换，化简成行最简，主元为1
    return G.T, inv, F.T


# 求解伪逆
def pinv(A):
    Gt, inv, Ft = pinv_split(A)
    return Gt @ inv @ Ft


# 求解最小二乘解及最小范数解
def lstsq(A, B):
    Gt, inv, Ft = pinv_split(A)
    print(pinv(A))
    return Gt @ (inv @ (Ft @ B))


if __name__ == '__main__':
    feature = np.array([
        [1, 2, 3],
        [2, 4, 3],
        [2, 4, 3],
        [3, 2, 4]
    ])  # 数据数量×自变量数量
    results = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 6],
        [2, 1, 5],
    ])  # 数据数量×因变量数量
    Xs = lstsq(
        np.concatenate([feature, np.ones((feature.shape[0], 1))], axis=1),
        results)
    W, b = Xs[:-1].T, Xs[-1]

    print(W[0], b[0])  # 第0个因变量的映射权重 w0 和偏执 b0
    print(W[1], b[1])  # 第1个因变量的映射权重 w1 和偏执 b1
    print(W[2], b[2])  # 第2个因变量的映射权重 w2 和偏执 b2
