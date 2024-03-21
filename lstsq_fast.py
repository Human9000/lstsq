import numpy as np


def swap_ensure_one(A, i, smooth):
    if abs(A[i, i]) > smooth:
        return True
    for k in range(i + 1, A.shape[0]):
        if abs(A[k, i]) > smooth:
            A[i], A[k] = A[k].copy(), A[i].copy()
            return k
    return False


def row_sample(A):
    """
    @param A: 矩阵
    @return R: 行变化矩阵
    @return Rs: 行阶梯最简矩阵，主元为1，零行全在下方
    @return r: 矩阵的秩
    """
    n, m = A.shape
    E = np.eye(n, n)  # 左乘伴随矩阵
    A_ext = np.hstack((A, E))  # 生成增广矩阵
    smooth = 1e-9
    for p in range(n):
        if not swap_ensure_one(A_ext, p, smooth):
            continue
        A_ext[p, :] = (A_ext[p, :]) / (A_ext[p, p])  # 当前p行归1化
        for q in range(p):  # 当前列上侧行归0化
            A_ext[q, :] = A_ext[q, :] - (A_ext[p, :] * A_ext[q, p]) / (A_ext[p, p])
        for q in range(p + 1, n):  # 当前列下侧行归0化
            A_ext[q, :] = A_ext[q, :] - (A_ext[p, :] * A_ext[q, p]) / (A_ext[p, p])
    # 非零行向上移动，零行向下移动
    r, c = 0, 0
    for c in range(n):
        if abs(A_ext[r, c]) > 1e-1:
            r += 1
        else:  # 将 r 行挪到最后，后面的提前
            A_ext[r:-1, :], A_ext[-1, :] = A_ext[r + 1:, :], A_ext[r, :].copy()
    R, Er = A_ext[:, m:], A_ext[:, :m]
    return R, Er, r


def pinv_split(A):
    # 满秩分解 A === F @ G,
    R, G, r = row_sample(A)  # 初等行变换，化简成行最简，主元为1
    F, E, _ = row_sample(R)  # 初等行变换，化简成行最简，主元为1
    # 启动满秩压缩，将F和G转化为列满秩和行满秩矩阵
    F = F[:, :r]
    G = G[:r, :]
    # 满秩分解 A === F @ G
    FT_F = F.T @ F  # rxr的矩阵
    G_GT = G @ G.T  # rxr的矩阵
    FT_F_G_GT = FT_F @ G_GT  # rxr的矩阵
    inv, _, _ = row_sample(FT_F_G_GT)  # 初等行变换，化简成行最简，主元为1
    pinv = G.T @ inv @ F.T
    return G.T, inv, F.T


def pinv_fast(A):
    Gt, inv, Ft = pinv_split(A)
    return Gt @ inv @ Ft


def lstsq2_fast(A, B):
    Gt, inv, Ft = pinv_split(A)
    return Gt @ (inv @ (Ft@ B))


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
Xs = lstsq2_fast(
    np.concatenate([feature, np.ones((feature.shape[0], 1))], axis=1),
    results)
W, b = Xs[:-1].T, Xs[-1]
print(W[0], b[0])  # 第0个因变量的映射权重 w0 和偏执 b0
print(W[1], b[1])  # 第1个因变量的映射权重 w1 和偏执 b1
print(W[2], b[2])  # 第2个因变量的映射权重 w2 和偏执 b2

