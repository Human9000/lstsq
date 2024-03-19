import numpy as np  
  
# 求行最简形式，及其行变换矩阵
def row_echelon_form(A: np.matrix):   
    ER = A.copy()
    # 获取矩阵的行数和列数  
    num_rows, num_cols = ER.shape 
    # 初始化行变换矩阵为单位矩阵  
    R = np.identity(num_rows, dtype='float')   
    smooth = 1e-9
    # 主元搜索和行变换
    for i in range(num_cols):
        # print(A)
        if abs(ER[i, i]) < smooth: # 找到i列值非零的行，与当前行交换
            flag = False
            for j in range(i+1, num_rows):
                if abs(ER[j, i]) > smooth:
                    flag = True
                    ER[j,:],ER[i,:] = ER[i,:].copy(),ER[j,:].copy()
                    R[j,:],R[i,:] = R[i,:].copy(),R[j,:].copy()
                    break
            if not flag:
                continue

        # 用当前行的主元归一化当前行  
        pivot = ER[i, i]   
        ER[i, :] /= pivot
        R[i, :] /= pivot  
          
        # 将当前行上方的行用当前行消元  
        for j in range(0, i):            
            factor = ER[j, i]
            if abs(factor) > smooth:
                ER[j, :] -= factor * ER[i, :]  
                R[j, :] -= factor * R[i, :]

        # 将当前行下方的行用当前行消元  
        for j in range(i + 1, num_rows):  
            factor = ER[j, i]  
            if abs(factor) > smooth:
                ER[j, :] -= factor * ER[i, :]  
                R[j, :] -= factor * R[i, :]  
      
    return ER, R  

def lstsq(A, B):
    E1, R1 = row_echelon_form(A.T @ A)
    X = R1 @ (A.T @ B) # 列满秩的条件下的最小二乘解
    if abs(E1[-1,-1]) < 1e-2: # 列不满秩, 拥有通解
        _, R2 = row_echelon_form(E1 @ E1.T)
        X = E1.T @ (R2 @ X) # 最小范式特解
    return X



# 示例使用  
A = np.matrix([[1, 2, 3],
                [2, 4, 6],
                  [1, 1, 2],
                  [1, 1, 2]] , dtype='float')
B = np.matrix([[1, 2, 3, 4]], dtype='float').T
ER, R = row_echelon_form(A.copy())
print(R @A)

print("Row-Echelon Form of A:")  
print(ER)  
print("Row Transformation Matrix E:")  
print(R)
print("lstsq AX=B:")  
print(np.linalg.lstsq(A, B, rcond=None))
print(lstsq(A, B))
