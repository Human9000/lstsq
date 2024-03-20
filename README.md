# lstsq

先看这个：[知乎：最小二乘解和最小范数解的一般公式](https://zhuanlan.zhihu.com/p/503664717)

然后是下面：
lstsq 最小二乘解，python实现，结果与numpy中的函数一致

求 $AX=B$ 的最小二乘解 $argmin||AX-B||_2$ 或最小范数特解 $argmin||X||_2$

已知，任意矩阵 $A$, 存在可逆的行变换 $R$ 使$A$化简为行最简形式$E_r$

$$
RA=E_r
$$

那么存在可逆行变换$R_1$和行最简行$E_{r1}$使下式成立

$$
R_1A^TA = E_{r1}
$$

$$
A^TA = R_1^{-1}E_{r1}
$$
$$
(A^TA)^{-1} = E_{r1}^{-1}R_1 \approx  R_1
$$

当 $E_{r1}$ 为满秩阵时，$E_{r1}=E$ ，$E$为单位阵，此时只有最小二乘解 $ X_{min_2} $，无通解，无最小范式解

$$
X_{min_2} = (A^TA)^{-1}A^TB= R_1A^TB
$$

代码

```python
Er1, R1 = row_echelon_form(A.T @ A)
X_min_2 = R1 @ (A.T @ B) # 列满秩的条件下的最小二乘解
```

当 $E_{r1}$ 为非满秩阵时，$E_{r1}$ 有通解，$X$ 有无穷多个解，有最小范数解
此时问题转化为求下面的，一个行最简阵为系数的最小范数解 $X_{min_f}$

$$
E_{r1}X_{min_f} = X_{min_2}
$$

$$
E_{r1}^TE_{r1}X_{min_f} = E_{r1}^TX_{min_2}
$$

此时存在可逆行变换矩阵 $R_2$ 和 左上角单位阵$E_{r2}$ 使下面的式子成立

$$
R_2E_{r1}^TE_{r1}=E_{r2}
$$

则

$$
E_{r1}^TE_{r1}=R_2^{-1}E_{r2}
$$

则

$$
(E_{r1}^TE_{r1})^{-1}=E_{r2}^{-1}R_2 \approx R_2
$$

则x的最小范数解为

$$
X_{min_f} = E_{r1}^T(E_{r1}^TE_{r1})^{-1}X_{min_2} = E_{r1}^TR_2X_{min_2} = E_{r1}^TR_2R_1A^TB
$$

代码

```python
Er2, R2 = row_echelon_form(Er1 @ Er1.T)
X_min_f = ER1.T @ (R2 @ X_min_2) # 列不满秩的条件下的最范数解
```

$$
AX = E_{r1}^TR_2R_1A^TB
$$

推论，任何矩阵的伪逆

$$
A^{-1} = E_{r1}^TR_2R_1A^T

 
R_1A^TA = E_{r1}


R_2E_{r1}^TE_{r1}=E_{r2}
$$

