### GMM 算法

就是说给定高斯函数为范本的模型，要求算出它对应的均值、方差，使之能够逼近已有的数据点。

这里还是考虑高维的GMM模型，低维的总归能够从高维来逼近。

#### 数据集

sklearn 自带的 iris 数据集。

```python
from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
import sys

iris = datasets.load_iris()

# print(iris.data)

n_dimention = 4
K = len(set(iris.target)) # 几个Gauss模型就对应几个类
```
观察一下`iris.data`就可以看到数据：
```python
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 ...
```
#### 主要函数
##### Gauss 概率函数
```python
def GMM(x, mu, sigma):
    try:
        div_term = (2 * (np.pi ** n_dimention) * abs(np.linalg.det(sigma))) ** 0.5
        exp_term = - 0.5 * np.dot((x - mu).reshape(1, n_dimention), np.dot(np.linalg.inv(sigma), (x - mu).reshape(n_dimention, 1)))
        exp_term = np.exp(exp_term)[0][0]
        return exp_term / div_term
    except np.linalg.LinAlgError:
        print("Singular matrix")
        print("x", x)
        print("mu", mu)
        print("Sigma", sigma)
```
**分析**：这部分完全按照公式变化为程序即可。但这里有非常严重的**溢出问题**：一旦因初始化、收敛不够快等因素导致当前指数函数中的指数稍大时，`np.exp`会运行出无限大`inf`的结果，影响到后续所有程序运行。以及**矩阵不可逆**的情形，原因同样可能是来自于初始化、收敛所得的中间状态等等。这里使用报错检查来检查是否存在这两种情形。如果遇到这两种情况之一，马上退出程序并记录当前数值。**因为暂时找不到更好的初始化程序…**

#### 初始化
```python
def init():
    mu_init = np.random.uniform(0, 1, (K, n_dimention))
    sigma_init = np.random.uniform(0, 1, (K, n_dimention, n_dimention))
    a1, b1 = np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)
    alpha_init = np.array([a1, b1, 1 - a1 - b1])
    return mu_init, sigma_init, alpha_init
```
根据所需的`shape`进行在`[0,1]`之间的随机采样。

#### EM算法
根据给定的算法步骤进行程序设计
##### E-step
```python
def EM_E_step(X, mu, sigma, alpha):
    n_data = np.zeros((X.shape[0], K)) # 储存 gamma_n_k
    for index in range(len(X)):
        N_list = np.zeros(K)
        sum_alpha = 0
        for k in range(K):
            N_list[k] = GMM(X[index], mu[k], sigma[k])
            n_data[index][k] = N_list[k]
            sum_alpha += alpha[k] * N_list[k]
        n_data[index] /=  sum_alpha
    return n_data
```

##### M-step
```python
def EM_M_step(X, n_data):
    N_k_list = np.sum(n_data, 0)
    N = np.sum(N_k_list)

    alpha_new = N_k_list / N

    mu_new = np.zeros((K, n_dimention))
    sigma_new = np.zeros((K, n_dimention, n_dimention))
    for k in range(K):
        mu_new[k] = np.array([
            X[index] * n_data[index][k] for index in range(X.shape[0])
            ]).sum(axis=0) / N_k_list[k]
        sigma_new[k] = np.array([
            np.dot((X[index] - mu_new[k]).reshape(n_dimention, 1), (X[index] - mu_new[k]).reshape(1, n_dimention)) * n_data[index][k] for index in range(X.shape[0])
            ]).sum(axis=0) / N_k_list[k]    
    return alpha_new, mu_new, sigma_new
```
#### 准确率评估
这是简单的准确率测试：`判断正确个数 / 所有样本数`
```python
def result(X, mu, sigma):
    # 计算测试正确的个数
    res = np.zeros(X.shape[0])
    for index in range(int(X.shape[0])):
        target = np.zeros(K)
        for k in range(K):
            target[k] = GMM(X[index], mu[k], sigma[k])
        possibility = target.argmax()
        if possibility == iris.target[index]:
            res[index] = 1
        else:
            res[index] = 0
    return res.sum() / res.shape[0]
```

#### 主程序
```python
def main():
    mu_init, sigma_init, alpha_init = init()
    X = iris.data
    # res = result(X, mu_init, sigma_init)
    possible = []

    mu_last, sigma_last, alpha_last = mu_init, sigma_init, alpha_init
    times = 0
    # print(np.exp(1000))
    # 需要的是 alpha, mu, sigma 收敛！
    while 1:
        times += 1
        if np.isnan(mu_init).any() or np.isnan(sigma_init).any():
            print("stop")
        n_data = EM_E_step(X, mu_last, sigma_last, alpha_last)
        alpha, mu, sigma = EM_M_step(X, n_data)
        loss = np.linalg.norm(alpha - alpha_last)
        for k in range(K):
            loss = np.linalg.norm(mu[k] - mu_last[k]) + \
                    np.linalg.norm(sigma[k] - sigma_last[k])
        if loss < 1e-4:
            print("好了", times, result(X, mu, sigma))
            break
        else:
            print(times, loss)
            mu_last, sigma_last, alpha_last = mu, sigma, alpha
```
#### 结果
我认为结果非常依赖于数据**初始化**，很不科学…也许是我需要导入 `torch._xavier`的初始化技术。在成功运行程序的前提下，不同的初始化值也会带来不同的收敛情况。因为收敛的标准是**均值、方差的变化小于某个阈值**。

这里每次运行都会得到完全不同的成果。我规定最后两次更新算法得到的均值、方差之间的差值为收敛标准（以下简称差值：小于1e-4时即认为收敛）。

附录包含了多次用例下的不同结果。

#### 附录

用例1：
```python 
# 运行次数；差值
...
23 0.0305009434478390424 
24 0.025632559656993043
25 0.022994234467901567
26 0.019454381378328264
27 0.01735936993237127
28 0.014758766980611907
29 0.013116264404712145
30 0.01119310396766798
# 运行次数：准确率
31 0.7866666666666666
```
该用例得到了较好的模型准确率。

用例2：
```python
# 运行次数；差值；当前各alpha_k之和
...
33 0.061904378174793176 1.0
34 0.02880933485188486 1.0
35 0.03171991678121982 1.0
36 0.0123916160963993 0.9999999999999999
37 0.01893359795115862 1.0
# 运行次数；判断正确率
好了 38 0.84
```
这一次的结果相当好。但存在非常糟糕的用例。

用例3：
```python
1 11.254714457056604
2 0.15844908203936142
Singular matrix
x [5.1 3.5 1.4 0.2]
mu [5.6  2.9  5.25 2.1 ]
Sigma [[0.49   0.28   0.525  0.28  ]
 [0.28   0.16   0.3    0.16  ]
 [0.525  0.3    0.5625 0.3   ]
 [0.28   0.16   0.3    0.16  ]]
```
运行两次后即告矩阵求逆失败。这是因为 `sigma` 矩阵是不可逆矩阵，这和一开始的初始化、之后的更新迭代中涉及到的参数关联性非常大。

用例4：
```python
...
9 0.20057175259081864
10 0.7666704820925674
11 0.27318559431174994
12 0.534348131911094
13 0.37156886414295326
14 0.27205963289390694
好了 15 0.03333333333333333
```
成功收敛但是准确率非常低。

用例5：

```python
...
5270 0.00010000675824269314
好了 5271 0.02666666666666667
```
收敛很慢，效果也很差。
