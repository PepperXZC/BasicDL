# LDA算法实现
本算法基于 PyTorch 框架完成。

## 数据集展示
西瓜数据集3.0的数据不便于观察，不同类别的离散程度不够强（需要加上中文特征才能识别，但中文特征非本算法可以解决），
所以这里采用随机生成的一组数，赋予明显差距的均值来展现。
```python
def create_dataset():
    X1 = torch.randn((8, 2)) * 5 + 15  # 类别A
    X2 = torch.randn((8, 2)) * 5 + 2 # 类别B
    return X1, X2
```
## 算法
首先考虑如何计算类内散度矩阵，涉及到的代码如下：
```python
def scatter_in_class(class_i):
    mean = class_i.mean(dim=0)
    # print(mean)
    n = len(class_i)
    sum = 0
    for sample in class_i:
        sum += torch.mul((sample - mean).reshape(-1, 1), (sample - mean))
        # 这里相当于自己的转置乘以自己，求得散度矩阵
    print(mean, n, sum.shape)
    return sum
```
须知，在程序中每个样本的格式默认 shape 为(2, 1)的数据，但是理论推导中考虑的样本都是 (1, 2) 的列向量。
所以我们需要对应地将其转置。

然后计算类间散度矩阵：
```python
def scatter_between_class(class_0, class_1):
    mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
    res = torch.mul((mean_0 - mean_1).reshape(-1, 1), (mean_0 - mean_1))
    print(res.shape)
    return res
```

能够计算这两个散度矩阵之后，我们考虑如何在 LDA 算法中使用。我们需要计算到类内散度矩阵的逆矩阵，
但为了数值稳定性，我们时常考虑做 SVD分解。（事实上，torch中直接求逆的函数也能保证一定的数值稳定性）
```python
def LDA(class_0, class_1):
    s_w = scatter_in_class(class_0) + scatter_in_class(class_1)
    s_b = scatter_between_class(class_0, class_1)

    mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
    U, S, V = torch.linalg.svd(s_w) # SVD分解
    s_w_inv = torch.mul(torch.mul(V.T, torch.linalg.pinv(torch.diag(S))), U.T) # 求得逆矩阵
    w = torch.matmul(s_w_inv , (mean_0 - mean_1).reshape(-1, 1))
    print(w, w[0], w[1])
    return w
```
这样我们就能够得到结果。

这里没有插入运行结果的图片。您可以在安装好 PyTorch 及 matplotlib 的环境中直接运行 LDA.py 文件，以查看图像化展示。