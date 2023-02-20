### PCA
这是一个过程上相对简单的算法，没有所谓的数据初始化，只要复现出函数程序就可以获得答案。

#### 数据集
仍然使用最熟悉的`iris`数据集。
```python
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
```

#### PCA函数
```python
def PCA_method(K,li):  
    matrix = np.mat(li.data)
    n_samples, n_features = matrix.shape
    mean_matrix = np.mean(matrix,axis=0)
    Dataadjust = matrix - mean_matrix
    covMatrix = np.cov(Dataadjust,rowvar=0)
    eigValues , eigVectors = np.linalg.eig(covMatrix)
    eig_pairs = [(np.abs(eigValues[i]), eigVectors[:,i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)
    feature = np.array([data[1] for data in eig_pairs[:K]])
    res = np.dot(matrix, np.transpose(feature))

    target = li.target
    print(target)
    draw_picture(li.target, res)
```
简单易懂的过程，直接按照课件上的过程进行。`numpy`库提供了非常完备的求均值、求特征值、求特征向量、求协方差等功能，直接调用就可以获得相应的数值数据。

#### sklearn 版 PCA 函数

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x = pca.fit_transform(iris.data)
```
更加简单的过程，只需要一行代码解决所有问题。

#### 画图函数
```python
def draw_picture(target, res):
    l0, r0 = 0, np.where(target == 1)[0][0]
    l1, r1 = np.where(target == 1)[0][0], np.where(target == 2)[0][0]
    l2, r2 = np.where(target == 2)[0][0], len(target)
    plt.scatter(res[l0:r0,0].tolist(), res[l0:r0,1].tolist(), color='g')
    plt.scatter(res[l1:r1,0].tolist(), res[l1:r1,1].tolist(), color='b')
    plt.scatter(res[l2:r2,0].tolist(), res[l2:r2,1].tolist(), color='r')
    plt.show()
    return
```
在这里为了直观地将源数据集中不同的类别按不同颜色区分，并展示降维后的结果，我们默认降维至二维。原数据集中共有150个样本，每个类别占50个。

#### 结果
在正确安装相应的外部包的基础上，直接运行`main.py`就可以获得程序直观图（降维至二维）。

需要注意的是，**两次数据可视化获得的结果有些差异**。可以看到，`sklearn`自带库所带来的结果与我们自己写的函数得到的结果不同，这可能是以为`sklearn`自带的库函数具有不同的计算方式（可能是为了加快程序运行速度、为了适配不同的降维维数要求等）。我并没有得到这两者差异的真正原因，但是我们的PCA函数并没有太大的问题，也能够得到降维的效果。
