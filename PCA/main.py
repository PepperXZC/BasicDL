import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

iris = datasets.load_iris()

def draw_picture(target, res):
    l0, r0 = 0, np.where(target == 1)[0][0]
    l1, r1 = np.where(target == 1)[0][0], np.where(target == 2)[0][0]
    l2, r2 = np.where(target == 2)[0][0], len(target)
    plt.scatter(res[l0:r0,0].tolist(), res[l0:r0,1].tolist(), color='g')
    plt.scatter(res[l1:r1,0].tolist(), res[l1:r1,1].tolist(), color='b')
    plt.scatter(res[l2:r2,0].tolist(), res[l2:r2,1].tolist(), color='r')
    plt.show()
    return

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

# file = load_txt('D:\Codebase\Python基础工具\data.txt')
file = iris
# file = [[1,1],[1,3],[2,3],[4,4],[2,4]]
print(PCA_method(2,file))

# 对比 sklearn 自带的 PCA
pca = PCA(n_components=2)
x = pca.fit_transform(iris.data)
draw_picture(iris.target, x)