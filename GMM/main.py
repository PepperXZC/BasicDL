from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(iris.data)

n_dimention = 4
K = 3 # 几个Gauss模型

def GMM(x, mu, sigma):
    div_term = (2 * (np.pi ** n_dimention) * np.linalg.det(sigma)) ** 0.5
    exp_term = - 0.5 * np.dot((x - mu).T, np.dot(np.linalg.inv(sigma), (x - mu)))
    return np.exp(exp_term) / div_term

def init(X):
    mu_init = np.random.rand(K, n_dimention)
    sigma_init = 2 * np.random.rand(K, n_dimention, n_dimention) + 1