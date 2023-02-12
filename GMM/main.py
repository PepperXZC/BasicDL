from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt
import sys

iris = datasets.load_iris()

# print(iris.data)

n_dimention = 4
K = len(set(iris.target)) # 几个Gauss模型就对应几个类

def GMM(x, mu, sigma):
    try:
        div_term = (2 * (np.pi ** n_dimention) * abs(np.linalg.det(sigma))) ** 0.5
        exp_term = - 0.5 * np.dot((x - mu).reshape(1, n_dimention), np.dot(np.linalg.inv(sigma), (x - mu).reshape(n_dimention, 1)))
        if np.isinf(np.exp(exp_term)[0][0]).any():
            print("inf")
            print(div_term, exp_term)
            sys.exit()
        exp_term = np.exp(exp_term)[0][0]
        return exp_term / div_term
    except np.linalg.LinAlgError:
        print("Singular matrix")
        print("x", x)
        print("mu", mu)
        print("Sigma", sigma)
        sys.exit()
    
def init():
    mu_init = np.random.uniform(0, 1, (K, n_dimention))
    sigma_init = np.random.uniform(0, 1, (K, n_dimention, n_dimention))
    a1, b1 = np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)
    alpha_init = np.array([a1, b1, 1 - a1 - b1])
    return mu_init, sigma_init, alpha_init

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
main()