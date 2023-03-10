import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#os.chdir(r"D:\mywork\test")


class SOMnetwork(object):
    """
    Self-Organizing Feature Map
    """
    def __init__(self, maxRound, minRound, maxRate, minRate, Iters):
        self.maxRound = maxRound
        self.minRound = minRound
        self.maxRate = maxRate
        self.minRate = minRate
        self.steps = Iters
        
        self.RateList = []                  #存放每轮迭代学习率的容器
        self.RoundList = []                 #存放每轮迭代优胜半径的容器
        self.X = 0                          #训练数据集
        self.normX = 0                      #归一化后的数据集
        
        self.gridLocation = 0               #竞争层的神经元节点位置坐标
        self.w = 0                          #神经元节点权重
        self.gridDist = 0                   #神经元节点之间的距离
    
    #归一化函数
    def normalize(self,X):
        return (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    
    
    #计算欧式距离函数
    def edist(self,X1,X2):
        return (np.linalg.norm(X1-X2))
    
    
    #计算各个节点之间的距离
    def calGdist(self, grid):
        m = len(grid)
        Gdist = np.zeros((m,m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    Gdist[i,j] = self.edist(grid[i], grid[j])
        return Gdist


    #初始化竞争层
    def init_grid(self,M,N):
        grid = np.zeros((M*N,2))      #分成M*N类，两个维度
        k = 0
        for i in range(M):
            for j in range(N):
                grid[k,:] = np.array([i,j])
                k += 1
        return grid
    
    #学习率和优胜半径的递减函数
    def changeRate(self,i):
        Rate = self.maxRate - (self.maxRate-self.minRate)*(i+1)/self.steps
        Round = self.maxRound - (self.maxRound-self.minRound)*(i+1)/self.steps
        return Rate, Round
    
    def train(self, X, M, N):
        self.X = X
        ##标准化数据集
        X = self.normalize(X)
        n_samples, n_features = X.shape
        ##1 初始化
        ###各个节点位置，以及各节点之间的位置
        self.gridLocation = self.init_grid(M,N)
        self.gridDist = self.calGdist(self.gridLocation)
        ###初始化各个节点对应的权值
        w = np.random.random((M*N, n_features))
        ###确定迭代次数，不小于样本数的5倍
        if self.steps<5*n_samples:
            self.steps = 5*n_samples
        for i in range(self.steps):
            ##2 竞争
            ###随机选取样本计算距离
            data = X[np.random.randint(0, n_samples, 1)[0], :]
            Xdist = [self.edist(data,w[row]) for row in range(len(w))]
            ###找到优胜节点
            winnerPointIdx = Xdist.index(min(Xdist))
            ##3 迭代
            ###确定学习率和节点优胜半径，并保存
            Rate, Round = self.changeRate(i)
            self.RateList.append(Rate)
            self.RoundList.append(Round)
            ###圈定优胜邻域内的所有节点
            winnerRoundIdx = np.nonzero(self.gridDist[winnerPointIdx]<Round)[0]
            ###对节点权值进行调整，为了简化运算这里暂不考虑节点的更新约束
            w[winnerRoundIdx] = w[winnerRoundIdx] + Rate*(data-w[winnerRoundIdx])
        self.w = w
        self.normX = X
    
    def cluster(self, X):
        X = self.normalize(X)
        m = X.shape[0]
        cluster_labels = []
        for i in range(m):
            yi = np.linalg.norm((X[i] - self.w), axis=1)
            cluster_labels.append(yi.argmin())
        return np.array(cluster_labels)

def draw(y, cluster, m, n, gridLocation):
    yset = np.unique(y)
    fig = plt.figure(dpi=100)
    for i in range(gridLocation.shape[0]):
        labelisi = sum(cluster==i)
        if labelisi > 0:
            #统计映射后的每个真实标签的个数
            yi = y[cluster==i]
            yiCount = np.bincount(yi)
            #统计比率
            if yiCount.size < yset.size:
                yiCount = yiCount.tolist() + [0]*(yset.size-yiCount.size)
            yiRatio = yiCount/labelisi
            plt.subplot(n, n, i+1)
            plt.pie(yiRatio)
            plt.text(x=0, y=0, s=str(labelisi), fontsize=15, horizontalalignment='center', verticalalignment='center')
    fig.legend(yset)
    plt.show()

if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    #自编的SOM网络对Iris进行6*6的映射
    n = 6
    som_self = SOMnetwork(5, 0.1, 0.5, 0.01, 1000)
    som_self.train(X, n, n)
    som_self_cluster = som_self.cluster(X)
    som_grid = som_self.gridLocation
    som_grid_Idx = [(som_grid[i,0], som_grid[i,1]) for i in range(som_grid.shape[0])]
    draw(y, som_self_cluster, n, n, som_grid)