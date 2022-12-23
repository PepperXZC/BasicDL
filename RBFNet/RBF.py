import d2l
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from d2l import torch as d2l
import sys
from torch.utils.data import  Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader_workers():  #@save
    """在非Windows的平台上，使用4个进程来读取数据"""
    return 0 if sys.platform.startswith('win') else 4

# def load_data_fashion_mnist(batch_size, resize=None):  #@save
#     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
#     # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
#     # 并除以255使得所有像素的数值均在0～1之间
#     trans = [transforms.ToTensor()]
#
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True)
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans, download=True)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=get_dataloader_workers()))

# num_inputs, num_outputs, num_hiddens = 784, 10, 256

class RBF(nn.Module):

    def __init__(self, centers, num_outputs):
        """
        :param centers: shape=[center_num,data_dim]
        :param n_out:
        """
        super(RBF, self).__init__()
        self.num_centers = centers.shape[0]
        self.n_dim = centers.shape[1] - 1
        self.num_output = num_outputs

        # 假设预先通过其它方式给定好 centers
        self.centers = centers

        self.beta = nn.Parameter(torch.ones(self.num_centers) / 2, requires_grad=True)

        self.Linear = nn.Linear(self.num_centers, 1)
        # self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x):
        hidden_out = torch.zeros(self.num_centers)
        for index in range(self.num_centers):
            mse_temp = (-1) * self.beta[index] * torch.norm(x - self.centers[index])
            hidden_out[index] = torch.exp(mse_temp)
        if torch.any(torch.isnan(self.beta)):
            print('hi')

        hidden_out = torch.where(torch.isinf(hidden_out), torch.full_like(hidden_out, 0), hidden_out)
        out = self.Linear(hidden_out)
        return out

batch_size = 256
# train_iter, test_iter = load_data_fashion_mnist(batch_size)
# trans = [transforms.ToTensor()]
# sampler = data.RandomSampler(torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans, download=True), replacement=False, num_samples=None)
# for i in sampler:
#     print(i)
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def make_dataset(batch_size):
    data = pd.read_excel('数据.xlsx', index_col=0)
    # print(data)
    data = data.to_numpy()
    sampler_train = torch.utils.data.RandomSampler(torch.arange(data.shape[0]))
    index_list = [index for index in sampler_train][:15]
    # for i in sampler_train:
    #     print(i)
    centers = torch.zeros(len(index_list), len(data[0]) - 1)
    for i in range(len(index_list)):
        centers[i] = torch.tensor(data[index_list[i]][:-1])
    test_num = len(data) // (batch_size + 1)
    # 中心点samples, train_set, test_set
    return centers, torch.tensor(data[:-test_num]), torch.tensor(data[-test_num:])

class iter_dataset:
    def __init__(self, data, batch_size):
        self.dataset = data
        self.num_data = len(self.dataset)
        self.num_dim = len(self.dataset[0])
        self.batch_size = batch_size
        # self.mode = mode

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        out, y = torch.zeros(batch_size, self.num_dim - 1), torch.zeros(batch_size)
        j = 0
        while self.i < self.num_data:
            out[j], y[j] = self.dataset[self.i][:-1], self.dataset[self.i][-1]
            self.i += 1
            j += 1
            if j == batch_size:
                return out, y
            if self.i >= self.num_data:
                # out = out[[not torch.all(out[i] == 0) for i in range(out.shape[0])], :]
                # y = y[[not torch.all(y[i] == 0) for i in range(len(y))]]
                for k in range(batch_size, j - 1, -1):
                    out = out[torch.arange(out.size(0))!=k]
                    y = y[torch.arange(y.size(0))!=k]
                self.i = 0
                return out, y
                # self.i, j = 0, 0
                # yield out, y
                # break
                # raise StopIteration
            # out, y = torch.zeros(batch_size, self.num_dim - 1), torch.zeros(batch_size)

def accuracy(y_hat, y):  #@save
    """由loss差值来计算精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    minus = nn.MSELoss()
    cmp = minus(y_hat.type(y.dtype), y)
    return float(cmp)

def evaluate_accuracy(net, data_iter, batch_size):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 每次的loss、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            if len(X) < batch_size:
                break
    return metric[0] / metric[1]

def train_per_epoch(model, train_iter, loss, updater, batch_size):
    model.train()
    metric = Accumulator(2)

    for X, y in train_iter:
        y_hat = torch.zeros_like(y)
        for i in range(len(X)):
            y_hat[i] = model(X[i])
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), y.numel())
        if len(X) < batch_size:
            break
    return metric[0] / metric[1]

if __name__ == '__main__':
    loss = nn.MSELoss()
    batch_size = 15
    centers, train_set, test_set = make_dataset(batch_size)
    num_epochs, lr,  = 200, 0.1

    train_iter, test_iter = iter_dataset(train_set, batch_size), iter_dataset(test_set, batch_size)

    model = RBF(centers, len(train_set[0]))
    updater = torch.optim.Adam(model.parameters(), lr=lr)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_per_epoch(model, train_iter, loss, updater, batch_size)
        test_acc = evaluate_accuracy(model, test_iter, batch_size)
        # animator.add(epoch + 1, train_metrics + test_acc)
        print(train_metrics, test_acc)
    train_loss = train_metrics
    # a = torch.tensor([0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 0, 0, 0, 0, 0])
    # a = a[[not torch.all(a[i] == 0) for i in range(len(a))]]
    # print(a)
    # print()
