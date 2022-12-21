import torch
import matplotlib.pyplot as plt

# def create_dataset():
#     chara_0 = torch.tensor([0.6970, 0.7740, 0.6340, 0.6080, 0.5560, 0.4030, 0.4810, 0.4370, 0.6660,
#      0.2430, 0.2450, 0.3430, 0.6390, 0.6570, 0.3600, 0.5930, 0.7190])
#     chara_1 = torch.tensor([0.4600, 0.3760, 0.2640, 0.3180, 0.2150, 0.2370, 0.1490, 0.2110, 0.0910,
#         0.2670, 0.0570, 0.0990, 0.1610, 0.1980, 0.3700, 0.0420, 0.1030])
#     count = 0
#     class_0, class_1 = torch.zeros((8, 2)), torch.zeros((9, 2)) # 设置类别
#     for (c0, c1) in zip(chara_0, chara_1):
#         if count < 8:
#             class_0[count] = torch.tensor([c0, c1])
#         else:
#             class_1[count - 8] = torch.tensor([c0, c1])
#         count += 1
#     return class_0, class_1

def create_dataset():
    X1 = torch.randn((8, 2)) * 5 + 15  # 类别A
    X2 = torch.randn((8, 2)) * 5 + 2 # 类别B
    return X1, X2

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

def scatter_between_class(class_0, class_1):
    mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
    res = torch.mul((mean_0 - mean_1).reshape(-1, 1), (mean_0 - mean_1))
    print(res.shape)
    return res

def LDA(class_0, class_1):
    s_w = scatter_in_class(class_0) + scatter_in_class(class_1)
    s_b = scatter_between_class(class_0, class_1)

    mean_0, mean_1 = class_0.mean(dim=0), class_1.mean(dim=0)
    U, S, V = torch.linalg.svd(s_w) # SVD分解
    s_w_inv = torch.mul(torch.mul(V.T, torch.linalg.pinv(torch.diag(S))), U.T) # 求得逆矩阵
    w = torch.matmul(s_w_inv , (mean_0 - mean_1).reshape(-1, 1))
    print(w, w[0], w[1])
    return w


def plot(class_0, class_1):
    result = torch.cat((class_0, class_1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = torch.arange(-10, 10, 0.1)
    w = LDA(class_0, class_1)
    y = w[1] * x / w[0]
    ax.scatter(result[:8, 0].tolist(), result[:8, 1].tolist(), color='red')
    ax.scatter(result[8:, 0].tolist(), result[8:, 1].tolist(), color='blue')
    ax.plot(x.tolist(), y.tolist(), 'r--', color='green')
    plt.show()

class_0, class_1 = create_dataset()
# print(class_0)
# print(class_1)
result = torch.cat((class_0, class_1))
# print(torch.mul((torch.tensor([0.6970, 0.4600]) - torch.tensor([0.5738, 0.2788])).reshape(-1, 1) , (torch.tensor([0.6970, 0.4600]) - torch.tensor([0.5738, 0.2788])).reshape(-1, 2)))
plot(class_0, class_1)