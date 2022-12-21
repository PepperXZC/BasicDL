import pandas as pd
import numpy as np
import sklearn.datasets as datasets
import torch
import csv
from sklearn.tree import DecisionTreeClassifier

def createDataSet():
    """
    创建测试的数据集，里面的数值中具有连续值
    :return:
    """
    dataSet = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
    ]

    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

    # 特征对应的所有可能的情况
    labels_full = {}

    for i in range(len(labels) - 2):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        uniqueLabel = [key for key in uniqueLabel]
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full

def dataset_to_tensor(dataset, labels, labels_full, result):
    data = torch.zeros((len(dataset), len(labels) - 2))
    label = torch.zeros(len(dataset))

    for index in range(len(dataset)):
        for j in range(len(labels) - 2):
            data[index][j] = labels_full[labels[j]].index(dataset[index][j])
        # for j in range(len(labels) - 2, len(labels)):
        #     data[index][j] = dataset[index][j]

        label[index] = result.index(dataset[index][-1])

    return data, label

result = ['好瓜', '坏瓜']
dataSet, labels, labels_full = createDataSet()
data, label = dataset_to_tensor(dataSet, labels, labels_full, result)
data = data.squeeze(dim=-1)
# print(data[[0, 2, 4, 5, 8]])
# print(labels[:-2])
# print(y.unique())
# print(torch.tensor([1,2,3]).shape)

class DecisionTree:
    def __init__(self, data:torch.tensor, label:torch.tensor, labels_full, max_depth:int):
        self.data = data
        self.label = label
        self.label_set = torch.unique(self.label) # 0, 1
        self.labels_full = labels_full
        self.max_depth = max_depth
        self.decide = {}

    def Ent(self, data_index):
        # if type(data_index) == torch.tensor:
        #     data_index = data_index.tolist()
        label_temp = self.label[data_index]
        count_sum = len(label_temp)
        result = torch.zeros(len(self.label_set))
        for prop in self.label_set:
            temp = torch.zeros(label_temp.shape)
            temp[label_temp == prop] = 1
            prop_count = torch.count_nonzero(temp)
            result[int(prop)] = prop_count / count_sum
        sum = 0
        if 0 in result:
            return 0
        for num in result:
            sum -= num * torch.log2(num)
        return sum

    def Gain(self, label:str, D_index):
        assert label in ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
        Ent_D = self.Ent(D_index)
        label_list = self.labels_full[label]
        label_index = list(self.labels_full).index(label)
        label_Ent_list = []
        count = torch.zeros(len(label_list))
        i = 0
        for label_key in label_list:
            label_key_index = list(self.labels_full[label]).index(label_key)
            chara_list = self.data[D_index, label_index]
            temp = torch.zeros_like(chara_list)
            temp[chara_list == label_key_index] = 1
            index = torch.nonzero(temp).reshape(-1)
            count[i] = torch.count_nonzero(temp).sum()
            i += 1
            label_Ent_list.append(self.Ent(index))
        count, label_Ent = torch.tensor(count), torch.tensor(label_Ent_list)
        if count.sum()==0:
            print('hellp')
        res = Ent_D - ((count / count.sum()) * label_Ent).sum()

        return res

    # def modify_value(self, dic, depth, key_seq, value):
    #     key_index = 0
    #     while depth != 0:
    #         temp = dic[key_seq[key_index]]
    #         key_index += 1
    #         depth -= 1
    #     temp = value

    # 计算IV
    def IV(self, label:str, D_index):
        gain = self.Gain(label, D_index)
        label_list = self.labels_full[label]
        label_index = list(self.labels_full).index(label)
        count = torch.zeros(len(label_list))
        i = 0
        for label_key in label_list:
            label_key_index = list(self.labels_full[label]).index(label_key)
            chara_list = self.data[D_index, label_index]
            temp = torch.zeros_like(chara_list)
            temp[chara_list == label_key_index] = 1
            count[i] = torch.count_nonzero(temp).sum()
            i += 1
        iv = - ((count / count.sum()) * torch.log2(count / count.sum())).sum()
        # 算得 IV 值与书本中示例相同
        return gain / iv

    # 计算基尼指数
    def Gini(self, data_index):
        label_temp = self.label[data_index]
        count_sum = len(label_temp)
        result = torch.zeros(len(self.label_set))
        for prop in self.label_set:
            temp = torch.zeros(label_temp.shape)
            temp[label_temp == prop] = 1
            prop_count = torch.count_nonzero(temp)
            result[int(prop)] = prop_count / count_sum
        sum = 1
        if 0 in result:
            return 0
        for num in result:
            sum -= torch.square(num)
        return sum

    def Gini_index(self, label:str, D_index):
        gain = self.Gain(label, D_index)
        label_list = self.labels_full[label]
        label_index = list(self.labels_full).index(label)
        count, gini = torch.zeros(len(label_list)), torch.zeros(len(label_list))
        i = 0
        for label_key in label_list:
            label_key_index = list(self.labels_full[label]).index(label_key)
            chara_list = self.data[D_index, label_index]
            temp = torch.zeros_like(chara_list)
            temp[chara_list == label_key_index] = 1
            index = torch.nonzero(temp).reshape(-1)
            count[i] = torch.count_nonzero(temp).sum()
            gini[i] = self.Gini(index)
            i += 1
        return ((count / count.sum()) * gini).sum()


    # 使用相应的度量标准 产生划分点
    def decision(self, depth=1, index=[], label_list=None, ch=''):
        if depth == 1:
            index = torch.arange(0, len(data), 1)

        if label_list==None:
            label_list = list(self.labels_full)

        # Gain_list = torch.zeros(len(labels_full))
        Gain_list = []
        iv_dict = {}
        if depth == 1:
            for label in label_list:
                iv_dict[label] = self.IV(label, index)
        print(iv_dict)
        for label in label_list:
            Gain_list.append(self.Gain(label, index))

        Gain_list = torch.tensor(Gain_list)
        max_index = torch.argmax(Gain_list)
        max_chara = label_list[max_index]
        print("第{num1}层决策树的划分属性为：{chara}".format(num1=depth, chara=max_chara))

        del label_list[max_index]
        chara_list = labels_full[max_chara]
        decision_dict = {}
        for chara in chara_list:
            chara_key_index = list(chara_list).index(chara)
            chara_index_list = self.data[:, max_index]
            temp = torch.zeros_like(chara_index_list)
            temp[chara_index_list == chara_key_index] = 1
            index = torch.nonzero(temp).reshape(-1).tolist()
            decision_dict[chara] = index

        # 第0层决策树的划分属性为：纹理
        # 划分树为： {'模糊': [10, 11, 15], '清晰': [0, 1, 2, 3, 4, 5, 7, 9, 14], '稍糊': [6, 8, 12, 13, 16]}
        # 与书本上结果相同。
        if depth==1 and ch =='清晰':
            print(Gain_list)
        for chara in decision_dict:
            if chara == '清晰':
                self.decision(depth+1, decision_dict[chara], label_list, chara)
            # 由这步操作，计算得到与书本上相同的数据。调用 Gain_list 可以获得。



prog = DecisionTree(data, label, labels_full,max_depth=3)
prog.decision()
# print(data[torch.nonzero(torch.tensor([1,0,0,0,4,6,23,1,0]))])
# print(torch.count_nonzero(torch.tensor([1,0,0,0,4,6,23,1,0])))
# print(torch.tensor([1,2,3]) * torch.tensor([4,5,6]))
# a = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
# print(torch.tensor([[1,2,4],2,3,4,5]))
# del a[2]
# print(a)
