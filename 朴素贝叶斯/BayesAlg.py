import torch

result = ['好瓜', '坏瓜']

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

    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        uniqueLabel = [key for key in uniqueLabel]
        labels_full[labels[i]] = uniqueLabel

    return dataSet, labels, labels_full

def dataset_to_tensor(dataset, labels, labels_full, result):
    data = torch.zeros((len(dataset), len(labels)))
    label = torch.zeros(len(dataset))

    for index in range(len(dataset)):
        for j in range(len(labels) - 2):
            data[index][j] = labels_full[labels[j]].index(dataset[index][j])
        for j in range(len(labels) - 2, len(labels)):
            data[index][j] = dataset[index][j]

        label[index] = result.index(dataset[index][-1])

    return data, label


def BayesAlg(x, dataset, label_list, labels, labels_full):

    count_0 = len([num for num in label if num == 0])
    count_1 = len(label) - count_0
    pos_0, pos_1 = count_0 / len(label), count_1 / len(label)

    poss_0_table = torch.zeros(len(label_list))
    poss_1_table = torch.zeros(len(label_list))
    for chara in range(len(label_list) - 2):
        table_0, table_1 = count_para(dataset, label_list, chara, labels, labels_full, count_0, count_1)
        # 查询该特征值对应labels_full中特征的下标，将中文特征值转换为数字
        index = labels_full[label_list[chara]].index(x[chara])
        # 通过下标值取得该特征在好瓜、坏瓜分类中分别出现的概率
        poss_0_table[chara], poss_1_table[chara] = table_0[index] / table_0.sum(), table_1[index] / table_1.sum()

    for chara in range(len(label_list) - 2, len(label_list)):
        table_0, table_1 = count_para(dataset, label_list, chara, labels, labels_full, count_0, count_1)
        # 查询该特征值在好瓜、坏瓜分类中距离（L1）最小的样本下标。
        index_0, index_1 = torch.argmin(torch.abs(torch.tensor(table_0) - x[chara])), torch.argmin(torch.abs(torch.tensor(table_1) - x[chara]))
        loss_0, loss_1 = abs(table_0[index_0] - x[chara]), abs(table_1[index_1] - x[chara])
        poss_0_table[chara], poss_1_table[chara] = ( 1 - loss_0 ) / (loss_1 + loss_0), ( 1 - loss_1 ) / (loss_1 + loss_0)
    poss_0 = torch.prod(poss_0_table) * pos_0
    poss_1 = torch.prod(poss_1_table) * pos_1
    return 0 if poss_0 >= poss_1 else 1

def count_para(dataset, label_list, chara, labels, labels_full, count_0, count_1):

    acc_0 = torch.zeros(len(labels_full[label_list[chara]]))
    acc_1 = torch.zeros(len(labels_full[label_list[chara]]))
    if chara == len(label_list) - 1 or chara == len(label_list) - 2:
        acc_0 = labels_full[label_list[chara]][:count_0]
        acc_1 = labels_full[label_list[chara]][count_0:]
    else:
        for num in range(len(acc_0)):
            chara_name = labels_full[label_list[chara]][num]
            temp = labels_full[label_list[chara]].index(chara_name)
            i = 0
            for sample in dataset:
                if sample[chara] == temp and labels[i] == 0:
                    acc_0[num] += 1
                elif sample[chara] == temp and labels[i] == 1:
                    acc_1[num] += 1
                i += 1
    # print(acc_0, acc_1)
    return acc_0, acc_1


    # 计算 P(X^j = x^j | Y = c_k)
    # for ck in range(2):
    #     for sample in dataset:
    #         if sample[-1] == 0:
    #             count_chara = 0
    #             for chara in sample:


        # ck_list = [sample for sample in dataset if sample[-1] == ck]

        # xj_ck_list = []

dataSet, labels, labels_full = createDataSet()
# print(dataSet)
# print(labels_full[labels[0]])
# print(labels_full['色泽'].index('青绿'))
# print(labels_full[labels[0]])
# print(labels_full[labels[0]][1])
# print(labels)
# print(labels_full)
data, label = dataset_to_tensor(dataSet, labels, labels_full, result)
# print(data)
TP, FN, FP, TN= 0, 0, 0, 0
for i in range(len(dataSet)):
    y = BayesAlg(dataSet[0], data, labels, label, labels_full)
    # print("这个瓜是好瓜") if y == 0 else print("这个瓜是坏瓜")
    if y == 0 and i < 8:
        TP += 1
    elif y == 0 and i >= 8:
        FP += 1
    elif y == 1 and i < 8:
        FN += 1
    elif y == 1 and i >= 8:
        TN += 1

print("查准率P={num1}, 查全率R={num2}".format(num1=(TP / (TP + FP)), num2=(TP / (TP + FN))))
