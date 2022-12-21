# 关于朴素贝叶斯算法的简单应用
##前置包
请在验证下列代码前，在代码文件的首段运行以下代码：
```python
import torch
```

##数据集
首先在网上导入西瓜数据集3.0：
```python
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
```

由这些代码得到 labels 的列表：`['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']`，
并且将所有数据合并、去重后，每个样本中的每维特征（中文特征）都从字典 labels_full 中取得：
```python
labels_full = {'色泽': ['青绿', '浅白', '乌黑'], '根蒂': ['蜷缩', '硬挺', '稍蜷'], ...}
```
根据这个字典中的每个 list 就可以将中文形式的特征转换为数字（例如：“色泽-青绿”就转化为在"色泽”维度下的值为0，这个0代表了索引的含义）.
再填入数字形式的特征得到数据集最终为：
```python
data = tensor([[0.0000, 0.0000, 2.0000, 1.0000, 0.0000, 0.0000, 0.6970, 0.4600],
        [2.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.7740, 0.3760],
        [2.0000, 0.0000, 2.0000, 1.0000, 0.0000, 0.0000, 0.6340, 0.2640],
        [0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.6080, 0.3180],
        [1.0000, 0.0000, 2.0000, 1.0000, 0.0000, 0.0000, 0.5560, 0.2150],
        [0.0000, 2.0000, 2.0000, 1.0000, 1.0000, 1.0000, 0.4030, 0.2370],
        [2.0000, 2.0000, 2.0000, 0.0000, 1.0000, 1.0000, 0.4810, 0.1490],
        [2.0000, 2.0000, 2.0000, 1.0000, 1.0000, 0.0000, 0.4370, 0.2110],
        [2.0000, 2.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.6660, 0.0910],
        [0.0000, 1.0000, 0.0000, 1.0000, 2.0000, 1.0000, 0.2430, 0.2670],
        [1.0000, 1.0000, 0.0000, 2.0000, 2.0000, 0.0000, 0.2450, 0.0570],
        [1.0000, 0.0000, 2.0000, 2.0000, 2.0000, 1.0000, 0.3430, 0.0990],
        [0.0000, 2.0000, 2.0000, 0.0000, 0.0000, 0.0000, 0.6390, 0.1610],
        [1.0000, 2.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.6570, 0.1980],
        [2.0000, 2.0000, 2.0000, 1.0000, 1.0000, 1.0000, 0.3600, 0.3700],
        [1.0000, 0.0000, 2.0000, 2.0000, 2.0000, 0.0000, 0.5930, 0.0420],
        [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.7190, 0.1030]])
```
其中每行代表每个样本，样本中每列为每个特征。不难看到，除最后两列外，其余列均由小数部分为0的浮点数表示。
最后两列为小数部分不为0的浮点数，故保留原数据。

同样的方式可以得到每个样本对应的标签：
```python
label = tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```
其中 0 表示好瓜，1表示坏瓜。

## 算法

接下来是朴素贝叶斯算法基于 PyTorch 框架的实现。
首先从朴素贝叶斯算法正文可见，我们的核心目标是要计算【对于前提已假设好标签 c_k 的前提下，样本 x 在样本空间 X 中各特征值出现的概率】。
我们要求的就是：对于待预测样本 x，哪种标签 c_k 能取到对应的概率最大，我们就认为这个样本最有可能属于这个标签。

因此，我们采用的函数如下：
```python
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
```

该函数引入`dataset`为上文中提到的数据集`data`，`label_list`为列表`['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']`，
`chara`代表所要考虑的第`chara`个维度的特征，`labels`为`dataset`中各样本对应的真实标签值，`labels_full`与上文中提及的`labels_full`一致。
`count_0`, `count_1`分别表示好瓜、坏瓜的数量，也就是对应各标签的样本个数。

该函数分以下两种情形计算：
1. 若所计算的特征最后两种特征之一：直接将对应的特征数据作为返回值返回。
2. 若所计算的特征为除最后两种特征的其余特征：记录在给定标签值的前提下，该特征在全体样本空间中出现过的所有值的个数。
如`'色泽': ['青绿', '浅白', '乌黑']`，假设全体样本空间中共8个好瓜，对应有4个青绿、4个浅白色泽；9个坏瓜，对应6个浅白、3个乌黑色泽，则返回值
为：`[4, 4, 0], [0, 6, 3]`。

得到具体个数后，主体函数如下：
```python
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
```
该函数将计算全体好瓜、坏瓜的情况。向该函数传递数据集（全体样本空间），以及待测样本 x， 特征名称列表 label_list， 数据集中样本的标签 labels， 
及labels_full，可以按如下方式计算各特征值对应的概率：
1. 计算除最后两列特征（密度、含糖率）以外的其它特征（文字特征：色泽、根蒂等）：由`count_para`函数传递该特征对应在好瓜分类、坏瓜分类的数据，
计算样本 x 中的该特征所取得的值在好瓜、坏瓜中分别出现的概率。
2. 计算除最后两列特征（密度、含糖率）：分别在好瓜、坏瓜分类中找到与该特征值距离最小的样本（记为距离0、距离1），然后按以下方式计算 x 归属于好瓜的概率：
`(1 - 距离0) / (距离0 + 距离1)`. 坏瓜类似。

由于这里出现了非0-1值的特征值情况，故需要采用不同的概率计算方式（引入样本 x 时，很难保证密度、含糖率特征值在其它样本里已经取到过）
，故采用了自定义的计算方式。我尚认为这个方式欠妥当，它目前只能确保分类至好、坏瓜概率大小关系的正确性，值本身很可能是有误的。

## 模型估计
使用传统的查全率、查准率标准计算朴素贝叶斯模型在该数据集下的分类能力：
```python
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
>>> 查准率P=0.47058823529411764, 查全率R=1.0
```