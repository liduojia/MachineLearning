'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


# 给定测试集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算给定数据集的 香农熵（集合信息的度量方式）
# 如果待分类的事务可能划分在多个分类中，则符号xi的信息定义为 l(xi) = -log2 p(xi)  其中p(xi)是选择该分类的概率
# 为了计算熵，我们需要计算所有类别所有可能值包含的信息期望值， H = - sigma(i=1 to n) p(xi) * log2 p(xi)
# 熵越高，则混合的数据越多
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 再创建一个数据字典
    labelCounts = {}
    for featVec in dataSet:
        # 字典的键值是最后一列的数值
        currentLabel = featVec[-1]
        # 如果当前键值不存在，则扩展字典并将当前键值加入字典 每个键值都记录了当前类别出现的次数
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 划分数据集  axis表示 划分数据集的特征   value表示 需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
# 要求数据必须是一种由列表元素组成的列表  且所有的列表元素具有相同的数据长度  数据的最后一列（或每个实例的最后一个元素）是当前实例的类别标签
# list中数据类型可以为int，也可以是String，无影响
def chooseBestFeatureToSplit(dataSet):
    # 判定当前数据集包含多少特征属性 （此处 3 - 1 = 2）
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    # 计算数据集原始的香农熵（用于与后续做比对）
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    # 遍历所有的features
    for i in range(numFeatures):  # iterate over all the features
        # 创建一个新的列表 将数据集中所有第i个特征值或者可能存在的值写入list中
        featList = [example[i] for example in dataSet]
        # 将list转成set（集合），以此来消除list中重复的元素，获取列表中唯一元素
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值 对每个唯一属性值划分一次数据集 然后计算新的熵
        # 并对所有唯一特征值的到的熵求和
        # 信息增益是熵的减少或者是数据无序度的减少
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # return第n个特征 表示第n个特征是最好的用于划分数据集的特征  （此处的bestFeature为序号）


# 如果数据集已经处理了所有的属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子结点 通常采用多数表决的方法决定该叶子结点的分类
def majorityCnt(classList):
    classCount = {}
    # 算法同classify0相似，该函数使用分类名称的列表，然后创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个类标签出现的频率
    # 最后用operator操作键值排序字典，并返回出现次数最多的分类名称
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树的函数代码
# 程序的输入为数据集和标签列表，其实算法本身没用到标签，但是为了便于理解，仍将它作为输入参数提供
def createTree(dataSet, labels):
    # 首先创建classList列表，其中包含了数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # 递归
    # 递归函数的第一个停止条件：所有的类标签完全相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 第二个停止条件：使用完了所有的特征，仍不能将数据集划分为仅包含唯一类别的分组，此时使用majorityCnt函数挑选出现次数最多的类别作为返回值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 字典变量myTree
    myTree = {bestFeatLabel: {}}
    # 此处与chooseBestFeatureToSplit中的部分代码相似，都是遍历将数据集中所有第i个特征值或者所有可能存在的值写入这个新list中，再调用set去重
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 首先新建列表subLabels复制标签列表，防止改变原表内容
        subLabels = labels[:]
        # 递归调用createTree
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    # 标签字符串
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# 存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,"rb+")
    return pickle.load(fr)
