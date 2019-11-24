'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from click._compat import raw_input
from numpy import *
import operator
from os import listdir
# k-近邻算法的一般流程
"""
（1）收集数据：可使用任何算法
（2）准备数据：距离计算所需要的数值，最好是结构化的数据格式
（3）分析数据：可使用任何方法
（4）训练算法：此步骤不适用于k-近邻算法
（5）测试算法：计算错误率
（6）使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个类别，最后应用对计算出的分类执行后续的处理
"""

# k-近邻算法
"""
    对未知类别属性的数据集中的每个点依次执行如下操作：
    （1）计算 已知类别 数据集中的点与当前点之间的距离
    （2）按照距离递增次序排序
    （3）选取与当前距离最小的k个点
    （4）确定前k个点所在的类别的出现频率
    （5）返回前k个点出现频率最高的类别作为当前点的预测分类
"""


def classify0(inX, dataSet, labels, k):
    # 1. 距离计算
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort()
    # print 'distances.argsort()=', sortedDistIndicies
    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    maxClassCount = max(classCount, key=classCount.get)
    return maxClassCount


# 创建测试数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


# -------------------------------------------demo 约会网站---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# 将文本记录转换成numpy的解析程序
# 注：此处的 3 是对应着txt文件中的features 具体使用时根据需要来改。
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #首先截取掉所有的回车字符
        line = line.strip()
        #再用tab字符\t将上一步得到的整行数据分割成一个元素列表
        listFromLine = line.split('\t')
        #我们选取前三个元素，将其存储到特征矩阵 returnMat中
        returnMat[index,:] = listFromLine[0:3]
        #python 使用 索引值-1表示列表中的最后一列元素，事实上根据不同的数据集可以变化此处的-1值
        #利用这种负索引，我们可以很方面的将列表的最后一列存储到向量classLabelVector中
        #此处必须明确告知解释器 存储的数据类型为int型，否则其会按照string类型处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        #注 ： 此处仍未转换成矩阵 应用的时候可能需要 dataset=mat(dataset)
    return returnMat,classLabelVector


# 均值归一化（避免某个特征值过大而对其他数据造成影响）    转换后的数值在0-1区间
# 数学表达式如下 ： newvalue = (oldvalue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    # 获得函数值的取值范围
    m = dataSet.shape[0]
    # 此处的 m 是1000
    # 该数据集的特征值举证是1000 * 3 的
    # 此处使用tile函数将变量内容复制成输入矩阵相同大小的矩阵
    # 注意这是具体特征值相除，对于某些软件 / 可能代表矩阵除法，在numpy库中，矩阵除法需要使用函数 linalg.solve(matA,matB)
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

# 测试代码，测试分类器的准确性 测试集是随机选取的
# 此处数据集是通过函数内部的 file2matrix函数读取的 不具有很好的扩展性 也违反了开闭原则
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    # normMat 是 均值归一化以后的特征值矩阵
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 此处 m = 1000
    m = normMat.shape[0]
    # 此处为 500
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 此处直接利用现成的数据集，分一些数据出去为测试集  classify0的参数分别为(intX , groups, labels, k)  很多时候k需要手动选取
        # normMat[i,:]代表着 某行的所有数据      normMat[numTestVecs:m,:] 代表着 从500-1000行的所有数据  datingLabels也代表着从500-1000行所有数据的标签
        # 所以500-1000行数据是作为检验的
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percenTats = float(raw_input("你喜欢每周打多少时间游戏"))
    ffMiles = float(raw_input("你每年花多久飞行"))
    icecream = float(raw_input("你每年吃多少冰淇淋"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percenTats, icecream])
    # 注意此处输入的新数据也需要进行 均值归一化
    newinArr = (inArr-minVals) / ranges
    # 再调用classify0时需要传入新的经过 autoNorm 转换后的数值
    classifierResult = classify0(newinArr, normMat, datingLabels, 3)
    print("you will probably like this personL",resultList[classifierResult-1])


# -------------------------------------------demo 手写数字集识别---------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# 将32*32的二进制图像矩阵转换为1*1024的向量（已有的数据集就已经是32*32的二进制图像矩阵了）
# 事实上在运用的时候还可写相关的图像处理函数 将 图片 转化为 32*32 的二进制矩阵
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    # 循环读出文件的前32行，并将每行的头32个字符值存储在numpy数组中  最后返回数组
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    # 结果是 1*1024的矩阵
    return returnVect

# 数据集的数据均在0-1之间，所以不再需要均值归一化处理
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')           #load the training set
    # 获取目录中有多少文件 为下文的遍历做准备
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # split 划分文件名 9_11 代表分类是9  它是数字9的第11个实例
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 获取 label  （0到9）
        hwLabels.append(classNumStr)
        # 字符串拼接 将其拼接程 img2vector能读懂的路径格式
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    # 对 test也执行相同的操作
    testFileList = listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
