from numpy.ma import array

from machinelearninginaction.Ch02 import kNN
import matplotlib
import matplotlib.pyplot as plt
group, labels = kNN.createDataSet()
print(kNN.classify0([0, 0], group, labels, 3))
print("------------------------------------------------------------------------------------------------------")
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)
print("------------------------------------------------------------------------------------------------------")
fig = plt.figure()
ax = fig.add_subplot(111)
# 查看特征值其中两个维度的 二维视图（第二列和第三列）   为了区分颜色，加以label的不同值进行区分
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()
print("------------------------------------------------------------------------------------------------------")
normalMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normalMat)
print(ranges)
print(minVals)
print("------------------------------------------------------------------------------------------------------")
kNN.datingClassTest()
print("------------------------------------------------------------------------------------------------------")
# kNN.classifyPerson()
print("------------------------------------------------------------------------------------------------------")
testVector = kNN.img2vector('digits/testDigits/0_12.txt')
print(testVector[0,0:31])
print(testVector[0,32:63])
print("------------------------------------------------------------------------------------------------------")
kNN.handwritingClassTest()