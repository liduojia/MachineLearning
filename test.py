# _*_ coding:utf-8 _*_
from machinelearninginaction.Ch03 import trees,treePlotter

myDat, labels = trees.createDataSet()
print(myDat)
print(labels)
print("------------------------------------------------------------------------------------------------------------")
print(trees.calcShannonEnt(myDat))
print("------------------------------------------------------------------------------------------------------------")
myDat[0][-1] = 'maybe'
print(myDat)
print(trees.calcShannonEnt(myDat))
myDat[1][-1] = 'no'
print(myDat)
print(trees.calcShannonEnt(myDat))
print("------------------------------------------------------------------------------------------------------------")
print(trees.splitDataSet(myDat,0,1))
print("------------------------------------------------------------------------------------------------------------")
print(trees.chooseBestFeatureToSplit(myDat))
print(myDat)
print("------------------------------------------------------------------------------------------------------------")
print(trees.createTree(myDat,labels))
print("------------------------------------------------------------------------------------------------------------")
myTree = treePlotter.retrieveTree(1)
print(myTree)
print(treePlotter.getNumLeafs(myTree))
print(treePlotter.getTreeDepth(myTree))
print("------------------------------------------------------------------------------------------------------------")
myTree = treePlotter.retrieveTree(1)
# treePlotter.createPlot(myTree)
print("------------------------------------------------------------------------------------------------------------")
myDat,labels = trees.createDataSet()
myTree = treePlotter.retrieveTree(0)
# treePlotter.createPlot(myTree)
# 利用classify做一个测试 可以与前面createPlot的树进行比较
print(trees.classify(myTree,labels,[1,0]))
print(trees.classify(myTree,labels,[1,1]))
print("------------------------------------------------------------------------------------------------------------")
trees.storeTree(myTree,'classifierStorage.txt')
trees.grabTree('classifierStorage.txt')
print("------------------------------------------------------------------------------------------------------------")
# 隐形眼镜数据集
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)



