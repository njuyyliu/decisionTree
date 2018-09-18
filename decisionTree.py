# coding=utf-8
import operator
from math import log
import time
import random


def createDataSet():    # 创造示例数据
    dataSet = []
    labels = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s0']  #两个特征

    fr = open('D:\\Second_2\\KNN\\traindata1.txt')
    for line in fr.readlines():
        line.strip()
        temp = list(map(float, line.split(' ')))
        dataSet.append(temp)

    return dataSet, labels


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 因为数据集的最后一项是标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 因为我们递归构建决策树是根据属性的消耗进行计算的，所以可能会存在最后属性用完了，但是分类
# 还是没有算完，这时候就会采用多数表决的方式计算节点分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 所有特征已经用完
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    #del (labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 为了不改变原始列表的内容复制了一下
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat, value), subLabels)
    return myTree


def classify(inputTree,featNames,testVec):

    firstStr = inputTree.keys()[0]  #当前树的根节点的特征名称
    # print "firstStr: ", firstStr
    secondDict = inputTree[firstStr]  #根节点的所有子节点
    # print secondDict
    featIndex = featNames.index(firstStr)  #找到根节点特征对应的下标
    # print "featIndex: ", featIndex
    key = testVec[featIndex]  #找出待测数据的特征值
    # print "key: ", key
    if secondDict.has_key(key):
        valueOfFeat = secondDict[key]  #拿这个特征值在根节点的子节点中查找，看它是不是叶节点
    else:
        return random.randint(0, len(featNames))
    # print "it's not feat"
    if isinstance(valueOfFeat, dict):  #如果不是叶节点
        classLabel = classify(valueOfFeat, featNames, testVec)  #递归地进入下一层节点
    else: classLabel = valueOfFeat  #如果是叶节点：确定待测数据的分类
    return classLabel


def testClassify(inputTree):
    dataSet = []
    labels = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's0']  # 两个特征
    # realResult = ['3', '3', '4', '6', '1', '6', '5', '6', '5', '6', '5']
    realResult = []

    fr = open('D:\\Second_2\\KNN\\testdata_tree')
    for line in fr.readlines():
        line.strip()
        temp = list(map(float, line.split(' ')))
        # print temp
        realResult.append(temp[len(temp)-1])
        del temp[(len(temp) - 1)]
        # print temp
        dataSet.append(classify(inputTree, labels, temp))
    print dataSet
    print realResult
    count = 0
    for index in range(len(dataSet)):
        if dataSet[index] == realResult[index]:
            count += 1
    print float(count)/float(len(dataSet))


def main():
    data, label = createDataSet()

    t1 = time.clock()
    myTree = createTree(data, label)
    t2 = time.clock()
    print myTree
    # print 'execute for ', t2 - t1
    # print classify(myTree, label, [142, 221, 23, 72, 7, 5, 10, 0, 0, 12])
    testClassify(myTree)


if __name__ == '__main__':
    main()