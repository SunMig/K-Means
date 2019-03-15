from numpy import *


# 导入数据函数
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # fltLine=list(map(float,curLine))
        fltLine = [float(i) for i in curLine]
        dataMat.append(fltLine)
    return dataMat


# 欧拉距离计算函数
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机选取k个质心的函数
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # K假设有K个中心，K*2
    for j in range(n):
        minJ = min(dataSet[:, j])  # 某列的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 大小范围
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    # print(centroids)
    return centroids


# kmeans算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 行数
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)  # 聚类中心坐标
    clusterChanged = True
    while clusterChanged:  # 迭代直到任一点的簇分配结果不在改变停止
        clusterChanged = False
        for i in range(m):
            minDist = inf;
            minIndex = -1
            # 这个循环是把每一对数据都拿来和选取的中心去比较，距离哪个最近就归为哪一类
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 求各点与聚类初始中心的欧拉距离
                if distJI < minDist:  # 如果满足条件，交换距离与索引值
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2;
            # print(clusterAssment)
        # print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 列方向取均值
    return centroids, clusterAssment


# 二分Kmeans
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 用于存储簇分配结果和误差
    # 当传入参数是矩阵时，tolist返回的是[[***]]元素是列表的列表，因此后面加[0]
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 求数据的均值
    centList = [centroid0]
    #循环，神坑。。。。
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    #二分开始循环
    while (len(centList) < k):  # 聚类的中心的数目小于k
        print(len(centList))
        lowestSSE = inf  # 定义一个误差项
        for i in range(len(centList)):
            # 矩阵名.A是把矩阵转为数组，nonzero()函数是得到数组array中非零元素的位置索引，返回的是元素是数组的元组，因此加[0]
            # 从dataSet中提取类别号为i的数据构成一个新数据集
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 得到分类结果是0,1的簇
            sseSplit = sum(splitClustAss[:, 1])  # 计算分类误差，距离平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print('sseSplit '+str(sseSplit))
            print('sseNotSplit ' + str(sseNotSplit))
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()  # 簇对应的结果和相应的误差
                lowestSSE = sseSplit + sseNotSplit


        # 更新簇的分配结果,此时通过误差项已经知道按照哪一类（1或者是0）再次划分数据集最好，因此需要更新簇分类的结果
        # 把分类最好的那一部分打上新的簇分类符号，1或者2或者3 等等
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: '+str(bestCentToSplit))
        print('the len of bestClustAss is: '+str(len(bestClustAss)))
        #把新的质心加入
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        print(centList)
        #更新簇分配结果，即按照bestCentToSpilt划分最好，就把他对应的数据用新的bestClustAss替换掉
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
        print(clusterAssment)
    return mat(centList), clusterAssment