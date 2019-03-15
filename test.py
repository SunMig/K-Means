from kMeans import *
import matplotlib.pyplot as plt
dataMat=mat(loadDataSet('testSet.txt'))
centList,mynewAssment=biKmeans(dataMat,4)
plt.scatter(array(centList[:,0])[:,0],array(centList[:,1])[:,0])
plt.scatter(array(dataMat[:,0])[:,0],array(dataMat[:,1])[:,0])
plt.show()

# myCentriods,clustAssing=kMeans(dataMat,4)
# print(myCentriods)

#代码测试
# m=shape(dataMat)[0]
# clusterAssment = mat(zeros((m, 2)))
# centroid0 = mean(dataMat, axis=0).tolist()[0]  # 求数据的均值
# centList = [centroid0]
# for j in range(m):
#     clusterAssment[j, 1] = distEclud(mat(centroid0), dataMat[j, :]) ** 2
# print(clusterAssment)
# a=dataMat[nonzero(clusterAssment[:,0].A!=0)[0],:]
# print(a)
#
# for i in range(1):
#     print(i)