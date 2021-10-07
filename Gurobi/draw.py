import matplotlib.pyplot as plt

x = [[1, 3], [2, 5]] # 要连接的两个点的坐标
y = [[4, 7], [6, 3]]

for i in range(len(x)):
    plt.plot(x[i], y[i], color='r')
    #plt.scatter(x[i], y[i], color='b')
    plt.show()