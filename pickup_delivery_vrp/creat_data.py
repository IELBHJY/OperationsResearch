import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(1)
n = 50
points = [(random.randint(0, 500), random.randint(0, 500)) for i in range(n)]
center_point = [0, 0]
# Dictionary of Euclidean distance between each pair of points
dist = {(i, j):
   math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
    for i in range(n) for j in range(i)}
import matplotlib.pyplot as plt
#plt.scatter([point[0] for point in points],[point[1] for point in points])
center_point[0] = np.mean([point[0] for point in points])
center_point[1] = np.mean([point[1] for point in points])
#plt.scatter(center_point[0], center_point[1])
#plt.show()
data = pd.DataFrame(columns=['id','x','y'])
data = data.append({'id':0,'x':center_point[0],'y':center_point[1]},ignore_index=True)
for index in range(n):
    data = data.append({'id':index+1,'x':points[index][0],'y':points[index][1]},ignore_index=True)
data.to_csv("vrp_"+str(n)+".csv")

