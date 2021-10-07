import random
import numpy as np

len =200

m = [random.randint(0,200) for _ in range(200)]
w = [round(20 * random.random(),2) for _ in range(200)]
y=[2.5 * m[index] + 0.25 * m[index] * w[index] + 100  for index in range(200)]

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
p_left_list = []
p_right_list=[]
p_left_list.append(0)
p_left_list.append(0)
p_left_list.append(0)
p_right_list.append(100)
p_right_list.append(10)
p_right_list.append(500)
for index in range(len):
    p_left_list.append(-np.inf)
    p_right_list.append(50)

#决策变量区直范围
bounds = Bounds(p_left_list,p_right_list)

#约束条件矩阵
matrixs=[]
matrix_l=[]
matrix_r=[]

for index in range(len):
    temp=[]
    temp.append(m[index])
    temp.append(m[index] * w[index])
    temp.append(1)
    for columns in range(len):
        if temp.__len__() == index + 3:
            temp.append(1)
        else:
            temp.append(0)
    matrixs.append(temp)

for index in range(len):
    matrix_r.append(50 + y[index])
    matrix_l.append(-np.inf)

lc = LinearConstraint(matrixs,matrix_l,matrix_r)

def objection(x):
    return sum((1+(1000 * np.e)**(x[0:]))**(-1) - 1)

x0 = np.ones([203])
res = minimize(objection, x0, method='trust-constr',
               constraints=[lc],
               options={'verbose': 1}, bounds=bounds)
print(res.x)