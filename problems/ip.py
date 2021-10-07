import gurobipy as gp
from gurobipy import GRB

c = [40, 20, 40, 30]

a = [[2, 1, 3, 2],
     [4, 2, 1, 2],
     [6, 2, 1, 2]]

b = [400, 600, 1000]
ub = [100, 200, 50, 100]

model = gp.Model('IP')
x = model.addVars(len(c),obj=c,lb=0,vtype=GRB.CONTINUOUS,name='x')
model.addConstrs((sum(x[i] * a[index][i] for i in range(len(a[index]))) <= b[index]
                  for index in range(len(a))), name='c')
model.addConstrs((x[i] <= ub[i] for i in range(len(ub))), name='ub')
model.write('IP.lp')
model.modelSense = GRB.MAXIMIZE
model.optimize()

if model.Status == GRB.OPTIMAL:
   print("object = ", model.objVal)
   for i in range(len(c)):
      print('x[{}]={}'.format(i, x[i].x))
   print('模型对偶变量值：', model.getAttr('Pi'))
   cons = model.getConstrs()
   #print("min{},max{}".format(cons[0].SARHSLow,cons[0].SARHSUp))
   #print("min{},max{}".format(cons[1].SARHSLow, cons[1].SARHSUp))
   #print("min{},max{}".format(cons[2].SARHSLow, cons[2].SARHSUp))

else:
   print("no solution")