import gurobipy as gp
from gurobipy import GRB
import math

var_cost=[12,9]
A_matrix = [[1,0],
            [0,1],
            [1,1],
            [4,2]]
b=[1000,1500,1750,4800]

model = gp.Model('LP')
vars = model.addVars(len(var_cost),obj= var_cost,lb=0,name='x')
model.modelSense = GRB.MAXIMIZE

#different method add constr
for row in range(len(A_matrix)):
    model.addConstr((A_matrix[row][0] * vars[0] + A_matrix[row][1] * vars[1] <= b[row]),'constr')

for row in range(len(A_matrix)):
    model.addConstr((sum(A_matrix[row][col]*vars[col] for col in range(len(var_cost))) <= b[row]),"con")

model.addConstrs(((sum(A_matrix[row][col]*vars[col] for col in range(len(var_cost))) <= b[row]) for row in range(len(A_matrix))),"cons")

model.write('lp.lp')
model.optimize()
print("optimial objection:",model.objval)
print("x",model.getAttr('x',model.getVars()))
print(vars[0].x)
print("dual value:",model.getAttr('PI',model.getConstrs()))