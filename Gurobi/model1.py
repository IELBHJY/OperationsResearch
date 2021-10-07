import gurobipy as gp
from gurobipy import GRB
import pandas as pd

c = pd.read_table('/Users/libaihe/Desktop/z2_100ctt.txt', delim_whitespace=True, header=None)
c.columns=['index','c1','c2']
c.index = range(c.shape[0])
c['c1'] = c['c1'].astype(float)
c['c2'] = c['c2'].astype(float)
c['c1_sq'] = c['c1'].apply(lambda x: x**2)
var_cost = c['c2']

model = gp.Model('MIP')
vars = model.addVars(len(var_cost), obj=var_cost, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='x')
model.modelSense = GRB.MINIMIZE

A = pd.read_table('/Users/libaihe/Desktop/z2_100att.txt', delim_whitespace=True, header=None)
b = pd.read_table('/Users/libaihe/Desktop/z2_100btt.txt', delim_whitespace=True, header=None)
b.columns = ['index', 'b']
A.columns = ['index', 'a1', 'a2']
A_matrix = A[['a1','a2']].T
A_matrix.index=range(A_matrix.shape[0])
for index in A_matrix.index:
    model.addConstr((sum(A_matrix.loc[index, col] * vars[col] for col in range(len(A_matrix.columns))) <= b['b'].iloc[index]),"con")

model.addConstr(sum(c.loc[index, 'c1_sq'] * vars[index] for index in range(len(c))) <= 4037, "con")

model.addConstr(
    (sum(c.loc[index, 'c1'] * vars[index] for index in range(len(c))) - 3235) * (sum(c.loc[index, 'c1'] * vars[index] for index in range(len(c))) - 3235) +
    (sum(c.loc[index, 'c2'] * vars[index] for index in range(len(c))) - 4037) * (sum(c.loc[index, 'c2'] * vars[index] for index in range(len(c))) - 4037) <= 1913 * 1913, "con")


model.write('models/model1.lp')
model.optimize()
print("optimial objection:",model.objval)
print("x:",model.getAttr('x',model.getVars()))
print(vars[0].x)
result = pd.DataFrame(columns=['index','value'])
result['index'] = range(1,101)
result['value'] = [vars[index].x for index in range(100)]
result.to_csv('result.csv',index=False)