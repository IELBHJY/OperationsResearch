import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


model = gp.Model()
data = pd.DataFrame()

n = model.addVars(data.shape[0],ub=100,vtype=GRB.CONTINUOUS,name='n')
k1=model.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name='k1')
k2=model.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name='k2')
b=model.addVar(lb=0,ub=20,vtype=GRB.CONTINUOUS,name='b')


def get_y_predict(m,w):
    return k1 * m + k2 * m * w + b

def F_function(x):
    return 1-(1+(np.e)**(1000 * (x + 0.05)))**(-1)




