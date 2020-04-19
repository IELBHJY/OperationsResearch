import gurobipy as gp
from gurobipy import GRB
import numpy as np

class Benders_Example(object):
    def __init__(self,i_num,j_num):
        self.j_num = j_num
        self.i_num = i_num
        self.f = [400, 250, 300]
        self.d = [75, 90, 81, 26, 57]
        self.cost = [[4, 7, 3, 12, 15],
                     [13, 11, 17, 9, 19],
                     [8, 12, 10, 7, 5]]

    def build_model(self):
        model = gp.Model("Benders")
        x_ij = model.addVars(self.i_num,self.j_num,vtype=GRB.CONTINUOUS,obj=self.cost,lb=0,ub=GRB.INFINITY,name="x")
        y_i = model.addVars(self.i_num,vtype=GRB.BINARY,obj=self.f,name="y")
        model.modelSense=GRB.MINIMIZE

        #add constr
        model.addConstrs((x_ij.sum(i,"*") <= y_i[i] * sum(self.d) for i in range(self.i_num)),name='c1')

        model.addConstrs((x_ij.sum("*",j) == self.d[j] for j in range(self.j_num)),name='c2')

        model.write("benders.lp")
        model.optimize()
        print("model obj = ",model.objVal)
        for i in range(self.i_num):
            print("y(%d)=%lf" % (i,y_i[i].x))
            for j in range(self.j_num):
                print(x_ij[i,j].x)




if __name__ == "__main__":
    benders = Benders_Example(3,5)
    benders.build_model()