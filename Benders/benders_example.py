import gurobipy as gp
from gurobipy import GRB
import random
from time import *
import numpy as np
from ortools.linear_solver import pywraplp

random.seed(2020)


class Benders_Example(object):
    def __init__(self, i_num, j_num):
        self.j_num = j_num  # 产品数量
        self.i_num = i_num  # 地点数量
        self.f = [random.randint(200, 500) for _ in range(self.i_num)]
        self.d = [random.randint(20, 100) for _ in range(self.j_num)]
        self.cost = []
        for _ in range(self.i_num):
            self.cost.append([random.randint(1, 20) for i in range(self.j_num)])

    def build_model(self):
        model = gp.Model("Benders")
        x_ij = model.addVars(self.i_num, self.j_num, vtype=GRB.CONTINUOUS, obj=self.cost, lb=0, ub=GRB.INFINITY,
                             name="x")
        y_i = model.addVars(self.i_num, vtype=GRB.BINARY, obj=self.f, name="y")
        model.modelSense = GRB.MINIMIZE

        # add constr
        model.addConstrs((x_ij.sum(i, "*") <= y_i[i] * sum(self.d) for i in range(self.i_num)), name='c1')

        model.addConstrs((x_ij.sum("*", j) >= self.d[j] for j in range(self.j_num)), name='c2')

        model.write("models/benders.lp")
        print("solving...")
        begin_time = time()
        model.optimize()
        end_time = time()
        print("Run time of Model:", end_time - begin_time)
        print("model obj = ", model.objVal)
        for i in range(self.i_num):
            print("y(%d)=%lf" % (i, y_i[i].x))
            for j in range(self.j_num):
                print(x_ij[i, j].x)

    def creat_master_model(self, input_y_i):
        # 创建主问题
        model = gp.Model("Benders")
        x_ij = model.addVars(self.i_num, self.j_num, vtype=GRB.CONTINUOUS, obj=self.cost, lb=0, ub=GRB.INFINITY,
                             name="x")
        y_i = model.addVars(self.i_num, vtype=GRB.BINARY, obj=self.f, name="y")
        model.modelSense = GRB.MINIMIZE
        # add constr
        model.addConstrs((x_ij.sum(i, "*") <= y_i[i] * sum(self.d) for i in range(self.i_num)), name='c1')

        model.addConstrs((x_ij.sum("*", j) >= self.d[j] for j in range(self.j_num)), name='c2')
        model.addConstrs((y_i[i] == input_y_i[i] for i in range(self.i_num)), name='y')
        model.write("models/MP_0.lp")
        model.optimize()
        if model.Status == GRB.Status.OPTIMAL:
            print("best obj:", model.objVal)

    def creat_dual_model(self, input_y_i, iteration):
        model = gp.Model("DualModel")
        v_i = model.addVars(self.i_num, obj=sum(self.d) * input_y_i, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=0,
                            name='v')
        w_j = model.addVars(self.j_num, obj=self.d, vtype=GRB.CONTINUOUS, name="w")
        for index in range(len(input_y_i)):
            model.addVar(lb=input_y_i[index], ub=input_y_i[index], obj=self.f[index], vtype=GRB.CONTINUOUS,
                         name='y_' + str(index))

        # add cons
        model.addConstrs((v_i[i] + w_j[j] <= self.cost[i][j] for i in range(self.i_num) for j in range(self.j_num)),
                         name='c')

        model.modelSense = GRB.MAXIMIZE
        model.Params.InfUnbdInfo = 1
        model.write("models/BD_" + str(iteration) + ".lp")
        return model

    def creat_BM_model(self):
        BM_Model = gp.Model("BM")
        Z = BM_Model.addVar(obj=1, vtype=GRB.CONTINUOUS, name='Z')
        y = BM_Model.addVars(self.i_num, vtype=GRB.BINARY, name='y')
        BM_Model.modelSense = GRB.MINIMIZE
        BM_Model.write('models/BM.lp')
        return BM_Model

    def add_constr_BM(self, model, iteration, rays, unbd_flag):
        Z = model.getVarByName("Z")
        vars = model.getVars()
        sum_cost = sum(self.d)
        sum_cons = 0
        if unbd_flag:
            for i in range(self.i_num):
                sum_cons += rays[i] * sum_cost * vars[i + 1]
            for j in range(self.j_num):
                sum_cons += rays[self.i_num + j] * self.d[j]
            model.addConstr((sum_cons <= 0), name='c' + str(iteration))
        else:
            for i in range(self.i_num):
                sum_cons += self.f[i] * vars[i + 1] + rays[i] * sum_cost * vars[i + 1]
            for j in range(self.j_num):
                sum_cons += rays[self.i_num + j] * self.d[j]
            model.addConstr((sum_cons <= Z), name='c' + str(iteration))
        model.write("models/BM_" + str(iteration) + ".lp")
        return model

    def benders(self):
        BM_last_obj = 0
        iteration = 1
        input_y_i = np.zeros([self.i_num])
        BM_Model = self.creat_BM_model()
        DP_Model = self.creat_dual_model(input_y_i, iteration)
        DP_Model.optimize()
        if DP_Model.Status == GRB.Status.UNBOUNDED:
            rays = DP_Model.unbdRay
            BM_Model = self.add_constr_BM(BM_Model, iteration, rays, 1)
        elif DP_Model.Status == GRB.Status.OPTIMAL:
            rays = DP_Model.getVars()
            rays = [item.x for item in rays]
            BM_Model = self.add_constr_BM(BM_Model, iteration, rays, 0)
        BM_Model.optimize()
        if BM_Model.Status == GRB.Status.OPTIMAL:
            BM_Model_obj = BM_Model.objVal
            BM_last_obj = BM_Model_obj
            vars = BM_Model.getVars()
            for index in range(len(vars)):
                if index == 0:
                    continue
                input_y_i[index - 1] = vars[index].x
        else:
            print("solve false")
            return
        iteration += 1
        DP_Model = self.creat_dual_model(input_y_i, iteration)
        DP_Model.optimize()
        DP_Model_obj = DP_Model.objVal
        while DP_Model_obj > BM_last_obj:
            if DP_Model.Status == GRB.Status.UNBOUNDED:
                rays = DP_Model.unbdRay
                BM_Model = self.add_constr_BM(BM_Model, iteration, rays, 1)
            elif DP_Model.Status == GRB.Status.OPTIMAL:
                rays = DP_Model.getVars()
                rays = [item.x for item in rays]
                BM_Model = self.add_constr_BM(BM_Model, iteration, rays, 0)
            BM_Model.optimize()
            if BM_Model.Status == GRB.Status.OPTIMAL:
                BM_Model_obj = BM_Model.objVal
                BM_last_obj = BM_Model_obj
                vars = BM_Model.getVars()
                for index in range(len(vars)):
                    if index == 0:
                        continue
                    input_y_i[index - 1] = vars[index].x
            else:
                print("solve false")
                print(iteration)
                return
            iteration += 1
            DP_Model = self.creat_dual_model(input_y_i, iteration)
            DP_Model.optimize()
            DP_Model_obj = DP_Model.objVal

        print("solve success")
        self.creat_master_model(input_y_i)


if __name__ == "__main__":
    # benders = Benders_Example(3, 5)
    # benders.build_model()
    # benders.benders()
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = solver.IntVar(0.0, solver.infinity(), 'x')
    y = solver.IntVar(0.0, solver.infinity(), 'y')
    z = solver.IntVar(0.0, solver.infinity(), 'z')
    constraint0 = solver.Constraint(-solver.infinity(), 50)
    constraint0.SetCoefficient(x, 2)
    constraint0.SetCoefficient(y, 7)
    constraint0.SetCoefficient(z, 3)
    constraint1 = solver.Constraint(-solver.infinity(), 45)
    constraint1.SetCoefficient(x, 3)
    constraint1.SetCoefficient(y, -5)
    constraint1.SetCoefficient(z, 7)

    objective = solver.Objective()
    objective.SetCoefficient(x, 2)
    objective.SetCoefficient(y, 2)
    objective.SetCoefficient(z, 3)
    objective.SetMaximization()

    solver.Solve()

    # Print the objective value of the solution.
    print('Maximum objective function value = %d' % solver.Objective().Value())
    print()