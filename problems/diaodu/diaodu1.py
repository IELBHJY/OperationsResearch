import random
import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB


class Param:
    def __init__(self):
        self.params = dict()

    def set_params(self):
        self.params['alpha'] = 0.1
        self.params['beta'] = 0.9
        self.params['Capacity'] = 100
        self.params['LU'] = 20
        self.params['HU'] = 100
        self.params['LL'] = 10
        self.params['HL'] = 50
        self.params['R_file_path'] = 'r_matrix.csv'
        return self.params


class Problem:
    def __init__(self, I, J, T, K,
                 ship_reach_time,
                 con_matrix,
                 dis_matrix,
                 R_matrix,
                 params_dict):
        self.U_matrix = None
        self.I = I
        self.J = J
        self.T = T
        self.K = K
        self.ship_reach_time = np.array(ship_reach_time, dtype=int)
        self.con_matrix = np.array(con_matrix, dtype=int)
        self.params_dict = params_dict
        self.dis_matrix = np.array(dis_matrix, dtype=float)
        self.R = np.array(R_matrix, dtype=int)
        print("=========ship reach time info==========")
        print(self.ship_reach_time)
        print("=========ship reach time info==========")

        print("=========R matrix info==========")
        print(self.R)
        print("=========R matrix info==========")

        print("=========con matrix info==========")
        print(self.con_matrix)
        print("=========con matrix info==========")

    def creat_model(self):
        model = gb.Model('MIP')
        # 初始化决策变量
        U_matrixs = np.zeros([len(self.I) + 1, len(self.J) + 1, len(self.T) + 1], dtype=object)
        for i in self.I:
            for j in self.J:
                for t in self.T:
                    U_matrixs[i][j][t] = model.addVar(0, GRB.INFINITY, vtype=GRB.INTEGER,
                                                      name="U_{}_{}_{}".format(i, j, t))
        z_array = np.zeros([len(self.T) + 1], dtype=object)
        z_lb_array = np.zeros([len(self.T) + 1], dtype=object)
        z_ub_array = np.zeros([len(self.T) + 1], dtype=object)
        Pou = np.zeros([len(self.J) + 1], dtype=int)
        h = np.zeros([len(self.I) + 1, len(self.T) + 1], dtype=object)

        # 设置Pou的值
        for j in self.J:
            if self.R[j][self.T[-1]] > self.R[j][1]:
                Pou[j] = 0
            else:
                Pou[j] = 1
        # 初始化变量z z' z''
        for t in self.T:
            z_array[t] = model.addVar(0, GRB.INFINITY, vtype=GRB.INTEGER, name="z_{}".format(t))
            z_lb_array[t] = model.addVar(0, GRB.INFINITY, vtype=GRB.INTEGER, name="z_lb_{}".format(t))
            z_ub_array[t] = model.addVar(0, GRB.INFINITY, vtype=GRB.INTEGER, name="z_ub_{}".format(t))

        # 初始化决策变量
        for i in self.I:
            for t in self.T:
                h[i][t] = model.addVar(vtype=GRB.BINARY, name="h_{}_{}".format(i, t))

        # 设置目标函数
        obj1 = 0
        for i in self.I:
            for j in self.J:
                obj1 += self.dis_matrix[i][j] * U_matrixs[i][j][self.ship_reach_time[j]]

        obj1 *= self.params_dict['alpha']

        obj2 = 0
        for t in self.T:
            obj2 += z_array[t]

        obj2 *= self.params_dict['beta']
        model.setObjective(obj1 + obj2)

        # 添加约束条件
        # (1) 船装货日之前，箱区内slot数量不减小
        for i in self.I:
            for j in self.J:
                for t in self.T[1:]:
                    if t > self.ship_reach_time[j]:
                        continue
                    model.addConstr((U_matrixs[i][j][t - 1] <= U_matrixs[i][j][t]), name="eq1")
        # (2)
        for i in self.I:
            for j in self.J:
                model.addConstr((Pou[j] * U_matrixs[i][j][self.T[-1]] <= U_matrixs[i][j][1]), name="eq2")

        # (3)
        for j in self.J:
            for t in self.T:
                model.addConstr((sum([U_matrixs[i][j][t] for i in self.I]) == self.R[j][t]), name="eq3")

        # (4)
        for i in self.I:
            for t in self.T:
                model.addConstr((sum([U_matrixs[i][j][t] for j in self.J]) <= self.params_dict['Capacity']), name="eq4")

        # (5) 是取t等于j船的到达时间吗？？
        for i in self.I:
            for t in self.T:
                model.addConstr((sum([U_matrixs[i][j][t]
                                      if t == self.ship_reach_time[j]
                                      else 0
                                      for j in self.J]) <= z_ub_array[t]), name="eq5")

        # (6) 是取t等于j船的到达时间吗？？
        for i in self.I:
            for t in self.T:
                model.addConstr((sum([U_matrixs[i][j][t]
                                      if t == self.ship_reach_time[j]
                                      else 0
                                      for j in self.J]) >= z_lb_array[t]), name="eq6")

        # (7)
        for t in self.T:
            model.addConstr((z_ub_array[t] - z_lb_array[t] <= z_array[t]), name="eq7")

        # (8) 是取t等于j船的到达时间吗？？
        # for i in self.I:
        #     for t in self.T:
        #         model.addConstr((sum([U_matrixs[i][j][t]
        #                               if t == self.ship_reach_time[j]
        #                               else 0
        #                               for j in self.J]) <= params['LU'] + (params['HU'] - params['LU']) * h[i][t]),
        #                         name="eq8")
        #         model.addConstr((sum([U_matrixs[i][j][t]
        #                               if t == self.ship_reach_time[j]
        #                               else 0
        #                               for j in self.J]) >= params['HL'] + (params['LL'] - params['HL']) * (1 - h[i][t]))
        #                         , name="eq8")
        # (9)
        # for i in self.I:
        #     for t in self.T:
        #         model.addConstr((sum([h[_i][t]
        #                               if self.con_matrix[i][_i] == 1
        #                               else 0
        #                               for _i in self.I]) <= 1), name="eq9")

        model.write("model.lp")
        model.modelSense = GRB.MINIMIZE
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print("object = ", model.objVal)
            for i in self.I:
                for j in self.J:
                    for t in self.T:
                        if U_matrixs[i][j][t].x > 0:
                            print("U[{}][{}][{}] = {}".format(i, j, t, U_matrixs[i][j][t].x))
            for i in self.I:
                for t in self.T:
                    if h[i][t].x > 0:
                        print("h[{}][{}] = {}".format(i, t, h[i][t].x))
        else:
            print("no solution")


if __name__ == '__main__':
    # 设置随机种子
    random.seed(2020)
    np.random.seed(2020)
    # 15 block 40 slots in every block
    I = np.array(np.arange(1, 16, 1))
    J = np.array(np.arange(1, 7, 1))
    T = np.array([1, 2, 3, 4, 5, 6, 7])
    K = 40

    # 设置参数
    params = Param().set_params()

    # R矩阵
    # r_data = pd.read_csv(params['R_file_path'], header=None)
    # R_matrix = r_data.values

    # 设置船到港时间
    ship_reach_time = np.random.choice(a=T[1:-1], size=len(J) + 1, replace=True)
    ship_num = len(J)
    day = len(T)
    ship_reach_time = np.zeros([ship_num + 1], dtype=int)
    one_ship = np.zeros([ship_num + 1, day + 1], dtype=int)
    for ship_id in range(1, ship_num + 1):
        temp = [np.random.randint(10, 80) for _ in range(7)]
        temp = sorted(temp)
        length = [value for value in range(1, day + 1, 1)].__len__()
        insert_day = np.random.choice([value for value in range(1, day + 1, 1)], 1, replace=True)[0]
        ship_reach_time[ship_id] = insert_day
        res = list(temp)
        res.insert(insert_day - 1, temp[-1])
        res = res[0:-1]
        direction = np.random.randint(0, 1)
        if direction == 0:  # 前向开始
            one_ship[ship_id, 1:] = res
    #
    # # 邻接矩阵
    con_matrix = np.ones([len(I) + 1, len(I) + 1])
    #
    # # 距离矩阵
    dis_matrix = np.random.randint(3, 10, [len(I) + 1, len(J) + 1], dtype=int)
    #
    problem = Problem(I, J, T, K,
                      ship_reach_time,
                      con_matrix,
                      dis_matrix,
                      one_ship,
                      params)
    problem.creat_model()