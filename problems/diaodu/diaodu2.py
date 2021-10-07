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
        self.params['U_file_path'] = 'U_matrix_2.csv'
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
        for i in self.I:
            for t in self.T:
                model.addConstr((sum([U_matrixs[i][j][t]
                                      if t == self.ship_reach_time[j]
                                      else 0
                                      for j in self.J]) <= params['LU'] + (params['HU'] - params['LU']) * h[i][t]),
                                name="eq8")
                model.addConstr((sum([U_matrixs[i][j][t]
                                      if t == self.ship_reach_time[j]
                                      else 0
                                      for j in self.J]) >= params['HL'] + (params['LL'] - params['HL']) * (1 - h[i][t]))
                                , name="eq8")
        # (9)
        for i in self.I:
            for t in self.T:
                model.addConstr((sum([h[_i][t]
                                      if self.con_matrix[i][_i] == 1
                                      else 0
                                      for _i in self.I]) <= 1), name="eq9")

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

    def creat_model2(self, i):
        # 模型是针对某一个 block而言的
        model = gb.Model("MIP2")
        X = np.zeros([self.K + 2, len(self.J) + 1, len(self.T) + 1], dtype=object)
        Y = np.zeros([self.K + 2, len(self.J) + 1, len(self.T) + 1], dtype=object)
        Pou = np.zeros([len(self.J) + 1], dtype=int)
        Z = model.addVar(0, obj=1, vtype=GRB.INTEGER, name="Z_{}".format(i))
        for k in range(1, self.K + 1, 1):
            for j in self.J:
                for t in self.T:
                    X[k][j][t] = model.addVar(0, 1, vtype=GRB.BINARY, name="X_{}_{}_{}_{}".format(i, k, j, t))
                    Y[k][j][t] = model.addVar(0, 1, vtype=GRB.BINARY, name="Y_{}_{}_{}_{}".format(i, k, j, t))
        # 添加约束条件
        # 约束1：slot摆放的item连续，除非装船日
        for k in range(1, self.K + 1, 1):
            for j in self.J:
                for t in self.T[1:]:
                    if t == self.ship_reach_time[j] + 1:
                        continue
                    model.addConstr((X[k][j][t-1] <= X[k][j][t]), name="eq1")
        # 约束2：
        # 设置Pou的值
        for j in self.J:
            if self.R[j][self.T[-1]] > self.R[j][1]:
                Pou[j] = 0
            else:
                Pou[j] = 1
        for k in range(1, self.K + 1, 1):
            for j in self.J:
                model.addConstr((Pou[j] * X[k][j][self.T[-1]] <= X[k][j][1]), name="eq2")
        # 约束3：
        for k in range(1, self.K + 1, 1):
            for t in self.T:
                model.addConstr((sum([X[k][j][t] for j in self.J]) <= 1), name="eq3")
        # 约束4：
        for j in self.J:
            for t in self.T:
                model.addConstr((sum([Y[k][j][t] for k in range(1, self.K + 1, 1)]) <= 1), name="eq4")
        # 约束5：
        for k in range(1, self.K + 1, 1):
            for j in self.J:
                for t in self.T:
                    model.addConstr((X[k][j][t] - X[k+1][j][t] <= Y[k][j][t]), name="eq5")
        # 约束6：
        for k in range(1, self.K + 1, 1):
            for j in self.J:
                for t in self.T:
                    model.addConstr((k * X[k][j][t] <= Z), name="eq6")
        # 约束7：
        # 这个约束有问题，这个约束是所有的block的关系，但是前面的模型是指某一个 block i
        for j in self.J:
            for t in self.T:
                model.addConstr((self.U_matrix[i][j][t] ==
                                 sum([X[k][j][t] for k in range(1, self.K + 1, 1)])), name="eq7")

        model.write("model2.lp")
        model.modelSense = GRB.MINIMIZE
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print("object = ", model.objVal)
            for k in range(1, self.K + 1, 1):
                for j in self.J:
                    for t in self.T:
                        if X[k][j][t].x > 0:
                            print("X[{}][{}][{}] = {}".format(k, j, t, X[k][j][t].x))
            for k in range(1, self.K + 1, 1):
                for j in self.J:
                    for t in self.T:
                        print("Y[{}][{}][{}] = {}".format(k, j, t, Y[k][j][t].x))
        else:
            print("no solution")

    # 1 是left 2 是 right
    def place_direction(self, begin_pos, end_pos, yard):
        lefts = np.zeros([self.U_matrix.shape[1] + 1], dtype=int)
        rights = np.zeros([self.U_matrix.shape[1] + 1], dtype=int)
        for t in range(1, self.U_matrix.shape[1] + 1):
            left_sum = 0
            for i in range(begin_pos, end_pos, 1):
                if yard[t, i] == 0:
                    left_sum += 1
                    lefts[t] = left_sum
                else:
                    lefts[t] = left_sum
                    break

        for t in range(1, self.U_matrix.shape[1] + 1):
            right_sum = 0
            for i in range(end_pos - 1, begin_pos - 1, -1):
                if yard[t, i] == 0:
                    right_sum += 1
                    rights[t] = right_sum
                else:
                    rights[t] = right_sum
                    break
        print("left:", lefts)
        print("right:", rights)
        return 1 if lefts.sum() >= rights.sum() else 2

    def place_one_ship(self, ship_id, begin_pos, end_pos, place_dir, yard):
        inside_num = 0
        yard_temp = np.array(yard, dtype=int)
        ship_need = self.U_matrix.loc[ship_id, :]
        ship_place_num = np.zeros([self.U_matrix.shape[1] + 1], dtype=int)
        ship_place_pos = np.zeros([self.U_matrix.shape[1] + 1], dtype=int)
        ship_place_pos1 = np.zeros([self.U_matrix.shape[1] + 1], dtype=int)
        for t in range(1, self.U_matrix.shape[1] + 1):
            num = 0
            if place_dir == 1:  # 从left开始
                for pos in range(begin_pos, end_pos, 1):
                    if yard[t][pos] == 0:
                        num += 1
                    else:
                        break
                ship_place_num[t] = num
            elif place_dir == 2:  # 从right开始
                for pos in range(end_pos - 1, begin_pos - 1, -1):
                    if yard[t][pos] == 0:
                        num += 1
                    else:
                        break
                ship_place_num[t] = num
            # 根据ship_id 的天需求量，确定每天的开始位置，最后选最外边的开始位置摆放
            ship_need_value = ship_need[t]
            if ship_need_value <= ship_place_num[t]:  # 内部空间恰好够摆放，则从边界开始摆放即可
                ship_place_pos[t] = begin_pos if place_dir == 1 else end_pos - 1
                ship_place_pos1[t] = begin_pos + ship_need_value - 1 if place_dir == 1 \
                    else end_pos - ship_need_value
            else:
                gap = ship_need_value - ship_place_num[t]
                ship_place_pos[t] = begin_pos - gap if place_dir == 1 else end_pos - 1 + gap
                ship_place_pos1[t] = begin_pos - gap + ship_need_value - 1 if place_dir == 1 \
                    else end_pos - ship_need_value + gap

        # 然后找出最终的摆放起始位置
        place_pos = ship_place_pos[1:].min() if place_dir == 1 else ship_place_pos[1:].max()

        # 装船日之后的位置是可以变的，保证装船日之前的位置连续即可。但是对于有多个选择的箱子也可以改变位置
        # 改变的范围在最长的到最短的中间
        # 先确定装船日，然后装船日之后的place_pos 需要被重新计算
        ship_reach_day = self.ship_reach_time[ship_id]
        second_place_pos = 0
        if ship_reach_day < len(self.T):
            second_place_pos = ship_place_pos[ship_reach_day + 1:].min() if place_dir == 1 else ship_place_pos[
                                                                                                ship_reach_day + 1:].max()
        # 开始摆放到yard_temp
        for t in range(1, self.U_matrix.shape[1] + 1):
            if t <= ship_reach_day:
                for i in range(ship_need[t]):
                    if place_dir == 1:
                        yard_temp[t, place_pos + i] = ship_id
                    elif place_dir == 2:
                        yard_temp[t, place_pos - i] = ship_id
            else:
                for i in range(ship_need[t]):
                    if place_dir == 1:
                        yard_temp[t, second_place_pos + i] = ship_id
                    elif place_dir == 2:
                        yard_temp[t, second_place_pos - i] = ship_id
        # print("摆放之后的yard:")
        # print(yard_temp)

        # 摆放后有多少在之前的内部区域
        old_inside_num = 0
        for t in range(1, yard_temp.shape[0]):
            for col in range(yard_temp.shape[1]):
                if yard_temp[t][col] == ship_id and \
                        begin_pos <= col < end_pos:
                    old_inside_num += 1
        # 计算摆放后的inside_num 和 新的边界
        new_begin_pos = yard_temp.shape[1] - 1
        new_end_pos = 0
        for t in range(1, yard_temp.shape[0]):
            for col in range(yard_temp.shape[1]):
                if yard_temp[t][col] == ship_id:
                    inside_num += 1
            for col in range(yard_temp.shape[1]):
                if yard_temp[t][col] > 0 and new_begin_pos > col:
                    new_begin_pos = col
                    break
            for col in range(yard_temp.shape[1] - 1, 0, -1):
                if yard_temp[t][col] > 0 and new_end_pos < col:
                    new_end_pos = col
                    break
        return old_inside_num, inside_num, new_begin_pos, new_end_pos + 1, yard_temp

    def next_ship(self, used_ship, begin_pos, end_pos, place_dir, yard):
        inside_num = -1
        next_ship_id = 0
        new_yard = None
        final_begin_pos = 0
        final_end_pos = 0
        for j in self.J:
            if j in used_ship:
                continue
            old_inside_num, inside_temp, new_begin_pos, new_end_pos, yard_temp = self.place_one_ship(j, begin_pos,
                                                                                                     end_pos, place_dir,
                                                                                                     yard)
            print("shipper_id:{};inside:{};begin:{};end:{}".format(j, old_inside_num, new_begin_pos, new_end_pos))
            if old_inside_num > inside_num:
                inside_num = old_inside_num
                new_yard = yard_temp
                next_ship_id = j
                final_begin_pos = new_begin_pos
                final_end_pos = new_end_pos
        return next_ship_id, inside_num, final_begin_pos, final_end_pos, new_yard

    @staticmethod
    def _get_bound(yard_item, ship_id):
        left_bound = 0
        right_bound = 0
        for i in range(len(yard_item)):
            if yard_item[i] == ship_id:
                left_bound = i
                break
        for i in range(len(yard_item) - 1, 0, -1):
            if yard_item[i] == ship_id:
                right_bound = i
                break
        return left_bound, right_bound

    def _change_some_bin(self, yard, ship_id, place_dir, begin_pos, end_pos):
        ship_reach_day = self.ship_reach_time[ship_id]
        # 摆放后，需要微调整
        yard_temp = np.array(yard, dtype=int)
        print(yard)
        max_len_pos, max_len_pos1 = self._get_bound(yard[ship_reach_day], ship_id)
        print("ship_id:{};left:{};right:{}".format(ship_id, max_len_pos, max_len_pos1))
        change_success = 1
        for t in range(ship_reach_day - 1, 0, -1):
            if place_dir == 1:
                # 看是否有必要移动
                left_bd, right_bd = self._get_bound(yard_temp[t], ship_id)
                if left_bd >= begin_pos or right_bd >= max_len_pos1:
                    continue
                # 有必要往右移动,需要判断是否能移动
                new_left_pos = begin_pos
                while True:
                    if new_left_pos + right_bd - left_bd > max_len_pos1:
                        new_left_pos -= 1
                        continue
                    if yard[t][new_left_pos + right_bd - left_bd] > 0 and \
                            yard[t][new_left_pos + right_bd - left_bd] != ship_id:
                        new_left_pos -= 1
                        continue
                    break
                # 开始摆放
                for i in range(left_bd, right_bd + 1, 1):
                    yard_temp[t][i] = 0
                for i in range(new_left_pos, new_left_pos + right_bd - left_bd + 1, 1):
                    yard_temp[t][i] = ship_id
                max_len_pos, max_len_pos1 = self._get_bound(yard_temp[t], ship_id)
            elif place_dir == 2:
                # 看是否有必要移动
                left_bd, right_bd = self._get_bound(yard_temp[t], ship_id)
                if right_bd < begin_pos or left_bd <= max_len_pos:
                    continue
                # 有必要往右移动,需要判断是否能移动
                new_left_pos = end_pos - 1
                while True:
                    if new_left_pos - (right_bd - left_bd) < max_len_pos:
                        new_left_pos += 1
                        continue
                    if yard[t][new_left_pos - (right_bd - left_bd)] > 0 and \
                            yard[t][new_left_pos - (right_bd - left_bd)] != ship_id:
                        new_left_pos += 1
                        continue
                    break
                # 开始摆放
                for i in range(left_bd, right_bd + 1, 1):
                    yard_temp[t][i] = 0
                for i in range(new_left_pos, new_left_pos - (right_bd - left_bd + 1), -1):
                    yard_temp[t][i] = ship_id
                max_len_pos, max_len_pos1 = self._get_bound(yard_temp[t], ship_id)

        left_bds = []
        right_bds = []
        for t in range(1, ship_reach_day, 1):
            left_bd, right_bd = self._get_bound(yard_temp[t], ship_id)
            # 判断是否产生冲突，产生冲突则不进行调整
            if left_bd == 0 or \
                    right_bd == 0:
                continue
            for index in range(len(left_bds)):
                if left_bd > left_bds[index] or right_bd < right_bds[index]:
                    change_success = 0
                    break
            if change_success == 0:
                break
            left_bds.append(left_bd)
            right_bds.append(right_bd)

        return yard_temp if place_dir == 2 and change_success else yard

    #  1、第一个船如何选择
    def bin_problem(self):
        self.U_matrix = pd.read_csv(self.params_dict['U_file_path'])
        self.U_matrix.columns = range(1, self.U_matrix.shape[1] + 1)
        if len(self.J) != self.U_matrix.shape[0] - 1:
            print("输入的船数和U矩阵不一致")
            return
        used_ship = []
        for j in self.J:
            self.ship_reach_time[j] = len(self.T)
            for col in self.U_matrix.columns[0:-1]:
                if self.U_matrix.iloc[j][col + 1] < self.U_matrix.iloc[j][col]:
                    self.ship_reach_time[j] = col
                    break

        print("ship reach time:")
        print(self.ship_reach_time)

        # step 1 peak period
        peak_period = 0
        cnt = 0
        for day in self.U_matrix.columns:
            if self.U_matrix[day].sum() > cnt:
                cnt = self.U_matrix[day].sum()
                peak_period = int(day)
        if peak_period == 0:
            print("peak period == 0")
            return None
        cnt = 0
        ship_need = np.zeros([len(self.U_matrix)], dtype=int)
        for j in self.J:
            for t in self.U_matrix.columns[0: -1]:
                if self.U_matrix.loc[j, t + 1] < self.U_matrix.loc[j, t]:
                    cnt += self.U_matrix.loc[j, t]
                    ship_need[j] = self.U_matrix.loc[j, t]
                    break
        print("peak period:{},cnt:{}".format(peak_period, cnt))
        # step 2 init first ship
        yard = np.zeros([self.U_matrix.shape[1] + 1, cnt + 10], dtype=int)
        # yard[0] = np.array(range(0, yard.shape[1], 1), dtype=int)
        # 暂定第一个船
        ship = 1
        item = self.U_matrix.loc[ship, :]
        begin_pos = int((yard.shape[1] - ship_need[ship]) / 2)
        end_pos = begin_pos + ship_need[ship]
        for t in range(len(item)):
            if item[t + 1] > 0:
                for i in range(item[t + 1]):
                    yard[t + 1, begin_pos + i] = ship

        used_ship.append(ship)
        print("======Step 2=============")
        print("begin pos:{},end pos:{}".format(begin_pos, end_pos))
        print(yard)
        # step 3
        place_dir = self.place_direction(begin_pos, end_pos, yard)
        print("place direction:{}".format(place_dir))
        old_begin_pos = begin_pos
        old_end_pos = end_pos
        next_ship_id, _, begin_pos, end_pos, new_yard = self.next_ship(used_ship, begin_pos, end_pos, place_dir, yard)
        new_yard = self._change_some_bin(new_yard, next_ship_id, place_dir, old_begin_pos, old_end_pos)
        print("begin pos:{},end pos:{}".format(begin_pos, end_pos))
        print(new_yard)
        while len(used_ship) < len(self.J) - 1:
            print("======Info=========")
            used_ship.append(next_ship_id)
            yard = np.array(new_yard, dtype=int)
            place_dir = self.place_direction(begin_pos, end_pos, yard)
            print("place direction:{}".format(place_dir))
            old_begin_pos = begin_pos
            old_end_pos = end_pos
            next_ship_id, _, begin_pos, end_pos, new_yard = self.next_ship(used_ship, begin_pos, end_pos, place_dir,
                                                                           yard)
            new_yard = self._change_some_bin(new_yard, next_ship_id, place_dir, old_begin_pos, old_end_pos)
            print("begin pos:{},end pos:{}".format(begin_pos, end_pos))
            print(new_yard)
        print("final use len:{}".format(end_pos - begin_pos))
        # print("======Step 4=========")
        # used_ship.append(next_ship_id)
        # yard = np.array(new_yard, dtype=int)
        # place_dir = self.place_direction(begin_pos, end_pos, yard)
        # print("place direction:{}".format(place_dir))
        # old_begin_pos = begin_pos
        # old_end_pos = end_pos
        # next_ship_id, _, begin_pos, end_pos, new_yard = self.next_ship(used_ship, begin_pos, end_pos, place_dir, yard)
        # new_yard = self._change_some_bin(new_yard, next_ship_id, place_dir, old_begin_pos, old_end_pos)
        # print("begin pos:{},end pos:{}".format(begin_pos, end_pos))
        # print(new_yard)
        # print("======Step 5=========")
        # used_ship.append(next_ship_id)
        # yard = np.array(new_yard, dtype=int)
        # place_dir = self.place_direction(begin_pos, end_pos, yard)
        # print("place direction:{}".format(place_dir))
        # old_begin_pos = begin_pos
        # old_end_pos = end_pos
        # next_ship_id, _, begin_pos, end_pos, new_yard = self.next_ship(used_ship, begin_pos, end_pos, place_dir, yard)
        # new_yard = self._change_some_bin(new_yard, next_ship_id, place_dir, old_begin_pos, old_end_pos)
        # print("begin pos:{},end pos:{}".format(begin_pos, end_pos))
        # print(new_yard)


if __name__ == '__main__':
    # 设置随机种子
    random.seed(2020)
    np.random.seed(2020)
    # 15 block 40 slots in every block
    I = np.array(np.arange(1, 16, 1))
    J = np.array(np.arange(1, 6, 1))
    T = np.array([1, 2, 3, 4, 5, 6, 7])
    K = 40

    # 设置参数
    params = Param().set_params()

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

    problem.bin_problem()
