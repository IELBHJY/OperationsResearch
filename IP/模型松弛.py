import pandas as pd
import numpy as np

np.random.seed(2020)

from ortools.linear_solver import pywraplp


class Model:
    def __init__(self, task_info, worker_info):
        self.task_info = task_info
        self.worker_info = worker_info
        self.task_num = len(self.task_info)
        self.worker_num = len(self.worker_info)
        self.task_cost = np.array(self.task_info['need_value'].values)
        self.task_profit = np.array(self.task_info['profit'].values)
        self.worker_capacity = np.array(self.worker_info['max_value'].values)
        self.ability_matrix = np.zeros((self.task_num, self.worker_num), dtype=int)
        self.get_ability_matrix()
        self.magic_value = np.random.random((self.worker_num,))
        self.assignment = np.zeros((self.task_num, self.worker_num))

    def get_ability_matrix(self):
        for i in range(self.worker_num):
            type = self.worker_info.loc[i]['type']
            type1 = type.split('_')[0]
            type2 = type.split('_')[1]
            for j in range(self.task_num):
                need_type = self.task_info.loc[j]['need_type']
                if int(type1) == int(need_type) or int(type2) == int(need_type):
                    self.ability_matrix[j][i] = 1

    def cal_upper_bound(self, tabu_table):
        # 建模
        solver = pywraplp.Solver.CreateSolver('SCIP')
        x = {}
        for i in range(self.task_num):
            for j in range(self.worker_num):
                x[i, j] = solver.NumVar(0, 1, '')

        # 约束1
        for i in range(self.task_num):
            solver.Add(solver.Sum([x[i, j] for j in range(self.worker_num)]) <= 1)

        # 约束2
        for i in range(self.task_num):
            for j in range(self.worker_num):
                solver.Add(x[i, j] <= self.ability_matrix[i, j])

        # 有一些限制
        if len(tabu_table) > 0:
            for item in tabu_table:
                solver.Add(x[item[0], item[1]] <= 0)

        # 建立目标函数
        objective_terms = []
        for i in range(self.task_num):
            for j in range(self.worker_num):
                objective_terms.append(self.task_profit[i] * x[i, j])
                objective_terms.append(x[i, j])
        for j in range(self.worker_num):
            objective_terms.append(self.magic_value[j] * (self.worker_info.loc[j]['max_value'] -
                                                          solver.Sum([x[i, j] * int(self.task_info.loc[i]['need_value'])
                                                                      for i in range(self.task_num)])))

        solver.Maximize(solver.Sum(objective_terms))
        status = solver.Solve()
        solution_assignment = np.zeros((self.task_num, self.worker_num))
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            print("status =", status)
            print('Total cost = ', solver.Objective().Value(), '\n')
            for i in range(self.task_num):
                for j in range(self.worker_num):
                    solution_assignment[i][j] = x[i, j].solution_value()
        else:
            print(status)
        return solution_assignment

    def process_one_worker(self):
        pass

    def get_obj(self):
        obj = 0
        assignment = self.assignment
        assignment[assignment <= 0.99] = 0
        obj += len(assignment > 0.99)
        obj += np.sum(np.dot(self.task_profit, assignment))
        return obj

    def cal_lower_bound(self):
        # 对于每个工人，检查是否满足最大负载约束，不满足的话，去掉任务直到满足
        tabu_table = []
        task_cost = self.task_cost.copy()
        np.sort(task_cost)
        for worker in range(self.worker_num):
            while np.sum(self.task_cost[(self.assignment[:, worker] > 0.99)]) > self.worker_capacity[worker]:
                task = np.argmax(self.assignment[:, worker])
                self.assignment[task, worker] = 0
                tabu_table.append([task, worker])
        cur_obj = self.get_obj()
        print("当前obj:{}".format(cur_obj))
        return tabu_table

    def show_solution(self):
        for i in range(self.task_num):
            if sum(self.assignment[i, :] > 0.99) > 0:
                print("task {} has worker {}".format(i, np.argmax(self.assignment[i, :])))

        for j in range(self.worker_num):
            print("worker {} has task num {}".format(j, sum(self.assignment[:, j] > 0.99)))

    def update_magic_value(self):
        pass

    def solver(self):
        assignment = self.cal_upper_bound([])
        self.assignment = assignment
        tabu_table = self.cal_lower_bound()
        for _ in range(10):
            assignment = self.cal_upper_bound(tabu_table)
            self.assignment = assignment
            tabu = self.cal_lower_bound()
            tabu_table.extend(tabu)
        self.show_solution()


task_info = pd.read_csv("TASK_1000.csv")
worker_info = pd.read_csv("WORKER_10.csv")
model = Model(task_info, worker_info)
model.solver()
