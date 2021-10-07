import pandas as pd
import numpy as np

np.random.seed(2020)


class Model:
    def __init__(self, task_info, worker_info):
        self.task_info = task_info
        self.worker_info = worker_info
        self.task_num = len(self.task_info)
        self.worker_num = len(self.worker_info)
        self.task_cost = np.array(self.task_info['need_value'].values)
        self.task_profit = np.array(self.task_info['profit'].values)
        self.worker_capacity = np.array(self.worker_info['max_value'].values)
        self.ability_matrix = np.zeros((self.worker_num, self.task_num), dtype=int)
        self.assignment = np.zeros((self.worker_num, self.task_num))
        self.feasible_solution = np.zeros((self.worker_num, self.task_num), dtype=int)
        self.gen_ability_matrix()

        self.v = np.random.random((self.worker_num,))
        self.u = np.random.random((self.task_num,))

    def gen_ability_matrix(self):
        for i in range(self.worker_num):
            type = self.worker_info.loc[i]['type']
            type1 = type.split('_')[0]
            type2 = type.split('_')[1]
            for j in range(self.task_num):
                need_type = self.task_info.loc[j]['need_type']
                if int(type1) == int(need_type) or int(type2) == int(need_type):
                    self.ability_matrix[i][j] = 1

    def solve_lagrange_relax_model(self):
        for j in range(self.worker_num):
            for i in range(self.task_num):
                if self.ability_matrix[j][i] > 0 and \
                        self.task_profit[i] - self.u[i] - self.v[j] * self.task_cost[i] > 0:
                    self.feasible_solution[j][i] = 1

    def get_obj(self):
        obj = 0
        assignment = self.feasible_solution
        if not self.check_solution():
            print("solution is not feasible")
            return obj
        obj += np.sum(np.multiply(assignment, self.task_profit))
        return obj

    def check_solution(self):
        for i in range(self.task_num):
            if np.sum(self.feasible_solution[:, i]) > 1:
                return False
        if np.sum(np.dot(self.task_cost, self.feasible_solution.T) > self.worker_capacity) > 0:
            return False

        return True

    def gen_feasible_solution(self):
        # worker身上超过负荷
        task_cost_dict = {idx: self.task_cost[idx] for idx in range(self.task_num)}
        task_cost_dict = sorted(task_cost_dict.items(), key=lambda item: item[1])
        for j in range(self.worker_num):
            if np.dot(self.task_cost, self.feasible_solution[j, :]) <= self.worker_capacity[j]:
                continue
            overload_num = np.dot(self.task_cost, self.feasible_solution[j, :]) - self.worker_capacity[j]
            cumsum = 0
            for item in task_cost_dict:
                if self.feasible_solution[j, item[0]] > 0:
                    cumsum += self.task_cost[item[0]]
                    if cumsum >= overload_num:
                        self.feasible_solution[j, item[0]] = 0
                        break
        # 一个任务被分派多次
        for i in range(self.task_num):
            if np.sum(self.feasible_solution[:, i]) > 1:
                overmatch_num = np.sum(self.feasible_solution[:, i]) - 1
                for j in range(self.worker_num):
                    if self.feasible_solution[j, i] > 0:
                        self.feasible_solution[j, i] = 0
                        overmatch_num -= 1
                        if overmatch_num == 0:
                            break

    def show_solution(self):
        for i in range(self.task_num):
            if sum(self.assignment[i, :] > 0.99) > 0:
                print("task {} has worker {}".format(i, np.argmax(self.assignment[i, :])))

        for j in range(self.worker_num):
            print("worker {} has task num {}".format(j, sum(self.assignment[:, j] > 0.99)))

    def update_magic_value(self):
        pass

    def print_importance_info(self):
        pass

    def solve(self, param_dict):
        iter_num = param_dict['iter_num']
        for _iter in range(iter_num):
            self.solve_lagrange_relax_model()
            if self.check_solution():
                self.assignment = self.feasible_solution.copy()
                break
            self.gen_feasible_solution()
            print("iter num:", _iter, ", obj:", self.get_obj())
            self.update_magic_value()
            self.print_importance_info()


task_info = pd.read_csv("TASK_1000.csv")
worker_info = pd.read_csv("WORKER_10.csv")
model = Model(task_info, worker_info)
param_dict = {'iter_num': 100}
model.solve(param_dict)
