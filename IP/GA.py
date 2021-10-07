import pandas as pd
import numpy as np
import random
import time

TYPE_NUM = 3
PENTY = 1000


class GA:
    def __init__(self, task_info, worker_info, param_dict, init_solution):
        self.task_info = task_info
        self.worker_info = worker_info
        self.task_num = len(self.task_info)
        self.worker_num = len(self.worker_info)

        # 遗传算法参数
        self._params_dict = param_dict
        self._param_init(param_dict)

        # 遗传算法数据结构
        self.task_parent_matrix = None
        self.worker_parent_matrix = None

        self.task_children_matrix = None
        self.worker_children_matrix = None

        self.pop_scores = None
        self.init_pop_scores = None
        self.sorted_item_info = None

        # 启发式信息
        self.init_solution = init_solution

        # 处理工人信息
        self.worker_type = pd.DataFrame(np.zeros([len(self.worker_info), TYPE_NUM + 1], dtype=int),
                                        columns=[v for v in range(0, TYPE_NUM + 1)])
        self._init_worker()

        # 初始化种群
        self._pop_init()
        self.pop_worker_value = np.zeros([self._params_dict['pop_size'], len(self.worker_info)], dtype=int)

        # 初始化适应度函数
        self.pop_scores = dict()
        self.init_pop_scores = dict()
        for i in range(self._params_dict['pop_size']):
            self.pop_scores[i] = 0.0
            self.init_pop_scores[i] = 0.0

    def _init_worker(self):
        self.worker_type['has_value'] = 0
        for j in range(0, len(self.worker_info), 1):
            type = self.worker_info.loc[j]['type']
            type1 = type.split("_")[0]
            type2 = type.split("_")[1]
            self.worker_type.loc[j][int(type1)] = 1
            self.worker_type.loc[j][int(type2)] = 1
            self.worker_type.loc[j]['has_value'] = self.worker_info.loc[j]['max_value']

    def _reset_worker_info(self):
        for j in range(0, len(self.worker_info), 1):
            self.worker_type.loc[j]['has_value'] = self.worker_info.loc[j]['max_value']

    def _param_init(self, params_dict):
        if "pop_size" not in params_dict.keys():
            self._params_dict['pop_size'] = 100
        else:
            self._params_dict['pop_size'] = params_dict['pop_size']
        if "mutate_rate" not in params_dict.keys():
            self._params_dict['mutate_rate'] = 0.4
        else:
            self._params_dict['mutate_rate'] = params_dict['mutate_rate']
        if "crossover_rate" not in params_dict.keys():
            self._params_dict['crossover_rate'] = 0.4
        else:
            self._params_dict['crossover_rate'] = params_dict['crossover_rate']
        if "inherit_rate" not in params_dict.keys():
            self._params_dict['inherit_rate'] = 0.2
        else:
            self._params_dict['inherit_rate'] = params_dict['inherit_rate']
        if "max_iter_num" not in params_dict.keys():
            self._params_dict['max_iter_num'] = 30
        else:
            self._params_dict['max_iter_num'] = params_dict['max_iter_num']
        if "problem_type" not in params_dict.keys():
            self._params_dict['problem_type'] = 'max'
        else:
            self._params_dict['problem_type'] = params_dict['problem_type']

    def _pop_init(self):
        x = self.task_info['task_id'].values
        self.task_parent_matrix = np.zeros([self._params_dict['pop_size'], len(x)], dtype=int)
        self.worker_parent_matrix = np.zeros([self._params_dict['pop_size'], len(x)], dtype=int)
        self.task_children_matrix = np.zeros([self._params_dict['pop_size'], len(x)], dtype=int)
        self.worker_children_matrix = np.zeros([self._params_dict['pop_size'], len(x)], dtype=int)

        # 设置启发式解
        for i in range(len(self.init_solution)):
            self.worker_parent_matrix[i] = np.array(self.init_solution[i], dtype=int)

        # 随机选择开始的点
        for index in range(len(self.init_solution), self._params_dict['pop_size']):
            self.task_parent_matrix[index] = np.array(x, dtype=int)
            start_point = random.randint(0, len(x) - 1)
            self._reset_worker_info()
            for i in range(start_point, len(x), 1):
                task_need_type = self.task_info.loc[i]['need_type']
                task_value = self.task_info.loc[i]['need_value']
                # 如果所有工人的剩余负荷 为 0 则提前退出
                if self.worker_type['has_value'].sum() == 0:
                    break
                # 找到工种匹配，且负荷还能做该任务的worker
                worker_list = self.worker_type[(self.worker_type[task_need_type] > 0) &
                                               (self.worker_type['has_value'] >= task_value)].index
                if len(worker_list) == 0:
                    self.worker_parent_matrix[index][i] = 0
                    continue
                worker_id = np.random.choice(worker_list) + 1
                self.worker_parent_matrix[index][i] = worker_id
                # 更新 worker的剩余 负荷
                self.worker_type.loc[worker_id - 1]['has_value'] -= self.task_info.loc[i]['need_value']
            for i in range(0, start_point, 1):
                if self.worker_type['has_value'].sum() == 0:
                    break
                task_need_type = self.task_info.loc[i]['need_type']
                task_value = self.task_info.loc[i]['need_value']
                # 找到工种匹配，且负荷还能做该任务的worker
                worker_list = self.worker_type[(self.worker_type[task_need_type] > 0) &
                                               (self.worker_type['has_value'] >= task_value)].index
                if len(worker_list) == 0:
                    self.worker_parent_matrix[index][i] = 0
                    continue
                worker_id = np.random.choice(worker_list) + 1
                self.worker_parent_matrix[index][i] = worker_id
                # 更新 worker的剩余 负荷
                self.worker_type.loc[worker_id - 1]['has_value'] -= self.task_info.loc[i]['need_value']

    def _update(self, index):
        score = 0
        self.pop_worker_value[index] = np.array(
            [self.worker_info.loc[j]['max_value'] for j in range(0, len(self.work_info), 1)], dtype=int)
        for i in range(0, len(self.task_info), 1):
            if self.worker_parent_matrix[index][i] > 0:
                score += self.task_info.loc[i]['profit'] * self._params_dict['alpha']
                score += self._params_dict['gamma']
                task_value = self.task_info.loc[i]['need_value']
                work_id = self.worker_parent_matrix[index][i]
                self.pop_worker_value[index][work_id - 1] -= task_value

        # 判断是否违反负荷约束
        for j in range(len(self.worker_info)):
            if self.pop_worker_value[index][j] < 0:
                score -= PENTY
        if score <= 0:
            score = 1
        return score

    def _output(self, pop_index, file_path):
        temp = self.pop_matrix[pop_index]
        res = pd.DataFrame(columns=['编号', '到达排序弧时间', '离开排序弧时间', '到达时间', 'ETA'])
        for id in temp:
            res = res.append({'编号': id,
                              '到达排序弧时间': self.reach_radis_time[pop_index][id],
                              '离开排序弧时间': self.leave_radis_time[pop_index][id],
                              '到达时间': self.reach_time[pop_index][id],
                              'ETA': self.ETA[pop_index][id]}, ignore_index=True)
        res.to_excel(file_path, index=False)

    def _cal_pop_score(self):
        for i in range(self._params_dict['pop_size']):
            score = self._update(i)
            self.pop_scores[i] = score
        reverse_flag = (True if self._params_dict['problem_type'] == 'max' else False)
        self.sorted_item_info = sorted(self.pop_scores.items(), key=lambda x: x[1], reverse=reverse_flag)

    def _choose_item(self):
        score_sum = sum(self.pop_scores.values())
        value = np.random.randint(0, int(score_sum))
        index = 0
        cumsum = self.pop_scores[index]
        while cumsum < value:
            index += 1
            cumsum += self.pop_scores[index]
        return index

    def _crossover(self, index1, index2, new_index1, new_index2):
        cv_point = np.random.randint(1, self.T - 1)
        child1_values = []
        child2_values = []
        for i in range(cv_point, self.T, 1):
            child1_values.append(self.pop_matrix[index1][i])

        for i in range(cv_point, self.T, 1):
            child2_values.append(self.pop_matrix[index2][i])

        child1_seq = []
        child2_seq = []
        for i in range(0, self.T, 1):
            if self.pop_matrix[index2][i] in child1_values:
                child1_seq.append(self.pop_matrix[index2][i])
        for i in range(0, self.T, 1):
            if self.pop_matrix[index1][i] in child2_values:
                child2_seq.append(self.pop_matrix[index1][i])

        item1 = np.array(np.hstack((self.pop_matrix[index1, :cv_point],
                                    np.array(child1_seq))), dtype=int)
        item2 = np.array(np.hstack((self.pop_matrix[index2, :cv_point],
                                    np.array(child2_seq))), dtype=int)
        self.children_matrix[new_index1] = item1
        self.children_matrix[new_index2] = item2

    # 会出现不可行的解
    # 解决方法，计算score时加惩罚  或者 修复
    def _one_point_crossover(self, index1, index2, new_index1, new_index2):
        cv_point = np.random.randint(1, self.task_num - 1)
        self.worker_children_matrix[new_index1][0:cv_point] = self.worker_parent_matrix[index1][0:cv_point]
        self.worker_children_matrix[new_index1][cv_point:] = self.worker_parent_matrix[index2][cv_point:]

        self.worker_children_matrix[new_index2][0:cv_point] = self.worker_parent_matrix[index2][0:cv_point]
        self.worker_children_matrix[new_index2][cv_point:] = self.worker_parent_matrix[index1][cv_point]

    def _mutation(self, index, new_index):
        item = np.array(self.pop_matrix[index], dtype=int)
        choosen_index = np.random.randint(0, self.T - 1)
        if choosen_index < self.T - 1:
            temp = item[choosen_index]
            item[choosen_index] = item[choosen_index + 1]
            item[choosen_index + 1] = temp
        self.children_matrix[new_index] = item

    # 两种变异方式，一种是加入未分配的任务，如果不能加入则 选择更好的任务替换已有的任务
    def _add_mutation(self, index, new_index):
        self.worker_children_matrix[new_index] = self.worker_parent_matrix[index]
        change_index = -1
        change_value = -1
        find_flag = 0
        for i in range(0, len(self.task_info), 1):
            if self.worker_parent_matrix[index][i] == 0:
                task_need_type = self.task_info.loc[i]['need_type']
                task_need_value = self.task_info.loc[i]['need_value']
                worker_list = self.worker_type[(self.worker_type[task_need_type] > 0)].index
                for worker_idx in worker_list:
                    if self.pop_worker_value[index][worker_idx] >= task_need_value:
                        change_index = i
                        change_value = worker_idx + 1
                        find_flag = 1
                        self.pop_worker_value[index][worker_idx] -= task_need_value
                        break
                if find_flag:
                    break
        self.worker_children_matrix[new_index][change_index] = change_value

    def _exchange_mutation(self, index, new_index):
        self.worker_children_matrix[new_index] = self.worker_parent_matrix[index]
        match_task_list = [i for i in range(len(self.task_info)) if self.worker_parent_matrix[index][i] > 0]
        unmatch_task_list = [i for i in range(len(self.task_info)) if self.worker_parent_matrix[index][i] == 0]
        index1 = np.random.choice(match_task_list)
        index2 = np.random.choice(unmatch_task_list)
        worker_id = self.worker_parent_matrix[index][index1]
        current_need_value = self.task_info.loc[index1]['need_value']
        task_need_type = self.task_info.loc[index2]['need_type']
        task_need_value = self.task_info.loc[index2]['need_value']
        if self.worker_type.loc[worker_id - 1][task_need_type] > 0 and \
                self.pop_worker_value[index][worker_id - 1] + current_need_value >= task_need_value:
            self.worker_children_matrix[new_index][index1] = 0
            self.worker_children_matrix[new_index][index2] = worker_id

    def _evolutionary_operator(self):
        # 先保留精英
        start_index = 0
        end_index = int(self._params_dict['pop_size'] * self._params_dict['inherit_rate'])
        for i in range(start_index, end_index, 1):
            if i == end_index - 1:
                self.worker_children_matrix[i] = np.array(self.init_solution, dtype=int)
            else:
                self.worker_children_matrix[i] = np.array(self.worker_parent_matrix[self.sorted_item_info[i][0]],
                                                          dtype=int)

        # 交叉
        start_index = end_index
        end_index = start_index + int(self._params_dict['pop_size'] * self._params_dict['crossover_rate'])
        for i in range(start_index, end_index, 2):
            index1 = self._choose_item()
            index2 = self._choose_item()
            while index1 == index2:
                index2 = self._choose_item()
            self._one_point_crossover(index1, index2, i, i + 1)

        # 变异
        start_index = end_index
        for i in range(start_index, self._params_dict['pop_size'], 2):
            index = self._choose_item()
            self._add_mutation(index, i)
            index = self._choose_item()
            self._exchange_mutation(index, i + 1)

    def _exchange(self):
        self.worker_parent_matrix = np.array(self.worker_children_matrix, dtype=int)

    def _genetic_algorithm_process(self):
        iter = 0
        self._cal_pop_score()
        while iter < self._params_dict['max_iter_num']:
            self._evolutionary_operator()
            self._exchange()
            self._cal_pop_score()
            self.best_index = self.sorted_item_info[0][0]
            iter += 1
            if iter % 10 == 0:
                print("最好的目标函数值:{}\n最好的个体:{}".format(self.sorted_item_info[0][1],
                                                     self.sorted_item_info[0][0]))


task_info = pd.read_csv("TASK_500.csv")
work_info = pd.read_csv("WORKER_10.csv")
param_dict = dict()
param_dict['alpha'] = 1
param_dict['gamma'] = 1
ga = GA(task_info, work_info, param_dict, [])
