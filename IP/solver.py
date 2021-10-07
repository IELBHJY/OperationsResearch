import pandas as pd
import numpy as np
import time
from ortools.linear_solver import pywraplp

TYPE_NUM = 3




def ckeck_solution(task_info, worker_info, result):
    # 每个任务是否被分配一次
    nums = np.zeros([len(task_info)], dtype=int)
    for i in range(0, len(task_info), 1):
        for j in range(0, len(worker_info), 1):
            if result[i, j] == 1:
                nums[i] += 1
    for i in range(0, len(task_info), 1):
        if nums[i] > 1:
            print("不满足任务最多分配一次约束")

    # 每个工人是否满足负荷约束
    nums = np.zeros([len(worker_info)], dtype=int)
    for i in range(0, len(task_info), 1):
        for j in range(0, len(worker_info), 1):
            if result[i, j] == 1:
                nums[j] += task_info.loc[i]['need_value']
    for j in range(0, len(worker_info), 1):
        if nums[j] > worker_info.loc[j]['max_value']:
            print("不满足工人负荷约束")

    # 分配是否满足工种约束
    for i in range(0, len(task_info), 1):
        for j in range(0, len(worker_info), 1):
            if result[i, j] == 1:
                type = worker_info.loc[j]['type']
                type1 = type.split('_')[0]
                type2 = type.split('_')[1]
                if task_info.loc[i]['need_type'] != int(type1) and \
                        task_info.loc[i]['need_type'] != int(type2):
                    print("不满足工种匹配约束")


def build_model(task_info, worker_info, param_dict):
    Y = np.zeros([len(worker_info), TYPE_NUM + 1], dtype=int)
    for i in range(0, len(worker_info), 1):
        type = worker_info.loc[i]['type']
        type1 = type.split('_')[0]
        type2 = type.split('_')[1]
        Y[i][int(type1)] = 1
        Y[i][int(type2)] = 1
    # 建模
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = {}
    for i in range(0, len(task_info), 1):
        for j in range(len(worker_info)):
            x[i, j] = solver.IntVar(0, 1, '')
    # 约束1
    for i in range(0, len(task_info), 1):
        solver.Add(solver.Sum([x[i, j] for j in range(0, len(worker_info), 1)]) <= 1)

    # 约束2
    for j in range(0, len(worker_info), 1):
        solver.Add(solver.Sum([x[i, j] * int(task_info.loc[i]['need_value'])
                               for i in range(0, len(task_info), 1)]) <= int(worker_info.loc[j]['max_value']))

    # 约束3
    for i in range(0, len(task_info), 1):
        for j in range(0, len(worker_info), 1):
            solver.Add(x[i, j] <= Y[j][task_info.loc[i]['need_type']])

    # 建立目标函数
    # 价值最大  分配数量最多
    objective_terms = []
    for i in range(0, len(task_info), 1):
        for j in range(0, len(worker_info), 1):
            objective_terms.append(param_dict['alpha'] * task_info.loc[i]['profit'] * x[i, j])
            objective_terms.append(param_dict['gamma'] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))
    # 求解模型
    status = solver.Solve()
    result = dict()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print("status =", status)
        print('Total cost = ', solver.Objective().Value(), '\n')
        for i in range(0, len(task_info), 1):
            for j in range(0, len(work_info), 1):
                if x[i, j].solution_value() > 0.9:
                    result[i, j] = 1
                    print('Task %d assigned to worker %d.' % (i, j))
    else:
        print(status)
    ckeck_solution(task_info, worker_info, result)

start = time.time()
task_info = pd.read_csv("TASK_500.csv")
work_info = pd.read_csv("WORKER_10.csv")
param_dict = dict()
param_dict['alpha'] = 1
param_dict['gamma'] = 1
param_dict['beta'] = 0.5
build_model(task_info, work_info, param_dict)
end = time.time()
print("运行时间:%.2f秒"% (end-start))