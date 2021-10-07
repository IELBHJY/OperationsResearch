import numpy as np
import pandas as pd
import random

random.seed(2021)

TASK_NUM = 1000
WORKER_NUM = 10
TYPE_SIZE = 3
need_values = [10, 20, 30, 40, 50, 60]
max_values = [200, 250, 300, 350, 400, 450, 500, 550, 600]
profits=[10, 20, 30, 40, 50]

result = pd.DataFrame(columns=['task_id', 'need_type', 'need_value', 'profit'])
for i in range(1, TASK_NUM + 1, 1):
    need_type = random.randint(1, TYPE_SIZE)
    idx = random.randint(0, len(need_values) - 1)
    need_value = need_values[idx]
    idx = random.randint(0, len(profits) - 1)
    profit = profits[idx]
    result = result.append({'task_id': i,
                            'need_type': need_type,
                            'need_value': need_value,
                            'profit': profit}, ignore_index=True)
result.to_csv("TASK_{}.csv".format(TASK_NUM), index=False)

result = pd.DataFrame(columns=['work_id', 'type', 'max_value'])
for i in range(1, WORKER_NUM + 1, 1):
    type1 = random.randint(1, TYPE_SIZE)
    type2 = random.randint(1, TYPE_SIZE)
    type = str(type1) + '_' + str(type2)
    idx = random.randint(0, len(max_values) - 1)
    max_value = max_values[idx]
    result = result.append({'work_id': i,
                            'type': type,
                            'max_value': max_value}, ignore_index=True)
result.to_csv("WORKER_{}.csv".format(WORKER_NUM), index=False)