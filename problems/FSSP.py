# -*- coding: utf-8 -*-
from threading import Timer

import numpy as np
import pandas as pd
import random
import time


def make_span(s, P):
    """Calculate the makespan given a sequence of jobs and a matrix of processing
   times. This function can also compute the makespan of a partial sequence but P must be the complete matrix.

    Arguments:
    s -- numpy vector containing a sequence of jobs. It must be a (possibly a subset) of a permutation of 0, 1, ..., number of jobs.
    P -- a NumPy matrix of shape (number of jobs, number of machines).
    """
    # We index the processing time matrix by s so that we can simply use 0 to
    # refer to s[0], 1 to refer to s[1], etc in the completion times matrix.
    # Otherwise we would have to use C[s[i], j] later.
    # This also allows handling partial sequences. In that case, C will contain fewer rows.
    C = P[s, :]
    # n: jobs
    # m: machines
    n, m = C.shape
    # Eq 1: Completion time of first job on all machines.
    # Note: np.cumsum is much faster than a loop.
    C[0, :] = np.cumsum(C[0, :])
    # Eq 2: Completion time of each job k on first machine.
    C[:, 0] = np.cumsum(C[:, 0])

    # It may be possible to remove these two loops with fancy indexing, but
    # let's keep it simple.
    for i in range(1, n):
        for j in range(1, m):
            # Eq 3. C[i,j] already contains P[s[i], j]
            # Note: np.maximum is faster than max()
            C[i, j] += np.maximum(C[i - 1, j], C[i, j - 1])
        # print(C[i,:])

    # We only need the makespan (completion time of the last job on the last
    # machine).
    return C[-1, -1]


def crossover_one(parent_one, parent_two, insert_point):
    child_one = np.array(parent_one)
    child_two = np.array(parent_two)
    if insert_point <= 0 or insert_point >= parent_two.size - 1:
        return child_one, child_two
    if parent_one.size != parent_two.size:
        return child_one, child_two
    p1_list = np.array(parent_one[insert_point + 1:])
    p1_cur_index = insert_point + 1
    for index in range(0, parent_two.size):
        if parent_two[index] in p1_list:
            child_one[p1_cur_index] = parent_two[index]
            p1_cur_index += 1
    p2_list = np.array(parent_two[insert_point + 1:])
    p2_cur_index = insert_point + 1
    for index in range(0, parent_one.size):
        if parent_one[index] in p2_list:
            child_two[p2_cur_index] = parent_one[index]
            p2_cur_index += 1
    return child_one, child_two


def crossover_two(parent_one, parent_two, insert_one, insert_two):
    child_one = np.array(parent_one)
    child_two = np.array(parent_two)
    if insert_one <= 0 or insert_one >= insert_two:
        return child_one, child_two
    if parent_one.size != parent_two.size:
        return child_one, child_two
    if insert_two >= parent_one.size - 1:
        return child_one, child_two
    p1_list = np.array(parent_one[insert_two + 1:])
    p1_cur_index = insert_two + 1
    for index in range(0, parent_two.size):
        if parent_two[index] in p1_list:
            child_one[p1_cur_index] = parent_two[index]
            p1_cur_index += 1
    temp = parent_one[insert_one + 1:insert_two + 1]
    np.random.shuffle(temp)
    child_one[insert_one + 1:insert_two + 1] = temp.copy()

    p2_list = np.array(parent_two[insert_two + 1:])
    p2_cur_index = insert_two + 1
    for index in range(0, parent_one.size):
        if parent_one[index] in p2_list:
            child_two[p2_cur_index] = parent_one[index]
            p2_cur_index += 1
    temp = parent_two[insert_one + 1:insert_two + 1]
    np.random.shuffle(temp)
    child_two[insert_one + 1:insert_two + 1] = temp.copy()

    return child_one, child_two


def exchange_mutation(parent, pos_one, pos_two):
    child = np.array(parent)
    if pos_one < 0 or \
            pos_one >= pos_two or \
            pos_two >= parent.size:
        return child
    temp = child[pos_one]
    child[pos_one] = child[pos_two]
    child[pos_two] = temp
    return child


def shift_mutation(parent, pos, new_pos, direct):
    '''
    pos上的元素插入到new_pos位置前面或者后边
    direct = 1 是前面
    direct = -1 是后边
    '''
    child = np.array(parent)
    if pos < 0 or pos >= parent.size:
        return child
    if new_pos < 0 or new_pos >= parent.size:
        return child
    temp = list()
    for index in range(pos + direct, new_pos, direct):
        if direct < 0:
            temp.insert(0, parent[index])
        else:
            temp.append(parent[index])
    if direct < 0:
        temp.insert(0, parent[pos])
    if direct > 0:
        temp.append(parent[pos])
    if direct > 0:
        child[pos:new_pos:1] = np.array(temp)
    else:
        child[new_pos + 1:pos + 1: 1] = np.array(temp)
    return child


def generate_random(job_list):
    init = np.array(job_list)
    np.random.shuffle(init)
    return init


def generate_NEH(job_list, time_matrix):
    '''
    job list, time matrix,row is job columns is machine
    '''
    job_seq = np.zeros(job_list.size)

    # step 1
    job_sum = dict()
    for index in range(job_list.size):
        job_sum[job_list[index]] = np.sum(time_matrix[job_list[index], :])

    # step 2
    seq_index = 0
    sorted_job_sum = sorted(job_sum.items(), key=lambda item: item[1], reverse=True)
    if make_span(np.array([sorted_job_sum[0][0], sorted_job_sum[1][0]]), time_matrix) > \
            make_span(np.array([sorted_job_sum[1][0], sorted_job_sum[0][0]]), time_matrix):
        job_seq[seq_index] = sorted_job_sum[1][0]
        seq_index += 1
        job_seq[seq_index] = sorted_job_sum[0][0]
        seq_index += 1
    else:
        job_seq[seq_index] = sorted_job_sum[0][0]
        seq_index += 1
        job_seq[seq_index] = sorted_job_sum[1][0]
        seq_index += 1
    sorted_job_sum[0] = (sorted_job_sum[0][0], -1)
    sorted_job_sum[1] = (sorted_job_sum[1][0], -1)

    # step 3
    for seq_index in range(2, job_list.size, 1):
        min_obj = np.inf
        target_index = -1
        for index in range(0, job_list.size, 1):
            if sorted_job_sum[index][1] < 0:
                continue
            aaa = np.hstack((job_seq[0:seq_index], np.array([sorted_job_sum[index][0]])))
            aaa = aaa.astype(int)
            if make_span(aaa, time_matrix) < min_obj:
                min_obj = make_span(aaa, time_matrix)
                target_index = index
        if target_index >= 0 and min_obj < np.inf:
            job_seq[seq_index] = sorted_job_sum[target_index][0]
            sorted_job_sum[target_index] = (sorted_job_sum[target_index][0], -1)

    return job_seq.astype(int)


def generate_init_pop(job_list, time_matrix, M):
    if M <= 0:
        return np.array([[]])
    pop = np.zeros([M, job_list.size], dtype=int)
    pop_num = 0
    NEH_one = generate_NEH(job_list, time_matrix)
    pop[pop_num, :] = np.array(NEH_one, dtype=int)
    pop_num += 1
    while pop_num < M:
        temp = generate_random(job_list)
        pop[pop_num, :] = np.array(temp, dtype=int)
        pop_num += 1
    return pop


def population_sort(pop_matrix, time_matrix, show_log=False):
    '''
    输入种群矩阵，row代表个体编号；col表示一个种群的基因
    return obj从优到差的index array
    '''
    dict_pop_obj = dict()
    for index in range(0, len(pop_matrix)):
        dict_pop_obj[index] = make_span(pop_matrix[index, :], time_matrix)
    sorted_dict_pop_obj = sorted(dict_pop_obj.items(), key=lambda x: x[1], reverse=True)
    if show_log:
        print("Best one:{}".format(sorted_dict_pop_obj[-1][1]))
    return sorted_dict_pop_obj


def select(sorted_dict_pop_obj, mode):
    if mode == 0:
        return int(np.random.uniform(0, len(sorted_dict_pop_obj)))
    if mode == 1:
        def rank_fun(k, M):
            return (2 * k) / (M * (M + 1))

        rank_scores = [rank_fun(k, len(sorted_dict_pop_obj)) \
                       for k in range(1, len(sorted_dict_pop_obj) + 1, 1)]

        return np.random.choice(a=range(0, len(sorted_dict_pop_obj)),
                                size=1, replace=False, p=rank_scores)[0]
    if mode == 2:
        return sorted_dict_pop_obj[0][0]


def evaluate(sol, time_matrix):
    return make_span(sol, time_matrix)


def GeneticAlgorithm(job_list, time_matrix, params, seed=2020):
    # 初始化参数值
    M = params['P']
    p_c = params['p_c']
    p_m_init = params['p_m']
    seita = 0.99
    D = params['D']
    iter_index = 0
    iter_num = 1000 * len(job_list)
    eval_index = 0
    evaluate_num = 1000 * len(job_list)
    random.seed(seed)
    np.random.seed(seed)

    # 初始化种群并计算排序，返回从最差到最好的顺序
    pop_init = generate_init_pop(job_list, time_matrix, M)
    sorted_seq = population_sort(pop_init, time_matrix)
    p_m = p_m_init
    eval_index += M
    while iter_index < iter_num:
        if random.random() < p_c:
            chosen_first = select(sorted_seq, mode=1)
            chosen_second = select(sorted_seq, mode=0)
            crossover_p1 = random.randint(0, len(job_list))
            crossover_p2 = random.randint(0, len(job_list))
            child_first, child_second = crossover_two(pop_init[chosen_first],
                                                      pop_init[chosen_second],
                                                      crossover_p1,
                                                      crossover_p2)
        else:
            child_first = chosen_first
            child_second = chosen_second

        mutation_flag = 0
        if random.random() < p_m:
            mutation_flag = 1
            mutation_pos1 = random.randint(0, len(job_list))
            mutation_pos2 = random.randint(0, len(job_list))
            child_first_after_mutation = exchange_mutation(child_first, mutation_pos1, mutation_pos2)
            child_second_after_mutation = exchange_mutation(child_second, mutation_pos1, mutation_pos2)
        good_index = 1
        if mutation_flag == 1:
            if evaluate(child_first_after_mutation, time_matrix) > \
                    evaluate(child_second_after_mutation, time_matrix):
                good_index = 2
        else:
            if evaluate(child_first, time_matrix) > \
                    evaluate(child_second, time_matrix):
                good_index = 2

        bad_index = select(sorted_seq, mode=2)
        if good_index == 1:
            pop_init[bad_index, :] = np.array(child_first_after_mutation,
                                              dtype=int) if mutation_flag == 1 else np.array(child_first, dtype=int)
        else:
            pop_init[bad_index, :] = np.array(child_second_after_mutation,
                                              dtype=int) if mutation_flag == 1 else np.array(child_second, dtype=int)

        if iter_index % 500 == 0:
            sorted_seq = population_sort(pop_init, time_matrix, show_log=True)
        else:
            sorted_seq = population_sort(pop_init, time_matrix)
        eval_index += M
        if eval_index >= evaluate_num:
            break
        p_m = p_m * seita
        v_min = sorted_seq[-1][1]
        v_mean = np.mean([x[1] for x in sorted_seq])
        if v_min / v_mean > D:
            p_m = p_m_init
        iter_index += 1
    print("迭代了{}次。best_obj:{}".format(iter_index, sorted_seq[-1][1]))
    return sorted_seq, pop_init


def RandomSearch(job_list):
    init = np.array(job_list)
    np.random.shuffle(init)
    return init


def load_data(file_path, N):
    data = pd.read_csv(file_path)
    P = data.T
    P.fillna(-100, inplace=True)
    P[P <= 0] = np.nan

    def get_col_mean(x):
        return np.mean(x)

    P['mean'] = P.apply(lambda x: get_col_mean(x), axis=1)
    values = dict()
    for index in P.index:
        values[index] = int(P.loc[index, 'mean'])
    data[data < 0] = np.nan
    data.fillna(values, inplace=True)
    n_data = data.sample(n=N, axis=1, random_state=2020)
    P = data.T
    return n_data, np.array(P, dtype=int)


def problem3(N, seed, result_path):
    random.seed(seed)
    np.random.seed(seed)
    data, time_matrix = load_data('dataset.csv', N)
    job_indexs = list()
    for column in data.columns:
        job_indexs.append(int(column[1:]) - 1)
    evaluate_num = 1000 * N
    num = 0
    best_obj = np.inf
    best_seq = np.zeros([N])
    start = time.time()
    while num < evaluate_num:
        new_seq = RandomSearch(job_indexs)
        obj = make_span(new_seq, time_matrix)
        if obj < best_obj:
            best_obj = obj
            best_seq = np.array(new_seq, dtype=int)
        num += 1
    end = time.time()
    using_time = end - start
    checkpoint = pd.read_csv(result_path)
    checkpoint = checkpoint.append({'data_size': N,
                                    'seed': seed,
                                    'best_obj': best_obj,
                                    'time:': using_time}, ignore_index=True)
    checkpoint.to_csv(result_path, index=False)

    return best_seq, best_obj


def run_problem3():
    df = pd.DataFrame(columns=['data_size', 'seed', 'best_obj', 'time'])
    df.to_csv('problem3.csv', index=False)
    for num in range(30):
        print("==========num:{}=============".format(num))
        best_seq, best_obj = problem3(100, 2020 + num, 'problem3.csv')
        print("==========best_obj:{}========".format(best_obj))


def run_problem4():
    df = pd.DataFrame(columns=['data_size', 'seed', 'best_obj', 'time'])
    df.to_csv('problem4.csv', index=False)
    for num in range(30):
        N = 10
        print("==========num:{}=============".format(num))
        data, time_matrix = load_data('dataset.csv', N)
        job_indexs = list()
        for column in data.columns:
            job_indexs.append(int(column[1:]) - 1)
        start = time.time()
        job_indexs = np.array(job_indexs)
        best_seq, best_obj = GeneticAlgorithm(job_indexs, time_matrix, seed=2020 + num,params=None)
        end = time.time()
        using_time = end - start
        res = pd.read_csv("problem4.csv")
        res = res.append({'data_size': N,
                          'seed': 2020 + num,
                          'best_obj': best_seq[-1][1],
                          'time': using_time}, ignore_index=True)
        res.to_csv("problem4.csv", index=False)
        print("==========best_obj:{}========".format(best_seq[-1][1]))


def run_problem5():
    df = pd.DataFrame(columns=['data_size', 'seed',
                               'P', 'p_c', 'p_m_init', 'D',
                               'best_obj', 'time'])
    df.to_csv('problem5.csv', index=False)
    P = [5, 10, 20, 50, 100]
    p_c = [0.0, 0.5, 0.7, 0.9]
    p_m = [0.0, 0.2, 0.4, 0.6, 1.0]
    D = [0.0, 0.5, 1.0]
    param_dict = {'P': 30,
                  'p_c':1.0,
                  'p_m':0.8,
                  'D':0.95}
    for p in P:
        param_dict['P'] = p
        for data_size in [10, 50, 100]:
            data, time_matrix = load_data('dataset.csv', data_size)
            job_indexs = list()
            for column in data.columns:
                job_indexs.append(int(column[1:]) - 1)
            start = time.time()
            job_indexs = np.array(job_indexs)
            best_seq, best_obj = GeneticAlgorithm(job_indexs, time_matrix,
                                                  params=param_dict,
                                                  seed=2020)
            end = time.time()
            using_time = end - start
            res = pd.read_csv("problem5.csv")
            res = res.append({'data_size': data_size,
                              'seed': 2020,
                              'P': param_dict['P'],
                              'p_c': param_dict['p_c'],
                              'p_m_init': param_dict['p_m'],
                              'D': param_dict['D'],
                              'best_obj': best_seq[-1][1],
                              'time': using_time}, ignore_index=True)
            res.to_csv("problem5.csv", index=False)
            print("==========best_obj:{}========".format(best_seq[-1][1]))

if __name__ == '__main__':
    # run_problem3()
    run_problem4()
