import pandas as pd
import numpy as np
import gurobipy as gb
from gurobipy import GRB


def Kantorovich_model(data):
    L = 32
    M = data['数量'].sum()
    M = 5000
    print(M)
    e = gb.Env()
    e.setParam('TimeLimit', 5 * 60)
    e.setParam('MIPGap', 0.02)

    model = gb.Model("Kantorovich_model", env=e)
    c = [1 for _ in range(M)]
    Y = model.addVars(len(c), obj=c, vtype=GRB.BINARY, name="y")
    item = {}
    index = 0
    for i in range(len(data)):
        for j in range(data.loc[i, "数量"]):
            item[index] = data.loc[i, "长度"]
            index += 1
    x = {(i, j): 1 for i in range(len(item)) for j in range(M)}
    X = model.addVars(x.keys(), vtype=GRB.BINARY, name="x")
    model.addConstrs((X.sum(i, '*') == 1 for i in range(len(item))), name="con1")
    model.addConstrs(((sum(item[i] * X[i, j] for i in range(len(item))) <= L * Y[j]) for j in range(M)), name="con2")
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print("object = ", model.objVal)
        vals = model.getAttr('x', X)
        selected = gb.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0)
        y = model.getAttr('x', Y)
        print("Print box used info,the First box is zero.")
        for key in y.keys():
            if y[key] > 0.5:
                print("box {} is used".format(key))
        print("Print all pack info.Frist item is zero First box is zero")
        for item in selected:
            print("item:{} in box:{}".format(item[0], item[1]))
    else:
        print("模型无解")
    model.write("Kantorovich_model.lp")


def creat_one_box(data):
    L = 32
    one_box = dict()
    item_lens = dict()
    for index in range(len(data)):
        item_lens[index] = data.loc[index, "长度"]
    count = 0
    # 每个长度占个箱子
    for index in range(len(data)):
        num = np.floor(L / item_lens[index])
        one_box[(index, count)] = num
        count += 1
    # 考虑箱子 A + B 组合情况
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data.loc[i, "长度"] + data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 1
                one_box[(j, count)] = 1
                count += 1
    # 考虑 2*A + B 组合的情况 和 A + 2 * B组合的情况
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if 6 * data.loc[i, "长度"] + 6 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 6
                one_box[(j, count)] = 6
                count += 1
                continue
            if 6 * data.loc[i, "长度"] + 5 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 6
                one_box[(j, count)] = 5
                count += 1
                continue
            if 5 * data.loc[i, "长度"] + 4 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 5
                one_box[(j, count)] = 4
                count += 1
                continue
            if 5 * data.loc[i, "长度"] + 5 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 5
                one_box[(j, count)] = 5
                count += 1
                continue
            if 4 * data.loc[i, "长度"] + 4 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 4
                one_box[(j, count)] = 4
                count += 1
                continue
            if 4 * data.loc[i, "长度"] + 3 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 4
                one_box[(j, count)] = 3
                count += 1
                continue
            if 3 * data.loc[i, "长度"] + 3 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 3
                one_box[(j, count)] = 3
                count += 1
                continue
            if 3 * data.loc[i, "长度"] + 4 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 3
                one_box[(j, count)] = 4
                count += 1
                continue
            if 4 * data.loc[i, "长度"] + 2 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 4
                one_box[(j, count)] = 2
                count += 1
                continue
            if 4 * data.loc[i, "长度"] + 1 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 4
                one_box[(j, count)] = 1
                count += 1
                continue
            if 3 * data.loc[i, "长度"] + 2 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 3
                one_box[(j, count)] = 2
                count += 1
                continue
            if 3 * data.loc[i, "长度"] + 1 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 3
                one_box[(j, count)] = 1
                count += 1
                continue
            if 2 * data.loc[i, "长度"] + 2 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 2
                one_box[(j, count)] = 2
                count += 1
                continue
            if 2 * data.loc[i, "长度"] + 1 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 2
                one_box[(j, count)] = 1
                count += 1
                continue
            if 1 * data.loc[i, "长度"] + 1 * data.loc[j, "长度"] <= L:
                one_box[(i, count)] = 1
                one_box[(j, count)] = 1
                count += 1
                continue
    print("one_box shape:{}".format(len(one_box.keys())))
    return one_box


def Gilmore_Gomory_model(data, one_box_dict):
    L = 32
    box_list_len = len(one_box_dict.keys())
    model = gb.Model("Gilmore_Gomory_model")
    c = [1 for _ in range(box_list_len)]
    X = model.addVars(len(c), obj=c, lb=0, vtype=GRB.INTEGER, name="x")
    b = [data.loc[index, "数量"] for index in range(len(data))]
    model.addConstrs((sum([X[j] * one_box_dict[(i, j)] if (i, j) in one_box_dict.keys() else 0 for j in range(box_list_len)]) == b[i] for i in range(len(data))), name='"con1')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print("object = ", model.objVal)
        x = model.getAttr('x', X)
        print("Print pattern used info,the First box is zero.")
        for key in x.keys():
            if x[key] > 0.5:
                print("pattern {} is used num:{}".format(key, x[key]))
    else:
        print("模型无解")
    model.write("Gilmore_Gomory_model.lp")


if __name__ == "__main__":
    # bin len:32
    data = pd.read_csv('1D-bin-big.csv')
    Kantorovich_model(data)
    # one_box_dict = creat_one_box(data)
    # Gilmore_Gomory_model(data, one_box_dict)