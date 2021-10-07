import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from time import *

n=200

# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < n:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour)-1)



# Given a tuplelist of edges, find the shortest subtour
def subtour(edges):
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle

#Given a tuplelist of edges, find all the subtour
def find_all_subtour(edges):
    unvisited = list(range(n))
    cycles=[]
    while unvisited:
        current_cycle=[]
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            current_cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i,j in edges.select(current,'*') if j in unvisited]
        cycles.append(current_cycle)
    return cycles


def creat_data(show_plot):
    # Create n random points
    random.seed(1)
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]
    # Dictionary of Euclidean distance between each pair of points
    dist = {(i, j):
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n) for j in range(i)}
    if show_plot:
        import matplotlib.pyplot as plt
        plt.scatter([point[0] for point in points],[point[1] for point in points])
        plt.show()
    return points,dist


def plot_result(points, tour):
    import matplotlib.pyplot as plt
    plt.scatter([point[0] for point in points], [point[1] for point in points],color='blue')
    tour.append(tour[0])
    for index in range(len(tour)):
        if index == len(tour) - 1:
            break
        plt.plot(
            [points[tour[index]][0], points[tour[index+1]][0]],
            [points[tour[index]][1], points[tour[index + 1]][1]],
            color='red'
        )
    plt.savefig("Gurobi/pics/tsp_{}_result.png".format(n))
    plt.show()


def build_model(dist):
    m = gp.Model()
    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction
    m.modelSense = GRB.MINIMIZE
    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))
    # Add degree-2 constraint

    m.addConstrs((vars.sum(i, '*') == 2 for i in range(n)), "city")
    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)
    m.write('Gurobi/models/tsp.lp')
    return m, vars


def solve_model(m, vars):
    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    begin_time = time()
    m.optimize(subtourelim)
    end_time = time()
    print("run time of linear:", end_time - begin_time)
    #m.optimize()

    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n
    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.objVal)
    print('')
    return tour


def build_model_linear(dist):
    m = gp.Model()
    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.CONTINUOUS, name='x')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction
    m.modelSense = GRB.MINIMIZE

    m.addConstrs((vars.sum(i, '*') == 2 for i in range(n)), "city")
    m.addConstrs((vars[i,j] <= 1 for i,j in vars.keys()),"x")
    m.addConstrs((vars[i,j] >=0 for i,j in vars.keys()),"x")


    m.write('Gurobi/models/tsp_linear.lp')
    return m,vars

def solve_model_linear(m,vars):
    print("solving...")
    begin_time = time()
    m.optimize()
    end_time = time()
    print("run time of linear:",end_time-begin_time)
    # vals = m.getAttr('x',vars)
    # print(len(vals.keys()))
    # for i,j in vars.keys():
    #     if vars[i,j].x > 0 and i > j:
    #         print("(",i,",",j,"):",vars[i,j].x)
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    #tour = subtour(selected)
    tour = find_all_subtour(selected)
    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.objVal)
    print('')
    return tour





if __name__ == "__main__":
    points, dist = creat_data(False)
    #model,vars = build_model(dist)
    #tour = solve_model(model,vars)
    #plot_result(points, tour)
    model,vars = build_model_linear(dist)
    tour = solve_model_linear(model,vars)
    # print(tour)
    #plot_result(points,tour)