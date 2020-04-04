import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB


n=10

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
    return dist

def build_model(dist):
    m = gp.Model()
    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, lb=0, ub=1, name='x')
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
    m.update()
    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)
    m.write('tsp.lp')
    return m,vars


def solve_model(m, vars):
    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)
    #m.optimize()

    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n
    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.objVal)
    print('')


def solve_model1(m,vars):
    m.optimize()
    vals = m.getAttr('x',vars)
    print(len(vals.keys()))
    for i,j in vars.keys():
        if vars[i,j].x > 0 and i > j:
            print("(",i,",",j,"):",vars[i,j].x)





if __name__ == "__main__":
    dist = creat_data(False)
    model,vars = build_model(dist)
    #solve_model(model,vars)
    solve_model1(model,vars)