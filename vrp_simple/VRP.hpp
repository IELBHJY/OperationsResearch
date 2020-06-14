//
//  VRP.hpp
//  OR
//
//  Created by libaihe on 2020/5/16.
//  Copyright © 2020 libaihe. All rights reserved.
//

#ifndef VRP_hpp
#define VRP_hpp
#include <iostream>
#include "TSP.h"
#include <math.h>
#include <stdlib.h>
#include <vector>

#define MAX_TASK_NUM_ONE_VEHICLE (50)
#define MAX_VEHICLE_NUM (100)
#define MAX_VRP_TASK_NUM (100)
#define MAX_PENTY_VALUE (180000)
#define MAX_WORK_TIME (3600 * 8)
#define MAX_PARENT_NUM (1000)
#define MAX_ITERATION_NUM (500)


class VRP
{
    double dist_one_task[MAX_VRP_TASK_NUM];
    double dist_between_tasks[MAX_VRP_TASK_NUM][MAX_VRP_TASK_NUM];
    double cross_rate=0.5;
    double mutate_rate=0.4;
    double inherit_rate=0.1;
    
public:
    typedef struct Sol
    {
        int vehicle_id;
        int type;
        int task_num;
        double finish_time;
        int tasks[MAX_TASK_NUM_ONE_VEHICLE];
    }Sol;
    
    typedef struct Task
    {
        int id;
        int type;
        int stay_time;
        Point start;
        Point end;
    }Task;
    
    typedef struct VRP_INPUT
    {
        int task_num;
        int vehicle_num;
        std::string data_path;
        Point depot;
        Task *task_list;
    }VRP_INPUT;
    
    typedef struct VRP_OUTPUT
    {
        int num;//车数
        Sol sol[MAX_VEHICLE_NUM];
    }VRP_OUTPUT;
    
    void cal_dist_matrix(VRP_INPUT *input);
    
    int cal_dis(Point x1,Point x2);
    
    void print_solution(Sol *sol);
    
    double random(double start, double end);
    
    void creat_inital_solution(VRP_INPUT *input, Sol* sol);
    
    void creat_inital_solution(VRP_INPUT *input,Sol *sol,std::vector<int> seq);
    
    void creat_inital_population(VRP_INPUT *input,Sol *sol[]);
    
    double cal_cost_one_vechile(VRP_INPUT *data,Sol sol);
    
    double cal_sol_cost(VRP_INPUT *data,Sol *sol);
    
    void cal_population_cost(VRP_INPUT *data, Sol *sol[],double *costs);
    
    int choose(const double *costs,int size);
    
    void crossover(Sol *sol[],Sol *children[], const double *costs,int start_index,int end_index);
    
    void mutation(Sol *sol[],Sol *children[],const double *costs,int start_index,int end_index);
    
    void elitism(Sol *sol[],Sol *children[],const double *costs,int start_index,int end_index);
    
    void inhert(Sol *sol[],Sol *chileren[],const double *costs,int start_index);
    
    void memcpy_solution(Sol *sol,Sol *children);
    
    void check_sol(VRP_INPUT *input,Sol *sol,int iter,int index);
    
    void write_result(Sol *sol);
    
    void Solve(VRP_INPUT *input,VRP_OUTPUT output);
};
#endif /* VRP_hpp */
