//
//  VRP.cpp
//  OR
//
//  Created by libaihe on 2020/5/16.
//  Copyright © 2020 libaihe. All rights reserved.
//

#include "VRP.hpp"
#include <math.h>
#include <vector>
#include <algorithm>
#include <fstream>


void VRP::cal_dist_matrix(VRP_INPUT *data)
{
    //cal task start->>end dist
    for(int i=0;i<data->task_num;i++)
    {
        dist_one_task[i] = sqrt(pow(abs(data->task_list[i].start.x - data->task_list[i].end.x),2) +
                                pow(abs(data->task_list[i].start.y - data->task_list[i].end.y),2));
        //std::cout<<data->task_list[i].id<<","<<dist_one_task[i]<<std::endl;
    }
    
    //cal task1 start -->> task2 start dist
    for(int i=0;i<data->task_num;i++)
    {
        for(int j=0;j<data->task_num;j++)
        {
            if(i==j){
                dist_between_tasks[i][j] = 0;
            }
            else{
                dist_between_tasks[i][j] = sqrt(pow(abs(data->task_list[i].start.x - data->task_list[j].start.x),2) +
                                               pow(abs(data->task_list[i].start.y - data->task_list[j].start.y),2));
                //std::cout<<data->task_list[i].id<<","<<data->task_list[j].id<<","<<dist_between_tasks[i][j]<<std::endl;
            }
        }
    }
    std::cout<<"cal dist success!"<<std::endl;
}

int VRP::cal_dis(Point x1, Point x2)
{
    return sqrt(pow(abs(x1.x - x2.x),2) + pow(abs(x1.y - x2.y), 2));
}

void VRP::print_solution(Sol *sol)
{
    std::cout<<"********************"<<std::endl;
    for(int i=0;i<MAX_VEHICLE_NUM;i++)
    {
        if(sol[i].vehicle_id <= 0)
        {
            continue;
        }
        std::cout<<sol[i].vehicle_id<<":";
        for(int j=0;j<sol[i].task_num;j++)
        {
            std::cout<<sol[i].tasks[j]<<",";
        }
        std::cout<<std::endl;
    }
    std::cout<<"********************"<<std::endl;
}

//贪心法生成初始解
void VRP::creat_inital_solution(VRP_INPUT *data,Sol* sol)
{
    for(int i=0;i<data->task_num;i++)
    {
        for(int j=0;j<MAX_VEHICLE_NUM;j++)
        {
            if(sol[j].finish_time < MAX_WORK_TIME)
            {
                if(sol[j].vehicle_id <= 0){
                    //车上没有任务
                    sol[j].vehicle_id = j+1;
                    sol[j].type = data->task_list[i].type;
                    sol[j].task_num=1;
                    sol[j].tasks[0] = i;
                    sol[j].finish_time = (cal_dis(data->depot, data->task_list[i].start) +
                                         dist_one_task[i] + data->task_list[i].stay_time + dist_one_task[i]);
                    break;
                }
                else if(sol[j].type == data->task_list[i].type){
                    int last_task_index = sol[j].tasks[sol[j].task_num-1];
                    double temp =(dist_between_tasks[last_task_index][i] + dist_one_task[i] +
                                  data->task_list[i].stay_time + dist_one_task[i]);
                    if(sol[j].finish_time + temp > MAX_WORK_TIME)
                    {
                        continue;
                    }
                    sol[j].tasks[sol[j].task_num] =i;
                    sol[j].finish_time += (dist_between_tasks[last_task_index][i] + dist_one_task[i] +
                                          data->task_list[i].stay_time + dist_one_task[i]);
                    sol[j].task_num++;
                    break;
                }
            }
        }
    }
    print_solution(sol);
    return;
}

void VRP::creat_inital_solution(VRP_INPUT *data, Sol *sol,std::vector<int> seq)
{
    int index=-1;
    std::vector<int> first_vec;
    std::vector<int> second_vec;
    for(int i=0;i< MAX_VEHICLE_NUM;i++)
    {
        first_vec.push_back(i + 1);
        second_vec.push_back(i + MAX_VEHICLE_NUM + 1);
    }
    for(int i=0;i<seq.size();i++)
    {
        index = seq.at(i);
        for(int j=0;j<MAX_VEHICLE_NUM;j++)
        {
            if(sol[j].finish_time < MAX_WORK_TIME)
            {
                if(sol[j].vehicle_id <= 0){
                    //车上没有任务
                    sol[j].type = data->task_list[index].type;
                    if(sol[j].type == 1)
                    {
                        sol[j].vehicle_id = first_vec.at(j);
                    }
                    else if(sol[j].type == 2)
                    {
                        sol[j].vehicle_id = second_vec.at(j);
                    }
                    sol[j].task_num=1;
                    sol[j].tasks[0] = index;
                    sol[j].finish_time = (cal_dis(data->depot, data->task_list[index].start) +
                                          dist_one_task[index] + data->task_list[index].stay_time + dist_one_task[index]);
                    break;
                }
                else if(sol[j].type == data->task_list[index].type){
                    int last_task_index = sol[j].tasks[sol[j].task_num-1];
                    double temp =(dist_between_tasks[last_task_index][index] + dist_one_task[index] +
                                  data->task_list[index].stay_time + dist_one_task[index]);
                    if(sol[j].finish_time + temp > MAX_WORK_TIME)
                    {
                        continue;
                    }
                    if(sol[j].task_num >= 6)
                    {
                        continue;
                    }
                    sol[j].tasks[sol[j].task_num] =index;
                    sol[j].finish_time += (dist_between_tasks[last_task_index][index] + dist_one_task[index] +
                                           data->task_list[index].stay_time + dist_one_task[index]);
                    sol[j].task_num++;
                    break;
                }
            }
        }
    }
    //print_solution(sol);
    return;
}

double VRP::random(double start, double end)
{
    return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

void VRP::creat_inital_population(VRP_INPUT *data, Sol *sol[MAX_PARENT_NUM])
{
    std::vector<int> seq;
    for(int i=0;i<data->task_num;i++)
    {
        seq.push_back(i);
    }
    for(int i=0;i<MAX_PARENT_NUM;i++)
    {
        random_shuffle(seq.begin(), seq.end());
        creat_inital_solution(data, sol[i], seq);
        check_sol(data, sol[i], 0, i);
    }
}

//计算一辆车的目标函数值（行驶距离总和），并check是否满足类型约束，以及工作时间约束
double VRP::cal_cost_one_vechile(VRP_INPUT *data,Sol sol)
{
    double cost=0.0;
    double finish_time=0.0;
    int i;
    if(sol.task_num <= 0)
    {
        return cost;
    }
    int first_task_index = sol.tasks[0];
    cost +=cal_dis(data->depot, data->task_list[first_task_index].start);//去家里
    finish_time +=cal_dis(data->depot, data->task_list[first_task_index].start);
    cost +=dist_one_task[first_task_index];//去目的地
    cost +=dist_one_task[first_task_index];//回家里
    finish_time +=dist_one_task[first_task_index] + data->task_list[first_task_index].stay_time + dist_one_task[first_task_index];
    for(i=0;i<sol.task_num - 1;i++)
    {
        cost+=dist_between_tasks[sol.tasks[i]][sol.tasks[i+1]];
        finish_time +=dist_between_tasks[sol.tasks[i]][sol.tasks[i+1]];
        cost +=dist_one_task[sol.tasks[i+1]];
        finish_time +=dist_one_task[sol.tasks[i+1]];
        finish_time +=data->task_list[sol.tasks[i+1]].stay_time;
        cost +=dist_one_task[sol.tasks[i+1]];
        finish_time +=dist_one_task[sol.tasks[i+1]];
    }
    //加上回到depot的距离
    cost += cal_dis(data->task_list[i].start, data->depot);
    if(finish_time > MAX_WORK_TIME)
    {
        cost += MAX_PENTY_VALUE;
        std::cout<<"penty success!"<<std::endl;
    }
    return cost;
}

double VRP::cal_sol_cost(VRP_INPUT *data, Sol *sol)
{
    double cost=0.0;
    for(int i=0;i<MAX_VEHICLE_NUM;i++)
    {
        if(sol[i].vehicle_id <= 0)
        {
            continue;
        }
        cost +=cal_cost_one_vechile(data, sol[i]);
    }
    return cost;
}

void VRP::cal_population_cost(VRP_INPUT *data, Sol *sol[],double *costs)
{
    for(int i=0;i<MAX_PARENT_NUM;i++)
    {
        costs[i] = cal_sol_cost(data, sol[i]);
        //std::cout<<"Function("<< i <<"):"<<costs[i] << std::endl;
    }
}

int VRP::choose(const double *costs,int size)
{
    int index=-1;
    double sum = 0.0;
    double value=0.0;
    for(int i=0;i<size;i++)
    {
        sum+=costs[i];
    }
    value = random(0, sum);
    sum=0.0;
    for(int i=0;i<size;i++)
    {
        sum+=costs[i];
        if(sum > value)
        {
            index=i;
            break;
        }
    }
    return index;
}

void print_sol(VRP::Sol *sol,int type)
{
    for(int i=0;i<MAX_VEHICLE_NUM;i++)
    {
        if(sol[i].vehicle_id <= 0 || (sol[i].type != type && type != 0))
        {
            continue;
        }
        std::cout<<sol[i].vehicle_id<<":";
        for(int j=0;j<sol[i].task_num;j++)
        {
            std::cout<<sol[i].tasks[j]<<",";
        }
        std::cout<<std::endl;
    }
}


void creat_vector_by_order(VRP::Sol *sol, int type, std::vector<int> &task,std::vector<int> &vechile)
{
    for(int j=0;j<MAX_VEHICLE_NUM;j++)
    {
        if(sol[j].vehicle_id <= 0 || sol[j].type!=type)
        {
            continue;
        }
        for(int z=0;z<sol[j].task_num;z++)
        {
            if(task.size()==0)
            {
                task.push_back(sol[j].tasks[z]);
                vechile.push_back(sol[j].vehicle_id);
                continue;
            }
            auto index=0;
            for(index=0;index<task.size();index++)
            {
                if(sol[j].tasks[z] < task.at(index))
                {
                    task.insert(task.begin() + index, sol[j].tasks[z]);
                    vechile.insert(vechile.begin()+index, sol[j].vehicle_id);
                    break;
                }
            }
            if(index == task.size())
            {
                task.push_back(sol[j].tasks[z]);
                vechile.push_back(sol[j].vehicle_id);
            }
        }
    }
}

void creat_vector(VRP::Sol *sol, int type, std::vector<int> &task,std::vector<int> &vechile)
{
    for(int j=0;j<MAX_VEHICLE_NUM;j++)
    {
        if(sol[j].vehicle_id <= 0 || sol[j].type!=type)
        {
            continue;
        }
        for(int z=0;z<sol[j].task_num;z++)
        {
            task.push_back(sol[j].tasks[z]);
            vechile.push_back(sol[j].vehicle_id);
        }
    }
}

void creat_sol(VRP::Sol *sol,int type,std::vector<int> task,std::vector<int> vechile)
{
    if(task.size() != vechile.size())
    {
        assert(0);
    }
    int task_offset=-1;
    for(int i=0;i<task.size();i++)
    {
        task_offset = task.at(i);
        for(int j=0;j<MAX_VEHICLE_NUM;j++)
        {
            if(sol[j].task_num == 0 && sol[j].vehicle_id <= 0)
            {
                sol[j].vehicle_id = vechile.at(i);
                sol[j].type = type;
                sol[j].tasks[0] = task_offset;
                sol[j].task_num++;
                break;
            }
            else if(sol[j].vehicle_id > 0 &&
                    sol[j].task_num > 0 &&
                    sol[j].vehicle_id == vechile.at(i)){
                int task_index = sol[j].task_num;
                sol[j].tasks[task_index] = task_offset;
                sol[j].task_num++;
                break;
            }
        }
    }
}


void print_vector(std::vector<int> task,std::vector<int> vechile)
{
    for(auto index = 0;index< task.size();index++)
    {
        std::cout<<task.at(index)<<",";
    }
    std::cout<<std::endl;
    for(auto index = 0;index< vechile.size();index++)
    {
        std::cout<<vechile.at(index)<<",";
    }
    std::cout<<std::endl;
}

void creat_cross_vector(std::vector<int> &task1,std::vector<int> &vec1,
                        std::vector<int> &task2,std::vector<int> &vec2,
                        std::vector<int> &new_task1,std::vector<int> &new_vec1,
                        std::vector<int> &new_task2,std::vector<int> &new_vec2,int pos)
{
    if(task1.size() != vec1.size() ||
       task2.size()!=vec2.size() ||
       task1.size()!=task2.size() ||
       pos < 0 || pos >= task1.size())
    {
        std::cout<<"creat_cross_vector"<<std::endl;
        assert(0);
    }
    int size = task1.size();
    for(int i=0;i<pos;i++)
    {
        new_task1.push_back(task1.at(i));
        new_vec1.push_back(vec1.at(i));
        new_task2.push_back(task2.at(i));
        new_vec2.push_back(vec2.at(i));
    }
    for(int i=pos;i<size;i++)
    {
        new_task1.push_back(task2.at(i));
        new_vec1.push_back(vec2.at(i));
        new_task2.push_back(task1.at(i));
        new_vec2.push_back(vec1.at(i));
    }
    return;
}

void clear_vector(std::vector<int> &vector)
{
    vector.clear();
}

void VRP::crossover(Sol *sol[],Sol *children[], const double *costs,int start_index,int end_index)
{
    using namespace std;
    int pos1=-1,pos2=-1;
    Sol *c1 = nullptr;
    Sol *c2 = nullptr;
    vector<int> task11;
    vector<int> vechile11;
    vector<int> task21;
    vector<int> vechile21;
    vector<int> task12;
    vector<int> vechile12;
    vector<int> task22;
    vector<int> vechile22;
    vector<int> new_task11;
    vector<int> new_vechile11;
    vector<int> new_task21;
    vector<int> new_vechile21;
    vector<int> new_task12;
    vector<int> new_vechile12;
    vector<int> new_task22;
    vector<int> new_vechile22;
    int pos=-1;
    for(int i=start_index;i< end_index;i+=2)
    {
        clear_vector(task11);
        clear_vector(vechile11);
        clear_vector(task12);
        clear_vector(vechile12);
        clear_vector(task21);
        clear_vector(vechile21);
        clear_vector(task22);
        clear_vector(vechile22);
        
        clear_vector(new_task11);
        clear_vector(new_vechile11);
        clear_vector(new_task12);
        clear_vector(new_vechile12);
        clear_vector(new_task21);
        clear_vector(new_vechile21);
        clear_vector(new_task22);
        clear_vector(new_vechile22);
        
        pos1 = choose(costs, MAX_PARENT_NUM);
        pos2 = choose(costs, MAX_PARENT_NUM);
        //std::cout<<"("<<pos1<<","<<pos2<<")"<<std::endl;
        c1 = sol[pos1];
        c2 = sol[pos2];
        //print_sol(c1,1);
        creat_vector_by_order(c1, 1, task11, vechile11);
        //print_vector(task11,vechile11);
        //print_sol(c2,1);
        creat_vector_by_order(c2, 1, task21, vechile21);
        //print_vector(task21,vechile21);
        if(task21.size()!=task11.size())
        {
            assert(0);
        }
        pos = random(0, task11.size());
        //std::cout<<"pos="<<pos<<std::endl;
        creat_cross_vector(task11, vechile11, task21, vechile21, new_task11, new_vechile11, new_task21, new_vechile21, pos);
        //print_vector(new_task11, new_vechile11);
        //print_vector(new_task21, new_vechile21);
        
        
        //print_sol(c1,2);
        creat_vector_by_order(c1, 2, task12, vechile12);
        //print_vector(task12,vechile12);
        //print_sol(c2,2);
        creat_vector_by_order(c2, 2, task22, vechile22);
        //print_vector(task22,vechile22);
        if(task22.size()!=task12.size())
        {
            assert(0);
        }
        pos = random(0, task12.size());
        //std::cout<<"pos="<<pos<<std::endl;
        creat_cross_vector(task12, vechile12, task22, vechile22, new_task12, new_vechile12, new_task22, new_vechile22, pos);
        //print_vector(new_task12, new_vechile12);
        //print_vector(new_task22, new_vechile22);
        
        creat_sol(children[i], 1, task11, vechile11);
        creat_sol(children[i], 2, task12, vechile12);
        creat_sol(children[i + 1], 1, task21, vechile21);
        creat_sol(children[i + 1], 2, task22, vechile22);
    }
}


void VRP::memcpy_solution(Sol *sol,Sol *children)
{
    for(int i=0;i<MAX_VEHICLE_NUM;i++)
    {
        (children + i)->vehicle_id = sol[i].vehicle_id;
        children[i].type = sol[i].type;
        children[i].finish_time = sol[i].finish_time;
        children[i].task_num = sol[i].task_num;
        for(int j=0;j<sol[i].task_num;j++)
        {
            children[i].tasks[j] = sol[i].tasks[j];
        }
    }
}

void VRP::mutation(Sol *sol[],Sol *children[], const double *costs,int start_index,int end_index)
{
    int index=-1;
    int pos=-1;
    int pos1=-1;
    int pos2=-1;
    int count=0;
    Sol *c=nullptr;
    for(int i=start_index;i<end_index;i++)
    {
        //std::cout<<"index="<<i<<std::endl;
        count=0;
        index = choose(costs, MAX_PARENT_NUM);
        //std::cout<<"pos="<<index<<std::endl;
        c = sol[index];
        for(int j=0;j<MAX_VEHICLE_NUM;j++)
        {
            if(c[j].vehicle_id > 0)
            {
                count++;
            }
        }
        pos = random(0, count);
        //std::cout<<"pos="<<pos<<std::endl;
        if(c[pos].vehicle_id <= 0 || c[pos].task_num <= 0)
        {
            std::cout<<"mutation"<<std::endl;
            assert(0);
        }
        if(c[pos].task_num == 1)
        {
            memcpy_solution(c, children[i]);
            continue;
        }
        pos1 = random(0, c[pos].task_num);
        pos2 = random(0, c[pos].task_num);
        while (pos2 == pos1) {
            pos2 = random(0, c[pos].task_num);
        }
        memcpy_solution(c, children[i]);
        int temp = children[i][pos].tasks[pos1];
        children[i][pos].tasks[pos1] = children[i][pos].tasks[pos2];
        children[i][pos].tasks[pos2] = temp;
    }
}

double get_best_solution(double *costs,int size,int *best_index)
{
    double temp=costs[0];
    int index=0;
    *best_index=0;
    for(int i=1;i<size;i++)
    {
        if(costs[i] < temp)
        {
            temp = costs[i];
            index = i;
            *best_index = i;
        }
    }
    //std::cout<<"best cost="<<temp<<",index="<<index<<std::endl;
    return temp;
}

bool compare(double a,double b)
{
    return a > b;
}

void VRP::elitism(Sol *sol[],Sol *children[],const double *costs,int start_index,int end_index)
{
    int num=0;
    int *indexs = (int *)malloc(sizeof(int) * (end_index - start_index));
    for(int i=0;i<MAX_PARENT_NUM;i++)
    {
        if(num < end_index - start_index)
        {
            indexs[num] = i;
            num++;
        }
        else
        {
            int max_index=0;
            double max_cost=costs[indexs[0]];
            for(int j=1;j<end_index - start_index;j++)
            {
                if(costs[indexs[j]] > max_cost)
                {
                    max_index=j;
                    max_cost = costs[indexs[j]];
                }
            }
            if(costs[i] < max_cost)
            {
                indexs[max_index] = i;
            }
        }
    }
    
    for(int i=start_index;i<end_index;i++)
    {
        if(indexs[i - start_index] < 0)
        {
            assert(0);
        }
        //std::cout<<costs[indexs[i-start_index]]<<std::endl;
        memcpy(children[i], sol[indexs[i-start_index]], sizeof(Sol) * MAX_VEHICLE_NUM);
    }
    free(indexs);
}


void VRP::check_sol(VRP_INPUT *data, Sol *sol,int iter,int index)
{
    int *indexs= (int *)malloc(sizeof(int) * data->task_num);
    memset(indexs, 0, sizeof(int) * data->task_num);
    for(int i=0;i<MAX_VEHICLE_NUM;i++)
    {
        if(sol[i].vehicle_id <= 0)
        {
            continue;
        }
        for(int j=0;j<sol[i].task_num;j++)
        {
            if(sol[i].type != data->task_list[sol[i].tasks[j]].type)
            {
                std::cout<<"("<<iter<<","<<index<<","<<sol[i].vehicle_id<<")"<<std::endl;
                assert(0);
            }
            if(indexs[sol[i].tasks[j]] == 0)
            {
                indexs[sol[i].tasks[j]] = 1;
            }else if(indexs[sol[i].tasks[j]] == 1)
            {
                std::cout<<"("<<iter<<","<<index<<","<<sol[i].vehicle_id<<")"<<std::endl;
                assert(0);
            }
        }
    }
    for(int i=0;i<data->task_num;i++)
    {
        if(indexs[i] == 0)
        {
            std::cout<<"("<<iter<<","<<index<<","<<sol[i].vehicle_id<<")"<<std::endl;
            assert(0);
        }
    }
    free(indexs);
}

void VRP::write_result(Sol *sol)
{
    using namespace std;
    string file_path="data/result.txt";
    ofstream out_file(file_path,ios::in);
    if(out_file)
    {
        for(int i=0;i<MAX_VEHICLE_NUM;i++)
        {
            if(sol[i].vehicle_id <= 0)
            {
                continue;
            }
            out_file<<"vehcile_id("<<(i+1)<<"),type="<<sol[i].type<<" task_seq(task_offset):";
            for(int j=0;j<sol[i].task_num;j++)
            {
                out_file<<"("<<sol[i].tasks[j]<<")";
            }
            out_file<<std::endl;
        }
    }
    out_file.close();
}

void VRP::Solve(VRP_INPUT *data, VRP_OUTPUT output)
{
    int best_index=0;
    double best_cost=0;
    Sol **sol = (Sol **)malloc(sizeof(Sol *) * MAX_PARENT_NUM);
    Sol **children=(Sol **)malloc(sizeof(Sol *) * MAX_PARENT_NUM);
    Sol *solution=(Sol *)malloc(sizeof(Sol) * MAX_VEHICLE_NUM);
    for(int i=0;i<MAX_PARENT_NUM;i++)
    {
        sol[i] = (Sol *)malloc(sizeof(Sol) * MAX_VEHICLE_NUM);
        memset(sol[i], 0, sizeof(Sol) * MAX_VEHICLE_NUM);
        children[i] = (Sol *)malloc(sizeof(Sol) * MAX_VEHICLE_NUM);
        memset(children[i], 0, sizeof(Sol) * MAX_VEHICLE_NUM);
    }
    double *costs=(double *)malloc(sizeof(double) * MAX_PARENT_NUM);
    memset(costs, 0, sizeof(double) * MAX_PARENT_NUM);
    srand(2020);
    (void)creat_inital_population(data, sol);
    (void)cal_population_cost(data, sol,costs);
    get_best_solution(costs, MAX_PARENT_NUM,&best_index);
    memcpy(solution, sol[best_index], sizeof(Sol) * MAX_VEHICLE_NUM);
    //print_sol(solution, 0);
    best_cost = costs[best_index];
    //print_sol(sol[best_index], 0);
    //轮盘赌选择下代交叉个体，变异个体，保留精英个体
    for(int iteration=0;iteration<MAX_ITERATION_NUM;iteration++)
    {
        //std::cout<<"iteration:"<<iteration<<std::endl;
        crossover(sol, children, costs, 0, (int)(MAX_PARENT_NUM * cross_rate));
        mutation(sol, children, costs, (int)(MAX_PARENT_NUM * cross_rate), (int)(MAX_PARENT_NUM * (cross_rate + mutate_rate)));
        elitism(sol, children, costs, (int)(MAX_PARENT_NUM * (cross_rate + mutate_rate)), MAX_PARENT_NUM);
        for(int i=0;i<MAX_PARENT_NUM;i++)
        {
            //std::cout<<"B"<<std::endl;
            check_sol(data, children[i], iteration, i);
        }
        
        memset(costs, 0, sizeof(double) * MAX_PARENT_NUM);
        (void)cal_population_cost(data, children, costs);
        get_best_solution(costs, MAX_PARENT_NUM,&best_index);
        //std::cout<<"best cost:"<<costs[best_index]<<std::endl;
        if(costs[best_index] < best_cost)
        {
            best_cost = costs[best_index];
            memcpy(solution, children[best_index], sizeof(Sol) * MAX_VEHICLE_NUM);
            std::cout<<"best cost:"<<best_cost<<std::endl;
        }
        for(int i=0;i<MAX_PARENT_NUM;i++)
        {
            memcpy(sol[i], children[i], sizeof(Sol) * MAX_VEHICLE_NUM);
            memset(children[i], 0, sizeof(Sol) * MAX_VEHICLE_NUM);
            //std::cout<<"A"<<std::endl;
            check_sol(data, sol[i], iteration, i);
        }
    }
    std::cout<<"over,cost ="<<best_cost<<std::endl;
    std::cout<<"write result to result.txt"<<std::endl;
    //print_sol(solution, 0);
    write_result(solution);
    free(sol);
    free(children);
    free(costs);
}
