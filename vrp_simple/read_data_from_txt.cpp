//
//  read_data_from_txt.cpp
//  OR
//
//  Created by libaihe on 2020/04/07.
//  Copyright © 2019 libaihe. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <cmath>

#include "read_data_from_txt.hpp"

//using namespace 
using namespace std;


void read_data_from_txt(std::string data_path,VRP::VRP_INPUT *input)
{
    
    ifstream in(data_path,ios::in);
    string line;
    int num=0;
    int begin_index=0;
    int end_index=0;
    VRP::Task task[100]={0};
    if(in)
    {
        //read first line
        getline(in, line);
        for(int i=0;i<line.size();i++)
        {
            if(line.at(i)==',')
            {
                num++;
                if(num == 3)
                {
                    end_index = i;
                    input->depot.id = 0;
                    //std::cout<<line.substr(begin_index + 1,end_index - begin_index);
                    input->depot.x = atof(line.substr(begin_index + 1,end_index-begin_index).c_str());
                }
                if(num == 4)
                {
                    end_index = i;
                    //std::cout<<line.substr(begin_index + 1,end_index - begin_index);
                    input->depot.y = atof(line.substr(begin_index + 1,end_index - begin_index).c_str());
                    break;
                }
                begin_index = i;
            }
        }
        num=-1;
        while(getline(in,line))
        {
            int count=0;
            num++;
            begin_index = 0;
            for(int i=0;i<line.size();i++)
            {
                if(line.at(i) == ',')
                {
                    count++;
                    end_index=i;
                    if(count == 2)
                    {
                        task[num].id = atoi( line.substr(begin_index,end_index-begin_index).c_str());
                    }
                    else if(count == 3)
                    {
                        task[num].start.x = atoi( line.substr(begin_index,end_index - begin_index).c_str());
                    }
                    else if(count == 4)
                    {
                        task[num].start.y = atoi( line.substr(begin_index,end_index - begin_index).c_str());
                    }
                    else if(count == 5)
                    {
                        task[num].end.x = atoi( line.substr(begin_index,end_index - begin_index).c_str());
                    }
                    else if(count == 6)
                    {
                        task[num].end.y = atoi( line.substr(begin_index,end_index - begin_index).c_str());
                    }
                    else if(count == 7)
                    {
                        task[num].type = atoi( line.substr(begin_index,end_index - begin_index).c_str());
                    }
                    begin_index = i+1;
                }
            }
            end_index = (int)line.size();
            task[num].stay_time = atoi( line.substr(begin_index,end_index - begin_index).c_str());
            input->task_list[num] = task[num];
        }
    }
    in.close();
}

