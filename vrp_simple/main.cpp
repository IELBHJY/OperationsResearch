//
//  main.cpp
//  OR
//
//  Created by libaihe on 2019/12/22.
//  Copyright © 2019 libaihe. All rights reserved.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include "input.hpp"

using namespace std;



int main(int argc, const char * argv[]) {
    // insert code here...
    //TreeNode *root = (TreeNode *)malloc(sizeof(TreeNode));
    //TreeNode *node1 = (TreeNode *)malloc(sizeof(TreeNode));
    //TreeNode *node2 = (TreeNode *)malloc(sizeof(TreeNode));
    //TreeNode *node3 = (TreeNode *)malloc(sizeof(TreeNode));
    //root->value =1;
    //root->parent = nullptr;
    //node1->value=2;
    //node1->left = nullptr;
    //node3->value = 4;
    //node3->left = nullptr;
    //node3->right = nullptr;
    //node1->right = node3;
    //node2->value = 3;
    //node2->left = node3;
    //node2->right = nullptr;
    //root->left = node1;
    //root->right = node2;
    //TreeNode root = creat_binary_tree(1);
    //add_left_child(&root, 2);
    //add_right_child(&root, 3);
    //std::cout<<root.value<<","<<root.left->value<<","<<root.right->value<<std::endl;
    //vector<int> res = getTreeValue(&root);
    //for(int i=0;i<res.size();i++)
    {
        //std::cout<<res[i]<<std::endl;
    }
    
    //int nums[4] = {2,3,-2,4};
    //vector<int> num;
    //for(int i=0;i<4;i++)
    //{
       // num.push_back(nums[i]);
    //}
    //int res = maxProduct(num);
    //std::cout<<res;
    //int res = find_k_value(nums, 10, 3);
    //std::cout<<res<<std::endl;
    //quick_sort(nums, 0, 9);
    //for(int i=0;i<10;i++)
    //{
        //std::cout<<nums[i]<<",";
    //}
    //"/Users/libaihe/Lab/Projects/cpp_codes/ORS/OR/OR/data/data.txt"
    //int nums[8] = {1,2,3,4,6};
    //cout << sizeof(nums) / sizeof(nums[0])<<endl;
    //string inputs;
    //getline(cin,inputs);
    //vector<int> res = split(inputs," ");
   // for(auto item : res)
    //{
        //cout<< item << endl;
    //}
    //int nums[11] = {10,8,9,7,6,5,5,4,3,2,1};
    //find2sum(nums, 11, 10);
    //MMD_INPUT mmd_input;
    //creat_input_mmd(&mmd_input,"/Users/libaihe/Lab/Projects/cpp_codes/ORS/OR/OR/data/mmd_data.txt");
    //MMD_Solution sol;
    //creat_inital_solution(&mmd_input,sol);
    //for(int iter=0;iter<20;iter++)
    //{
        //MMD_change_last_dispatch(&mmd_input, &sol);
    //}
    //std::cout<<"machine_1 finish time:"<<sol.machine[0].day<<","<<sol.machine[0].finish_time<<std::endl;
    //std::cout<<"machine_2 finish time:"<<sol.machine[1].day<<","<<sol.machine[1].finish_time<<std::endl;
    //std::cout<<"machine_3 finish time:"<<sol.machine[2].day<<","<<sol.machine[2].finish_time<<std::endl;
    //write2file(sol,"/Users/libaihe/Lab/Projects/cpp_codes/ORS/OR/OR/data/result.txt");
    VRP::VRP_INPUT vrp_input;
    vrp_input.task_num=50;
    vrp_input.vehicle_num=50;
    vrp_input.data_path ="data/vrp_50.txt";
    vrp_input.task_list = (VRP::Task *)malloc(sizeof(VRP::Task) * vrp_input.task_num);
    creat_input_vrp(&vrp_input, "data/vrp_50.txt");
    VRP vrp_problem;
    VRP::VRP_OUTPUT output;
    vrp_problem.cal_dist_matrix(&vrp_input);
    vrp_problem.Solve(&vrp_input,output);
    
    return 0;
}
