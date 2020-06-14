//
//  input.cpp
//  OR
//
//  Created by libaihe on 2019/12/22.
//  Copyright © 2019 libaihe. All rights reserved.
//
#include <iostream>

#include "input.hpp"
#include "read_data_from_txt.hpp"

using namespace std;

void creat_input_vrp(VRP::VRP_INPUT *input,std::string fila_path)
{
    read_data_from_txt(fila_path,input);
}
