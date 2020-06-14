//
//  input.hpp
//  OR
//
//  Created by libaihe on 2019/12/22.
//  Copyright © 2019 libaihe. All rights reserved.
//

#ifndef input_hpp
#define input_hpp

#include <stdio.h>

#include "TSP.h"
#include "VRP.hpp"

#define MAX_CITY_NUM 500


void new_input(TSP_INPUT *input);

void creat_input_vrp(VRP::VRP_INPUT *input,std::string file_path);

#endif /* input_hpp */
