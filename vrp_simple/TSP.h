//
//  TSP.h
//  OR
//
//  Created by libaihe on 2019/12/22.
//  Copyright © 2019 libaihe. All rights reserved.
//

#ifndef TSP_h
#define TSP_h

typedef struct Point
{
    /* data */
    int id;
    int x;
    int y;
}Point;

typedef struct TSP_INPUT
{
    int city_num;
    std::string data_path;
    Point *citys;
}TSP_INPUT;

#endif /* TSP_h */
