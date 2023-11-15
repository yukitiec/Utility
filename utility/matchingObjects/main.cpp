// sortData.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "matching.h"

extern const bool debug = false;
extern const int dif_threshold = 2; //difference between 2 cams


int main()
{
    //construct class
    Matching mt;
    std::vector<std::vector<std::vector<int>>> vec_left{ {{1,0,1,1},{2,0,1,1},{3,0,1,1}},{{1,0,1,5},{2,0,1,5},{3,0,1,5}},{{1,1,1,6},{2,1,1,6},{3,1,1,6}},{{1,0,1,4},{2,0,1,4},{3,0,1,4}},{{1,1,1,1},{2,1,1,1},{3,1,1,1}} };
    std::vector<std::vector<std::vector<int>>> vec_right{ {{1,0,2,5},{2,0,2,5},{3,0,2,5}},{{1,0,2,4},{2,0,2,4},{3,0,2,4}},{{1,1,2,7},{2,1,2,7},{3,1,2,7}},{{1,0,2,9},{2,0,2,9},{3,0,2,9}},{{1,1,1,6},{2,1,1,6},{3,1,1,6}} };
    std::vector<int> classes_left{ 0,0,1,0,1 };
    std::vector<int> classes_right{ 0,0,1,0,1 };
    std::vector<std::vector<int>> matching; //matching indexes
    //extract position and index, and then sort
    auto start_time = std::chrono::high_resolution_clock::now();
    mt.main(vec_left, vec_right, classes_left, classes_right, matching);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken by arrange : " << duration_ms.count() << " microseconds" << std::endl; // 156 microseconds
}

