#pragma once

#ifndef MATCHING_H
#define MATCHING_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>


class Matching
{
private:
    const bool debug = true;
    const int dif_threshold = 1; //difference between 2 cams
public:
    Matching()
    {
        std::cout << "construct Matching class" << std::endl;
    }

    void main(std::vector<std::vector<std::vector<int>>>& vec_left, std::vector<std::vector<std::vector<int>>>& vec_right, std::vector<int>& classes_left, std::vector<int>& classes_right, std::vector<std::vector<int>>& matching)
    {
        std::vector<std::vector<int>> ball_left, ball_right, box_left, box_right; //storage for data
        std::vector<int> ball_index_left, box_index_left, ball_index_right, box_index_right; //storage for index
        //extract position and index, and then sort -> 300 microseconds for 2 
        arrangeData(vec_left, classes_left, ball_left, box_left, ball_index_left, box_index_left);
        arrangeData(vec_right, classes_right, ball_right, box_right, ball_index_right, box_index_right);
        //matching data in y value
        //ball
        int num_left = ball_index_left.size();
        int num_right = ball_index_right.size();
        auto start_time2 = std::chrono::high_resolution_clock::now();
        // more objects detected in left camera -> 79 microseconds for 2
        matchingObj(ball_left, ball_right, ball_index_left, ball_index_right, matching);
        //box
        matchingObj(box_left, box_right, box_index_left, box_index_right, matching);
        if (debug)
        {
            for (int i = 0; i < matching.size(); i++)
                std::cout << i << "-th matching :: left : " << matching[i][0] << ", right: " << matching[i][1] << std::endl;
        }
    }

    void arrangeData(std::vector<std::vector<std::vector<int>>> vec_left, std::vector<int> classes_left, std::vector<std::vector<int>>& ball_left, std::vector<std::vector<int>>& box_left,
        std::vector<int>& ball_index_left, std::vector<int>& box_index_left)
    {
        for (int i = 0; i < classes_left.size(); i++)
        {
            // ball
            if (classes_left[i] == 0)
            {
                ball_left.push_back(vec_left[i].back());
                ball_index_left.push_back(i);
            }
            // box
            else if (classes_left[i] == 1)
            {
                box_left.push_back(vec_left[i].back());
                box_index_left.push_back(i);
            }
        }
        // sort data 
        sortData(ball_left, ball_index_left);
        sortData(box_left, box_index_left);
    }

    void sortData(std::vector<std::vector<int>>& data, std::vector<int>& classes)
    {
        // Create an index array to remember the original order
        std::vector<size_t> index(data.size());
        for (size_t i = 0; i < index.size(); ++i)
        {
            index[i] = i;
        }
        // Sort data1 based on centerX values and apply the same order to data2
        std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
            { return data[a][3] >= data[b][3]; });

        std::vector<std::vector<int>> sortedData(data.size());
        std::vector<int> sortedClasses(classes.size());

        for (size_t i = 0; i < index.size(); ++i)
        {
            sortedData[i] = data[index[i]];
            sortedClasses[i] = classes[index[i]];
        }

        data = sortedData;
        classes = sortedClasses;
    }

    void matchingObj(std::vector<std::vector<int>>& ball_left, std::vector<std::vector<int>>& ball_right, std::vector<int>& ball_index_left, std::vector<int>& ball_index_right, std::vector<std::vector<int>>& matching)
    {
        int dif_min = dif_threshold;
        int matchIndex_right;
        int i = 0;
        int startIndex = 0; //from which index to start comparison
        //for each object
        while (i < ball_left.size() && startIndex < ball_right.size())
        {
            int j = 0;
            bool boolMatch = false;
            dif_min = dif_threshold;
            std::cout << "startIndex = " << startIndex << std::endl;
            //continue right object y-value is under threshold
            while (startIndex + j < ball_right.size())
            {
                if ((ball_left[i][3] - ball_right[startIndex + j][3]) > dif_threshold) break;
                std::cout << ball_left[i][3] - ball_right[startIndex + j][3] << std::endl;
                int dif = std::abs(ball_left[i][3] - ball_right[startIndex + j][3]);
                if (dif < dif_min)
                {
                    dif_min = dif;
                    matchIndex_right = startIndex + j;
                    boolMatch = true;
                }
                j++;
            }
            std::cout << boolMatch << std::endl;
            //match index is the last value
            if (boolMatch && (matchIndex_right == (startIndex + j - 1))) startIndex += j;
            else startIndex += std::max(j - 1, 0);
            /* matching successful*/
            if (boolMatch)
            {
                matching.push_back({ ball_index_left[i],ball_index_right[matchIndex_right] }); //save matching pair
                //delete selected data
                ball_index_left.erase(ball_index_left.begin() + i);
                ball_left.erase(ball_left.begin() + i);
                //ball_index_right.erase(ball_index_right.begin() + matchIndex_right);
                //ball_right.erase(ball_right.begin() + matchIndex_right);
            }
            // can't find matching object
            else
                i++;
        }
    }

};

#endif 