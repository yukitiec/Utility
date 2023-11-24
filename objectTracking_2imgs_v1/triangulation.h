#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"
#include "matching.h"

//3d storage
extern const int numObjects;
std::vector<std::vector<std::vector<int>>> data_3d(numObjects); //{num of objects, sequential, { frameIndex, X,Y,Z }}

extern const std::string file_3d;

//Yolo detect signal
extern std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;
extern std::queue<bool> queue_tri2predict;
std::queue<std::vector<std::vector<int>>> queueMatchingIndexes; //for saving matching indexes


/*3D position*/
extern std::vector<std::vector<std::vector<int>>> seqData_left, seqData_right; //storage for sequential data
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdateLabels_right;

/* 3D triangulation */
extern const int BASELINE; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
/* save file*/
extern const std::string file_3d;

class Triangulation
{
private:
    const float fX = cameraMatrix.at<double>(0, 0);
    const float fY = cameraMatrix.at<double>(1, 1);
    const float fSkew = cameraMatrix.at<double>(0, 1);
    const float oX = cameraMatrix.at<double>(0, 2);
    const float oY = cameraMatrix.at<double>(1, 2);
    const int numJoint = 6; //number of joints
public:
    Triangulation()
    {
        std::cout << "construct Triangulation class" << std::endl;
    }

    void main()
    {
        Utility utTri;//constructor
        Matching match; //matching algorithm
        while (true)
        {
            if (!queueUpdateLabels_left.empty() && !queueUpdateLabels_right.empty()) break;
            //std::cout << "wait for target data" << std::endl;
        }
        std::cout << "start calculating 3D position" << std::endl;

        int counterIteration = 0;
        int counterFinish = 0;
        int counter = 0;//counter before delete new data
        while (true) // continue until finish
        {
            counterIteration++;
            if (queueFrame.empty() && queueUpdateLabels_left.empty() && queueUpdateLabels_right.empty())
            {
                if (counterFinish == 10) break;
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
                continue;
            }
            else
            {
                /* new detection data available */
                if (!queueUpdateLabels_left.empty() && !queueUpdateLabels_right.empty())
                {
                    counter = 0; //reset
                    auto start = std::chrono::high_resolution_clock::now();
                    std::cout << "start 3d positioning" << std::endl;
                    //get latest classes
                    std::vector<int> labels_left = queueUpdateLabels_left.front(); queueUpdateLabels_left.pop();
                    std::vector<int> labels_right = queueUpdateLabels_right.front(); queueUpdateLabels_right.pop();
                    std::vector<std::vector<int>> matchingIndexes; //list for matching indexes
                    //matching algorithm
                    if (!queueYolo_seq2tri_left.empty() && !queueYolo_seq2tri_right.empty())
                    {
                        std::cout << "update matching" << std::endl;
                        queueYolo_seq2tri_left.pop(); 
                        queueYolo_seq2tri_right.pop();
                        match.main(seqData_left, seqData_right, labels_left, labels_right, matchingIndexes); //matching objects in 2 images
                        sortData(matchingIndexes);
                    }
                    //use previous matching indexes
                    else if (!queueMatchingIndexes.empty())
                    {
                        matchingIndexes = queueMatchingIndexes.front();
                        queueMatchingIndexes.pop();
                    }
                    //calculate 3d positions based on matchingIndexes
                    if (!matchingIndexes.empty())
                    {
                        std::cout << "start 3D positioning" << std::endl;
                        triangulation(seqData_left, seqData_right, matchingIndexes,data_3d);
                        queue_tri2predict.push(true);
                        std::cout << "calculate 3d position" << std::endl;
                        auto stop = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                        std::cout << "time taken by 3d positioning=" << duration.count() << " milliseconds" << std::endl;
                    }
                 
                }
                /* at least one data can't be available -> delete data */
                else
                {
                    //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                    if (counter == 5)
                    {
                        if (!queueUpdateLabels_left.empty()) queueUpdateLabels_left.pop();
                        if (!queueUpdateLabels_right.empty()) queueUpdateLabels_right.pop();
                    }
                    else if (!queueUpdateLabels_left.empty() || !queueUpdateLabels_right.empty()) counter++;
                   
                }
            }
        }
        std::cout << "***triangulation data***" << std::endl;
        utTri.save3d(data_3d,file_3d);
    }

    void sortData(std::vector<std::vector<int>>& data)
    {
        // Sorting in ascending way
        std::sort(data.begin(), data.end(), compareVectors);
    }

    bool compareVectors(const std::vector<int>& a, const std::vector<int>& b) 
    {
        // Compare based on the first element of each vector
        return a[0] < b[0];
    }

    void triangulation(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right, std::vector<std::vector<int>>& matchingIndexes,std::vector<std::vector<std::vector<int>>>& data_3d)
    {
        //for all matching data
        for (std::vector<int>& matchIndex : matchingIndexes)
        {
            int index = matchIndex[0]; //left object's index
            //calculate objects
            std::vector<std::vector<int>> left = data_left[matchIndex[0]]; //{num frames, {frameIndex, classLabel,xCenter,yCenter}}
            std::vector<std::vector<int>> right = data_right[matchIndex[1]]; //{num frames, {frameIndex, classLabel,xCenter,yCenter}}
            calulate3Ds(index,left, right,data_3d); //{num of objects, sequential, {frameIndex, X,Y,Z}}
        }
    }

    void calulate3Ds(int& index, std::vector<std::vector<int>>& left, std::vector<std::vector<int>>& right, std::vector<std::vector<std::vector<int>>>& data_3d)
    {
        std::vector<std::vector<int>> temp;
        int left_frame_start = left[0][0];
        int right_frame_start = right[0][0];
        int num_frames_left = left.size();
        int num_frames_right = right.size();
        int it_left = 0; int it_right = 0;
        //check whether previous data is in data_3d -> if exist, check last frame Index
        if (!data_3d[index].empty())
        {
            int last_frameIndex = (data_3d[index].back())[0]; //get last frame index
            //calculate 3d position for all left data
            std::vector<std::vector<int>> temp_3d;
            while (it_left < num_frames_left && it_right < num_frames_right)
            {
                if (left[it_left][0] == right[it_right][0] && left[it_left][0] > last_frameIndex)
                {
                    std::vector<int> result;
                    cal3D(left[it_left], right[it_right], result);
                    if (result[0] != -1) temp_3d.push_back(result); //add 3D data
                    it_left++;
                    it_right++;
                }
                else if (left[it_left][0] > right[it_right][0]) it_right++;
                else if (left[it_left][0] < right[it_right][0]) it_left++;
            }
            if (!temp_3d.empty())
            {
                for (std::vector<int>& newData : temp_3d)
                {
                    data_3d.at(index).push_back(newData);
                }
            }
        }
        //no previous data
        else
        {
            //calculate 3d position for all left data
            std::vector<std::vector<int>> temp_3d;
            while (it_left < num_frames_left && it_right < num_frames_right)
            {
                if (left[it_left][0] == right[it_right][0])
                {
                    std::vector<int> result;
                    cal3D(left[it_left], right[it_right], result);
                    if (result[0] != -1) temp_3d.push_back(result); //add 3D data
                    it_left++;
                    it_right++;
                }
                else if (left[it_left][0] > right[it_right][0]) it_right++;
                else if (left[it_left][0] < right[it_right][0]) it_left++;
            }
            if (!temp_3d.empty()) data_3d.at(index) = temp_3d;
        }
    }

    void cal3D(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result)
    {
        int xl = left[2]; int xr = right[2]; int yl = left[3]; int yr = right[3];
        int disparity = (int)(xl - xr);
        int X = (int)(BASELINE / disparity) * (xl - oX - (fSkew / fY) * (yl - oY));
        int Y = (int)(BASELINE * (fX / fY) * (yl - oY) / disparity);
        int Z = (int)(fX * BASELINE / disparity);
        result = std::vector<int>{ left[0],X,Y,Z };
    }

};

#endif