#pragma once

#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "stdafx.h"
#include "global_parameters.h"
#include "utility.h"

/*3D position*/
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;

/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;

/* 3D triangulation */
extern const int BASELINE; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix;
extern const cv::Mat distCoeffs;
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base;
/* save file*/
extern const std::string file_3d;

class Triangulation
{
private:
    const float fX = cameraMatrix.at<float>(0, 0);
    const float fY = cameraMatrix.at<float>(1, 1);
    const float fSkew = cameraMatrix.at<float>(0, 1);
    const float oX = cameraMatrix.at<float>(0, 2);
    const float oY = cameraMatrix.at<float>(1, 2);
    const int numJoint = 6; //number of joints
    const float epsiron = 0.001;
public:
    Triangulation()
    {
        std::cout << "construct Triangulation class" << std::endl;
    }

    void main()
    {
        Utility utTri;//constructor
        while (true)
        {
            if (!queueTriangulation_left.empty() && !queueTriangulation_right.empty()) break;
            //std::cout << "wait for target data" << std::endl;
        }
        std::cout << "start calculating 3D position" << std::endl;


        std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_3d; //{num of sequence, num of human, joints, {frameIndex, X,Y,Z}}
        int counterIteration = 0;
        int counterFinish = 0;
        int counterNextIteration = 0;
        while (true) // continue until finish
        {
            counterIteration++;
            if (queueFrame.empty() && queueTriangulation_left.empty() && queueTriangulation_right.empty())
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
                if (!queueTriangulation_left.empty() && !queueTriangulation_right.empty())
                {
                    counterNextIteration = 0;
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<std::vector<int>>> data_left, data_right;
                    std::cout << "start 3d positioning" << std::endl;
                    getData(data_left, data_right);
                    std::cout << "start" << std::endl;
                    std::vector<std::vector<std::vector<int>>> data_3d; //{num of human, joints, {frameIndex, X,Y,Z}}
                    triangulation(data_left,data_right,data_3d);
                    std::cout << "calculate 3d position" << std::endl;
                    /* arrange posSaver -> sequence data */
                    arrangeData(data_3d, posSaver_3d);
                    queueJointsPositions.push(posSaver_3d.back());
                    std::cout << "arrange data" << std::endl;
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    std::cout << "time taken by 3d positioning=" << duration.count() << " milliseconds" << std::endl;
                }
                /* at least one data can't be available -> delete data */
                else
                {
                    //std::cout << "both data can't be availble :: left " << !queueTriangulation_left.empty() << ", right=" << queueTriangulation_right.empty() << std::endl;
                    if (!queueTriangulation_left.empty() || !queueTriangulation_right.empty())
                    {
                        if (counterNextIteration == 10)
                        {
                            if (!queueTriangulation_left.empty()) queueTriangulation_left.pop();
                            if (!queueTriangulation_right.empty()) queueTriangulation_right.pop();
                        }
                        else
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            counterNextIteration++;
                        }
                        
                    }
                    
                    
                }
            }
        }
        std::cout << "***triangulation data***" << std::endl;
        utTri.save3d(posSaver_3d, file_3d);
    }

    void getData(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right)
    {
        data_left = queueTriangulation_left.front();
        data_right = queueTriangulation_right.front();
        queueTriangulation_left.pop();
        queueTriangulation_right.pop();
    }

    void triangulation(std::vector<std::vector<std::vector<int>>>& data_left, std::vector<std::vector<std::vector<int>>>& data_right,
        std::vector<std::vector<std::vector<int>>>& data_3d)
    {
        int numHuman = std::min(data_left.size(), data_right.size());
        // for each human
        for (int i = 0; i < numHuman; i++)
        {
            std::vector<std::vector<int>> temp;
            // for each joint
            for (int j = 0; j < numJoint; j++)
            {
                std::vector<int> result;
                std::cout << "cal3D" << std::endl;
                cal3D(data_left[i][j],data_right[i][j],result);
                std::cout << "finish" << std::endl;
                temp.push_back(result);
            }
            data_3d.push_back(temp);
        }
    }

    void cal3D(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result)
    {
        //both joints is detected
        if (left[1] != -1 && right[1] != -1)
        {
            // frameIndex is same
            if (left[0] == right[0])
            {
                int xl = left[1]; int xr = right[1]; int yl = left[2]; int yr = right[2];
                float disparity = xl - xr + epsiron;
                if (disparity < 0.1) disparity += epsiron;
                int X = (int)(BASELINE / disparity) * (xl - oX - (fSkew / fY) * ((yl+yr)/2 - oY));
                int Y = (int)(BASELINE * (fX / fY) * ((yl + yr) / 2 - oY) / disparity);
                int Z = (int)(fX * BASELINE / disparity);
                /* convert Camera coordinate to robot base coordinate */
                X = static_cast<int>(transform_cam2base[0][0] * X + transform_cam2base[0][1] * Y + transform_cam2base[0][2] * Z + transform_cam2base[0][3]);
                Y = static_cast<int>(transform_cam2base[1][0] * X + transform_cam2base[1][1] * Y + transform_cam2base[1][2] * Z + transform_cam2base[1][3]);
                Z = static_cast<int>(transform_cam2base[2][0] * X + transform_cam2base[2][1] * Y + transform_cam2base[2][2] * Z + transform_cam2base[2][3]);
                result = std::vector<int>{ left[0],X,Y,Z };
            }
            else
            {
                std::cout << "frameIndex is different" << std::endl;
                int X = -1; int Y = -1; int Z = -1;
                result = std::vector<int>{ -1,X,Y,Z };
            }
        }
        // at least one isn't detected
        else
        {
            std::cout << "frameIndex is different" << std::endl;
            int X = -1; int Y = -1; int Z = -1;
            result = std::vector<int>{ -1,X,Y,Z };
        }
        
    }

    void arrangeData(std::vector<std::vector<std::vector<int>>>& data_3d,std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver)
    {
        // already first 3d calculation done
        std::cout << "posSaver.size()=" << posSaver.size() << std::endl;
        if (!posSaver.empty())
        {
            std::vector<std::vector<std::vector<int>>> all; //all human data
            std::cout << "data3d.size()=" << data_3d.size() << std::endl;
            // for each human
            for (int i = 0; i < data_3d.size(); i++)
            {
                std::vector<std::vector<int>> tempHuman;
                /* same human */
                if (posSaver[posSaver.size() - 1].size() > i)
                {
                    // for each joint
                    for (int j = 0; j < data_3d[i].size(); j++)
                    {
                        // detected
                        if (data_3d[i][j][1] != -1)
                        {
                            tempHuman.push_back(data_3d[i][j]);
                        }
                        // not detected
                        else
                        {
                            // already detected
                            if (posSaver[posSaver.size() - 1][i][j][1] != -1)
                            {
                                tempHuman.push_back(posSaver[posSaver.size() - 1][i][j]); //adapt last detection
                            }
                            // not detected yet
                            else
                            {
                                tempHuman.push_back(data_3d[i][j]); //-1
                            }
                        }
                    }
                }
                //new human
                else
                {
                    tempHuman = data_3d[i];
                }
                all.push_back(tempHuman); //push human data
            }
            posSaver.push_back(all);
        }
        // first detection
        else
        {
            std::cout << "first 3d points :: data3d.size()=" << data_3d.size() << std::endl;
            posSaver.push_back(data_3d);
        }
    }

};

#endif