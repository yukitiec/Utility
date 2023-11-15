#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"

extern const std::string file_yolo;
extern const std::string file_of;
extern const std::string file_seq;

class Utility
{
public:
	Utility()
	{
		std::cout << "construct Utility class" << std::endl;
	}

    void pushImg(cv::Mat1b& frame, int& frameIndex)
    {
        // std::unique_lock<std::mutex> lock(mtxImg);
        queueFrame.push(frame);
        queueFrameIndex.push(frameIndex);
    }

    void removeFrame()
    {
        int counterStart = 0;
        while (counterStart < 2)
        {
            if (!queueYoloSearchRoi.empty())
            {
                while (true)
                {
                    if (queueYoloSearchRoi.empty())
                    {
                        counterStart += 1;
                        break;
                    }
                }
            }
        }
        /* remove imgs */
        while (!queueFrame.empty())
        {
            queueFrame.pop();
            queueFrameIndex.pop();
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            std::cout << "remove image" << std::endl;
        }
    }

    void getImages(cv::Mat1b& frame, int& frameIndex)
    {
        std::unique_lock<std::mutex> lock(mtxImg); // exclude other accesses
        frame = queueFrame.front();
        // cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        frameIndex = queueFrameIndex.front();
        queueFrame.pop();
        queueFrameIndex.pop();
    }

	void saveYolo(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver)
	{
        // Open the file for writing
        std::ofstream outputFile(file_yolo);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        /* write posSaver data to csv file */
        std::cout << "estimated position :: YOLO :: " << std::endl;
        /*sequence*/
        for (int i = 0; i < posSaver.size(); i++)
        {
            std::cout << i << "-th sequence data ::: " << std::endl;
            /*num of humans*/
            for (int j = 0; j < posSaver[i].size(); j++)
            {
                std::cout << j << "-th human detection:::" << std::endl;
                /*num of joints*/
                for (int k = 0; k < posSaver[i][j].size(); k++)
                {
                    std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                    outputFile << posSaver[i][j][k][0];
                    outputFile << ",";
                    outputFile << posSaver[i][j][k][1];
                    outputFile << ",";
                    outputFile << posSaver[i][j][k][2];
                    if (k != posSaver[i][j].size() - 1)
                    {
                        outputFile << ",";
                    }
                }
                outputFile << "\n";
            }
        }
        // Close the file
        outputFile.close();
	}

    void saveOF(std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver)
    {
        // Open the file for writing
        std::ofstream outputFile(file_of);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "estimated position :: Optical Flow :: " << std::endl;
        /*sequence*/
        for (int i = 0; i < posSaver.size(); i++)
        {
            std::cout << i << "-th sequence data ::: " << std::endl;
            /*num of humans*/
            for (int j = 0; j < posSaver[i].size(); j++)
            {
                std::cout << j << "-th human detection:::" << std::endl;
                /*num of joints*/
                for (int k = 0; k < posSaver[i][j].size(); k++)
                {
                    std::cout << k << "-th joint :: frameIndex=" << posSaver[i][j][k][0] << ", xCenter=" << posSaver[i][j][k][1] << ", yCenter=" << posSaver[i][j][k][2] << std::endl;
                    outputFile << posSaver[i][j][k][0];
                    outputFile << ",";
                    outputFile << posSaver[i][j][k][1];
                    outputFile << ",";
                    outputFile << posSaver[i][j][k][2];
                    if (k != posSaver[i][j].size() - 1)
                    {
                        outputFile << ",";
                    }
                }
                outputFile << "\n";
            }
        }
        // close file
        outputFile.close();
    }
};

#endif
