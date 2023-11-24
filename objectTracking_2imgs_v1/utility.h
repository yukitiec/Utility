#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "stdafx.h"

/* file name setting */
extern const std::string file_yolo_bbox;
extern const std::string file_yolo_class;
extern const std::string file_tm_bbox;
extern const std::string file_tm_class;
extern const std::string file_seq_bbox;
extern const std::string file_seq_class;
// queue definition
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

class Utility
{
public:
    Utility()
    {
        std::cout << "construct Utility" << std::endl;
    }

    bool getImagesFromQueueYolo(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
    {
        // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
        if (!queueFrame.empty())
        {
            imgs = queueFrame.front();
            frameIndex = queueFrameIndex.front();
            // remove frame from queue
            queueFrame.pop();
            queueFrameIndex.pop();
            return true;
        }
        return false;
    }

    void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string& fileName)
    {

        // Open the file for writing
        std::ofstream outputFile(fileName);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "estimated position :: Yolo :: " << std::endl;
        int count = 1;
        std::cout << "posSaverYolo :: Contensts ::" << std::endl;
        for (int i = 0; i < posSaverYolo.size(); i++)
        {
            std::cout << (i + 1) << "-th iteration : " << std::endl;
            for (int j = 0; j < posSaverYolo[i].size(); j++)
            {
                std::cout << detectedFrame[i] << "-th frame :: left=" << posSaverYolo[i][j].x << ", top=" << posSaverYolo[i][j].y << ", width=" << posSaverYolo[i][j].width << ", height=" << posSaverYolo[i][j].height << std::endl;
                outputFile << detectedFrame[i];
                outputFile << ",";
                outputFile << posSaverYolo[i][j].x;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].y;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].width;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].height;
                if (j != posSaverYolo[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
        // close file
        outputFile.close();
    }

    void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string& fileName)
    {
        // Open the file for writing
        std::ofstream outputFile(fileName);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        int count = 1;
        std::cout << "Class saver :: Contensts ::" << std::endl;
        for (int i = 0; i < classSaverYolo.size(); i++)
        {
            std::cout << detectedFrame[i] << "-th frame : " << std::endl;
            for (int j = 0; j < classSaverYolo[i].size(); j++)
            {
                std::cout << classSaverYolo[i][j] << " ";
                outputFile << detectedFrame[i];
                outputFile << ",";
                outputFile << classSaverYolo[i][j];
                if (j != classSaverYolo[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";



            std::cout << std::endl;
        }
        // close file
        outputFile.close();
    }

    void checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame, std::string& fileName)
    {
        // Open the file for writing
        std::ofstream outputFile(fileName);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "estimated position :: Yolo :: " << std::endl;
        int count = 1;
        std::cout << "posSaverYolo :: Contents ::" << std::endl;
        for (int i = 0; i < posSaverYolo.size(); i++)
        {
            std::cout << (i + 1) << "-th iteration : " << std::endl;
            for (int j = 0; j < posSaverYolo[i].size(); j++)
            {
                std::cout << detectedFrame[i] << "-th frame :: left=" << posSaverYolo[i][j].x << ", top=" << posSaverYolo[i][j].y << ", width=" << posSaverYolo[i][j].width << ", height=" << posSaverYolo[i][j].height << std::endl;
                outputFile << detectedFrame[i];
                outputFile << ",";
                outputFile << posSaverYolo[i][j].x;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].y;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].width;
                outputFile << ",";
                outputFile << posSaverYolo[i][j].height;
                if (j != posSaverYolo[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
        // close file
        outputFile.close();
    }

    void checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame, std::string& fileName)
    {
        // Open the file for writing
        std::ofstream outputFile(fileName);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        int count = 1;
        std::cout << "Class saver :: Contensts ::" << std::endl;
        for (int i = 0; i < classSaverYolo.size(); i++)
        {
            std::cout << detectedFrame[i] << "-th frame : " << std::endl;
            for (int j = 0; j < classSaverYolo[i].size(); j++)
            {
                std::cout << classSaverYolo[i][j] << " ";
                outputFile << detectedFrame[i];
                outputFile << ",";
                outputFile << classSaverYolo[i][j];
                if (j != classSaverYolo[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
            std::cout << std::endl;
        }
        // close file
        outputFile.close();
    }

    void checkSeqData(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<int>>& classesLeft, std::string& fileName)
    {
        // Open the file for writing
        /* bbox data */
        std::ofstream outputFile(fileName);
        std::vector<int> frameIndexes;
        frameIndexes.reserve(2000);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "file open :: save sequence data in csv file" << std::endl;
        for (int i = 0; i < dataLeft.size(); i++)
        {
            std::cout << (i + 1) << "-th objects : " << std::endl;
            for (int j = 0; j < dataLeft[i].size(); j++)
            {
                std::cout << j << ":: frameIndex=" << dataLeft[i][j][0] << "class label=" << dataLeft[i][j][1] << ", x=" << dataLeft[i][j][2] << ", y=" << dataLeft[i][j][3] << " ";
                auto it = std::find(frameIndexes.begin(), frameIndexes.end(), dataLeft[i][j][0]);
                /* new frame index */
                if (it == frameIndexes.end()) frameIndexes.push_back(dataLeft[i][j][0]);
                outputFile << dataLeft[i][j][0];
                outputFile << ",";
                outputFile << dataLeft[i][j][1];
                outputFile << ",";
                outputFile << dataLeft[i][j][2];
                outputFile << ",";
                outputFile << dataLeft[i][j][3];
                if (j != dataLeft[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
            std::cout << std::endl;
        }
        // close file
        outputFile.close();

        /* sort frame indexes */
        std::sort(frameIndexes.begin(), frameIndexes.end());
        /* bbox data */
        std::ofstream outputFile_class(file_seq_class);
        if (!outputFile_class.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "frameIndexes size = " << frameIndexes.size() << ", classesLeft size = " << classesLeft.size() << std::endl;
        for (int i = 0; i < classesLeft.size(); i++)
        {
            std::cout << (i + 1) << "-th time : " << std::endl;
            for (int j = 0; j < classesLeft[i].size(); j++)
            {
                std::cout << "frameIndex=" << frameIndexes[i] << ", label=" << classesLeft[i][j];
                outputFile_class << frameIndexes[i];
                outputFile_class << ",";
                outputFile_class << classesLeft[i][j];
                if (j != classesLeft[i].size() - 1)
                {
                    outputFile_class << ",";
                }
            }
            outputFile_class << "\n";
            std::cout << std::endl;
        }
        // close file
        outputFile_class.close();
    }

    void save3d(std::vector<std::vector<std::vector<int>>>& posSaver, const std::string& file)
    {
        // Open the file for writing
        std::ofstream outputFile(file);
        if (!outputFile.is_open())
        {
            std::cerr << "Error: Could not open the file." << std::endl;
        }
        std::cout << "estimated position :: Optical Flow :: " << std::endl;
        /*num of objects */
        for (int i = 0; i < posSaver.size(); i++)
        {
            std::cout << i << "-th objects :: " << std::endl;
            /*num of sequence*/
            for (int j = 0; j < posSaver[i].size(); j++)
            {
                std::cout << j << "-th timestep :: frameIndex=" << posSaver[i][j][0] << ", x=" << posSaver[i][j][1]<<", y=" << posSaver[i][j][2] << ", z=" << posSaver[i][j][3] << std::endl;
                outputFile << posSaver[i][j][0];
                outputFile << ",";
                outputFile << posSaver[i][j][1];
                outputFile << ",";
                outputFile << posSaver[i][j][2];
                outputFile << ",";
                outputFile << posSaver[i][j][3];
                if (j != posSaver[i].size() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
        // close file
        outputFile.close();
    }


    bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
    {
        // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
        if (queueFrame.empty() || queueFrameIndex.empty())
        {
            return false;
        }
        else
        {
            imgs = queueFrame.front();
            frameIndex = queueFrameIndex.front();
            queueTargetFrameIndex.push(frameIndex);
            // remove frame from queue
            queueFrame.pop();
            queueFrameIndex.pop();
            return true;
        }
    }

    /* read imgs */
    void pushFrame(std::array<cv::Mat1b, 2>& src, const int frameIndex)
    {
        // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
        //  std::cout << "push imgs" << std::endl;
        //  cv::Mat1b undistortedImgL, undistortedImgR;
        //  cv::undistort(srcs[0], undistortedImgL, cameraMatrix, distCoeffs);
        //  cv::undistort(srcs[1], undistortedImgR, cameraMatrix, distCoeffs);
        //  std::array<cv::Mat1b, 2> undistortedImgs = { undistortedImgL,undistortedImgR };
        //  queueFrame.push(undistortedImgs);
        queueFrame.push(src);
        queueFrameIndex.push(frameIndex);
    }

    void removeFrame()
    {
        int counterStart = 0;
        while (counterStart < 2)
        {
            if (!queueYoloBboxLeft.empty())
            {
                while (true)
                {
                    if (queueYoloBboxLeft.empty())
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
            std::this_thread::sleep_for(std::chrono::microseconds(2000);
            std::cout << "remove image" << std::endl;
        }
    }

};

#endif
