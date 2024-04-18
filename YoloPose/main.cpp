#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "yolo_pose.h"


int main()
{
    YOLOPose yoloPoseEstimator; //construct
    //video load
    const std::string filename = "video/yolotest.mp4";
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    cv::VideoCapture capture(filename);
    if (!capture.isOpened()) {
        //error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 1;

    //iteration
    while (true) {
        // Read the next frame
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        cv::Mat1b frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        //YOLO detection start
        auto start = std::chrono::high_resolution_clock::now();
        yoloPoseEstimator.detect(frameGray, counter, posSaver);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
        //YOLO detection finishes
        counter++;
    }

    //save data
    //prepare file
    std::string filePath = "detection.csv";
    // Open the file for writing
    std::ofstream outputFile(filePath);
    if (!outputFile.is_open()) {
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


    return 0;
}
