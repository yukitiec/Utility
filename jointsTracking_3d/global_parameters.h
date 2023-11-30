#pragma once

#pragma once

#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"

std::mutex mtxRobot;
/* queueu definition */
/* frame queue */
std::queue<std::array<cv::Mat1b, 2>> queueFrame;
std::queue<int> queueFrameIndex;
/* yolo and optical flow */
/* left */
std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_left;      // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_left;        // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_left;          // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_left; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
/* right */
std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_right;      // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_right;        // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_right;          // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_right; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
/*3D position*/
std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;
/* from joints to robot control */
std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
/* notify danger */
std::queue<bool> queueDanger;

/* constant valude definition */
extern const std::string filename_left = "joints_front_left.mp4";
extern const std::string filename_right = "joints_front_right.mp4";
extern const int LEFT = 0;
extern const int RIGHT = 1;
extern const bool save = true;
extern const bool boolSparse = false;
extern const bool boolGray = true;
extern const bool boolBatch = true; //if yolo inference is run in concatenated img
extern const std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"
extern const int dense_vel_method = 3; //0: average, 1:max, 2 : median, 3 : third-quarter, 4 : first-quarter
extern const float qualityCorner = 0.01;
/* roi setting */
extern const int roiWidthOF = 35;
extern const int roiHeightOF = 35;
extern const int roiWidthYolo = 35;
extern const int roiHeightYolo = 35;
extern const int MoveThreshold = 0.25; //cancell background
extern const float epsironMove = 0.1;//half range of back ground effect:: a-epsironMove<=flow<=a+epsironMove
/* dense optical flow skip rate */
extern const int skipPixel = 2;
extern const float DIF_THRESHOLD = roiWidthOF / 4; //threshold for adapting yolo detection's roi
extern const float MIN_MOVE = 0.5; //minimum opticalflow movement
/*if exchange template of Yolo */
extern const bool boolChange = true;
/* save date */
extern const std::string file_yolo_left = "yolo_left.csv";
extern const std::string file_yolo_right = "yolo_right.csv";
extern const std::string file_of_left = "opticalflow_left.csv";
extern const std::string file_of_right = "opticalflow_right.csv";
extern const std::string file_3d = "triangulation.csv";

/* 3D triangulation */
extern const int BASELINE = 280; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */
extern const cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 297.0, 0, 151.5, // fx: focal length in x, cx: principal point x
    0, 297.5, 149.0,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
extern const cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << -0.00896 ,-0.215 ,0.00036 ,0.0043, 0.391);
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base{ {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };
#endif
