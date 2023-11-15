#pragma once

#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"

std::mutex mtxImg, mtxYolo;
/* queueu definition */
/* frame queue */
std::queue<cv::Mat1b> queueFrame;
std::queue<int> queueFrameIndex;
/* yolo and optical flow */
std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch;      // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2d>>> queueYoloSearchRoi;        // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch;        // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2d>>> queueOFSearchRoi;          // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]

/* constant valude definition */
extern const std::string filename = "video/yolotest.mp4";
extern const bool save = true;
extern const bool boolSparse = false;
extern const bool boolGray = true;
extern const std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"
extern const float qualityCorner = 0.01;
/* roi setting */
extern const int roiWidthOF = 12;
extern const int roiHeightOF = 12;
extern const int roiWidthYolo = 12;
extern const int roiHeightYolo = 12;
extern const int MoveThreshold = 1.0;
extern const float epsironMove = MoveThreshold / 10;
/* dense optical flow skip rate */
extern const int skipPixel = 1;
/*if exchange template of Yolo */
extern const bool boolChange = false;
/* save date */
extern const std::string file_yolo = "yolo.csv";
extern const std::string file_of = "opticalflow.csv";
extern const std::string file_seq = "sequence.csv";

#endif
