#pragma once


#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"

// camera : constant setting
extern const int LEFT_CAMERA = 0;
extern const int RIGHT_CAMERA = 1;
extern const int FPS = 300;
// YOLO label
extern const int BALL = 0;
extern const int BOX = 1;
/* 3d positioning by stereo camera */
extern const int BASELINE = 280; // distance between 2 cameras
/* camera calibration result */
extern const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 179, 0, 160, // fx: focal length in x, cx: principal point x
    0, 179, 160,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
extern const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 1, 1, 1, 1, 1);

/* UR catching point */
extern const int TARGET_DEPTH = 400; // catching point is 40 cm away from camera position

/* save file setting */
extern const std::string file_yolo_bbox= "yolo_bbox_test300fpsmp4.csv";
extern const std::string file_yolo_class = "yolo_class_test300fpsmp4.csv";
extern const std::string file_tm_bbox = "tm_bbox_test300fpsmp4.csv";
extern const std::string file_tm_class = "tm_class_test300fpsmp4.csv";
extern const std::string file_seq_bbox = "seqData_bbox_test300fpsmp4.csv";
extern const std::string file_seq_class = "seqData_class_test300fpsmp4.csv";
// queue definition
std::queue<cv::Mat1b> queueFrame; // queue for frame
std::queue<int> queueFrameIndex;  // queue for frame index

// left cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
std::queue<bool> queueStartYolo; //if new Yolo inference can start

// right cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

// 3D positioning ~ trajectory prediction
std::queue<int> queueTargetFrameIndex;                      // TM estimation frame
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency
std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex




#endif