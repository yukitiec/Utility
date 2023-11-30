#pragma once


#ifndef GLOBAL_PARAMETERS_H
#define GLOBAL_PARAMETERS_H

#include "stdafx.h"
#include "mosse.h"

extern const bool boolGroundTruth=false;
//video path
extern const std::string filename_left = "test300fps_1114_left.mp4";
extern const std::string filename_right = "test300fps_1114_right.mp4";
// camera : constant setting
extern const int LEFT_CAMERA = 0;
extern const int RIGHT_CAMERA = 1;
extern const int FPS = 300;
// YOLO label
extern const int BALL = 0;
extern const int BOX = 1;
// tracker
extern const bool boolMOSSE = true; //Use MOSSE or Template matching :: true->> MOSSE, false->> template matching
extern const double threshold_mosse = 2.0;
/* 3d positioning by stereo camera */
extern const int BASELINE = 280; // distance between 2 cameras
/* camera calibration result */
extern const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 179, 0, 160, // fx: focal length in x, cx: principal point x
    0, 179, 160,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
extern const cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 1, 1, 1, 1, 1);
/* transformation matrix from camera coordinate to robot base coordinate */
extern const std::vector<std::vector<float>> transform_cam2base{ {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };
//3d objects number
extern const int numObjects=50;

/* UR catching point */
extern const int TARGET_DEPTH = 400; // catching point is 40 cm away from camera position

/* save file setting */
extern const std::string file_yolo_bbox_left = "yolo_bbox_test300fpsmp4_left.csv";
extern const std::string file_yolo_class_left = "yolo_class_test300fpsmp4_left.csv";
extern const std::string file_tm_bbox_left = "tm_bbox_test300fpsmp4_left.csv";
extern const std::string file_tm_class_left = "tm_class_test300fpsmp4_left.csv";
extern const std::string file_seq_bbox_left = "seqData_bbox_test300fpsmp4_left.csv";
extern const std::string file_seq_class_left = "seqData_class_test300fpsmp4_left.csv";
extern const std::string file_yolo_bbox_right = "yolo_bbox_test300fpsmp4_right.csv";
extern const std::string file_yolo_class_right = "yolo_class_test300fpsmp4_right.csv";
extern const std::string file_tm_bbox_right = "tm_bbox_test300fpsmp4_right.csv";
extern const std::string file_tm_class_right = "tm_class_test300fpsmp4_right.csv";
extern const std::string file_seq_bbox_right = "seqData_bbox_test300fpsmp4_right.csv";
extern const std::string file_seq_class_right = "seqData_class_test300fpsmp4_right.csv";
extern const std::string file_3d = "triangulation.csv";
extern const std::string file_target = "target.csv";

// queue definitions
std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
std::queue<int> queueFrameIndex;  // queue for frame index

//Yolo signals
std::queue<bool> queueYolo_tracker2seq_left,queueYolo_tracker2seq_right;
std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;
std::queue<bool> queue_tri2predict;

//mosse
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_left;
std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_right;

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
std::queue<bool> queueStartYolo_left; //if new Yolo inference can start
std::queue<bool> queueStartYolo_right; //if new Yolo inference can start

// right cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

// sequential data
std::vector<std::vector<std::vector<int>>> seqData_left, seqData_right; //storage for sequential data
std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
std::queue<int> queueTargetFrameIndex_right;                      // TM estimation frame
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

//matching
std::queue<std::vector<int>> queueUpdateLabels_left;
std::queue<std::vector<int>> queueUpdateLabels_right;
/* for predict */
std::queue< std::vector<std::vector<std::vector<int>>>> queue3DData;
//mutex
std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex


#endif