#pragma once

#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include "stdafx.h"
#include "global_parameters.h"

extern std::queue<cv::Mat1b> queueFrame;
extern std::queue<int> queueFrameIndex;
/* yolo and optical flow */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2d>>> queueYoloSearchRoi;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2d>>> queueOFSearchRoi;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]

class OpticalFlow
{
public:
    OpticalFlow()
    {
        std::cout << "construct OpticalFlow class" << std::endl;
    }

    void getPreviousData(std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2d>>& searchRoi, std::vector<std::vector<std::vector<float>>>& moveDists)
    {
        /*
         *  if yolo data available -> update if tracking was failed
         *  else -> if tracking of optical flow was successful -> update features
         *  else -> if tracking of optical flow was failed -> wait for next update of yolo
         *  [TO DO]
         *  get Yolo data and OF data from queue if possible
         *  organize data
         */
        if (!queueOFOldImgSearch.empty())
        {
            previousImg = queueOFOldImgSearch.front();
            searchRoi = queueOFSearchRoi.front();
            moveDists = queuePreviousMove.front();
            queueOFOldImgSearch.pop();
            queueOFSearchRoi.pop();
            queuePreviousMove.front();
        }
        else if (queueOFOldImgSearch.empty() && !queueOFSearchRoi.empty())
        {
            searchRoi = queueOFSearchRoi.front();
            queueOFSearchRoi.pop();
        }
        std::vector<std::vector<cv::Mat1b>> previousYoloImg;
        std::vector<std::vector<cv::Rect2d>> searchYoloRoi;
        if (!queueYoloOldImgSearch.empty())
        {
            std::cout << "yolo data is available" << std::endl;
            getYoloData(previousYoloImg, searchYoloRoi);
            /* update data here */
            /* iterate for all human detection */
            std::cout << "searchYoloRoi size : " << searchYoloRoi.size() << std::endl;
            // std::cout << "searchRoi by optical flow size : " << searchRoi.size() << ","<<searchRoi[0].size()<<std::endl;
            // std::cout << "successful trackers of optical flow : " << previousImg.size() << std::endl;
            for (int i = 0; i < searchYoloRoi.size(); i++)
            {
                std::cout << i << "-th human" << std::endl;
                /* some OF tracking were successful */
                if (!previousImg.empty())
                {
                    /* existed human detection */
                    if (i < previousImg.size())
                    {
                        std::cout << "previousImg : num of human : " << previousImg.size() << std::endl;
                        /* for all joints */
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        int counterTrackerOF = 0; // number of successful trackers by Optical Flow
                        for (const cv::Rect2d& roi : searchRoi[i])
                        {
                            std::cout << "update data with Yolo detection : " << counterJoint << "-th joint" << std::endl;
                            /* tracking is failed -> update data with yolo data */
                            if (roi.x == -1)
                            {
                                /* yolo detect joints -> update data */
                                if (searchYoloRoi[i][counterJoint].x != -1)
                                {
                                    std::cout << "update OF tracker features with yolo detection" << std::endl;
                                    searchRoi[i].insert(searchRoi[i].begin() + counterJoint, searchYoloRoi[i][counterJoint]);
                                    previousImg[i].insert(previousImg[i].begin() + counterTrackerOF, previousYoloImg[i][counterYoloImg]);
                                    counterJoint++;
                                    counterYoloImg++;
                                    counterTrackerOF++;
                                }
                                /* yolo can't detect joint -> not updated data */
                                else
                                {
                                    std::cout << "Yolo didn't detect joint" << std::endl;
                                    counterJoint++;
                                }
                            }
                            /* tracking is successful */
                            else
                            {
                                std::cout << "tracking was successful" << std::endl;
                                /* update template image with Yolo's one */
                                if (boolChange)
                                {
                                    if (searchYoloRoi[i][counterJoint].x != -1)
                                    {
                                        std::cout << "update OF tracker with yolo detection" << std::endl;
                                        previousImg[i].at(counterTrackerOF) = previousYoloImg[i][counterYoloImg];
                                        counterYoloImg++;
                                    }
                                }
                                /* not update template images -> keep tracking */
                                else
                                {
                                    if (searchYoloRoi[i][counterJoint].x != -1)
                                    {
                                        counterYoloImg++;
                                    }
                                }
                                counterJoint++;
                                counterTrackerOF++;
                                std::cout << "update iterator" << std::endl;
                            }
                        }
                    }
                    /* new human detecte3d */
                    else
                    {
                        std::cout << "new human was detected by Yolo " << std::endl;
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        std::vector<cv::Rect2d> joints;
                        std::vector<cv::Mat1b> imgJoints;
                        std::vector<std::vector<float>> moveJoints;
                        /* for every joints */
                        for (const cv::Rect2d& roi : searchYoloRoi[i])
                        {
                            /* keypoint is found */
                            if (roi.x != -1)
                            {
                                joints.push_back(roi);
                                imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
                                moveJoints.push_back({ 0.0, 0.0 });
                                counterJoint++;
                                counterYoloImg++;
                            }
                            /* keypoints not found */
                            else
                            {
                                joints.push_back(roi);
                                counterJoint++;
                            }
                        }
                        searchRoi.push_back(joints);
                        if (!imgJoints.empty())
                        {
                            previousImg.push_back(imgJoints);
                            moveDists.push_back(moveJoints);
                        }
                    }
                }
                /* no OF tracking was successful or first yolo detection */
                else
                {
                    searchRoi = std::vector<std::vector<cv::Rect2d>>(); // initialize searchRoi for avoiding data
                    std::cout << "Optical Flow :: failed or first Yolo detection " << std::endl;
                    int counterJoint = 0;
                    int counterYoloImg = 0;
                    std::vector<cv::Rect2d> joints;
                    std::vector<cv::Mat1b> imgJoints;
                    std::vector<std::vector<float>> moveJoints;
                    std::vector<std::vector<cv::Point2f>> features;
                    /* for every joints */
                    for (const cv::Rect2d& roi : searchYoloRoi[i])
                    {
                        /* keypoint is found */
                        if (roi.x != -1)
                        {
                            joints.push_back(roi);
                            imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
                            moveJoints.push_back({ 0.0, 0.0 });
                            counterJoint++;
                            counterYoloImg++;
                        }
                        /* keypoints not found */
                        else
                        {
                            joints.push_back(roi);
                            counterJoint++;
                        }
                    }
                    searchRoi.push_back(joints);
                    if (!imgJoints.empty())
                    {
                        previousImg.push_back(imgJoints);
                        moveDists.push_back(moveJoints);
                    }
                }
            }
        }
    }

    void getYoloData(std::vector<std::vector<cv::Mat1b>>& previousYoloImg, std::vector<std::vector<cv::Rect2d>>& searchYoloRoi)
    {
        //std::unique_lock<std::mutex> lock(mtxYolo);
        previousYoloImg = queueYoloOldImgSearch.front();
        searchYoloRoi = queueYoloSearchRoi.front();
        queueYoloOldImgSearch.pop();
        queueYoloSearchRoi.pop();
    }

    void opticalFlow(const cv::Mat1b& frame, const int& frameIndex, cv::Mat1b& previousImg, cv::Rect2d& searchRoi, std::vector<float>& previousMove,
        cv::Mat1b& updatedImg, cv::Rect2d& updatedSearchRoi, std::vector<float>& updatedMove, std::vector<int>& updatedPos)
    {
        // Calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
        cv::Mat1b croppedImg = frame(searchRoi);
        // Calculate Optical Flow
        cv::Mat flow(previousImg.size(), CV_32FC2);                                           // prepare matrix for saving dense optical flow
        cv::calcOpticalFlowFarneback(previousImg, croppedImg, flow, 0.5, 2, 4, 4, 3, 1.6, 0); // calculate dense optical flow
        /*
         *void cv::calcOpticalFlowFarneback(
         *InputArray prev,         // Input previous frame (grayscale image)
         *InputArray next,         // Input next frame (grayscale image)
         *InputOutputArray flow,   // Output optical flow image (2-channel floating-point)
         *double pyr_scale,        // Image scale (< 1) to build pyramids
         *int levels,              // Number of pyramid levels
         *int winsize,             // Average window size
         *int iterations,          // Number of iterations at each pyramid level
         *int poly_n,              // Polynomial expansion size
         *double poly_sigma,       // Standard deviation for Gaussian filter
         *int flags                // Operation flags
         *);
         */
         // calculate velocity

        float vecX = 0;
        float vecY = 0;
        int numPixels = 0;
        int numBackGround = 0;
        int rows = static_cast<int>(flow.rows / (skipPixel + 1)); // number of rows adapted pixels
        int cols = static_cast<int>(flow.cols / (skipPixel + 1)); // number of cols adapted pixels
        for (int y = 1; y <= rows; ++y)
        {
            for (int x = 1; x <= cols; ++x)
            {
                cv::Point2f flowVec = flow.at<cv::Point2f>(std::max((y - 1) * (skipPixel + 1), 0), std::max((x - 1) * (skipPixel + 1), 0)); // get velocity of position (y,x)
                // Access flowVec.x and flowVec.y for the horizontal and vertical components of velocity.
                if ((flowVec.x >= -previousMove[0] - epsironMove && flowVec.x <= -previousMove[0] + epsironMove) && (flowVec.y >= -previousMove[1] - epsironMove && flowVec.y <= -previousMove[1] + epsironMove))
                {
                    /* this may seem background optical flow */
                    numBackGround += 1;
                }
                else
                {
                    vecX += flowVec.x;
                    vecY += flowVec.y;
                    numPixels += 1;
                }
            }
        }
        std::cout << "background = " << numBackGround << ", adapted optical flow = " << numPixels << std::endl;
        vecX /= numPixels;
        vecY /= numPixels;
        if (std::pow(vecX, 2) + std::pow(vecY, 2) >= MoveThreshold)
        {
            int updatedLeft = searchRoi.x + static_cast<int>(vecX);
            int updatedTop = searchRoi.y + static_cast<int>(vecY);
            cv::Rect2d roi(updatedLeft, updatedTop, roiWidthOF, roiHeightOF);
            updatedSearchRoi = roi;
            updatedMove = std::vector<float>{ vecX, vecY };
            // Update the previous frame and previous points
            updatedImg = croppedImg.clone();
            updatedPos = std::vector<int>{ frameIndex, static_cast<int>(updatedLeft + roiWidthOF / 2), static_cast<int>(updatedTop + roiHeightOF / 2) };
        }
        /* not move -> tracking was failed */
        else
        {
            updatedSearchRoi.x = -1;
            updatedSearchRoi.y = -1;
            updatedSearchRoi.width = -1;
            updatedSearchRoi.height = -1;
            updatedPos = std::vector<int>{ frameIndex, -1, -1 };
        }
    }
};

#endif