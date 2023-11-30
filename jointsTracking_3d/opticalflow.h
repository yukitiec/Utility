#pragma once

#pragma once

#ifndef OPTICALFLOW_H
#define OPTICALFLOW_H

#include "stdafx.h"
#include "global_parameters.h"

std::vector<float> defaultMove{ 0.0,0.0 };
extern const float DIF_THRESHOLD;
extern const float MIN_MOVE; //minimum opticalflow movement
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int dense_vel_method;

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

class OpticalFlow
{
private:
    const cv::Size ROISize{ roiWidthOF, roiHeightOF };
    const int originalWidth = 320;
    const int originalHeight = 320;
public:
    OpticalFlow()
    {
        std::cout << "construct OpticalFlow class" << std::endl;
    }

    void main(cv::Mat1b& frame, const int& frameIndex, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueOFOldImgSearch, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueOFSearchRoi,
        std::queue<std::vector<std::vector<std::vector<float>>>>& queuePreviousMove, std::queue<std::vector<std::vector<std::vector<int>>>>& queueTriangulation)
    {
        /* optical flow process for each joints */
        std::vector<std::vector<cv::Mat1b>> previousImg;           //[number of human,0~6,cv::Mat1b]
        std::vector<std::vector<cv::Rect2i>> searchRoi;            //[number of human,6,cv::Rect2i], if tracker was failed, roi.x == -1
        std::vector<std::vector<std::vector<float>>> previousMove; //[number of human,6,movement in x and y] ROI movement of each joint
        getPreviousData(previousImg, searchRoi, previousMove, queueYoloOldImgSearch, queueYoloSearchRoi, queueOFOldImgSearch, queueOFSearchRoi, queuePreviousMove);
        //std::cout << "finish getting previous data " << std::endl;
        /* start optical flow process */
        /* for every human */
        std::vector<std::vector<cv::Mat1b>> updatedImgHuman;
        std::vector<std::vector<cv::Rect2i>> updatedSearchRoiHuman;
        std::vector<std::vector<std::vector<float>>> updatedMoveDists;
        std::vector<std::vector<std::vector<int>>> updatedPositionsHuman;
        //std::cout << "human =" << searchRoi.size() << " !!!!!!!!!!!!" << std::endl;
        //for (cv::Rect2i& roi : searchRoi[0])
        //    std::cout << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
        //std::cout << "previousMove.size()=" << previousMove.size() << ", previousImg.size()=" << previousImg.size() << std::endl;
        if (!searchRoi.empty())
        {
            for (int i = 0; i < searchRoi.size(); i++)
            {
                //std::cout << i << "-th human::" << "previousMove.size()=" << previousMove[i].size() << std::endl;
                //std::cout<<"previousImg.size()=" << previousImg[i].size() << std::endl;
                /* for every joints */
                std::vector<cv::Mat1b> updatedImgJoints;
                std::vector<cv::Rect2i> updatedSearchRoi;
                std::vector<std::vector<float>> moveJoints; //roi movement
                std::vector<std::vector<int>> updatedPositions;
                std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
                cv::Mat1b updatedImgLeftShoulder, updatedImgRightShoulder, updatedImgLeftElbow, updatedImgRightElbow, updatedImgLeftWrist, updatedImgRightWrist;
                cv::Rect2i updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
                std::vector<float> moveLS, moveRS, moveLE, moveRE, moveLW, moveRW;
                bool boolLeftShoulder = false;
                bool boolRightShoulder = false;
                bool boolLeftElbow = false;
                bool boolRightElbow = false;
                bool boolLeftWrist = false;
                bool boolRightWrist = false;
                std::vector<std::thread> threadJoints;
                /* start optical flow process for each joints */
                /* left shoulder */
                int counterTracker = 0;
                if (searchRoi[i][0].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][0]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(moveLS), std::ref(updatedPosLeftShoulder));
                    boolLeftShoulder = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftShoulder.x = -1;
                    updatedSearchRoiLeftShoulder.y = -1;
                    updatedSearchRoiLeftShoulder.width = -1;
                    updatedSearchRoiLeftShoulder.height = -1;
                    updatedPosLeftShoulder = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right shoulder */
                if (searchRoi[i][1].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][1]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightShoulder), std::ref(updatedSearchRoiRightShoulder), std::ref(moveRS), std::ref(updatedPosRightShoulder));
                    boolRightShoulder = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightShoulder.x = -1;
                    updatedSearchRoiRightShoulder.y = -1;
                    updatedSearchRoiRightShoulder.width = -1;
                    updatedSearchRoiRightShoulder.height = -1;
                    updatedPosRightShoulder = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* left elbow */
                if (searchRoi[i][2].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][2]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(moveLE), std::ref(updatedPosLeftElbow));
                    boolLeftElbow = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftElbow.x = -1;
                    updatedSearchRoiLeftElbow.y = -1;
                    updatedSearchRoiLeftElbow.width = -1;
                    updatedSearchRoiLeftElbow.height = -1;
                    updatedPosLeftElbow = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right elbow */
                if (searchRoi[i][3].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][3]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(moveRE), std::ref(updatedPosRightElbow));
                    boolRightElbow = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightElbow.x = -1;
                    updatedSearchRoiRightElbow.y = -1;
                    updatedSearchRoiRightElbow.width = -1;
                    updatedSearchRoiRightElbow.height = -1;
                    updatedPosRightElbow = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* left wrist */
                if (searchRoi[i][4].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][4]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(moveLW), std::ref(updatedPosLeftWrist));
                    boolLeftWrist = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftWrist.x = -1;
                    updatedSearchRoiLeftWrist.y = -1;
                    updatedSearchRoiLeftWrist.width = -1;
                    updatedSearchRoiLeftWrist.height = -1;
                    updatedPosLeftWrist = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right wrist */
                if (searchRoi[i][5].x >= 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, this, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][5]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(moveRW), std::ref(updatedPosRightWrist));
                    boolRightWrist = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightWrist.x = -1;
                    updatedSearchRoiRightWrist.y = -1;
                    updatedSearchRoiRightWrist.width = -1;
                    updatedSearchRoiRightWrist.height = -1;
                    updatedPosRightWrist = std::vector<int>{ frameIndex, -1, -1 };
                }
                std::cout << "all threads have started :: " << threadJoints.size() << std::endl;
                /* wait for all thread has finished */
                int counterThread = 0;
                if (!threadJoints.empty())
                {
                    for (std::thread& thread : threadJoints)
                    {
                        thread.join();
                        counterThread++;
                    }
                    std::cout << counterThread << " threads have finished!" << std::endl;
                }
                else
                {
                    std::cout << "no thread has started" << std::endl;
                    //std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                //std::cout << i << "-th human" << std::endl;
                /* combine all data and push data to queue */
                /* search roi */
                updatedSearchRoi.push_back(updatedSearchRoiLeftShoulder);
                updatedSearchRoi.push_back(updatedSearchRoiRightShoulder);
                updatedSearchRoi.push_back(updatedSearchRoiLeftElbow);
                updatedSearchRoi.push_back(updatedSearchRoiRightElbow);
                updatedSearchRoi.push_back(updatedSearchRoiLeftWrist);
                updatedSearchRoi.push_back(updatedSearchRoiRightWrist);
                updatedPositions.push_back(updatedPosLeftShoulder);
                updatedPositions.push_back(updatedPosRightShoulder);
                updatedPositions.push_back(updatedPosLeftElbow);
                updatedPositions.push_back(updatedPosRightElbow);
                updatedPositions.push_back(updatedPosLeftWrist);
                updatedPositions.push_back(updatedPosRightWrist);
                /* updated img */
                /* left shoulder */
                if (updatedSearchRoi[0].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftShoulder);
                    moveJoints.push_back(moveLS);
                }
                /* right shoulder*/
                if (updatedSearchRoi[1].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgRightShoulder);
                    moveJoints.push_back(moveRS);
                }
                /*left elbow*/
                if (updatedSearchRoi[2].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftElbow);
                    moveJoints.push_back(moveLE);
                }
                /*right elbow */
                if (updatedSearchRoi[3].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgRightElbow);
                    moveJoints.push_back(moveRE);
                }
                /* left wrist*/
                if (updatedSearchRoi[4].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftWrist);
                    moveJoints.push_back(moveLW);
                }
                /*right wrist*/
                if (updatedSearchRoi[5].x >= 0)
                {
                    updatedImgJoints.push_back(updatedImgRightWrist);
                    moveJoints.push_back(moveRW);
                }
                /* combine all data for one human */
                updatedSearchRoiHuman.push_back(updatedSearchRoi);
                updatedPositionsHuman.push_back(updatedPositions);
                if (!updatedImgJoints.empty())
                {
                    updatedImgHuman.push_back(updatedImgJoints);
                    updatedMoveDists.push_back(moveJoints);
                }
            }
            /* push updated data to queue */
            queueOFSearchRoi.push(updatedSearchRoiHuman);
            if (!updatedImgHuman.empty())
            {
                queueOFOldImgSearch.push(updatedImgHuman);
                queuePreviousMove.push(updatedMoveDists);
            }
            //std::cout << "save to posSaver! posSaver size=" << posSaver.size() << std::endl;

            /* arrange posSaver */
            if (!posSaver.empty())
            {
                std::vector<std::vector<std::vector<int>>> all; //all human data
                // for each human
                for (int i = 0; i < updatedPositionsHuman.size(); i++)
                {
                    std::vector<std::vector<int>> tempHuman;
                    /* same human */
                    if (posSaver[posSaver.size() - 1].size() > i)
                    {
                        // for each joint
                        for (int j = 0; j < updatedPositionsHuman[i].size(); j++)
                        {
                            // detected
                            if (updatedPositionsHuman[i][j][1] != -1) tempHuman.push_back(updatedPositionsHuman[i][j]);
                            // not detected
                            else
                            {
                                // already detected
                                if (posSaver[posSaver.size() - 1][i][j][1] != -1)
                                    tempHuman.push_back(posSaver[posSaver.size() - 1][i][j]); //adapt last detection
                                // not detected yet
                                else
                                    tempHuman.push_back(updatedPositionsHuman[i][j]); //-1
                            }
                        }
                    }
                    //new human
                    else
                        tempHuman = updatedPositionsHuman[i];
                    all.push_back(tempHuman); //push human data
                }
                posSaver.push_back(all);
                if (!queueTriangulation.empty()) queueTriangulation.pop();
                queueTriangulation.push(all);
            }
            // first detection
            else
            {
                posSaver.push_back(updatedPositionsHuman);
                if (!queueTriangulation.empty()) queueTriangulation.pop();
                queueTriangulation.push(updatedPositionsHuman);
            }
        }
        // no data
        else
        {
            //nothing to do
        }
    }

    void getPreviousData(std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2i>>& searchRoi, std::vector<std::vector<std::vector<float>>>& moveDists,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi, 
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueOFOldImgSearch,std::queue<std::vector<std::vector<cv::Rect2i>>>& queueOFSearchRoi,
        std::queue<std::vector<std::vector<std::vector<float>>>>& queuePreviousMove)
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
            queuePreviousMove.pop();
        }
        else if (queueOFOldImgSearch.empty() && !queueOFSearchRoi.empty())
        {
            searchRoi = queueOFSearchRoi.front();
            queueOFSearchRoi.pop();
        }
        std::vector<std::vector<cv::Mat1b>> previousYoloImg;
        std::vector<std::vector<cv::Rect2i>> searchYoloRoi;
        if (!queueYoloOldImgSearch.empty())
        {
            std::cout << "yolo data is available" << std::endl;
            getYoloData(previousYoloImg, searchYoloRoi, queueYoloOldImgSearch, queueYoloSearchRoi);
            /* update data here */
            /* iterate for all human detection */
            //std::cout << "searchYoloRoi size : " << searchYoloRoi.size() << std::endl;
            // std::cout << "searchRoi by optical flow size : " << searchRoi.size() << ","<<searchRoi[0].size()<<std::endl;
            // std::cout << "successful trackers of optical flow : " << previousImg.size() << std::endl;
            for (int i = 0; i < searchYoloRoi.size(); i++)
            {
                //std::cout << i << "-th human" << std::endl;
                /* some OF tracking were successful */
                if (!previousImg.empty())
                {
                    /* existed human detection */
                    if (i < previousImg.size())
                    {
                        //std::cout << "previousImg : num of human : " << previousImg.size() << std::endl;
                        /* for all joints */
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        int counterTrackerOF = 0; // number of successful trackers by Optical Flow
                        for (const cv::Rect2i& roi : searchRoi[i])
                        {
                            //std::cout << "update data with Yolo detection : " << counterJoint << "-th joint" << std::endl;
                            /* tracking is failed -> update data with yolo data */
                            if (roi.x < 0)
                            {
                                /* yolo detect joints -> update data */
                                if (searchYoloRoi[i][counterJoint].x >= 0)
                                {
                                    //std::cout << "update OF tracker features with yolo detection" << std::endl;
                                    searchRoi[i].insert(searchRoi[i].begin() + counterJoint, searchYoloRoi[i][counterJoint]);
                                    previousImg[i].insert(previousImg[i].begin() + counterTrackerOF, previousYoloImg[i][counterYoloImg]);
                                    moveDists[i].insert(moveDists[i].begin() + counterTrackerOF, { 0.0,0.0 });
                                    counterJoint++;
                                    counterYoloImg++;
                                    counterTrackerOF++;
                                }
                                /* yolo can't detect joint -> not updated data */
                                else
                                {
                                    //std::cout << "Yolo didn't detect joint" << std::endl;
                                    counterJoint++;
                                }
                            }
                            /* tracking is successful */
                            else
                            {
                                //std::cout << "tracking was successful" << std::endl;
                                /* update template image with Yolo's one */
                                if (boolChange)
                                {
                                    if (searchYoloRoi[i][counterJoint].x >= 0)
                                    {
                                        //std::cout << "update OF tracker with yolo detection" << std::endl;
                                        previousImg[i].at(counterTrackerOF) = previousYoloImg[i][counterYoloImg];
                                        moveDists[i].at(counterTrackerOF) = std::vector<float>{ 0.0,0.0 };
                                        if (static_cast<float>(roi.x - searchYoloRoi[i][counterJoint].x) > DIF_THRESHOLD || static_cast<float>(searchRoi[i][counterJoint].y - searchYoloRoi[i][counterJoint].y) > DIF_THRESHOLD ||
                                            (std::pow(static_cast<float>(roi.x - searchYoloRoi[i][counterJoint].x),2)+std::pow(static_cast<float>(roi.x - searchYoloRoi[i][counterJoint].x),2) > DIF_THRESHOLD*DIF_THRESHOLD))
                                        {
                                            searchRoi[i].at(counterJoint) = searchYoloRoi[i][counterJoint];
                                        }
                                        counterYoloImg++;
                                    }
                                }
                                /* not update template images -> keep tracking */
                                else
                                {
                                    if (searchYoloRoi[i][counterJoint].x >= 0)
                                        counterYoloImg++;
                                }
                                counterJoint++;
                                counterTrackerOF++;
                                //std::cout << "update iterator" << std::endl;
                            }
                        }
                    }
                    /* new human detecte3d */
                    else
                    {
                        //std::cout << "new human was detected by Yolo " << std::endl;
                        int counterJoint = 0;
                        int counterYoloImg = 0;
                        std::vector<cv::Rect2i> joints;
                        std::vector<cv::Mat1b> imgJoints;
                        std::vector<std::vector<float>> moveJoints;
                        /* for every joints */
                        for (const cv::Rect2i& roi : searchYoloRoi[i])
                        {
                            /* keypoint is found */
                            if (roi.x >= 0)
                            {
                                joints.push_back(roi);
                                imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
                                moveJoints.push_back(defaultMove);
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
                    searchRoi = std::vector<std::vector<cv::Rect2i>>(); // initialize searchRoi for avoiding data
                    //std::cout << "Optical Flow :: failed or first Yolo detection " << std::endl;
                    int counterJoint = 0;
                    int counterYoloImg = 0;
                    std::vector<cv::Rect2i> joints;
                    std::vector<cv::Mat1b> imgJoints;
                    std::vector<std::vector<float>> moveJoints;
                    std::vector<std::vector<cv::Point2f>> features;
                    /* for every joints */
                    for (const cv::Rect2i& roi : searchYoloRoi[i])
                    {
                        /* keypoint is found */
                        if (roi.x >= 0)
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
        

    void getYoloData(std::vector<std::vector<cv::Mat1b>>& previousYoloImg, std::vector<std::vector<cv::Rect2i>>& searchYoloRoi,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi)
    {
        //std::unique_lock<std::mutex> lock(mtxYolo);
        previousYoloImg = queueYoloOldImgSearch.front();
        searchYoloRoi = queueYoloSearchRoi.front();
        queueYoloOldImgSearch.pop();
        queueYoloSearchRoi.pop();
    }

    void opticalFlow(const cv::Mat1b& frame, const int& frameIndex, cv::Mat1b& previousImg, cv::Rect2i& searchRoi, std::vector<float>& previousMove,
        cv::Mat1b& updatedImg, cv::Rect2i& updatedSearchRoi, std::vector<float>& updatedMove, std::vector<int>& updatedPos)
    {
        // Calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
        cv::Mat1b croppedImg = frame(searchRoi);
        if (croppedImg.rows!=roiHeightOF || croppedImg.cols!=roiWidthOF) cv::resize(croppedImg, croppedImg, ROISize);
        if (previousImg.rows != roiHeightOF || previousImg.cols != roiWidthOF) cv::resize(previousImg, previousImg, ROISize);
        // Calculate Optical Flow
        cv::Mat flow(previousImg.size(), CV_32FC2);                                           // prepare matrix for saving dense optical flow
        cv::calcOpticalFlowFarneback(previousImg, croppedImg, flow, 0.5, 2, 5, 3, 2, 1.6, 0); // calculate dense optical flow default :: 0.5, 2, 4, 4, 3, 1.6, 0
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
        std::vector<float> velocities;
        std::vector<std::vector<float>> candidates_vel;
        float max = MIN_MOVE;
        for (int y = 1; y <= rows; ++y)
        {
            for (int x = 1; x <= cols; ++x)
            {
                cv::Point2f flowVec = flow.at<cv::Point2f>(std::max((y - 1) * (skipPixel + 1), 0), std::max((x - 1) * (skipPixel + 1), 0)); // get velocity of position (y,x)
                // Access flowVec.x and flowVec.y for the horizontal and vertical components of velocity.
                if (((flowVec.x >= -previousMove[0] - epsironMove && flowVec.x <= -previousMove[0] + epsironMove) && (flowVec.y >= -previousMove[1] - epsironMove && flowVec.y <= -previousMove[1] + epsironMove))
                    || ((flowVec.x * flowVec.x + flowVec.y * flowVec.y) <= MIN_MOVE))
                {
                    numBackGround += 1;/* this may seem background optical flow */
                }
                else
                {
                    //adapt average value
                    if (dense_vel_method == 0)
                    {
                        vecX += flowVec.x;
                        vecY += flowVec.y;
                        numPixels += 1;
                    }
                    //adapt max value
                    else if(dense_vel_method == 1)
                    {
                        if (std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2) > max)
                        {
                            vecX = flowVec.x;
                            vecY = flowVec.y;
                            max = std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2);
                        }
                    }
                    //adapt first, second, or third quarter value
                    else if (dense_vel_method == 2 || dense_vel_method == 3 || dense_vel_method == 4)
                    {
                        velocities.push_back(std::pow(flowVec.x, 2) + std::pow(flowVec.y, 2));
                        candidates_vel.push_back({ flowVec.x,flowVec.y });
                    }
                    
                }
            }
        }
        //std::cout << "background = " << numBackGround << ", adapted optical flow = " << numPixels << std::endl;
        //average method
        if (dense_vel_method == 0)
        {
            vecX /= numPixels;
            vecY /= numPixels;
        }
        if (!velocities.empty())
        {
            //median value
            if (dense_vel_method == 2 || ((dense_vel_method == 3 || dense_vel_method == 4) && velocities.size() <=3))
            {
                float median = calculateMedian(velocities);
                // Iterate over the vector and find indices with the specified value
                auto it = std::find(velocities.begin(), velocities.end(), median);
                size_t index = std::distance(velocities.begin(), it);
                vecX = candidates_vel[index][0];
                vecY = candidates_vel[index][1];
            }
            //third quarter value
            else if (dense_vel_method == 3 && velocities.size() >= 4)
            {
                float thirdQuarter = calculateThirdQuarter(velocities);
                // Iterate over the vector and find indices with the specified value
                auto it = std::find(velocities.begin(), velocities.end(), thirdQuarter);
                size_t index = std::distance(velocities.begin(), it);
                vecX = candidates_vel[index][0];
                vecY = candidates_vel[index][1];
            }
            //first quarter value
            else if (dense_vel_method == 4 && velocities.size()>=4)
            {
                float firstQuarter = calculateFirstQuarter(velocities);
                // Iterate over the vector and find indices with the specified value
                auto it = std::find(velocities.begin(), velocities.end(), firstQuarter);
                size_t index = std::distance(velocities.begin(), it);
                vecX = candidates_vel[index][0];
                vecY = candidates_vel[index][1];
            }
        }
        
        std::cout << "vecX=" << vecX << ", vecY=" << vecY << std::endl;
        if (std::pow(vecX, 2) + std::pow(vecY, 2) >= MoveThreshold)
        {
            int updatedLeft = searchRoi.x + static_cast<int>(vecX);
            int updatedTop = searchRoi.y + static_cast<int>(vecY);
            int left = std::min(std::max(updatedLeft, 0),frame.cols);
            int top = std::min(std::max(updatedTop, 0), frame.rows);
            int right = std::max(std::min(left + roiWidthOF, frame.cols), 0);
            int bottom = std::max(std::min(top + roiHeightOF, frame.rows), 0);
            cv::Rect2i roi(left,top,right-left,bottom-top);
            //tracking was successful
            if (roi.width > roiWidthOF / 2 && roi.height > roiHeightOF / 2)
            {
                std::cout << "roi.x=" << roi.x << ", roi.y=" << roi.y << ", roi.width=" << roi.width << ", roi.height=" << roi.height << std::endl;
                updatedSearchRoi = roi;
                updatedMove = std::vector<float>{ vecX, vecY };
                // Update the previous frame and previous points
                updatedImg = croppedImg.clone();
                updatedPos = std::vector<int>{ frameIndex, static_cast<int>((left + right) / 2), static_cast<int>((top + bottom) / 2) };
            }
            //out of range
            else
            {
                updatedSearchRoi.x = -1;
                updatedSearchRoi.y = -1;
                updatedSearchRoi.width = -1;
                updatedSearchRoi.height = -1;
                updatedPos = std::vector<int>{ frameIndex, -1, -1 };
            }
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

    float calculateMedian(std::vector<float> vec) {
        // Check if the vector is not empty
        if (vec.empty()) {
            throw std::invalid_argument("Vector is empty");
        }

        // Calculate the middle index
        size_t size = vec.size();
        size_t middleIndex = size / 2;

        // Use std::nth_element to find the median
        std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

        return vec[middleIndex];
    }

    float calculateThirdQuarter(std::vector<float> vec) {
        // Check if the vector is not empty
        if (vec.empty()) {
            throw std::invalid_argument("Vector is empty");
        }

        // Calculate the middle index
        size_t size = vec.size();
        size_t middleIndex = static_cast<size_t>(size *(3/ 4));

        // Use std::nth_element to find the median
        std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

        return vec[middleIndex];
    }

    float calculateFirstQuarter(std::vector<float> vec) {
        // Check if the vector is not empty
        if (vec.empty()) {
            throw std::invalid_argument("Vector is empty");
        }

        // Calculate the middle index
        size_t size = vec.size();
        size_t middleIndex = static_cast<size_t>(size * (1 / 4));

        // Use std::nth_element to find the median
        std::nth_element(vec.begin(), vec.begin() + middleIndex, vec.end());

        return vec[middleIndex];
    }
};

#endif