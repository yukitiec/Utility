#pragma once

#pragma once

#ifndef YOLOPOSE_BATCH_H
#define YOLOPOSE_BATCH_H

#include "stdafx.h"
#include "global_parameters.h"

extern const int LEFT;
extern const int RIGHT;
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;

/*  YOLO class definition  */
class YOLOPoseBatch
{

private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8m-pose_640_1280.torchscript";
    const int originalWidth = 640;
    const int originalHeight = 640;
    int frameWidth = 640;
    int frameHeight = 1280;
    const int yoloWidth = 640;
    const int yoloHeight = 1280;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.1;
    const float ConfThreshold = 0.4;
    const float IoUThresholdIdentity = 0.25; // for maitainig consistency of tracking
    /* initialize function */
    void initializeDevice()
    {
        // set device
        if (torch::cuda::is_available())
        {
            // device = new torch::Device(devicetype, 0);
            device = new torch::Device(torch::kCUDA);
            std::cout << "set cuda" << std::endl;
        }
        else
        {
            device = new torch::Device(torch::kCPU);
            std::cout << "set CPU" << std::endl;
        }
    }

    void loadModel()
    {
        // read param
        mdl = torch::jit::load(yolofilePath, *device);
        mdl.to(*device);
        mdl.eval();
        std::cout << "load model" << std::endl;
    }

public:
    // constructor for YOLODetect
    YOLOPoseBatch()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLOBatch construtor has finished!" << std::endl;
    };
    ~YOLOPoseBatch() { delete device; }; // Deconstructor

    void detect(cv::Mat1b& frame, int& frameIndex, int& counter, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_left, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_left,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_right, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_right) //, const int frameIndex, std::vector<std::vector<cv::Rect2i>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /* inference by YOLO
         *  Args:
         *      frame : img
         *      posSaver : storage for saving detected position
         *      queueYoloTemplate : queue for pushing detected img
         *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
         *      queueClassIndex : queue for pushing detected
         */

         /* preprocess img */
        torch::Tensor imgTensor;
        preprocessImg(frame, imgTensor);
        // std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
        // std::cout << "finish preprocess" << std::endl;

        // std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;

        std::vector<cv::Rect2i> roiLatest;
        /* get latest roi for comparing new detection */
        /*
        if (!queueOFSearchRoi.empty())
        {
          getLatestRoi(roiLatest);
        }
        */
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        // std::cout << "finish inference" << std::endl;
        preds = preds.permute({ 0, 2, 1 }); // change order : (1,56,2100) -> (1,2100,56)
        // std::cout << "preds size:" << preds.sizes() << std::endl;
        // std::cout << "preds size : " << preds << std::endl;
        std::vector<torch::Tensor> detectedBoxesHuman; //(n,56)
        /*detect human */
        nonMaxSuppressionHuman(preds, detectedBoxesHuman, ConfThreshold, IoUThreshold);
        std::cout << "detectedBboxesHuman size=" << detectedBoxesHuman.size() << std::endl;
        /* get keypoints from detectedBboxesHuman -> shoulder,elbow,wrist */
        std::vector<std::vector<std::vector<int>>> keyPoints; // vector for storing keypoints
        std::vector<int> humanPos; //whether human is in left or right
        /* if human detected, extract keypoints */
        if (!detectedBoxesHuman.empty())
        {
            //std::cout << "Human detected!" << std::endl;
            keyPointsExtractor(detectedBoxesHuman, keyPoints,humanPos, ConfThreshold);
            /*push updated data to queue*/
            push2Queue(frame, frameIndex, keyPoints, roiLatest, humanPos,posSaver_left,posSaver_right, queueYoloOldImgSearch_left, queueYoloSearchRoi_left, queueYoloOldImgSearch_right, queueYoloSearchRoi_right);
            // std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
            /* draw keypoints in the frame */
            //drawCircle(frame, keyPoints, counter);
        }
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        cv::Mat yoloimg; // define yolo img type
        cv::imwrite("input.jpg", frame);
        cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
        cv::resize(yoloimg, yoloimg, YOLOSize);
        //cv::imwrite("yoloimg.jpg", yoloimg);
        //std::cout << "yoloImg.height" << yoloimg.rows << ", yoloimg.width" << yoloimg.cols << std::endl;
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
        imgTensor = imgTensor.div(255);                                                            // normalization
        imgTensor = imgTensor.unsqueeze(0);                                                        //(1,3,320,320)
        imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
    }

    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold)
    {
        /* non max suppression : remove overlapped bbox
         * Args:
         *   prediction : (1,2100,,6)
         * Return:
         *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
         */

        torch::Tensor xc = prediction.select(2, 4) > confThreshold;                       // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor x = prediction.index_select(1, torch::nonzero(xc[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
        x = x.index_select(1, x.select(2, 4).argsort(1, true).squeeze());                 // ball : sorted in descending order
        x = x.squeeze();                                                                  //(1,n,56) -> (n,56)
        bool boolLeft = false;
        bool boolRight = false;
        if (x.size(0) != 0)
        {
            /* 1 dimension */
            if (x.dim() == 1)
            {
                // std::cout << "top defined" << std::endl;
                detectedBoxesHuman.push_back(x.cpu());
            }
            /* more than 2 dimensions */
            else
            {
                // std::cout << "top defined" << std::endl;
                if (x[0][0].item<int>()<= originalWidth)
                {
                    //std::cout << "first Human is left" << std::endl;
                    detectedBoxesHuman.push_back(x[0].cpu());
                    boolLeft = true;
                }
                else if (x[0][0].item<int>() > originalWidth)
                {
                    //std::cout << "first human is right" << std::endl;
                    detectedBoxesHuman.push_back(x[0].cpu());
                    boolRight = true;
                }
                // std::cout << "push back top data" << std::endl;
                // for every candidates
                /* if adapt many humans, validate here */
                if (x.size(0) >= 2)
                {
                    //std::cout << "nms start" << std::endl;
                    nms(x, detectedBoxesHuman, iouThreshold,boolLeft,boolRight); // exclude overlapped bbox : 20 milliseconds
                    //std::cout << "num finished" << std::endl;
                }
            }
        }
    }

    torch::Tensor xywh2xyxy(torch::Tensor& x)
    {
        torch::Tensor y = x.clone();
        y[0] = x[0] - x[2] / 2; // left
        y[1] = x[1] - x[3] / 2; // top
        y[2] = x[0] + x[2] / 2; // right
        y[3] = x[1] + x[3] / 2; // bottom
        return y;
    }

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold,bool& boolLeft, bool& boolRight)
    {
        /* calculate IoU for excluding overlapped bboxes
         *
         * bbox1,bbox2 : [left,top,right,bottom,score0,score1]
         *
         */
        int numBoxes = x.size(0);
        torch::Tensor box;
        int counter = 0;
        // there are some overlap between two bbox
        for (int i = 1; i < numBoxes; i++)
        {
            if (boolLeft && boolRight) break;
            //detect only 1 human in each image
            if ((x[i][0].item<int>() <= originalWidth && !boolLeft) || (x[i][0].item<int>() > originalWidth && !boolRight))
            {
                box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

                bool addBox = true; // if save bbox as a new detection

                for (const torch::Tensor& savedBox : detectedBoxes)
                {
                    float iou = calculateIoU(box, savedBox); // calculate IoU
                    /* same bbox : already found -> not add */
                    if (iou > iouThreshold)
                    {
                        addBox = false;
                        break; // next iteration
                    }
                }
                /* new tracker */
                if (addBox)
                {
                    detectedBoxes.push_back(x[i].cpu());
                    if (x[i][0].item<int>() <= originalWidth && !boolLeft) boolLeft = true;
                    if (x[i][0].item<int>() > originalWidth && !boolRight) boolRight = true;
                }
            }
        }
    }

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2)
    {
        float left = std::max(box1[0].item<float>(), box2[0].item<float>());
        float top = std::max(box1[1].item<float>(), box2[1].item<float>());
        float right = std::min(box1[2].item<float>(), box2[2].item<float>());
        float bottom = std::min(box1[3].item<float>(), box2[3].item<float>());

        if (left < right && top < bottom)
        {
            float intersection = (right - left) * (bottom - top);
            float area1 = ((box1[2] - box1[0]) * (box1[3] - box1[1])).item<float>();
            float area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])).item<float>();
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints,std::vector<int>& humanPos, const int& ConfThreshold)
    {
        int numDetections = detectedBboxesHuman.size();
        bool boolLeft=false;
        /* iterate for all detections of humand */
        for (int i = 0; i < numDetections; i++)
        {
            std::vector<std::vector<int>> keyPointsTemp;
            /* iterate for 3 joints positions */
            for (int j = 5; j < 11; j++)
            {
                /* if keypoints score meet criteria */
                if (detectedBboxesHuman[i][3 * j + 7].item<float>() > ConfThreshold)
                {
                    //left
                    if ((static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()) <= originalWidth))
                    {
                        boolLeft = true; //left person
                        keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()), static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                    }
                    //right
                    else
                    {
                        boolLeft = false;
                        keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>() - originalWidth), static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                    }
                }
                else
                {
                    keyPointsTemp.push_back({ -1, -1 });
                }
            }
            //std::cout << "boolLeft=" << boolLeft << std::endl;
            keyPoints.push_back(keyPointsTemp);
            if (boolLeft) humanPos.push_back(LEFT);
            else humanPos.push_back(RIGHT);
        }
    }

    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter)
    {
        /*number of detections */
        for (int k = 0; k < ROI.size(); k++)
        {
            /*for all joints */
            for (int i = 0; i < ROI[k].size(); i++)
            {
                if (ROI[k][i][0] != -1)
                {
                    cv::circle(frame, cv::Point(ROI[k][i][0], ROI[k][i][1]), 5, cv::Scalar(125), -1);
                }
            }
        }
        std::string save_path = std::to_string(counter) + ".jpg";
        cv::imwrite(save_path, frame);
    }

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2i>& roiLatest, std::vector<int>& humanPos,std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_left, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver_right,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_left, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_left,
        std::queue<std::vector<std::vector<cv::Mat1b>>>& queueYoloOldImgSearch_right, std::queue<std::vector<std::vector<cv::Rect2i>>>& queueYoloSearchRoi_right)
    {
        /* check roi Latest
         * if tracking was successful -> update and
         * else : update roi and imgSearch and calculate features. push data to queue
         */
        if (!keyPoints.empty())
        {
            std::vector<std::vector<cv::Rect2i>> humanJoints_left,humanJoints_right; // for every human
            std::vector<std::vector<cv::Mat1b>> imgHuman_left, imgHuman_right;
            std::vector<std::vector<std::vector<int>>> humanJointsCenter_left, humanJointsCenter_right;
            /* for every person */
            for (int i = 0; i < keyPoints.size(); i++)
            {
                std::vector<cv::Rect2i> joints; // for every joint
                std::vector<cv::Mat1b> imgJoint;
                std::vector<std::vector<int>> jointsCenter;
                //left person
                if (humanPos[i] == LEFT)
                {
                    //std::cout << "left" << std::endl;
                    /* for every joints */
                    for (int j = 0; j < keyPoints[i].size(); j++)
                    {
                        organize_left(frame, frameIndex, keyPoints[i][j], joints, imgJoint, jointsCenter);
                    }
                    humanJoints_left.push_back(joints);
                    if (!imgJoint.empty())
                    {
                        imgHuman_left.push_back(imgJoint);
                    }
                    humanJointsCenter_left.push_back(jointsCenter);
                }
                //right person
                else
                {
                    //std::cout << "right" << std::endl;
                    /* for every joints */
                    for (int j = 0; j < keyPoints[i].size(); j++)
                    {
                        organize_right(frame, frameIndex, keyPoints[i][j], joints, imgJoint, jointsCenter);
                    }
                    humanJoints_right.push_back(joints);
                    if (!imgJoint.empty())
                    {
                        imgHuman_right.push_back(imgJoint);
                    }
                    humanJointsCenter_right.push_back(jointsCenter);
                }
                
            }
            // push data to queue
            //std::unique_lock<std::mutex> lock(mtxYolo); // exclude other accesses
            queueYoloSearchRoi_left.push(humanJoints_left);
            queueYoloSearchRoi_right.push(humanJoints_right);
            if (!imgHuman_left.empty()) queueYoloOldImgSearch_left.push(imgHuman_left);
            if (!imgHuman_right.empty()) queueYoloOldImgSearch_right.push(imgHuman_right);
            posSaver_left.push_back(humanJointsCenter_left);
            posSaver_right.push_back(humanJointsCenter_right);
        }
    }

    void organize_left(cv::Mat1b& frame, int& frameIndex,std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        if (static_cast<int>(pos[0]) >= 0)
        {
            int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
            int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0),originalHeight);
            int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth),0);
            int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight),0);
            cv::Rect2i roi(left, top, right - left, bottom - top);
            jointsCenter.push_back({ frameIndex, pos[0], pos[1] });
            joints.push_back(roi);
            imgJoint.push_back(frame(roi));
        }
        /* keypoints can't be detected */
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }

    void organize_right(cv::Mat1b& frame, int& frameIndex, std::vector<int>& pos, std::vector<cv::Rect2i>& joints, std::vector<cv::Mat1b>& imgJoint, std::vector<std::vector<int>>& jointsCenter)
    {
        if (static_cast<int>(pos[0]) >= 0)
        {
            int left = std::min(std::max(static_cast<int>(pos[0] - roiWidthYolo / 2), 0), originalWidth);
            int top = std::min(std::max(static_cast<int>(pos[1] - roiHeightYolo / 2), 0), originalHeight);
            int right = std::max(std::min(static_cast<int>(pos[0] + roiWidthYolo / 2), originalWidth), 0);
            int bottom = std::max(std::min(static_cast<int>(pos[1] + roiHeightYolo / 2), originalHeight), 0);
            cv::Rect2i roi(left, top, right - left, bottom - top);
            joints.push_back(roi);
            roi.x += originalWidth;
            imgJoint.push_back(frame(roi));
            jointsCenter.push_back({ frameIndex, pos[0], pos[1] });
        }
        /* keypoints can't be detected */
        else
        {
            jointsCenter.push_back({ frameIndex, -1, -1 });
            joints.emplace_back(-1, -1, -1, -1);
        }
    }
};

#endif