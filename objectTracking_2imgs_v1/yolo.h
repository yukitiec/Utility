#pragma once


#ifndef YOLO_H
#define YOLO_H

#include "stdafx.h"
#include "global_parameters.h"
#include "mosse.h"

// queue definition
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

//mosse
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;

// left cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
extern std::queue<bool> queueStartYolo; //if new Yolo inference can start

// right cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
extern std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

// 3D positioning ~ trajectory prediction
extern std::queue<int> queueTargetFrameIndex;                      // TM estimation frame
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

/*  YOLO class definition  */
class YOLODetect
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8m.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloWidth = 320;
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.4;
    const float ConfThreshold = 0.3;
    const float IoUThresholdIdentity = 0.4; // for maitainig consistency of tracking
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
    YOLODetect()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLODetect() { delete device; }; // Deconstructor

    void detectLeft(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass, int counterIteration)
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
        // std::cout << "finish preprocess" << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMLeft; // for limiting detection area
        std::vector<int> classIndexesTMLeft;
        if (!queueTMClassIndexLeft.empty() || counterIteration >= 3)
        {
            getLatestDataLeft(bboxesCandidateTMLeft, classIndexesTMLeft); // get latest data
        }
        // std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;
        // auto start = std::chrono::high_resolution_clock::now();
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        // auto stop = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // std::cout << "Time taken by Yolo inference: " << duration.count() << " milliseconds" << std::endl;
        //  std::cout << "finish inference" << std::endl;
        //  candidate class from TM
        /* get latest data from Template Matching */
        /* postProcess */
        // std::cout << "post process" << std::endl;
        preds = preds.permute({ 0, 2, 1 }); // change order : (1,6,2100) -> (1,2100,6)
        // std::cout << "permute" << std::endl;
        // std::cout << "preds size : " << preds.sizes() << std::endl;
        std::vector<torch::Tensor> detectedBoxes0Left, detectedBoxes1Left; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Left, detectedBoxes1Left, ConfThreshold, IoUThreshold);

        // std::cout << "BBOX for Ball : " << detectedBoxes0Left.size() << " BBOX for BOX : " << detectedBoxes1Left.size() << std::endl;
        std::vector<cv::Rect2d> existedRoi, newRoi;
        std::vector<int> existedClass, newClass;
        /* Roi Setting : take care of dealing with TM data */
        /* ROI and class index management */
        roiSetting(detectedBoxes0Left, existedRoi, existedClass, newRoi, newClass, BALL, bboxesCandidateTMLeft, classIndexesTMLeft);
        /*if (!existedClass.empty())
        {
            std::cout << "existed class after roisetting of Ball:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        */
        roiSetting(detectedBoxes1Left, existedRoi, existedClass, newRoi, newClass, BOX, bboxesCandidateTMLeft, existedClass);
        /* in Ball roisetting update all classIndexesTMLeft to existedClass, so here adapt existedClass as a reference class */
        /*if (!existedClass.empty())
        {
            std::cout << "existedClass after roiSetting of Box:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }*/
        /* push and save data */
        push2QueueLeft(existedRoi, newRoi, existedClass, newClass, frame, posSaver, classSaver, frameIndex, detectedFrame, detectedFrameClass);
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        cv::Mat yoloimg; // define yolo img type
        cv::resize(frame, yoloimg, YOLOSize);
        cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
        imgTensor = imgTensor.div(255);                                                            // normalization
        imgTensor = imgTensor.unsqueeze(0);                                                        // expand dims for Convolutional layer (height,width,1)
        imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
    }

    void non_max_suppression2(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxes0, std::vector<torch::Tensor>& detectedBoxes1, float confThreshold, float iouThreshold)
    {
        /* non max suppression : remove overlapped bbox
         * Args:
         *   prediction : (1,2100,,6)
         * Return:
         *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
         */

        torch::Tensor xc0 = prediction.select(2, 4) > confThreshold; // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor xc1 = prediction.select(2, 5) > confThreshold; // get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"

        torch::Tensor x0 = prediction.index_select(1, torch::nonzero(xc0[0]).select(1, 0)); // box, x0.shape : (1,n,6) : n: number of candidates
        torch::Tensor x1 = prediction.index_select(1, torch::nonzero(xc1[0]).select(1, 0)); // ball x1.shape : (1,m,6) : m: number of candidates

        x0 = x0.index_select(1, x0.select(2, 4).argsort(1, true).squeeze()); // ball : sorted in descending order
        x1 = x1.index_select(1, x1.select(2, 5).argsort(1, true).squeeze()); // box : sorted in descending order

        x0 = x0.squeeze(); //(1,n,6) -> (n,6)
        x1 = x1.squeeze(); //(1,m,6) -> (m,6)
        // std::cout << "sort detected data" << std::endl;
        // ball : non-max suppression
        if (x0.size(0) != 0)
        {
            /* 1 dimension */
            if (x0.dim() == 1)
            {
                torch::Tensor bbox0Top = xywh2xyxy(x0.slice(0, 0, 4));
                // std::cout << "top defined" << std::endl;
                detectedBoxes0.push_back(bbox0Top.cpu());
            }
            /* 2 dimension */
            else
            {
                torch::Tensor bbox0Top = xywh2xyxy(x0[0].slice(0, 0, 4));
                // std::cout << "top defined" << std::endl;
                detectedBoxes0.push_back(bbox0Top.cpu());
                // std::cout << "push back top data" << std::endl;
                // for every candidates
                if (x0.size(0) >= 2)
                {
                    // std::cout << "nms start" << std::endl;
                    nms(x0, detectedBoxes0, iouThreshold); // exclude overlapped bbox : 20 milliseconds
                    // std::cout << "num finished" << std::endl;
                }
            }
        }

        // box
        // std::cout << "x1 size:" << x1.size(0) << std::endl;
        if (x1.size(0) != 0)
        {
            if (x1.dim() == 1)
            {
                torch::Tensor bbox1Top = xywh2xyxy(x1.slice(0, 0, 4));
                // std::cout << "top defined" << std::endl;
                // std::cout << "bbox1Top:" << bbox1Top.sizes() << std::endl;
                detectedBoxes1.push_back(bbox1Top.cpu());
            }
            else
            {
                torch::Tensor bbox1Top = xywh2xyxy(x1[0].slice(0, 0, 4));
                // std::cout << "top defined" << std::endl;
                // std::cout << "bbox1Top:" << bbox1Top.sizes() << std::endl;
                detectedBoxes1.push_back(bbox1Top.cpu());
                // std::cout << "push back top data" << std::endl;
                if (x1.size(0) >= 2)
                {
                    // std::cout << "nms start" << std::endl;
                    nms(x1, detectedBoxes1, iouThreshold); // exclude overlapped bbox : 20 milliseconds
                    // std::cout << "nms finish" << std::endl;
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

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold)
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
            box = xywh2xyxy(x[i].slice(0, 0, 4)); //(xCenter,yCenter,width,height) -> (left,top,right,bottom)

            bool addBox = true; // if save bbox as a new detection

            for (const torch::Tensor& savedBox : detectedBoxes)
            {
                float iou = calculateIoU(box, savedBox); // calculate IoU
                /* same bbox : already found -> nod add */
                if (iou > iouThreshold)
                {
                    addBox = false;
                    break; // next iteration
                }
            }
            /* new tracker */
            if (addBox)
            {
                detectedBoxes.push_back(box.cpu());
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

    void roiSetting(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& existedRoi, std::vector<int>& existedClass, std::vector<cv::Rect2d>& newRoi, std::vector<int>& newClass,
        int candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM)
    {
        /*
         * Get current data before YOLO inference started.
         * First : Compare YOLO detection and TM detection
         * Second : if match : return new templates in the same order with TM
         * Third : if not match : adapt as a new templates and add after TM data
         * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
         */
         // std::cout << "bboxesYolo size=" << detectedBoxes.size() << std::endl;
         /* detected by Yolo */
        if (!detectedBoxes.empty())
        {
            // std::cout << "yolo detection exists" << std::endl;
            /* some trackers exist */
            if (!classIndexesTM.empty())
            {
                // std::cout << "template matching succeeded" << std::endl;
                /* constant setting */
                std::vector<cv::Rect2d> bboxesYolo; // for storing cv::Rect2d
                /* start comparison Yolo and TM data -> search for existed tracker */
                // std::cout << "start comparison of YOLO and TM" << std::endl;
                comparisonTMYolo(detectedBoxes, classIndexesTM, candidateIndex, bboxesCandidate, bboxesYolo, existedRoi, existedClass);
                /* finish comparison */
                // std::cout << "finish comparison with yolo " << std::endl;
                /* deal with new trackers */
                int numNewDetection = bboxesYolo.size(); // number of new detections
                /* if there is a new detection */
                if (numNewDetection != 0)
                {
                    for (int i = 0; i < numNewDetection; i++)
                    {
                        newRoi.push_back(bboxesYolo[i]);
                        newClass.push_back(candidateIndex);
                    }
                }
            }
            /* No TM tracker exist */
            else
            {
                // std::cout << "No TM tracker exist " << std::endl;
                int numBboxes = detectedBoxes.size(); // num of detection
                int left, top, right, bottom;         // score0 : ball , score1 : box
                cv::Rect2d roi;

                /* convert torch::Tensor to cv::Rect2d */
                std::vector<cv::Rect2d> bboxesYolo;
                bboxesYolo.reserve(100);
                for (int i = 0; i < numBboxes; ++i)
                {
                    float expandrate[2] = { static_cast<float>(frameWidth) / static_cast<float>(yoloWidth), static_cast<float>(frameHeight) / static_cast<float>(yoloHeight) }; // resize bbox to fit original img size
                    // std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
                    left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
                    top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
                    right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
                    bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
                    newRoi.emplace_back(left, top, (right - left), (bottom - top));
                    newClass.push_back(candidateIndex);
                }
            }
        }
        /* No object detected in Yolo -> return -1 class label */
        else
        {
            // std::cout << "yolo detection doesn't exist" << std::endl;
            /* some TM trackers exist */
            /* if class label is equal to 0, return -1 if existed label == 0 or -1 and return the same label if classIndex is otherwise*/
            if (candidateIndex == 0)
            {
                if (!classIndexesTM.empty())
                {
                    int counterCandidate = 0;
                    int counterIteration = 0;
                    for (const int& classIndex : classIndexesTM)
                    {
                        /* if same label -> failed to track in YOLO  */
                        if (classIndex == candidateIndex)
                        {
                            existedClass.push_back(-1);
                            if (!bboxesCandidate.empty())
                            {
                                bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate); // erase existed roi to maintain roi order
                            }
                        }
                        /* else classIndex != candidateIndex */
                        else if (classIndex != candidateIndex && classIndex != -1)
                        {
                            existedClass.push_back(classIndex);
                            counterCandidate++;
                        }
                        /* else classIndex != candidateIndex */
                        else if (classIndex == -1)
                        {
                            existedClass.push_back(-1);
                        }
                    }
                }
                /* No TM tracker */
                else
                {
                    // std::cout << "No Detection , no tracker" << std::endl;
                    /* No detection ,no trackers -> Nothing to do */
                }
            }
            /* if candidateIndex is other than 0 */
            else
            {
                if (!existedClass.empty())
                {
                    int counterCandidate = 0;
                    int counterIteration = 0;
                    for (const int& classIndex : existedClass)
                    {
                        // std::cout << "bboxesCandidate size = " << bboxesCandidate.size() << std::endl;
                        // std::cout << "counterCandidate=" << counterCandidate << std::endl;
                        /* if same label -> failed to track in YOLO  */
                        if (classIndex == candidateIndex)
                        {
                            existedClass.at(counterIteration) = -1;
                            if (!bboxesCandidate.empty())
                            {
                                bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate);
                            }
                        }
                        /* if classIndex is greater than candidateIndex -> keep bbox*/
                        else if (candidateIndex < classIndex)
                        {
                            counterCandidate++;
                        }
                        counterIteration++;
                    }
                }
                /* No TM tracker */
                else
                {
                    // std::cout << "No Detection , no tracker" << std::endl;
                    /* No detection ,no trackers -> Nothing to do */
                }
            }
        }
    }

    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2)
    {
        float left = std::max(box1.x, box2.x);
        float top = std::max(box1.y, box2.y);
        float right = std::min((box1.x + box1.width), (box2.x + box2.width));
        float bottom = std::min((box1.y + box1.height), (box2.y + box2.height));

        if (left < right && top < bottom)
        {
            float intersection = (right - left) * (bottom - top);
            float area1 = box1.width * box1.height;
            float area2 = box2.width * box2.height;
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void comparisonTMYolo(std::vector<torch::Tensor>& detectedBoxes, std::vector<int>& classIndexesTM, int& candidateIndex,
        std::vector<cv::Rect2d>& bboxesCandidate, std::vector<cv::Rect2d>& bboxesYolo,
        std::vector<cv::Rect2d>& existedRoi, std::vector<int>& existedClass)
    {
        /* constant setting */
        int numBboxes = detectedBoxes.size(); // num of detection
        bboxesYolo.reserve(numBboxes);        // reserve space to avoid reallocation
        int left, top, right, bottom;         // score0 : ball , score1 : box
        cv::Rect2d roi;                       // for updated Roi
        bool boolCurrentPosition = false;     // if current position is available

        /* convert torch::Tensor to cv::Rect2d */
        /* iterate for numBboxes */
        for (int i = 0; i < numBboxes; ++i)
        {
            float expandrate[2] = { static_cast<float>(frameWidth) / static_cast<float>(yoloWidth), static_cast<float>(frameHeight) / static_cast<float>(yoloHeight) }; // resize bbox to fit original img size
            // std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
            left = static_cast<int>(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
            top = static_cast<int>(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
            right = static_cast<int>(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
            bottom = static_cast<int>(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
            bboxesYolo.emplace_back(left, top, (right - left), (bottom - top));
        }
        // std::cout << "finish converting torch::Tensor to cv::Rect2d" << std::endl;

        /*  compare detected bbox and TM bbox  */
        float max = IoUThresholdIdentity;      // set max value as threshold for lessening process volume
        bool boolIdentity = false;             // if found the tracker
        int indexMatch = 0;                    // index match
        std::vector<cv::Rect2d> newDetections; // for storing newDetection

        /* start comparison */
        /* if found same things : push_back detected template and classIndex, else: make sure that push_back only -1 */
        // std::cout << "classIndexesTM size" << classIndexesTM.size() << std::endl;
        int counterCandidateTM = 0; // number of candidate bbox
        // std::cout << "Comparison of TM and YOLO :: CandidateIndex:" << candidateIndex << std::endl;
        /* there is classes in TM trackeing */
        int counterIteration = 0;
        /*iterate for each clssIndexes of TM */
        for (const int classIndex : classIndexesTM)
        {

            /* bboxes Yolo still exist */
            if (!bboxesYolo.empty())
            {
                // std::cout << "Couter Check :: counterCandidateTM : " << counterCandidateTM << " bboxes YOLO size : " << bboxesYolo.size() << std::endl;
                /* TM tracker exist */
                if (classIndex != -1)
                {
                    /* Tracker labels match like 0=0,1=1 */
                    if (candidateIndex == classIndex)
                    {
                        boolIdentity = false; // initialize
                        cv::Rect2d bboxTemp;  // temporal bbox storage
                        /* iterate for number of detected bboxes */
                        for (int counterCandidateYolo = 0; counterCandidateYolo < bboxesYolo.size(); counterCandidateYolo++)
                        {
                            float iou = calculateIoU_Rect2d(bboxesCandidate[counterCandidateTM], bboxesYolo[counterCandidateYolo]);
                            if (iou >= max) // found similar bbox
                            {
                                max = iou;
                                bboxTemp = bboxesYolo[counterCandidateYolo]; // save top iou candidate
                                indexMatch = counterCandidateYolo;
                                boolIdentity = true;
                            }
                        }
                        /* find matched tracker */
                        if (boolIdentity)
                        {
                            if (candidateIndex == 0)
                            {
                                // add data
                                existedClass.push_back(candidateIndex);
                            }
                            /* other label */
                            else
                            {
                                existedClass.at(counterIteration) = candidateIndex;
                            }
                            // std::cout << "TM and Yolo Tracker matched!" << std::endl;
                            existedRoi.push_back(bboxTemp);
                            // delete candidate bbox
                            bboxesYolo.erase(bboxesYolo.begin() + indexMatch); // erase detected bbox from bboxes Yolo -> number of bboxesYolo decrease
                        }
                        /* not found matched tracker -> return classIndex -1 to updatedClassIndexes */
                        else
                        {
                            /* class label is 0 */
                            if (candidateIndex == 0)
                            {
                                // std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                                existedClass.push_back(-1);
                            }
                            /* class label is other than 0 */
                            else
                            {
                                // std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                                existedClass.at(counterIteration) = -1;
                            }
                        }
                        /* delete candidate bbox */
                        bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                    }
                    /* other labels -> push back same label to maintain order only when candidateIndex=0 */
                    else if (candidateIndex != classIndex && candidateIndex == 0)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidateTM++; // for maintain order of existed roi
                    }
                    /* only valid if classIndex != 0 */
                    else if (candidateIndex < classIndex && candidateIndex != 0)
                    {
                        counterCandidateTM++; // for maintaining order of existed roi
                    }
                }
                /* templateMatching Tracking was fault : classIndex = -1 */
                else
                {
                    /* if existedClass label is -1, return -1 only when candidate Index is 0. Then otherwise,
                     * only deal with the case when class label is equal to candidate index
                    */
                    if (candidateIndex == 0)
                    {
                        // std::cout << "add label -1 because we are label 1 even if we didn't experince" << std::endl;
                        existedClass.push_back(-1);
                    }
                }
            }
            /* bboxes Yolo already disappear -> previous trackers was failed here */
            else
            {
                /* candidate index == 0 */
                if (candidateIndex == 0)
                {
                    /* if same label -> failed to track in YOLO  */
                    if (classIndex == candidateIndex)
                    {
                        existedClass.push_back(-1);
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                        }
                    }
                    /* class label is 1,2,3,... */
                    else if (classIndex > candidateIndex)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidateTM++; // maintain existedROI order
                    }
                    /* else classIndex != candidateIndex */
                    else if (classIndex == -1)
                    {
                        existedClass.push_back(-1);
                    }
                }
                /* class label is other than 0 */
                else
                {
                    if (classIndex == candidateIndex)
                    {
                        existedClass.at(counterIteration) = -1; // update label as -1
                        if (!bboxesCandidate.empty())
                        {
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                        }
                    }
                    else if (candidateIndex < classIndex)
                    {
                        counterCandidateTM++; // for maintaining order of existed roi
                    }
                }
            }
            counterIteration++;
        }
    }

    void push2QueueLeft(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver,
        const int& frameIndex, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass)
    {
        /*
         * push detection data to queueLeft
         */
        std::vector<cv::Rect2d> updatedRoi;
        updatedRoi.reserve(100);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(100);
        std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>> updatedTrackers;
        updatedTrackers.reserve(100);
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(300);
        /* update data */
        //MOSSE
        if (boolMOSSE) updateData_MOSSE(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTrackers, updatedClassIndexes);
        //Template Matching
        else updateData(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTemplates, updatedClassIndexes);
        /* detection is successful */
        if (!updatedRoi.empty())
        {
            // std::cout << "detection succeeded" << std::endl;
            // save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            detectedFrame.push_back(frameIndex);
            detectedFrameClass.push_back(frameIndex);
            // push detected data
            // std::unique_lock<std::mutex> lock(mtxYoloLeft);
            /* initialize queueYolo for maintaining data consistency */
            if (!queueYoloBboxLeft.empty())
                queueYoloBboxLeft.pop();
            if (!queueYoloTemplateLeft.empty())
                queueYoloTemplateLeft.pop();
            if (!queueYoloClassIndexLeft.empty())
                queueYoloClassIndexLeft.pop();
            if (!queueTrackerYolo_left.empty())
                queueTrackerYolo_left.pop();
            /* finish initialization */
            queueYoloBboxLeft.push(updatedRoi);
            //MOSSE
            if (boolMOSSE)
            {
                //std::cout << "Yolo :: updatedTrackers :: size=" << updatedTrackers.size() << std::endl;
                queueTrackerYolo_left.push(updatedTrackers);
            }
            //Template Matching
            else queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
            //int numLabels = updatedClassIndexes.size();
            //if (!queueNumLabels.empty())
            //    queueNumLabels.pop();
            //queueNumLabels.push(numLabels);
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                // std::unique_lock<std::mutex> lock(mtxYoloLeft);
                /* initialize queueYolo for maintaining data consistency */
                if (!queueYoloClassIndexLeft.empty())
                    queueYoloClassIndexLeft.pop();
                queueYoloClassIndexLeft.push(updatedClassIndexes);
                detectedFrameClass.push_back(frameIndex);
                classSaver.push_back(updatedClassIndexes);
                //int numLabels = updatedClassIndexes.size();
                //queueNumLabels.push(numLabels);
            }
            /* no class Indexes -> nothing to do */
            else
            {
                /* go trough */
            }
        }
    }

    void updateData(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Mat1b>& updatedTemplates,
        std::vector<int>& updatedClassIndexes)
    {
        // std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
        // std::cout << "Existed class" << std::endl;
        if (!existedRoi.empty())
        {
            /* update bbox and templates */
            for (const cv::Rect2d& roi : existedRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
            }
            for (const int& classIndex : existedClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            // std::cout << "new detection" << std::endl;
            for (const cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
    }

    void updateData_MOSSE(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi, std::vector<int>& existedClass, std::vector<int>& newClass,
        cv::Mat1b& frame, std::vector<cv::Rect2d>& updatedRoi, std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>& updatedTrackers,
        std::vector<int>& updatedClassIndexes)
    {
        // std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
        // std::cout << "Existed class" << std::endl;
        if (!existedRoi.empty())
        {
            /* update bbox and templates */
            for (cv::Rect2d& roi : existedRoi)
            {
                updatedRoi.push_back(roi);
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
                //std::cout << "Yolo :: updatedTrackers :: size=" << updatedTrackers.size() << std::endl;
            }
            for (const int& classIndex : existedClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            // std::cout << "new detection" << std::endl;
            for (cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                cv::Ptr<cv::mytracker::TrackerMOSSE> tracker = cv::mytracker::TrackerMOSSE::create();
                tracker->init(frame, roi);
                updatedTrackers.push_back(tracker);
                //std::cout << "Yolo :: updatedTrackers :: size=" << updatedTrackers.size() << std::endl;
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                // std::cout << classIndex << " ";
            }
            // std::cout << std::endl;
        }
        else
        {
            if (!newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    // std::cout << classIndex << " ";
                }
                // std::cout << std::endl;
            }
        }
    }

    void getLatestDataLeft(std::vector<cv::Rect2d>& bboxes, std::vector<int>& classes)
    {
        // std::unique_lock<std::mutex> lock(mtxTMLeft); // Lock the mutex
        /* still didn't synchronize -> wait for next data */
        /* getting iteratively until TM class labels are synchronized with Yolo data */
        while (true) // >(greater than for first exchange between Yolo and TM)
        {
            if (!queueStartYolo.empty())
            {
                bool start = queueStartYolo.front();
                queueStartYolo.pop();
                if (start)
                {
                    std::cout << "start yolo inference" << std::endl;
                    break;
                }
            }
            //std::cout << "wait for new TM data" << std::endl;
        }

        // std::cout << "Left Img : Yolo bbox available from TM " << std::endl;
        if (!queueTMClassIndexLeft.empty())
        {
            classes = queueTMClassIndexLeft.front();
            if (!queueTMBboxLeft.empty()) bboxes = queueTMBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            /* for debug */
            /*std::cout << ":: Left :: latest data " << std::endl;
            for (const int& label : classes)
                std::cout << label << " ";
            std::cout << std::endl;
            */
        }
    }
};

#endif 