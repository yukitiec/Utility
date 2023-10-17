// ximea_test.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <ctime>
#include <direct.h>
#include <sys/stat.h>
#include <algorithm>


std::mutex mtxImg, mtxYoloLeft, mtxTMLeft, mtxTarget; // define mutex

// camera : constant setting
const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
const int FPS = 300;
// YOLO label
const int BALL = 0;
const int BOX = 1;
// template matching constant value setting
const float scaleXTM = 1.2; // search area scale compared to roi
const float scaleYTM = 1.2;
const float scaleXYolo = 2.0;
const float scaleYYolo = 2.0;
const float matchingThreshold = 0.5;             // matching threshold
const int MATCHINGMETHOD = cv::TM_SQDIFF_NORMED; // TM_SQDIFF_NORMED is good for small template
/* 3d positioning by stereo camera */
const int BASELINE = 280; // distance between 2 cameras
// std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration

/* revise here based on camera calibration */

const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 179, 0, 160, // fx: focal length in x, cx: principal point x
    0, 179, 160,                           // fy: focal length in y, cy: principal point y
    0, 0, 1                                // 1: scaling factor
    );
cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 1, 1, 1, 1, 1);

/* revise here based on camera calibration */

const int TARGET_DEPTH = 400; // catching point is 40 cm away from camera position
/*matching method of template Matching
 * const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM CCOEFF \n 5: TM CCOEFF NORMED"; 2 is not so good
 */

 // queue definition
std::queue<cv::Mat1b> queueFrame; // queue for frame
std::queue<int> queueFrameIndex;                 // queue for frame index

// left cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
std::queue<int> queueNumLabels; // current labels number -> for maintaining label number consistency

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

// declare function
/* utility */
void checkStorage(std::vector<std::vector<cv::Rect2d>>&, std::vector<int>&);
void checkClassStorage(std::vector<std::vector<int>>&, std::vector<int>&);
void checkStorageTM(std::vector<std::vector<cv::Rect2d>>&, std::vector<int>&);
void checkClassStorageTM(std::vector<std::vector<int>>&, std::vector<int>&);
/* get img */
bool getImagesFromQueueYolo(cv::Mat1b&, int&); // get image from queue
bool getImagesFromQueueTM(cv::Mat1b&, int&);
/* template matching */
void templateMatching();                                                                                                                                  // void* //proto definition
void templateMatchingForLeft(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&, std::vector<std::vector<int>>&, std::vector<int>&, std::vector<int>&);  //, int, int, float, const int);
void organizeData(std::vector<bool>&, bool&, std::vector<int>&, std::vector<cv::Mat1b>&,std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, int&);
void combineYoloTMData(std::vector<int>&, std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,
    std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&, const int&);
void getTemplateMatchingDataLeft(bool&, std::vector<int>&, std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<bool>&, int&);
void processTM(std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<bool>&, std::vector<cv::Rect2d>&, cv::Mat1b&,std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&);
/* access yolo data */
void getYoloDataLeft(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&);
//void push2YoloDataLeft(std::vector<cv::Rect2d>&, std::vector<int>&);
/* read img */
void pushFrame(cv::Mat1b&, const int);
void removeFrame();

/*  YOLO class definition  */
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
    const float IoUThresholdIdentity = 0.1; // for maitainig consistency of tracking
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

    void detectLeft(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass)
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
        if (!queueTMClassIndexLeft.empty())
        {
            getLatestDataLeft(bboxesCandidateTMLeft, classIndexesTMLeft); // get latest data
        }
        // std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;
        //auto start = std::chrono::high_resolution_clock::now();
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        //auto stop = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        //std::cout << "Time taken by Yolo inference: " << duration.count() << " milliseconds" << std::endl;
        // std::cout << "finish inference" << std::endl;
        // candidate class from TM
        /* get latest data from Template Matching */
        /* postProcess */
        // std::cout << "post process" << std::endl;
        preds = preds.permute({ 0, 2, 1 }); // change order : (1,6,2100) -> (1,2100,6)
        // std::cout << "permute" << std::endl;
        // std::cout << "preds size : " << preds.sizes() << std::endl;
        std::vector<torch::Tensor> detectedBoxes0Left, detectedBoxes1Left; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Left, detectedBoxes1Left, ConfThreshold, IoUThreshold);

        //std::cout << "BBOX for Ball : " << detectedBoxes0Left.size() << " BBOX for BOX : " << detectedBoxes1Left.size() << std::endl;
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
         //std::cout << "bboxesYolo size=" << detectedBoxes.size() << std::endl;
         /* detected by Yolo */
        if (!detectedBoxes.empty())
        {
            //std::cout << "yolo detection exists" << std::endl;
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
            //std::cout << "yolo detection doesn't exist" << std::endl;
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
                        //std::cout << "bboxesCandidate size = " << bboxesCandidate.size() << std::endl;
                        //std::cout << "counterCandidate=" << counterCandidate << std::endl;
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
                            //std::cout << "TM and Yolo Tracker matched!" << std::endl;
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
                                //std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                                existedClass.push_back(-1);
                            }
                            /* class label is other than 0 */
                            else
                            {
                                //std::cout << "TM and Yolo Tracker didn't match" << std::endl;
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
                        existedClass.at(counterIteration) = -1;                                  // update label as -1
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
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(300);
        /* update data */
        updateData(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTemplates, updatedClassIndexes);
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
            if (!queueYoloBboxLeft.empty()) queueYoloBboxLeft.pop();
            if (!queueYoloTemplateLeft.empty()) queueYoloTemplateLeft.pop();
            if (!queueYoloClassIndexLeft.empty()) queueYoloClassIndexLeft.pop();
            /* finish initialization */
            queueYoloBboxLeft.push(updatedRoi);
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
            int numLabels = updatedClassIndexes.size();
            if (!queueNumLabels.empty()) queueNumLabels.pop();
            queueNumLabels.push(numLabels);
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                // std::unique_lock<std::mutex> lock(mtxYoloLeft);
                /* initialize queueYolo for maintaining data consistency */
                if (!queueYoloClassIndexLeft.empty()) queueYoloClassIndexLeft.pop();
                queueYoloClassIndexLeft.push(updatedClassIndexes);
                detectedFrameClass.push_back(frameIndex);
                classSaver.push_back(updatedClassIndexes);
                int numLabels = updatedClassIndexes.size();
                queueNumLabels.push(numLabels);
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
        //std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
        //std::cout << "Existed class" << std::endl;
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
                //std::cout << classIndex << " ";
            }
            //std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    //std::cout << classIndex << " ";
                }
                //std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            //std::cout << "new detection" << std::endl;
            for (const cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                //std::cout << classIndex << " ";
            }
            //std::cout << std::endl;
        }
        else
        {
            if (!newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    //std::cout << classIndex << " ";
                }
                //std::cout << std::endl;
            }
        }
    }

    void getLatestDataLeft(std::vector<cv::Rect2d>& bboxes, std::vector<int>& classes)
    {
        //std::unique_lock<std::mutex> lock(mtxTMLeft); // Lock the mutex
        std::cout << "Yolo detection : get latest data!" << std::endl;
        if (!queueYoloClassIndexLeft.empty())
        {
            classes = queueTMClassIndexLeft.front(); // get current tracking status
            int labelCounter = queueNumLabels.front();
            queueNumLabels.pop();
            int numClassesTM = classes.size();
            if (labelCounter > numClassesTM) // number of labels doesn't match -> have to check label data
            {
                /* getting iteratively until TM class labels are synchronized with Yolo data */
                while (numClassesTM >= labelCounter) // >(greater than for first exchange between Yolo and TM)
                {
                    if (!queueTMClassIndexLeft.empty())
                    {
                        classes = queueTMClassIndexLeft.front(); // get current tracking status
                        numClassesTM = classes.size();
                    }
                }
            }
            std::unique_lock<std::mutex> lock(mtxTMLeft);
            std::cout << "::class label :: ";
            for (const int& classIndex : classes)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        // std::cout << "Left Img : Yolo bbox available from TM " << std::endl;
        if (!queueTMBboxLeft.empty())
        {
            bboxes = queueTMBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            /* for debug */
            std::cout << ":: Left :: latest yolo data " << std::endl;
            for (const cv::Rect2d& bbox : bboxes)
            {
                std::cout << "BBOX ::" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << std::endl;
            }
        }
    }
};

/*
 * Yolo thread function definition
 */

void yoloDetect()
{
    /* Yolo Detection Thread
     * Args:
     *   queueFrame : frame
     *   queueFrameIndex : frame index
     *   queueYoloTemplate : detected template img
     *   queueYoloBbox : detected template bbox
     */

     // YoloDetector initialization
    YOLODetect yolodetectorLeft;
    std::cout << "yolo initialization has finished" << std::endl;
    /* initialization */
    if (!queueYoloBboxLeft.empty())
    {
        std::cout << "queueYoloBboxLeft isn't empty" << std::endl;
        while (!queueYoloBboxLeft.empty())
        {
            queueYoloBboxLeft.pop();
        }
    }
    if (!queueYoloTemplateLeft.empty())
    {
        std::cout << "queueYoloTemplateLeft isn't empty" << std::endl;
        while (!queueYoloTemplateLeft.empty())
        {
            queueYoloTemplateLeft.pop();
        }
    }
    if (!queueYoloClassIndexLeft.empty())
    {
        std::cout << "queueYoloClassIndexesLeft isn't empty" << std::endl;
        while (!queueYoloClassIndexLeft.empty())
        {
            queueYoloClassIndexLeft.pop();
        }
    }
    // vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    posSaverYoloLeft.reserve(300);
    std::vector<int> detectedFrame;
    detectedFrame.reserve(300);
    std::vector<int> detectedFrameClass;
    detectedFrame.reserve(300);
    std::vector<std::vector<int>> classSaverYoloLeft;
    classSaverYoloLeft.reserve(300);
    int frameIndex;
    int countIteration = 1;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat1b img;
        int frameIndex;
        bool boolImgs = getImagesFromQueueYolo(img, frameIndex);
        if (!boolImgs)
        {
            if (counterFinish > 10)
            {
                break;
            }
            // No more frames in the queue, exit the loop
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::cout << "wait for image pushed" << std::endl;
            counterFinish++;
            continue;
        }
        std::cout << " YOLO -- " << countIteration << " -- " << std::endl;

        /*start yolo detection */
        yolodetectorLeft.detectLeft(img, frameIndex, posSaverYoloLeft, classSaverYoloLeft, detectedFrame, detectedFrameClass);
        // std::thread threadLeftYolo(&YOLODetect::detectLeft, &yolodetectorLeft,std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(posSaverYoloLeft),std::ref(classSaverYoloLeft));
        // std::thread threadRightYolo(&YOLODetect::detectRight, &yolodetectorRight,std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(posSaverYoloRight),std::ref(classSaverYoloRight));
        // wait for each thread finishing
        // threadLeftYolo.join();
        // threadRightYolo.join();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
        countIteration++;
    }
    /* check data */
    std::cout << "position saver : Yolo : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverYoloLeft size:" << posSaverYoloLeft.size() << ", detectedFrame size:" << detectedFrame.size() << std::endl;
    checkStorage(posSaverYoloLeft, detectedFrame);
    std::cout << " : Left : " << std::endl;
    std::cout << "classSaverYoloLeft size:" << classSaverYoloLeft.size() << ", detectedFrameClass size:" << detectedFrameClass.size() << std::endl;
    checkClassStorage(classSaverYoloLeft, detectedFrameClass);
}

bool getImagesFromQueueYolo(cv::Mat1b& img, int& frameIndex)
{
    // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    if (!queueFrame.empty())
    {
        img = queueFrame.front();
        frameIndex = queueFrameIndex.front();
        // remove frame from queue
        queueFrame.pop();
        queueFrameIndex.pop();
        return true;
    }
    return false;
}

void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame)
{
    //prepare file
    std::string filePathOF = "yoloData_yoloTMMOT_bbox_IoU0.1_test300fpsmp4.csv";
    // Open the file for writing
    std::ofstream outputFile(filePathOF);
    if (!outputFile.is_open()) {
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

void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame)
{
    //prepare file
    std::string filePathOF = "yoloData_yoloTMMOT_class_IoU0.1_test300fpsmp4.csv";
    // Open the file for writing
    std::ofstream outputFile(filePathOF);
    if (!outputFile.is_open()) {
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

/* template matching thread function definition */
void templateMatching() // void*
{
    /* Template Matching
     * return most similar region
     */

     // vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    posSaverTMLeft.reserve(2000);
    std::vector<std::vector<int>> classSaverTMLeft;
    classSaverTMLeft.reserve(2000);
    std::vector<int> detectedFrame;
    detectedFrame.reserve(300);
    std::vector<int> detectedFrameClass;
    detectedFrame.reserve(300);

    int countIteration = 0;
    int counterFinish = 0;
    /* sleep for 2 seconds */
    // std::this_thread::sleep_for(std::chrono::seconds(30));
    //  for each img iterations
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
    std::cout << "start template matching" << std::endl;
    while (true) // continue until finish
    {
        countIteration++;
        std::cout << " -- " << countIteration << " -- " << std::endl;
        // get img from queue
        cv::Mat1b img;
        int frameIndex;
        bool boolImgs = getImagesFromQueueTM(img, frameIndex);
        std::cout << "get imgs" << std::endl;
        if (!boolImgs)
        {
            if (counterFinish == 10)
            {
                break;
            }
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else
        {
            counterFinish = 0; // reset
        }
        std::vector<cv::Mat1b> templateImgsLeft;
        templateImgsLeft.reserve(100);
        bool boolLeft = false;
        /*start template matching process */
        auto start = std::chrono::high_resolution_clock::now();
        templateMatchingForLeft(img, frameIndex, templateImgsLeft, posSaverTMLeft,classSaverTMLeft, detectedFrame, detectedFrameClass);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by template matching: " << duration.count() << " milliseconds" << std::endl;
    }
    // check data
    std::cout << "position saver : TM : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverTMLeft size:" << posSaverTMLeft.size() << ", detectedFrame size:" << detectedFrame.size() << std::endl;
    checkStorageTM(posSaverTMLeft,detectedFrame);
    std::cout << "Class saver : TM : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "classSaverTMLeft size:" << classSaverTMLeft.size() << ", detectedFrameClass size:" << detectedFrameClass.size() << std::endl;
    checkClassStorageTM(classSaverTMLeft,detectedFrameClass); 
}

void checkStorageTM(std::vector<std::vector<cv::Rect2d>>& posSaverYolo, std::vector<int>& detectedFrame)
{
    //prepare file
    std::string filePathOF = "TMData_yoloTMMOT_bbox_IoU0.1_test300fpsmp4.csv";
    // Open the file for writing
    std::ofstream outputFile(filePathOF);
    if (!outputFile.is_open()) {
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

void checkClassStorageTM(std::vector<std::vector<int>>& classSaverYolo, std::vector<int>& detectedFrame)
{
    //prepare file
    std::string filePathOF = "TMData_yoloTMMOT_class_IoU0.1_test300fpsmp4.csv";
    // Open the file for writing
    std::ofstream outputFile(filePathOF);
    if (!outputFile.is_open()) {
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

bool getImagesFromQueueTM(cv::Mat1b& img, int& frameIndex)
{
    // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    if (queueFrame.empty() || queueFrameIndex.empty())
    {
        return false;
    }
    else
    {
        img = queueFrame.front();
        frameIndex = queueFrameIndex.front();
        queueTargetFrameIndex.push(frameIndex);
        // remove frame from queue
        queueFrame.pop();
        queueFrameIndex.pop();
        return true;
    }
}

/*  Template Matching Function  */

/* Template Matching :: Left */
void templateMatchingForLeft(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, 
            std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass)
{
    // for updating templates
    std::vector<cv::Rect2d> updatedBboxes;
    updatedBboxes.reserve(30);
    std::vector<cv::Mat1b> updatedTemplates;
    updatedTemplates.reserve(30);
    std::vector<int> updatedClasses;
    updatedClasses.reserve(300);

    // get Template Matching data
    std::vector<int> classIndexTMLeft;
    classIndexTMLeft.reserve(30);
    int numTrackersTM = 0;
    std::vector<cv::Rect2d> bboxesTM;
    bboxesTM.reserve(30);
    std::vector<cv::Mat1b> templatesTM;
    templatesTM.reserve(30);
    std::vector<bool> boolScalesTM;
    boolScalesTM.reserve(30);   // search area scale
    bool boolTrackerTM = false; // whether tracking is successful

    /* Template Matching bbox available */
    getTemplateMatchingDataLeft(boolTrackerTM, classIndexTMLeft, bboxesTM, templatesTM, boolScalesTM, numTrackersTM);

    // template from yolo is available
    bool boolTrackerYolo = false;
    /* Yolo Template is availble */
    if (!queueYoloTemplateLeft.empty())
    {
        /* get Yolo data and update Template matchin data */
        organizeData(boolScalesTM, boolTrackerYolo, classIndexTMLeft, templatesTM, bboxesTM, updatedTemplates, updatedBboxes, updatedClasses, numTrackersTM);
    }
    /* template from yolo isn't available but TM tracker exist */
    else if (boolTrackerTM)
    {
        updatedTemplates = templatesTM;
        updatedBboxes = bboxesTM;
        updatedClasses = classIndexTMLeft;
    }
    /* no template is available */
    else
    {
        // nothing to do
    }
    std::cout << "BoolTrackerTM :" << boolTrackerTM << ", boolTrackerYolo : " << boolTrackerYolo << std::endl;
    /*  Template Matching Process */
    if (boolTrackerTM || boolTrackerYolo)
    {
        std::cout << "template matching process has started" << std::endl;
        int counterTracker = 0;
        // for storing TM detection results
        std::vector<cv::Mat1b> updatedTemplatesTM;
        updatedTemplatesTM.reserve(100);
        std::vector<cv::Rect2d> updatedBboxesTM;
        updatedBboxesTM.reserve(100);
        std::vector<int> updatedClassesTM;
        updatedClassesTM.reserve(300);
        std::vector<bool> updatedSearchScales;
        updatedSearchScales.reserve(100); // if scale is Yolo or TM
        // template matching
        std::cout << "processTM start" << std::endl;
        processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);
        std::cout << "processTM finish" << std::endl;
        if (!updatedSearchScales.empty())
        {
            if (!queueTMScalesLeft.empty()) queueTMScalesLeft.pop(); //pop before push
            queueTMScalesLeft.push(updatedSearchScales);
        }
        /* if yolo data is avilable -> send signal to target predict to change labels*/
        if (boolTrackerYolo)
        {
            queueLabelUpdateLeft.push(true);
        }
        else
        {
            queueLabelUpdateLeft.push(false);
        }
        if (!updatedBboxesTM.empty())
        {
            if (!queueTMBboxLeft.empty()) queueTMBboxLeft.pop();
            queueTMBboxLeft.push(updatedBboxesTM); // push roi
            posSaver.push_back(updatedBboxes);     // save current position to the vector
        }
        if (!updatedTemplatesTM.empty())
        {
            if (!queueTMTemplateLeft.empty()) queueTMTemplateLeft.pop();
            queueTMTemplateLeft.push(updatedTemplatesTM); // push template image
        }
        if (!updatedClassesTM.empty())
        {
            if (!queueTMClassIndexLeft.empty()) queueTMClassIndexLeft.pop();
            queueTMClassIndexLeft.push(updatedClassesTM);
            classSaver.push_back(updatedClassesTM); // save current class to the saver
        }
        detectedFrame.push_back(frameIndex);
        detectedFrameClass.push_back(frameIndex);
    }
    else // no template or bbox -> nothing to do
    {
        if (!classIndexTMLeft.empty())
        {
            if (!queueTMClassIndexLeft.empty()) queueTMClassIndexLeft.pop();
            queueTMClassIndexLeft.push(classIndexTMLeft);
            classSaver.push_back(classIndexTMLeft); // save current class to the saver
            detectedFrameClass.push_back(frameIndex);
        }
        else
        {
            // nothing to do
        }
    }
}

void getTemplateMatchingDataLeft(bool& boolTrackerTM, std::vector<int>& classIndexTMLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, int& numTrackersTM)
{
    if (!queueTMClassIndexLeft.empty())
    {
        classIndexTMLeft = queueTMClassIndexLeft.front();
        numTrackersTM = classIndexTMLeft.size();
    }
    if (!queueTMBboxLeft.empty() && !queueTMTemplateLeft.empty() && !queueTMScalesLeft.empty())
    {
        std::cout << "previous TM tracker is available" << std::endl;
        boolTrackerTM = true;

        bboxesTM = queueTMBboxLeft.front();
        templatesTM = queueTMTemplateLeft.front();
        boolScalesTM = queueTMScalesLeft.front();
    }
    else
    {
        boolTrackerTM = false;
    }
}

void organizeData(std::vector<bool>& boolScalesTM, bool& boolTrackerYolo, std::vector<int>& classIndexTMLeft, std::vector<cv::Mat1b>& templatesTM,
    std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& updatedTemplates,std::vector<cv::Rect2d>& updatedBboxes, std::vector<int>& updatedClasses, int& numTrackersTM)
{
    std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
    std::cout << "TM :: Yolo data is available" << std::endl;
    boolTrackerYolo = true;
    if (!boolScalesTM.empty())
    {
        boolScalesTM.clear(); // clear all elements of scales
    }
    // get Yolo data
    std::vector<cv::Mat1b> templatesYoloLeft;
    templatesYoloLeft.reserve(10); // get new data
    std::vector<cv::Rect2d> bboxesYoloLeft;
    bboxesYoloLeft.reserve(10); // get current frame data
    std::vector<int> classIndexesYoloLeft;
    classIndexesYoloLeft.reserve(300);
    getYoloDataLeft(templatesYoloLeft, bboxesYoloLeft, classIndexesYoloLeft); // get new frame
    // combine Yolo and TM data, and update latest data
    combineYoloTMData(classIndexesYoloLeft, classIndexTMLeft, templatesYoloLeft, templatesTM, bboxesYoloLeft, bboxesTM,
        updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
}

void getYoloDataLeft(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
{
    newTemplates = queueYoloTemplateLeft.front();
    newBboxes = queueYoloBboxLeft.front();
    newClassIndexes = queueYoloClassIndexLeft.front();
    std::cout << "TM get data from Yolo ::class label :: ";
    for (const int& classIndex : newClassIndexes)
    {
        std::cout << classIndex << " ";
    }
    std::cout << std::endl;
}

void combineYoloTMData(std::vector<int>& classIndexesYoloLeft, std::vector<int>& classIndexTMLeft, std::vector<cv::Mat1b>& templatesYoloLeft, std::vector<cv::Mat1b>& templatesTM,
    std::vector<cv::Rect2d>& bboxesYoloLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes,
    std::vector<int>& updatedClasses, std::vector<bool>& boolScalesTM, const int& numTrackersTM)
{
    int counterYolo = 0;
    int counterTM = 0;      // for counting TM adaptations
    int counterClassTM = 0; // for counting TM class counter
    // organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
    // think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)

    /* should check carefully -> compare num of detection */
    for (const int& classIndex : classIndexesYoloLeft)
    {
        /* after 2nd time from YOLO data */
        if (numTrackersTM != 0)
        {
            std::cout << "TM tracker already exist" << std::endl;
            /* first numTrackersTM is existed Templates -> if same label, update else, unless tracking was successful lost tracker */
            if (counterClassTM < numTrackersTM) // numTrackersTM : num ber of class indexes
            {
                /*update tracker*/
                if (classIndex != -1)
                {
                    /* if classIndex != -1, update tracker. and if tracker of TM is successful, search aream can be limited */
                    if (classIndex == classIndexTMLeft[counterClassTM])
                    {
                        updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                        updatedBboxes.push_back(bboxesTM[counterTM]);               // update bbox to TM one
                        updatedClasses.push_back(classIndex);                       // update class
                        boolScalesTM.push_back(true);                               // scale is set to TM
                        counterTM++;
                        counterYolo++;
                        counterClassTM++;
                    }
                    /* trakcer of TM was failed */
                    else
                    {
                        updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                        updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                        updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                        boolScalesTM.push_back(false);                              // scale is set to Yolo
                        counterYolo++;
                        counterClassTM++;
                    }
                }
                /* tracker not found in YOLO*/
                else
                {
                    /* template matching was successful -> keep tracking */
                    if (classIndexTMLeft[counterClassTM] != -1)
                    {
                        updatedTemplates.push_back(templatesTM[counterTM]); // update tracker to TM's one
                        updatedBboxes.push_back(bboxesTM[counterTM]);       // update bbox to TM's one
                        updatedClasses.push_back(classIndexTMLeft[counterClassTM]);
                        boolScalesTM.push_back(true); // scale is set to TM
                        counterTM++;
                        counterClassTM++;
                    }
                    /* both tracking was failed -> lost */
                    else
                    {
                        updatedClasses.push_back(classIndex);
                        counterClassTM++;
                    }
                }
            }
            /* new tracker -> add new templates * maybe in this case all calss labels should be positive, not -1 */
            else
            {
                if (classIndex != -1)
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                    updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                    boolScalesTM.push_back(false);                              // scale is set to Yolo
                    counterYolo++;
                }
                /* this is for exception, but prepare for emergency*/
                else
                {
                    std::cout << "this is exception:: even if new tracker, class label is -1. Should revise code " << std::endl;
                    updatedClasses.push_back(classIndex);
                }
            }
        }
        /* for the first time from YOLO data */
        else
        {
            std::cout << "first time of TM" << std::endl;
            /* tracker was successful */
            if (classIndex != -1)
            {
                updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                boolScalesTM.push_back(false);                              // scale is set to Yolo
                counterYolo++;
            }
            /* tracker was not found in YOLO */
            else
            {
                updatedClasses.push_back(classIndex);
            }
        }
    }
}

void processTM(std::vector<int>& updatedClasses, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, cv::Mat1b& img,
    std::vector<cv::Mat1b>& updatedTemplatesTM, std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales)
{
    int counterTracker = 0;
    // get bbox from queue for limiting search area
    int leftSearch, topSearch, rightSearch, bottomSearch;
    // iterate for each tracking classes
    std::cout << "check updateClasses:" << std::endl;
    for (const int& classIndex : updatedClasses)
    {
        std::cout << classIndex << " ";
    }
    std::cout << std::endl;
    for (const int& classIndexTM : updatedClasses)
    {
        /* template exist -> start template matching */
        if (classIndexTM != -1)
        {
            const cv::Mat1b& templateImg = updatedTemplates[counterTracker]; // for saving memory, using reference data
            // std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;
            // search area setting
            std::cout << "scale of TM : " << boolScalesTM[counterTracker] << std::endl;
            if (boolScalesTM[counterTracker]) // scale is set to TM : smaller search area
            {
                leftSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].x - (scaleXTM - 1) * updatedBboxes[counterTracker].width / 2));
                topSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].y - (scaleYTM - 1) * updatedBboxes[counterTracker].height / 2));
                rightSearch = std::min(img.cols, static_cast<int>(updatedBboxes[counterTracker].x + (scaleXTM + 1) * updatedBboxes[counterTracker].width / 2));
                bottomSearch = std::min(img.rows, static_cast<int>(updatedBboxes[counterTracker].y + (scaleYTM + 1) * updatedBboxes[counterTracker].height / 2));
            }
            else // scale is set to YOLO : larger search area
            {
                leftSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].x - (scaleXYolo - 1) * updatedBboxes[counterTracker].width / 2));
                topSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].y - (scaleYYolo - 1) * updatedBboxes[counterTracker].height / 2));
                rightSearch = std::min(img.cols, static_cast<int>(updatedBboxes[counterTracker].x + (scaleXYolo + 1) * updatedBboxes[counterTracker].width / 2));
                bottomSearch = std::min(img.rows, static_cast<int>(updatedBboxes[counterTracker].y + (scaleYYolo + 1) * updatedBboxes[counterTracker].height / 2));
            }
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            std::cout << "img size : width = " << img.cols << ", height = " << img.rows << std::endl;
            cv::Mat1b croppedImg = img.clone();
            croppedImg = croppedImg(searchArea); // crop img
            std::cout << "crop img" << std::endl;
            std::cout << "croppdeImg size: " << croppedImg.cols << ", height:" << croppedImg.rows << std::endl;
            std::cout << "templateImg size:" << templateImg.cols << ", height:" << templateImg.rows << std::endl;
            // std::cout << "img search size:" << img.rows << "," << img.cols << std::endl;

            // search area in template matching
            // int result_cols = img.cols - templ.cols + 1;
            // int result_rows = img.rows - templ.rows + 1;
            cv::Mat result; // for saving template matching results
            int result_cols = croppedImg.cols - templateImg.cols + 1;
            int result_rows = croppedImg.rows - templateImg.rows + 1;
            std::cout << "result_cols :" << result_cols << ", result_rows:" << result_rows << std::endl;
            // template seems to go out of frame 
            if (result_cols <= 0 || result_rows <= 0)
            {
                std::cout << "template seems to go out from frame" << std::endl;
                std::cout << "croppedImg :: left=" << leftSearch << ", top=" << topSearch << ", right=" << rightSearch << ", bottom=" << bottomSearch << std::endl;
                updatedClassesTM.push_back(-1); // tracking fault
                counterTracker++;
            }
            else
            {
                result.create(result_rows, result_cols, CV_32FC1); // create result array for matching quality
                std::cout << "make result cv::Mat" << std::endl;
                // std::cout << "create result" << std::endl;
                // const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
                cv::matchTemplate(croppedImg, templateImg, result, MATCHINGMETHOD); // template Matching
                std::cout << "finish matchTemplate" << std::endl;
                // std::cout << "templateMatching" << std::endl;
                cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat()); // normalize result score between 0 and 1
                std::cout << "normalization finish" << std::endl;
                // std::cout << "normalizing" << std::endl;
                double minVal;    // minimum score
                double maxVal;    // max score
                cv::Point minLoc; // minimum score left-top points
                cv::Point maxLoc; // max score left-top points
                cv::Point matchLoc;

                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); // In C++, we should prepare type-defined box for returns, which is usually pointer
                // std::cout << "minmaxLoc: " << maxVal << std::endl;
                if (MATCHINGMETHOD == cv::TM_SQDIFF || MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
                {
                    matchLoc = minLoc;
                }
                else
                {
                    matchLoc = maxLoc;
                }

                // meet matching criteria :: setting bbox
                // std::cout << "max value : " << maxVal << std::endl;
                /* find matching object */
                std::cout << "finish matchin template" << std::endl;
                if (minVal <= matchingThreshold)
                {
                    int leftRoi, topRoi, rightRoi, bottomRoi;
                    cv::Mat1b newTemplate;
                    cv::Rect2d roi;
                    // std::cout << "search area limited" << std::endl;
                    //
                    // convert search region coordinate to frame coordinate
                    leftRoi = std::max(0, static_cast<int>(matchLoc.x + leftSearch));
                    topRoi = std::max(0, static_cast<int>(matchLoc.y + topSearch));
                    rightRoi = std::min(img.cols, static_cast<int>(leftRoi + templateImg.cols));
                    bottomRoi = std::min(img.rows, static_cast<int>(topRoi + templateImg.rows));
                    std::cout << "updated template Roi : left=" << leftRoi << ", top =" << topRoi << ", right=" << rightRoi << ", bottom=" << bottomRoi << std::endl;
                    // update roi
                    roi.x = leftRoi;
                    roi.y = topRoi;
                    roi.width = rightRoi - leftRoi;
                    roi.height = bottomRoi - topRoi;
                    /* moving constraints */
                    if (std::pow((updatedBboxes[counterTracker].x - roi.x), 2) + std::pow((updatedBboxes[counterTracker].y - roi.y), 2) >= 2.0)
                    {
                        // update template image
                        newTemplate = img(roi);
                        std::cout << "new ROI : left=" << roi.x << ", top=" << roi.y << ", width=" << roi.width << ", height=" << roi.height << std::endl;
                        std::cout << "new template size :: width = " << newTemplate.cols << ", height = " << newTemplate.rows << std::endl;

                        // update information
                        updatedBboxesTM.push_back(roi);
                        updatedTemplatesTM.push_back(newTemplate);
                        updatedClassesTM.push_back(classIndexTM);
                        updatedSearchScales.push_back(true);
                        counterTracker++;
                    }
                    /* not moving */
                    else
                    {
                        updatedClassesTM.push_back(-1); // tracking fault
                        counterTracker++;
                    }
                    
                }
                /* doesn't meet matching criteria */
                else
                {
                    updatedClassesTM.push_back(-1); // tracking fault
                    counterTracker++;
                }
            }
        }
        /* template doesn't exist -> don't do template matching */
        else
        {
            updatedClassesTM.push_back(classIndexTM);
        }
    }
}

/* read imgs */
void pushFrame(cv::Mat1b& src, const int frameIndex)
{
    //std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    // std::cout << "push imgs" << std::endl;
    // cv::Mat1b undistortedImgL, undistortedImgR;
    // cv::undistort(srcs[0], undistortedImgL, cameraMatrix, distCoeffs);
    // cv::undistort(srcs[1], undistortedImgR, cameraMatrix, distCoeffs);
    // std::array<cv::Mat1b, 2> undistortedImgs = { undistortedImgL,undistortedImgR };
    // queueFrame.push(undistortedImgs);
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
        std::this_thread::sleep_for(std::chrono::microseconds(3333));
        std::cout << "remove image" << std::endl;
    }
}

/*
 * main function
 */
int main()
{
    /* video inference */
    const std::string filename = "test300fps.mp4";
    cv::VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 0;
    /* start multiThread */
    // multi thread code
    std::thread threadYolo(yoloDetect);
    std::cout << "start Yolo thread" << std::endl;
    std::thread threadTemplateMatching(templateMatching);
    std::cout << "start template matching thread" << std::endl;
    std::thread threadRemoveImg(removeFrame);
    while (true)
    {
        // Read the next frame
        cv::Mat frame;
        capture >> frame;
        counter++;
        if (frame.empty())
            break;
        cv::Mat1b frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_RGB2GRAY);
        //cv::Mat1b frameGray;
        // cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        pushFrame(frameGray, counter);
    }
    
    // std::thread threadTargetPredict(targetPredict);
    threadYolo.join();
    threadTemplateMatching.join();
    threadRemoveImg.join();
    return 0;
}
