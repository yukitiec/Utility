#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <chrono>
#include <fstream>

/* constant valude definition */
const std::string filename = "video/yolotest.mp4";
const bool save = true;
bool boolSparse = false;
bool boolGray = true;
std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"
const float qualityCorner = 0.01;
/* roi setting */
const int roiWidthOF = 12;
const int roiHeightOF = 12;
const int roiWidthYolo = 12;
const int roiHeightYolo = 12;
const int MoveThreshold = 0.5;
/* dense optical flow skip rate */
const int skipPixel = 2;
/*if exchange template of Yolo */
const bool boolChange = true;

std::mutex mtxImg, mtxYolo;
/* queueu definition */
/* frame queue */
std::queue<cv::Mat1b> queueFrame;
std::queue<int> queueFrameIndex;
/* yolo and optical flow */
std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch; // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2d>>> queueYoloSearchRoi;   // queue for search roi for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch;   // queue for old image for optical flow. vector size is [num human,6]
std::queue<std::vector<std::vector<cv::Rect2d>>> queueOFSearchRoi;     // queue for search roi for optical flow. vector size is [num human,6]

/* Declaration of function */
void yolo();
void getImages(cv::Mat1b&, int&);
void denseOpticalFlow();
void getPreviousData(std::vector<std::vector<cv::Mat1b>>&, std::vector<std::vector<cv::Rect2d>>&);
void getYoloData(std::vector<std::vector<cv::Mat1b>>&, std::vector<std::vector<cv::Rect2d>>&);
void opticalFlow(const cv::Mat1b&, const int&, cv::Mat1b&, cv::Rect2d&, cv::Mat1b&, cv::Rect2d&, std::vector<int>&);

void pushImg(cv::Mat1b&, int&);
void removeFrame();

/*  YOLO class definition  */
class YOLOPose
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8m-pose.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloWidth = 320;
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.45;
    const float ConfThreshold = 0.45;
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
    YOLOPose()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLOPose() { delete device; }; // Deconstructor

    void detect(cv::Mat1b& frame, int& frameIndex, int& counter, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver) //, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
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

        std::vector<cv::Rect2d> roiLatest;
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
        /* get keypoints from detectedBboxesHuman -> shoulder,elbow,wrist */
        std::vector<std::vector<std::vector<int>>> keyPoints; // vector for storing keypoints
        /* if human detected, extract keypoints */
        if (!detectedBoxesHuman.empty())
        {
            std::cout << "Human detected!" << std::endl;
            keyPointsExtractor(detectedBoxesHuman, keyPoints, ConfThreshold);
            /*push updated data to queue*/
            push2Queue(frame, frameIndex, keyPoints, roiLatest, posSaver);
            /*
            for (int i = 0; i < keyPoints.size(); i++)
            {
                std::cout << i << "-th human" << std::endl;
                for (int j = 0; j < keyPoints[i].size(); j++)
                {
                    std::cout << (j+5)<<"-th keypoints :: "<<"x=" << keyPoints[i][j][0] << ", y=" << keyPoints[i][j][1] << std::endl;
                }
            }
            */
            // std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
            /* draw keypoints in the frame */
            drawCircle(frame, keyPoints, counter);
        }
        //std::string save_path = "video/poseEstimation" + std::to_string(counter) + ".jpg";
        //cv::imwrite(save_path, frame);
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        cv::Mat yoloimg; // define yolo img type
        cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
        cv::resize(yoloimg, yoloimg, YOLOSize);
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); // vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 });                                                  // Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat);                                               // convert to float type
        imgTensor = imgTensor.div(255);                                                            // normalization
        imgTensor = imgTensor.unsqueeze(0);                                                        //(1,3,320,320)
        imgTensor = imgTensor.to(*device);                                                         // transport data to GPU
    }

    /*
    void getLatestRoi(std::vector<cv::Rect2d>& roiLatest)
    {
        //std::unique_lock<std::mutex> lock(mtxRoi); // exclude other accesses
        roiLatest = queueOFSearchRoi.front();        // get latest data
    }
    */

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
                detectedBoxesHuman.push_back(x[0].cpu());
                // std::cout << "push back top data" << std::endl;
                // for every candidates
                /* if adapt many humans, validate here */
                /*
                if (x.size(0) >= 2)
                {
                    //std::cout << "nms start" << std::endl;
                    nms(x, detectedBoxesHuman, iouThreshold); // exclude overlapped bbox : 20 milliseconds
                    //std::cout << "num finished" << std::endl;
                }
                */
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

    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, const int& ConfThreshold)
    {
        int numDetections = detectedBboxesHuman.size();
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
                    keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()), static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                }
                else
                {
                    keyPointsTemp.push_back({ -1, -1 });
                }
            }
            keyPoints.push_back(keyPointsTemp);
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
        std::string save_path = "video/poseEstimation" + std::to_string(counter) + ".jpg";
        cv::imwrite(save_path, frame);
    }

    void push2Queue(cv::Mat1b& frame, int& frameIndex, std::vector<std::vector<std::vector<int>>>& keyPoints,
        std::vector<cv::Rect2d>& roiLatest, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver)
    {
        /* check roi Latest
         * if tracking was successful -> update and
         * else : update roi and imgSearch and calculate features. push data to queue
         */
        if (!keyPoints.empty())
        {
            std::vector<std::vector<cv::Rect2d>> humanJoints; // for every human
            std::vector<std::vector<cv::Mat1b>> imgHuman;
            std::vector<std::vector<std::vector<int>>> humanJointsCenter;
            /* for every person */
            for (int i = 0; i < keyPoints.size(); i++)
            {
                std::vector<cv::Rect2d> joints; // for every joint
                std::vector<cv::Mat1b> imgJoint;
                std::vector<std::vector<int>> jointsCenter;
                /* for every joints*/
                for (int j = 0; j < keyPoints[i].size(); j++)
                {
                    if (static_cast<int>(keyPoints[i][j][0]) != -1)
                    {
                        int left = std::max(static_cast<int>(keyPoints[i][j][0] - roiWidthYolo / 2), 0);
                        int top = std::max(static_cast<int>(keyPoints[i][j][1] - roiHeightYolo / 2), 0);
                        int right = std::min(static_cast<int>(keyPoints[i][j][0] + roiWidthYolo / 2), frame.cols);
                        int bottom = std::min(static_cast<int>(keyPoints[i][j][1] + roiHeightYolo / 2), frame.rows);
                        cv::Rect2d roi(left, top, right - left, bottom - top);
                        jointsCenter.push_back({ frameIndex,keyPoints[i][j][0],keyPoints[i][j][1] });
                        joints.push_back(roi);
                        imgJoint.push_back(frame(roi));
                    }
                    /* keypoints can't be detected */
                    else
                    {
                        jointsCenter.push_back({ frameIndex,-1,-1 });
                        joints.emplace_back(-1, -1, -1, -1);
                    }

                }
                humanJoints.push_back(joints);
                if (!imgJoint.empty())
                {
                    imgHuman.push_back(imgJoint);
                }
                humanJointsCenter.push_back(jointsCenter);
            }
            //push data to queue
            std::unique_lock<std::mutex> lock(mtxYolo); // exclude other accesses 
            queueYoloSearchRoi.push(humanJoints);
            if (!imgHuman.empty())
            {
                queueYoloOldImgSearch.push(imgHuman);
            }
            posSaver.push_back(humanJointsCenter);
            std::cout << "push Yolo detection data to queue" << std::endl;
        }
    }
};

void yolo()
{
    /* constructor of YOLOPoseEstimator */
    YOLOPose yoloPoseEstimator;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    if (queueFrame.empty())
    {
        while (queueFrame.empty())
        {
            if (!queueFrame.empty())
            {
                break;
            }
            std::cout << "wait for images" << std::endl;
        }
    }
    /* frame is available */
    else
    {
        int counter = 1;
        int counterFinish = 0;
        while (true)
        {
            if (counterFinish == 10)
            {
                break;
            }
            /* frame can't be available */
            if (queueFrame.empty())
            {
                counterFinish++;
                /* waiting */
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            /* frame available -> start yolo pose estimation */
            else
            {
                counterFinish = 0;
                cv::Mat1b frame;
                int frameIndex;
                auto start = std::chrono::high_resolution_clock::now();
                getImages(frame, frameIndex);
                yoloPoseEstimator.detect(frame, frameIndex, counter, posSaver);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
            }
        }
    }
    //prepare file
    std::string filePath = "jointsPositionYolo_roi12_moveThreshold0.5_changeTemplate.csv";
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
}

void getImages(cv::Mat1b& frame, int& frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg); // exclude other accesses 
    frame = queueFrame.front();
    //cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frameIndex = queueFrameIndex.front();
    queueFrame.pop();
    queueFrameIndex.pop();
}

void denseOpticalFlow()
{
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}

    int counterStart = 0;
    while (true)
    {
        if (counterStart == 2)
        {
            break;
        }
        if (!queueYoloOldImgSearch.empty())
        {
            counterStart++;
            int countIteration = 1;
            while (!queueYoloOldImgSearch.empty())
            {
                std::cout << countIteration << " :: remove yolo data" << std::endl;
                countIteration++;
                queueYoloOldImgSearch.pop();
            }
            while (!queueYoloSearchRoi.empty())
            {
                queueYoloSearchRoi.pop();
            }
        }
        cv::Mat1b frame;
        int frameIndex;
        if (!queueFrame.empty())
        {
            //getImages(frame, frameIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "wait for yolo detection" << std::endl;
        }
    }
    /* frame is available */
    int counter = 1;
    int counterFinish = 0;
    while (true)
    {
        if (counterFinish == 10)
        {
            break;
        }
        if (queueFrame.empty())
        {
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        /* frame available */
        else
        {
            std::cout << "start optical flow tracking" << std::endl;
            /* get images from queue */
            cv::Mat1b frame;
            int frameIndex;
            auto start = std::chrono::high_resolution_clock::now();
            getImages(frame, frameIndex);
            /* optical flow process for each joints */
            std::vector<std::vector<cv::Mat1b>> previousImg; //[number of human,0~6,cv::Mat1b]
            std::vector<std::vector<cv::Rect2d>> searchRoi; //[number of human,6,cv::Rect2d], if tracker was failed, roi.x == -1
            getPreviousData(previousImg, searchRoi);
            std::cout << "finish getting previous data " << std::endl;
            /* start optical flow process */
            /* for every human */
            std::vector<std::vector<cv::Mat1b>> updatedImgHuman;
            std::vector<std::vector<cv::Rect2d>> updatedSearchRoiHuman;
            std::vector<std::vector<std::vector<int>>> updatedPositionsHuman;
            for (int i = 0; i < searchRoi.size(); i++)
            {

                /* for every joints */
                std::vector<cv::Mat1b> updatedImgJoints;
                std::vector<cv::Rect2d> updatedSearchRoi;
                std::vector<std::vector<int>> updatedPositions;
                std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
                cv::Mat1b updatedImgLeftShoulder, updatedImgRightShoulder, updatedImgLeftElbow, updatedImgRightElbow, updatedImgLeftWrist, updatedImgRightWrist;
                cv::Rect2d updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
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
                if (searchRoi[i][0].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][0]),
                        std::ref(updatedImgLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(updatedPosLeftShoulder));
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
                if (searchRoi[i][1].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][1]), 
                        std::ref(updatedImgRightShoulder), std::ref(updatedSearchRoiRightShoulder),  std::ref(updatedPosRightShoulder));
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
                if (searchRoi[i][2].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][2]),
                        std::ref(updatedImgLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(updatedPosLeftElbow));
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
                if (searchRoi[i][3].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][3]),
                        std::ref(updatedImgRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(updatedPosRightElbow));
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
                if (searchRoi[i][4].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][4]),
                        std::ref(updatedImgLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(updatedPosLeftWrist));
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
                if (searchRoi[i][5].x > 0)
                {
                    threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][5]),
                        std::ref(updatedImgRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(updatedPosRightWrist));
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
                std::cout << "all threads have started" << std::endl;
                /* wait for all thread has finished */
                int counterThread = 0;
                if (!threadJoints.empty())
                {
                    for (std::thread& thread : threadJoints) {
                        thread.join();
                        counterThread++;
                    }
                    std::cout << counterThread << " threads have finished!" << std::endl;
                }
                else
                {
                    std::cout << "no thread has started" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
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
                /* left shoulder*/
                if (updatedSearchRoi[0].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftShoulder);
                }
                /* right shoulder*/
                if (updatedSearchRoi[1].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightShoulder);
                }
                /*left elbow*/
                if (updatedSearchRoi[2].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftElbow);
                }
                /*right elbow */
                if (updatedSearchRoi[3].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightElbow);
                }
                /* left wrist*/
                if (updatedSearchRoi[4].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftWrist);
                }
                /*right wrist*/
                if (updatedSearchRoi[5].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightWrist);
                }
                /* combine all data for one human */
                updatedSearchRoiHuman.push_back(updatedSearchRoi);
                updatedPositionsHuman.push_back(updatedPositions);
                if (!updatedImgJoints.empty())
                {
                    updatedImgHuman.push_back(updatedImgJoints);
                }
            }
            /* push updated data to queue */
            queueOFSearchRoi.push(updatedSearchRoiHuman);
            posSaver.push_back(updatedPositionsHuman);
            if (!updatedImgHuman.empty())
            {
                queueOFOldImgSearch.push(updatedImgHuman);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << " Time taken by OpticalFlow : " << duration.count() << " milliseconds" << std::endl;
        }
    }
    //prepare file
    std::string filePathOF = "jointsPositionDenseOpticalFlow_roi12_moveThreshold0.5_changeTemplate.csv";
    // Open the file for writing
    std::ofstream outputFile(filePathOF);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
    }
    std::cout << "estimated position :: Optical Flow :: " << std::endl;
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
    // close file
    outputFile.close();
}

void getPreviousData(std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2d>>& searchRoi)
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
        queueOFOldImgSearch.pop();
        queueOFSearchRoi.pop();
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
        //std::cout << "searchRoi by optical flow size : " << searchRoi.size() << ","<<searchRoi[0].size()<<std::endl;
        //std::cout << "successful trackers of optical flow : " << previousImg.size() << std::endl;
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
                    int counterTrackerOF = 0; //number of successful trackers by Optical Flow
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
                    std::vector<std::vector<cv::Point2f>> features;
                    /* for every joints */
                    for (const cv::Rect2d& roi : searchYoloRoi[i])
                    {
                        /* keypoint is found */
                        if (roi.x != -1)
                        {
                            joints.push_back(roi);
                            imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
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
                    }
                }

            }
            /* no OF tracking was successful or first yolo detection */
            else
            {
                searchRoi = std::vector<std::vector<cv::Rect2d>>(); //initialize searchRoi for avoiding data
                std::cout << "Optical Flow :: failed or first Yolo detection " << std::endl;
                int counterJoint = 0;
                int counterYoloImg = 0;
                std::vector<cv::Rect2d> joints;
                std::vector<cv::Mat1b> imgJoints;
                std::vector<std::vector<cv::Point2f>> features;
                /* for every joints */
                for (const cv::Rect2d& roi : searchYoloRoi[i])
                {
                    /* keypoint is found */
                    if (roi.x != -1)
                    {
                        joints.push_back(roi);
                        imgJoints.push_back(previousYoloImg[i][counterYoloImg]);
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
                }
            }
        }
    }
}

void getYoloData(std::vector<std::vector<cv::Mat1b>>& previousYoloImg, std::vector<std::vector<cv::Rect2d>>& searchYoloRoi)
{
    std::unique_lock<std::mutex> lock(mtxYolo);
    previousYoloImg = queueYoloOldImgSearch.front();
    searchYoloRoi = queueYoloSearchRoi.front();
    queueYoloOldImgSearch.pop();
    queueYoloSearchRoi.pop();
}

void opticalFlow(const cv::Mat1b& frame, const int& frameIndex, cv::Mat1b& previousImg, cv::Rect2d& searchRoi,
    cv::Mat1b& updatedImg, cv::Rect2d& updatedSearchRoi, std::vector<int>& updatedPos)
{
    // Calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
    cv::Mat1b croppedImg = frame(searchRoi);
    // Calculate Optical Flow
    cv::Mat flow(previousImg.size(), CV_32FC2); //prepare matrix for saving dense optical flow
    cv::calcOpticalFlowFarneback(previousImg, croppedImg, flow, 0.5, 2, 3, 3, 5, 1.2, 0); //calculate dense optical flow
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
    //calculate velocity


    float vecX = 0;
    float vecY = 0;
    int numPixels = 0;
    int rows = static_cast<int>(flow.rows / (skipPixel + 1)); //number of rows adapted pixels
    int cols = static_cast<int>(flow.cols / (skipPixel + 1)); //number of cols adapted pixels
    for (int y = 1; y <= rows; ++y) {
        for (int x = 1; x <= cols; ++x) {
            cv::Point2f flowVec = flow.at<cv::Point2f>(std::max((y-1)*(skipPixel+1),0), std::max((x - 1) * (skipPixel + 1), 0)); //get velocity of position (y,x)
            // Access flowVec.x and flowVec.y for the horizontal and vertical components of velocity.
            vecX += flowVec.x;
            vecY += flowVec.y;
            numPixels += 1;
        }
    }
    vecX /= numPixels;
    vecY /= numPixels;
    if (std::pow(vecX, 2) + std::pow(vecY, 2) >= MoveThreshold)
    {
        int updatedLeft = searchRoi.x + static_cast<int>(vecX);
        int updatedTop = searchRoi.y + static_cast<int>(vecY);
        cv::Rect2d roi(updatedLeft, updatedTop, roiWidthOF, roiHeightOF);
        updatedSearchRoi = roi;
        // Update the previous frame and previous points
        updatedImg = croppedImg.clone();
        updatedPos = { frameIndex,static_cast<int>(updatedLeft+roiWidthOF/2),static_cast<int>(updatedTop+roiHeightOF/2) };
    }
    /* not move -> tracking was failed */
    else
    {
        updatedSearchRoi.x = -1;
        updatedSearchRoi.y = -1;
        updatedSearchRoi.width = -1;
        updatedSearchRoi.height = -1;
        updatedPos = { frameIndex,-1,-1 };
    }
}

void pushImg(cv::Mat1b& frame, int& frameIndex)
{
    //std::unique_lock<std::mutex> lock(mtxImg);
    queueFrame.push(frame);
    queueFrameIndex.push(frameIndex);
}

void removeFrame()
{
    int counterStart = 0;
    while (counterStart < 2)
    {
        if (!queueYoloSearchRoi.empty())
        {
            while (true)
            {
                if (queueYoloSearchRoi.empty())
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
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        std::cout << "remove image" << std::endl;
    }
}

int main()
{
    /* image inference */
    /*
    cv::Mat img = cv::imread("video/0019.jpg");
    cv::Mat1b imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    std::cout << img.size()<< std::endl;
    int counter = 1;
    yoloPoseEstimator.detect(imgGray,counter);
    */
    /* video inference */
    const std::string filename = "yolotest.mp4";
    cv::VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 0;
    /* start multiThread */
    std::thread threadYolo(yolo);
    std::thread threadOF(denseOpticalFlow);
    std::thread threadRemoveFrame(removeFrame);
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
        pushImg(frameGray, counter);
    }
    threadYolo.join();
    threadOF.join();
    threadRemoveFrame.join();

    return 0;
}
