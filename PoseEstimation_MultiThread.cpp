#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <chrono>

/* constant valude definition */
const std::string filename = "video/yolotest.mp4";
const bool save = true;
bool boolSparse = false;
bool boolGray = true;
std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"
const float qualityCorner = 0.2;
/* roi setting */
const int roiWidth = 32;
const int roiHeight = 32;
const int MoveThreshold = 4;

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
std::queue<std::vector<std::vector<std::vector<cv::Point2f>>>> queueOFFeatures;       // queue for OF features. vector size is [num human,6]

/* Declaration of function */
void yolo();
void getImages(cv::Mat1b&, int&);
void sparseOpticalFlow();
void getPreviousData(std::vector<std::vector<cv::Mat1b>>&
    
    , std::vector<std::vector<cv::Rect2d>>&,std::vector<std::vector<std::vector<cv::Point2f>>>&);
void opticalFlow(const cv::Mat1b&, const int&, cv::Mat1b&, cv::Rect2d&, std::vector<cv::Point2f>&,cv::Mat1b&, cv::Rect2d&, std::vector<cv::Point2f>&, std::vector<int>&);

void pushImg(cv::Mat1b, int);

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
        }*/
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
                        int left = std::max(static_cast<int>(keyPoints[i][j][0] - roiWidth / 2), 0);
                        int top = std::max(static_cast<int>(keyPoints[i][j][1] - roiHeight / 2), 0);
                        int right = std::min(static_cast<int>(keyPoints[i][j][0] + roiWidth / 2), frame.cols);
                        int bottom = std::min(static_cast<int>(keyPoints[i][j][1] + roiHeight / 2), frame.rows);
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
            }
        }
    }
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

void sparseOpticalFlow()
{
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    if (queueYoloOldImgSearch.empty())
    {
        int counterStart = 0;
        while (queueYoloOldImgSearch.empty())
        {
            if (counterStart == 3)
            {
                break;
            }
            if (!queueYoloOldImgSearch.empty())
            {
                counterStart++;
                while (!queueYoloOldImgSearch.empty())
                {
                    queueYoloOldImgSearch.pop();
                }
                while (!queueYoloSearchRoi.empty())
                {
                    queueYoloSearchRoi.pop();
                }
            }
            cv::Mat1b frame;
            int frameIndex;
            getImages(frame, frameIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            std::cout << "wait for yolo detection" << std::endl;
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
            if (queueFrame.empty())
            {
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            /* frame available */
            else
            {
                /* get images from queue */
                cv::Mat1b frame;
                int frameIndex;
                getImages(frame, frameIndex);
                /* optical flow process for each joints */
                std::vector<std::vector<cv::Mat1b>> previousImg; //[number of human,0~6,cv::Mat1b]
                std::vector<std::vector<cv::Rect2d>> searchRoi; //[number of human,6,cv::Rect2d], if tracker was failed, roi.x == -1
                std::vector<std::vector<std::vector<cv::Point2f>>> previousFeatures;//[number of human,0~6,num of features,cv::Point2f]
                getPreviousData(previousImg, searchRoi, previousFeatures);
                /* start optical flow process */
                /* for every human */
                std::vector<std::vector<cv::Mat1b>> updatedImgHuman;
                std::vector<std::vector<cv::Rect2d>> updatedSearchRoiHuman;
                std::vector<std::vector<std::vector<cv::Point2f>>> updatedFeaturesHuman;
                std::vector<std::vector<std::vector<int>>> updatedPositionsHuman;
                for (int i = 0; i < searchRoi.size(); i++)
                {
                    /* for every joints */
                    std::vector<cv::Mat1b> updatedImgJoints;
                    std::vector<cv::Rect2d> updatedSearchRoi;
                    std::vector<std::vector<cv::Point2f>> updatedFeatures;
                    std::vector<std::vector<int>> updatedPositions;
                    std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
                    cv::Mat1b updatedImgLeftShoulder, updatedImgRightShoulder, updatedImgLeftElbow, updatedImgRightElbow, updatedImgLeftWrist, updatedImgRightWrist;
                    cv::Rect2d updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
                    std::vector<cv::Point2f> updatedFeaturesLeftShoulder, updatedFeaturesRightShoulder, updatedFeaturesLeftElbow, updatedFeaturesRightElbow, updatedFeaturesLeftWrist, updatedFeaturesRightWrist;
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
                    if (searchRoi[i][0].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow,std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][0]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(updatedFeaturesLeftShoulder), std::ref(updatedPosLeftShoulder));
                        boolLeftShoulder = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiLeftShoulder.x = -1;
                        updatedSearchRoiLeftShoulder.y = -1;
                        updatedSearchRoiLeftShoulder.width = -1;
                        updatedSearchRoiLeftShoulder.height = -1;
                    }
                    /* right shoulder */
                    if (searchRoi[i][1].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][1]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgRightShoulder), std::ref(updatedSearchRoiRightShoulder), std::ref(updatedFeaturesRightShoulder), std::ref(updatedPosRightShoulder));
                        boolRightShoulder = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiRightShoulder.x = -1;
                        updatedSearchRoiRightShoulder.y = -1;
                        updatedSearchRoiRightShoulder.width = -1;
                        updatedSearchRoiRightShoulder.height = -1;
                    }
                    /* left elbow */
                    if (searchRoi[i][2].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][2]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(updatedFeaturesLeftElbow), std::ref(updatedPosLeftElbow));
                        boolLeftElbow = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiLeftElbow.x = -1;
                        updatedSearchRoiLeftElbow.y = -1;
                        updatedSearchRoiLeftElbow.width = -1;
                        updatedSearchRoiLeftElbow.height = -1;
                    }
                    /* right elbow */
                    if (searchRoi[i][3].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][3]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(updatedFeaturesRightElbow), std::ref(updatedPosRightElbow));
                        boolRightElbow = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiRightElbow.x = -1;
                        updatedSearchRoiRightElbow.y = -1;
                        updatedSearchRoiRightElbow.width = -1;
                        updatedSearchRoiRightElbow.height = -1;
                    }
                    /* left wrist */
                    if (searchRoi[i][4].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][4]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(updatedFeaturesLeftWrist), std::ref(updatedPosLeftWrist));
                        boolLeftWrist = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiLeftWrist.x = -1;
                        updatedSearchRoiLeftWrist.y = -1;
                        updatedSearchRoiLeftWrist.width = -1;
                        updatedSearchRoiLeftWrist.height = -1;
                    }
                    /* right wrist */
                    if (searchRoi[i][5].x != -1)
                    {
                        threadJoints.emplace_back(opticalFlow, std::ref(frame), std::ref(frameIndex), std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][5]), std::ref(previousFeatures[i][counterTracker]),
                            std::ref(updatedImgRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(updatedFeaturesRightWrist), std::ref(updatedPosRightWrist));
                        boolRightWrist = true;
                        counterTracker++;
                    }
                    else
                    {
                        updatedSearchRoiRightWrist.x = -1;
                        updatedSearchRoiRightWrist.y = -1;
                        updatedSearchRoiRightWrist.width = -1;
                        updatedSearchRoiRightWrist.height = -1;
                    }
                    /* wait for all thread has finished */
                    for (std::thread& thread : threadJoints) {
                        thread.join();
                    }
                    /* combine all data and push data to queue */
                    /* search roi */
                    updatedSearchRoi.push_back(updatedSearchRoiLeftShoulder);
                    updatedSearchRoi.push_back(updatedSearchRoiRightShoulder);
                    updatedSearchRoi.push_back(updatedSearchRoiLeftElbow);
                    updatedSearchRoi.push_back(updatedSearchRoiRightElbow);
                    updatedSearchRoi.push_back(updatedSearchRoiLeftWrist);
                    updatedSearchRoi.push_back(updatedSearchRoiRightWrist);
                    /* updated img */
                    /* left shoulder*/
                    if (updatedSearchRoi[0].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgLeftShoulder);
                        updatedFeatures.push_back(updatedFeaturesLeftShoulder);
                        updatedPositions.push_back(updatedPosLeftShoulder);
                    }
                    /* right shoulder*/
                    if (updatedSearchRoi[1].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgRightShoulder);
                        updatedFeatures.push_back(updatedFeaturesRightShoulder);
                        updatedPositions.push_back(updatedPosRightShoulder);
                    }
                    /*left elbow*/
                    if (updatedSearchRoi[2].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgLeftElbow);
                        updatedFeatures.push_back(updatedFeaturesLeftElbow);
                        updatedPositions.push_back(updatedPosLeftElbow);
                    }
                    /*right elbow */
                    if (updatedSearchRoi[3].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgRightElbow);
                        updatedFeatures.push_back(updatedFeaturesRightElbow);
                        updatedPositions.push_back(updatedPosRightElbow);
                    }
                    /* left wrist*/
                    if (updatedSearchRoi[4].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgLeftWrist);
                        updatedFeatures.push_back(updatedFeaturesLeftWrist);
                        updatedPositions.push_back(updatedPosLeftWrist);
                    }
                    /*right wrist*/
                    if (updatedSearchRoi[5].x != -1)
                    {
                        updatedImgJoints.push_back(updatedImgRightWrist);
                        updatedFeatures.push_back(updatedFeaturesRightWrist);
                        updatedPositions.push_back(updatedPosRightWrist);
                    }
                    /* combine all data for one human */
                    updatedSearchRoiHuman.push_back(updatedSearchRoi);
                    if (!updatedImgJoints.empty())
                    {
                        updatedImgHuman.push_back(updatedImgJoints);
                        updatedFeaturesHuman.push_back(updatedFeatures);
                        updatedPositionsHuman.push_back(updatedPositions);
                    }
                }
                /* push updated data to queue */
                queueOFSearchRoi.push(updatedSearchRoiHuman);
                if (!updatedPositionsHuman.empty())
                {
                    queueOFOldImgSearch.push(updatedImgHuman);
                    queueOFFeatures.push(updatedFeaturesHuman);
                    posSaver.push_back(updatedPositionsHuman);
                }
            }
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
                }
            }
        }
    }
}

void getPreviousData(std::vector<std::vector<cv::Mat1b>>& previousImg, std::vector<std::vector<cv::Rect2d>>& searchRoi,
    std::vector<std::vector<std::vector<cv::Point2f>>>& previousFeatures)
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
        previousFeatures = queueOFFeatures.front();
        queueOFOldImgSearch.pop();
        queueOFSearchRoi.pop();
        queueOFFeatures.pop();
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
        getYoloData(previousYoloImg, searchYoloRoi);
        /* update data here */
        /* OF tracking has started */
        if (!searchRoi.empty())
        {
            /* iterate for all human detection */
            for (int i = 0; i < searchRoi.size(); i++)
            {
                /* for all joints */
                int counterJoint = 0;
                int counterYoloImg = 0;
                for (const cv::Rect2d& roi : searchRoi[i])
                {
                    /* tracking is failed -> updated data with yolo data */
                    if (roi.x == -1)
                    {
                        /* yolo detect joints -> update of data */
                        if (searchYoloRoi[i][counterJoint].x != -1)
                        {
                            searchRoi[i].insert(searchRoi[i].begin() + counterJoint, searchYoloRoi[i][counterJoint]);
                            previousImg[i].insert(previousImg[i].begin() + counterJoint, previousYoloImg[i][counterYoloImg]);
                            /* calculate features? */
                            std::vector<cv::Point2f> p0;
                            /* shi-tomashi method. if not good -> try */
                            /* Shi-tomashi corner method */
                            cv::goodFeaturesToTrack(previousYoloImg[i][counterYoloImg], p0, 20, qualityCorner, 5, cv::Mat(), 5, false, 0.04);
                            /* Harris Corner detection */
                            //cv::goodFeaturesToTrack(previousYoloImg[i][counterYoloImg], p0, 20, qualityCorner, 5, cv::Mat(), 5, true);
                            /* FAST corner detection */
                            /*
                            int thresholdFast = 100;
                            bool boolnonMax = true;
                            std::vector<cv::KeyPoint> keypoints;
                            cv::Fast(previousYoloImg[i][counterYoloImg],thresholdFast,boolNonMax);
                            */
                            previousFeatures[i].insert(previousFeatures[i].begin() + counterJoint, p0);
                            counterJoint++;
                            counterYoloImg++;
                        }
                        /* yolo can't detect joint -> not updated data */
                        else
                        {
                            counterJoint++;
                        }

                    }
                    /* tracking is successful */
                    else
                    {
                        /* not update features */
                        counterJoint++;
                    }
                }
            }
        }
        /* first yolo detection */
        else
        {
            /* for every human */
            for (int i = 0; i < searchYoloRoi.size(); i++)
            {
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
                        /* calculate features? */
                        std::vector<cv::Point2f> p0;
                        /* Shi-tomashi corner method */
                        cv::goodFeaturesToTrack(previousYoloImg[i][counterYoloImg], p0, 20, qualityCorner, 5, cv::Mat(), 5, false, 0.04);
                        /* Harris Corner detection */
                        //cv::goodFeaturesToTrack(previousYoloImg[i][counterYoloImg], p0, 20, qualityCorner, 5, cv::Mat(), 5, true);
                        features.push_back(p0);
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
                searchRoi.push_back( joints );
                if (!imgJoints.empty())
                {
                    previousImg.push_back(imgJoints);
                    previousFeatures.push_back(features);
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

void opticalFlow(const cv::Mat1b& frame, const int& frameIndex, cv::Mat1b& previousImg, cv::Rect2d& searchRoi, std::vector<cv::Point2f>& previousFeatures,
    cv::Mat1b& updatedImg, cv::Rect2d& updatedSearchRoi, std::vector<cv::Point2f>& updatedFeatures, std::vector<int>& updatedPos)
{
    // Calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
    cv::Mat1b croppedImg = frame(searchRoi);
    std::vector<cv::Point2f> p1;
    calcOpticalFlowPyrLK(previousImg, croppedImg, previousFeatures, p1, status, err, cv::Size(5, 5), 2, criteria);
    std::vector<cv::Point2f> good_new;

    // Visualization part
    int sumX = 0;
    int sumY = 0;
    for (uint i = 0; i < previousFeatures.size(); i++)
    {
        // Select good points
        if (status[i] == 1)
        {
            good_new.push_back(p1[i]);
            sumX += p1[i].x;
            sumY += p1[i].y;
            // Draw the tracks
            //cv::line(mask, p1[i], p0[i], colors[i], 2);
            //cv::circle(frame, p1[i], 5, colors[i], -1);
        }
    }
    if (!good_new.empty())
    {
        /* convert search coordinate to img coordinate */
        int meanX = searchRoi.x + static_cast<int>(sumX / good_new.size());
        int meanY = searchRoi.y + static_cast<int>(sumY / good_new.size());
        cv::Rect2d roi(meanX - roiWidth / 2, meanY - roiHeight / 2, roiWidth, roiHeight);
        /* move -> tracking is successful */
        if (std::pow((roi.x - searchRoi.x), 2) + std::pow((roi.y - searchRoi.y), 2) > MoveThreshold)
        {
            updatedSearchRoi = roi;
            // Update the previous frame and previous points
            updatedImg = croppedImg.clone();
            updatedFeatures = good_new;
            updatedPos = { frameIndex,meanX,meanY };
        }
        /*not move -> tracking was failed */
        else
        {
            updatedSearchRoi.x = -1;
            updatedSearchRoi.y = -1;
            updatedSearchRoi.width = -1;
            updatedSearchRoi.height = -1;
            updatedPos = { frameIndex,-1,-1 };
        }
    }
    else
    {
        updatedSearchRoi.x = -1;
        updatedSearchRoi.y = -1;
        updatedSearchRoi.width = -1;
        updatedSearchRoi.height = -1;
        updatedPos = { frameIndex,-1,-1 };
    }
}

void pushImg(cv::Mat1b frame, int frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg);
    queueFrame.push(frame);
    queueFrameIndex.push(frameIndex);
}

int main()
{
    YOLOPose yoloPoseEstimator; // construct
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
    const std::string filename = "video/yolotest.mp4";
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
    std::thread threadOF(sparseOpticalFlow);
    while (true)
    {
        // Read the next frame
        cv::Mat frame;
        capture >> frame;
        counter++;
        if (frame.empty())
            break;
        //cv::Mat1b frameGray;
        // cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        pushImg(frame, counter);
    }
    threadYolo.join();
    threadOF.join();

    return 0;
}
