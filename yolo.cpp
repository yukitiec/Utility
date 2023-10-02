// ximea_test.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"

/**
* @brief サイズを考慮してROIを作成
*
* @param[in] center トラッキング点の前フレームの位置（ROIの中心）
* @param[in] src 入力画像
* @param[in] roiBase ROIの枠
* @return ROI画像
*/
cv::Mat createROI(cv::Point2d& center, const cv::Mat& src, const cv::Rect& roiBase)
{
    return src((roiBase + cv::Point(center)) & cv::Rect(cv::Point(0, 0), src.size()));
}

std::mutex mtxImg, mtxYoloLeft, mtxYoloRight, mtxTarget; // define mutex

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
std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
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
void checkStorage(std::vector<std::vector<cv::Rect2d>>&);
void checkClassStorage(std::vector<std::vector<int>>&);
/* get img */
bool getImagesFromQueueYolo(std::array<cv::Mat1b, 2>&, int&); // get image from queue
bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>&, int&);
/* template matching */
void templateMatching();                                                                                                                                  // void* //proto definition
void templateMatchingForLeft(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&, std::vector<std::vector<int>>&);  //, int, int, float, const int);
void templateMatchingForRight(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&, std::vector<std::vector<int>>&); // , int, int, float, const int);
void combineYoloTMData(std::vector<int>&, std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,
    std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&, const int&);
void getTemplateMatchingDataLeft(bool&, std::vector<int>&, std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<bool>&, int&);
void getTemplateMatchingDataRight(bool&, std::vector<int>&, std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<bool>&, int&);
void processTM(std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<bool>&, std::vector<cv::Rect2d>&, cv::Mat1b&,
    std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&);
/* access yolo data */
void getYoloDataLeft(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&);
void getYoloDataRight(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&);
void push2YoloDataRight(std::vector<cv::Rect2d>&, std::vector<int>&);
void push2YoloDataLeft(std::vector<cv::Rect2d>&, std::vector<int>&);
/* read img */
void pushFrame(std::array<cv::Mat1b, 2>&, const int);
/* Target Prediction */
void targetPredict();
void updateTrackingData(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&);
bool getTargetData(int&, std::vector<int>&, std::vector<int>&, std::vector<cv::Rect2d>&, std::vector<cv::Rect2d>&, bool&, bool&);
void convertRoi2Center(std::vector<int>&, std::vector<cv::Rect2d>&, const int&, std::vector<std::vector<int>>&);
void adjustData(std::vector<std::vector<int>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&, std::vector<int>&, std::vector<int>&);
void getLatestData(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&);
void getLatestClass(std::vector<std::vector<int>>&, std::vector<int>&);
bool sortByCenterX(const std::vector<int>&, const std::vector<int>&);
void sortData(std::vector<std::vector<int>>&, std::vector<int>&);
/* trajectory prediction */
void curveFitting(const std::vector<std::vector<int>>&, std::vector<float>&);     // curve fitting
void linearRegression(const std::vector<std::vector<int>>&, std::vector<float>&); // linear regression
void linearRegressionZ(const std::vector<std::vector<int>>&, std::vector<float>&);
void trajectoryPredict2D(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<int>&);
float calculateME(std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
void dataMatching(std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&,
    std::vector<int>&, std::vector<int>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<std::vector<int>>>>&);
void predict3DTargets(std::vector<std::vector<std::vector<std::vector<int>>>>&, std::vector<std::vector<int>>&);

/*  YOLO class definition  */
class YOLODetect
{
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8n_epoch2200.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloWidth = 320;
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth, yoloHeight };
    const float IoUThreshold = 0.45;
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
    YOLODetect()
    {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLODetect() { delete device; }; // Deconstructor

    void detectLeft(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
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

        // std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        // std::cout << "finish inference" << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMLeft; // for limiting detection area
        std::vector<int> classIndexesTMLeft;           // candidate class from TM
        /* get latest data from Template Matching */
        if (!queueYoloClassIndexLeft.empty())
        {
            // std::cout << "get latest data" << std::endl;
            getYoloDataLeft(bboxesCandidateTMLeft, classIndexesTMLeft); // get latest data
            // std::cout << "bboxesCandidateTMLeft size : " << bboxesCandidateTMLeft.size() << ", classIndexesTMLeft size : " << classIndexesTMLeft.size() << std::endl;
        }
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
        if (!existedClass.empty())
        {
            std::cout << "existed class after roisetting of Ball:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        roiSetting(detectedBoxes1Left, existedRoi, existedClass, newRoi, newClass, BOX, bboxesCandidateTMLeft, existedClass);
        /* in Ball roisetting update all classIndexesTMLeft to existedClass, so here adapt existedClass as a reference class */
        if (!existedClass.empty())
        {
            std::cout << "existedClass after roiSetting of Box:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        /* push and save data */
        push2QueueLeft(existedRoi, newRoi, existedClass, newClass, frame, posSaver, classSaver);
    };

    void detectRight(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /* inference by YOLO
         *  Args:
         *      frame : img
         *      posSaver : storage for saving detected position
         *      queueYoloTemplate : queue for pushing detected img
         *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
         *      queueClassIndex : queue for pushing detected
         */
         // bbox0.resize(0);
         // bbox1.resize(0);
         // clsidx.resize(0);
         // predscore.resize(0);

         /* preprocess img */
        torch::Tensor imgTensor;
        preprocessImg(frame, imgTensor);
        /* inference */
        // auto startInference = std::chrono::high_resolution_clock::now();
        torch::Tensor preds;
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); // preds shape : [1,6,2100]
        }
        // auto stopInference = std::chrono::high_resolution_clock::now();
        // auto durationInference = std::chrono::duration_cast<std::chrono::milliseconds>(stopInference - startInference);
        // std::cout << " Time taken by YOLO inference : " << durationInference.count() << " milliseconds" << std::endl;
        // preds shape : [1,6,2100]
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMRight; // for limiting detection area
        std::vector<int> classIndexesTMRight;           // candidate class from TM
        if (!queueYoloClassIndexRight.empty())
        {
            getYoloDataRight(bboxesCandidateTMRight, classIndexesTMRight); // get latest data
        }
        /* postProcess*/
        preds = preds.permute({ 0, 2, 1 });                                    // change order : (1,6,8400) -> (1,8400,6)
        std::vector<torch::Tensor> detectedBoxes0Right, detectedBoxes1Right; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Right, detectedBoxes1Right, ConfThreshold, IoUThreshold);

        std::vector<cv::Rect2d> existedRoi, newRoi;
        std::vector<int> existedClass, newClass;
        /* Roi Setting : take care of dealing with TM data */
        /* ROI and class index management */
        roiSetting(detectedBoxes0Right, existedRoi, existedClass, newRoi, newClass, BALL, bboxesCandidateTMRight, classIndexesTMRight);
        if (!existedClass.empty())
        {
            std::cout << "existed class after roisetting of Ball:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        roiSetting(detectedBoxes1Right, existedRoi, existedClass, newRoi, newClass, BOX, bboxesCandidateTMRight, existedClass);
        /* in Ball roisetting update all classIndexesTMRight to existedClass, so here adapt existedClass as a reference class */
        if (!existedClass.empty())
        {
            std::cout << "existedClass after roiSetting of Box:" << std::endl;
            for (const int& classIndex : existedClass)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        /* push and save data */
        push2QueueRight(existedRoi, newRoi, existedClass, newClass, frame, posSaver, classSaver);
    };

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

         /* detected by Yolo */
        if (!detectedBoxes.empty())
        {
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
            /* Mo TM tracker exist */
            else
            {
                // std::cout << "No TM tracker exist " << std::endl;
                int numBboxes = detectedBoxes.size(); // num of detection
                int left, top, right, bottom;         // score0 : ball , score1 : box
                cv::Rect2d roi;

                /* convert torch::Tensor to cv::Rect2d */
                std::vector<cv::Rect2d> bboxesYolo;
                bboxesYolo.reserve(20);
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
            /* some TM trackers exist */
            if (!classIndexesTM.empty())
            {
                /* if class label is equal to 0, return -1 if existed label == 0 or -1 and return the same label if classIndex is otherwise*/
                if (candidateIndex == 0)
                {
                    int counterCandidate = 0;
                    int counterIteration = 0;
                    for (const int classIndex : classIndexesTM)
                    {
                        /* if same label -> failed to track in YOLO  */
                        if (classIndex == candidateIndex)
                        {
                            existedClass.push_back(-1);
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate); // erase existed roi to maintain roi order
                        }
                        /* else classIndex != candidateIndex */
                        else if (classIndex != -1)
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
                /* if candidateIndex is other than 0 */
                else
                {
                    int counterCandidate = 0;
                    int counterIteration = 0;
                    for (const int classIndex : classIndexesTM)
                    {
                        /* if same label -> failed to track in YOLO  */
                        if (classIndex == candidateIndex)
                        {
                            existedClass.at(counterIteration) = -1;
                            bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidate);
                        }
                        else if (candidateIndex < classIndex)
                        {
                            counterCandidate++;
                        }
                        counterIteration++;
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
                            std::cout << "TM and Yolo Tracker matched!" << std::endl;
                            existedRoi.push_back(bboxTemp);
                            // delete candidate bbox
                            bboxesYolo.erase(bboxesYolo.begin() + indexMatch); // erase detected bbox from bboxes Yolo -> number of bboxesYolo decrease
                        }
                        /* not found matched tracker -> return classIndex -1 to updatedClassIndexes */
                        else
                        {
                            std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                            existedClass.push_back(-1);
                        }
                        bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                    }
                    /* other labels -> push back same label to maintain order only when candidateIndex=0 */
                    else if (candidateIndex < classIndex && candidateIndex == 0)
                    {
                        existedClass.push_back(classIndex);
                        counterCandidateTM++; // for maintain order of existed roi
                    }
                    /* only valid if classIndex != 0 */
                    else if (candidateIndex < classIndex && classIndex != 0)
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
                /* if same label -> failed to track in YOLO  */
                if (classIndex == candidateIndex && candidateIndex == 0)
                {
                    existedClass.push_back(-1);
                    bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                }
                else if (classIndex == candidateIndex && candidateIndex != 0)
                {
                    existedClass.at(counterIteration) = -1;                                  // update label as -1
                    bboxesCandidate.erase(bboxesCandidate.begin() + counterCandidateTM); // delete TM latest roi to maintain roi order
                }
                /* else classIndex != candidateIndex */
                else if (classIndex != -1 && candidateIndex == 0)
                {
                    existedClass.push_back(classIndex);
                    counterCandidateTM++; // maintain existedROI order
                }
                /* else classIndex != candidateIndex */
                else if (classIndex == -1 && candidateIndex == 0)
                {
                    existedClass.push_back(-1);
                }
            }
            counterIteration++;
        }
    }

    void push2QueueLeft(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /*
         * push detection data to queueuLeft
         */
        std::vector<cv::Rect2d> updatedRoi;
        updatedRoi.reserve(20);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(20);
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(100);
        /* update data */
        updateData(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTemplates, updatedClassIndexes);
        /* detection is successful */
        if (!updatedRoi.empty())
        {
            // std::cout << "detection succeeded" << std::endl;
            // save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);

            // push detected data
            std::unique_lock<std::mutex> lock(mtxYoloLeft);
            queueYoloBboxLeft.push(updatedRoi);
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                std::unique_lock<std::mutex> lock(mtxYoloLeft);
                queueYoloClassIndexLeft.push(updatedClassIndexes);
            }
            /* no class Indexes -> nothing to do */
            else
            {
                /* go trough */
            }
        }
    }

    void push2QueueRight(std::vector<cv::Rect2d>& existedRoi, std::vector<cv::Rect2d>& newRoi,
        std::vector<int>& existedClass, std::vector<int>& newClass, cv::Mat1b& frame,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /*
         * push detection data to queueuRight
         */

         /* both objects detected */
        std::vector<cv::Rect2d> updatedRoi;
        updatedRoi.reserve(20);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(20);
        std::vector<int> updatedClassIndexes;
        updatedClassIndexes.reserve(100);
        updateData(existedRoi, newRoi, existedClass, newClass, frame, updatedRoi, updatedTemplates, updatedClassIndexes);
        /* detection is successful */
        if (!updatedRoi.empty())
        {
            // std::cout << "detection succeeded" << std::endl;
            // save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);

            // push detected data
            std::unique_lock<std::mutex> lock(mtxYoloRight);
            queueYoloBboxRight.push(updatedRoi);
            queueYoloTemplateRight.push(updatedTemplates);
            queueYoloClassIndexRight.push(updatedClassIndexes);
        }
        /* no object detected -> return class label -1 if TM tracker exists */
        else
        {
            if (!updatedClassIndexes.empty())
            {
                std::unique_lock<std::mutex> lock(mtxYoloRight);
                queueYoloClassIndexRight.push(updatedClassIndexes);
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
        std::cout << "updateData function" << std::endl;
        /* firstly add existed class and ROi*/
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
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        else
        {
            if (!existedClass.empty())
            {
                for (const int& classIndex : existedClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    std::cout << classIndex << " ";
                }
                std::cout << std::endl;
            }
        }
        /* secondly add new roi and class */
        if (!newRoi.empty())
        {
            for (const cv::Rect2d& roi : newRoi)
            {
                updatedRoi.push_back(roi);
                updatedTemplates.push_back(frame(roi));
            }
            for (const int& classIndex : newClass)
            {
                updatedClassIndexes.push_back(classIndex);
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
        else
        {
            if (newClass.empty())
            {
                for (const int& classIndex : newClass)
                {
                    updatedClassIndexes.push_back(classIndex);
                    std::cout << classIndex << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    void drawRectangle(cv::Mat1b frame, std::vector<cv::Rect2d>& ROI, int index)
    {
        if (ROI.size() != 0)
        {
            if (index == 0) // ball
            {
                for (int k = 0; k < ROI.size(); k++)
                {
                    cv::rectangle(frame, cv::Point((int)ROI[k].x, (int)ROI[k].y), cv::Point((int)(ROI[k].x + ROI[k].width), (int)(ROI[k].y + ROI[k].height)), cv::Scalar(255, 0, 0), 3);
                }
            }
            if (index == 1) // box
            {
                for (int k = 0; k < ROI.size(); k++)
                {
                    cv::rectangle(frame, cv::Point((int)ROI[k].x, (int)ROI[k].y), cv::Point((int)(ROI[k].x + ROI[k].width), (int)(ROI[k].y + ROI[k].height)), cv::Scalar(0, 255, 255), 3);
                }
            }
        }
    }

    void getYoloDataLeft(std::vector<cv::Rect2d>& bboxes, std::vector<int>& classes)
    {
        std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
        // std::cout << "Left Img : Yolo bbox available from TM " << std::endl;
        if (!queueYoloBboxLeft.empty())
        {
            bboxes = queueYoloBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            queueYoloBboxLeft.pop();            // remove yolo bbox
            /* for debug */
            std::cout << ":: Left :: latest yolo data " << std::endl;
            for (const cv::Rect2d& bbox : bboxes)
            {
                std::cout << "BBOX ::" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << std::endl;
            }
        }
        if (!queueYoloClassIndexLeft.empty())
        {
            classes = queueYoloClassIndexLeft.front(); // get current tracking status
            queueYoloClassIndexLeft.pop();
            std::cout << "::class label :: ";
            for (const int& classIndex : classes)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
        }
    }

    void getYoloDataRight(std::vector<cv::Rect2d>& bboxes, std::vector<int>& classes)
    {
        std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
        std::cout << ":: Right Img :: Yolo bbox available from TM " << std::endl;
        if (!queueYoloBboxRight.empty())
        {
            bboxes = queueYoloBboxRight.front(); // get new yolodata : {{x,y.width,height},...}
            queueYoloBboxRight.pop();            // remove yolo bbox
            /* for debug */
            std::cout << "Left :: latest yolo data " << std::endl;
            for (const cv::Rect2d& bbox : bboxes)
            {
                std::cout << "BBOX :" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << std::endl;
            }
        }
        if (!queueYoloClassIndexRight.empty())
        {
            classes = queueYoloClassIndexRight.front(); // get current tracking status
            queueYoloClassIndexRight.pop();
            std::cout << ":: class label :: ";
            for (const int& classIndex : classes)
            {
                std::cout << classIndex << " ";
            }
            std::cout << std::endl;
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
    YOLODetect yolodetectorLeft, yolodetectorRight;
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
    if (!queueYoloBboxRight.empty())
    {
        std::cout << "queueYoloBboxRight isn't empty" << std::endl;
        while (!queueYoloBboxRight.empty())
        {
            queueYoloBboxRight.pop();
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
    if (!queueYoloTemplateRight.empty())
    {
        std::cout << "queueYoloTemplateRight isn't empty" << std::endl;
        while (!queueYoloTemplateRight.empty())
        {
            queueYoloTemplateRight.pop();
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
    if (!queueYoloClassIndexRight.empty())
    {
        std::cout << "queueClassIndexRight isn't empty" << std::endl;
        while (!queueYoloClassIndexRight.empty())
        {
            queueYoloClassIndexRight.pop();
        }
    }

    // vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    posSaverYoloLeft.reserve(300);
    std::vector<std::vector<int>> classSaverYoloLeft;
    classSaverYoloLeft.reserve(300);
    std::vector<std::vector<cv::Rect2d>> posSaverYoloRight;
    posSaverYoloRight.reserve(300);
    std::vector<std::vector<int>> classSaverYoloRight;
    classSaverYoloRight.reserve(300);
    std::array<cv::Mat1b, 2> imgs;
    int frameIndex;
    int countIteration = 1;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    while (true)
    {
        if (queueFrame.empty())
        {
            /* waiting */
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::cout << "waiting for frame pushing " << std::endl;
            continue;
        }
        auto start = std::chrono::high_resolution_clock::now();
        bool boolImgs = getImagesFromQueueYolo(imgs, frameIndex);
        if (!boolImgs)
        {
            if (counterFinish > 10)
            {
                break;
            }
            // No more frames in the queue, exit the loop
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "wait for image pushed" << std::endl;
            counterFinish++;
        }
        std::cout << " YOLO -- " << countIteration << " -- " << std::endl;

        /*start yolo detection */
        yolodetectorLeft.detectLeft(imgs[LEFT_CAMERA], frameIndex, posSaverYoloLeft, classSaverYoloLeft);
        yolodetectorRight.detectRight(imgs[RIGHT_CAMERA], frameIndex, posSaverYoloRight, classSaverYoloRight);
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
    checkStorage(posSaverYoloLeft);
    std::cout << " : Right : " << std::endl;
    checkStorage(posSaverYoloRight);
    std::cout << "Class saver : Yolo : " << std::endl;
    std::cout << " : Left : " << std::endl;
    checkClassStorage(classSaverYoloLeft);
    std::cout << " : Right : " << std::endl;
    checkClassStorage(classSaverYoloRight);
}

bool getImagesFromQueueYolo(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    if (!queueFrame.empty())
    {
        imgs = queueFrame.front();
        frameIndex = queueFrameIndex.front();
        // remove frame from queue
        queueFrame.pop();
        queueFrameIndex.pop();
        return true;
    }
    return false;
}

void checkStorage(std::vector<std::vector<cv::Rect2d>>& posSaverYolo)
{
    int count = 1;
    std::cout << "posSaverYolo :: Contensts ::" << std::endl;
    for (int i = 0; i < posSaverYolo.size(); i++)
    {
        std::cout << (i + 1) << "-th iteration : " << std::endl;
        for (int j = 0; j < posSaverYolo[i].size(); j++)
        {
            std::cout << (j + 1) << "::" << posSaverYolo[i][j].x << "," << posSaverYolo[i][j].y << "," << posSaverYolo[i][j].width << "," << posSaverYolo[i][j].height << std::endl;
        }
    }
}

void checkClassStorage(std::vector<std::vector<int>>& classSaverYolo)
{
    int count = 1;
    std::cout << "Class saver :: Contensts ::" << std::endl;
    for (int i = 0; i < classSaverYolo.size(); i++)
    {
        std::cout << (i + 1) << "-th iteration : " << std::endl;
        for (int j = 0; j < classSaverYolo[i].size(); j++)
        {
            std::cout << classSaverYolo[i][j] << " ";
        }
        std::cout << std::endl;
    }
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
    std::vector<std::vector<cv::Rect2d>> posSaverTMRight;
    posSaverTMRight.reserve(2000);
    std::vector<std::vector<int>> classSaverTMRight;
    classSaverTMRight.reserve(2000);

    int countIteration = 0;
    int counterFinish = 0;
    /* sleep for 2 seconds */
    // std::this_thread::sleep_for(std::chrono::seconds(30));
    //  for each img iterations
    int startCounter = 0; // after yolo detection has succeeded second times -> start template matching
    if (queueYoloBboxLeft.empty() && queueYoloBboxRight.empty())
    {
        std::cout << "- Template Matching idling -" << std::endl;
        /* get rid of imgs until yolo inference started */
        while (queueYoloBboxLeft.empty() && queueYoloBboxRight.empty())
        {
            if (!queueYoloBboxLeft.empty() || !queueYoloBboxRight.empty())
            {
                /* YOLO detection has succeede! */
                startCounter++;
                if (startCounter >= 3)
                {
                    break;
                }
            }
            // get img from queue
            if (!queueFrame.empty())
            {
                // std::cout << "get images from queue" << std::endl;
                std::array<cv::Mat1b, 2> imgs;
                int frameIndex;
                getImagesFromQueueTM(imgs, frameIndex);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000 / FPS)));
            }
            // std::cout << "wait for YOLO inference " << std::endl;
        }
        std::cout << "start template matching" << std::endl;
        while (true) // continue until finish
        {
            countIteration++;
            std::cout << " -- " << countIteration << " -- " << std::endl;
            // get img from queue
            std::array<cv::Mat1b, 2> imgs;
            int frameIndex;
            bool boolImgs = getImagesFromQueueTM(imgs, frameIndex);
            std::cout << "get imgs" << std::endl;
            if (!boolImgs)
            {
                if (counterFinish > 10)
                {
                    break;
                }
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
            }
            else
            {
                counterFinish = 0; // reset
            }

            std::vector<cv::Mat1b> templateImgsLeft, templateImgsRight;
            templateImgsLeft.reserve(10);
            templateImgsRight.reserve(10);
            bool boolLeft = false;
            bool boolRight = false;
            /*start template matching process */
            auto start = std::chrono::high_resolution_clock::now();
            std::thread threadLeftTM(templateMatchingForLeft, std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(templateImgsLeft), std::ref(posSaverTMLeft), std::ref(classSaverTMLeft));
            std::thread threadRightTM(templateMatchingForRight, std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(templateImgsRight), std::ref(posSaverTMRight), std::ref(classSaverTMRight));
            threadLeftTM.join();
            threadRightTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by template matching: " << duration.count() << " milliseconds" << std::endl;
        }
        // check data
        std::cout << "position saver : TM : " << std::endl;
        std::cout << " : Left : " << std::endl;
        checkStorage(posSaverTMLeft);
        std::cout << " : Right : " << std::endl;
        checkStorage(posSaverTMRight);
        std::cout << "Class saver : TM : " << std::endl;
        std::cout << " : Left : " << std::endl;
        checkClassStorage(classSaverTMLeft);
        std::cout << " : Right : " << std::endl;
        checkClassStorage(classSaverTMRight);
    }
}

bool getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    // std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    if (queueFrame.empty() || queueFrameIndex.empty())
    {
        return false;
    }
    imgs = queueFrame.front();
    frameIndex = queueFrameIndex.front();
    queueTargetFrameIndex.push(frameIndex);
    // remove frame from queue
    queueFrame.pop();
    queueFrameIndex.pop();
    return true;
}

/*  Template Matching Function  */

/* Template Matching :: Left */
void templateMatchingForLeft(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
{
    // for updating templates
    std::vector<cv::Rect2d> updatedBboxes;
    updatedBboxes.reserve(20);
    std::vector<cv::Mat1b> updatedTemplates;
    updatedTemplates.reserve(20);
    std::vector<int> updatedClasses;
    updatedClasses.reserve(100);

    // get Template Matching data
    std::vector<int> classIndexTMLeft;
    classIndexTMLeft.reserve(100);
    int numTrackersTM = 0;
    std::vector<cv::Rect2d> bboxesTM;
    bboxesTM.reserve(20);
    std::vector<cv::Mat1b> templatesTM;
    templatesTM.reserve(20);
    std::vector<bool> boolScalesTM;
    boolScalesTM.reserve(20);   // search area scale
    bool boolTrackerTM = false; // whether tracking is successful

    /* Template Matching bbox available */
    getTemplateMatchingDataLeft(boolTrackerTM, classIndexTMLeft, bboxesTM, templatesTM, boolScalesTM, numTrackersTM);

    // template from yolo is available
    bool boolTrackerYolo = false;
    /* Yolo Template is availble */
    if (!queueYoloTemplateLeft.empty())
    {
        boolTrackerYolo = true;
        if (!boolScalesTM.empty())
        {
            boolScalesTM.clear(); // clear all elements of scales
        }
        // get Yolo data
        std::vector<cv::Mat1b> templatesYoloLeft;
        templatesYoloLeft.reserve(20); // get new data
        std::vector<cv::Rect2d> bboxesYoloLeft;
        bboxesYoloLeft.reserve(20); // get current frame data
        std::vector<int> classIndexesYoloLeft;
        classIndexesYoloLeft.reserve(100);
        getYoloDataLeft(templatesYoloLeft, bboxesYoloLeft, classIndexesYoloLeft); // get new frame
        // combine Yolo and TM data, and update latest data
        combineYoloTMData(classIndexesYoloLeft, classIndexTMLeft, templatesYoloLeft, templatesTM, bboxesYoloLeft, bboxesTM,
            updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
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
    std::cout << "BoolTrackeTM :" << boolTrackerTM << ", boolTrackerYolo : " << boolTrackerYolo << std::endl;
    /*  Template Matching Process */
    if (boolTrackerTM || boolTrackerYolo)
    {
        std::cout << "template matching process has started" << std::endl;
        int counterTracker = 0;
        // for storing TM detection results
        std::vector<cv::Mat1b> updatedTemplatesTM;
        updatedTemplatesTM.reserve(20);
        std::vector<cv::Rect2d> updatedBboxesTM;
        updatedBboxesTM.reserve(20);
        std::vector<int> updatedClassesTM;
        updatedClassesTM.reserve(100);
        std::vector<bool> updatedSearchScales;
        updatedSearchScales.reserve(20); // if scale is Yolo or TM
        // template matching
        std::cout << "processTM start" << std::endl;
        processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);
        std::cout << "processTM finish" << std::endl;

        if (!updatedBboxesTM.empty())
        {
            queueTMBboxLeft.push(updatedBboxesTM); // push roi
            posSaver.push_back(updatedBboxes);     // save current position to the vector
            std::unique_lock<std::mutex> lock(mtxTarget);
            queueTargetBboxesLeft.push(updatedBboxesTM); // push roi to target
        }
        if (!updatedTemplatesTM.empty())
        {
            queueTMTemplateLeft.push(updatedTemplatesTM); // push template image
        }
        if (!updatedClassesTM.empty())
        {
            queueTMClassIndexLeft.push(updatedClassesTM);
            queueTargetClassIndexesLeft.push(updatedClassesTM);
            classSaver.push_back(updatedClassesTM); // save current class to the saver
        }
        if (!updatedSearchScales.empty())
        {
            queueTMScalesLeft.push(updatedSearchScales);
        }
        push2YoloDataLeft(updatedBboxesTM, updatedClassesTM); // push latest data to queueYoloBboes and queueYoloClassIndexes
        /* if yolo data is avilable -> send signal to target predict to change labels*/
        if (boolTrackerYolo)
        {
            queueLabelUpdateLeft.push(true);
        }
    }
    else // no template or bbox -> nothing to do
    {
        if (!classIndexTMLeft.empty())
        {
            std::vector<cv::Rect2d> temp{};
            queueTMClassIndexLeft.push(classIndexTMLeft);
            queueTargetClassIndexesLeft.push(classIndexTMLeft);
            classSaver.push_back(classIndexTMLeft); // save current class to the saver
            std::unique_lock<std::mutex> lock(mtxTarget);
            queueTargetClassIndexesLeft.push(classIndexTMLeft);
            push2YoloDataRight(temp, classIndexTMLeft);
        }

        // nothing to do
    }
}

/*
 * Template Matching :: Right camera
 */
void templateMatchingForRight(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
{
    // for updating templates
    std::vector<cv::Rect2d> updatedBboxes;
    updatedBboxes.reserve(20);
    std::vector<cv::Mat1b> updatedTemplates;
    updatedTemplates.reserve(20);
    std::vector<int> updatedClasses;
    updatedClasses.reserve(100);

    // get Template Matching data
    std::vector<int> classIndexTMRight;
    classIndexTMRight.reserve(100);
    int numTrackersTM = 0;
    std::vector<cv::Rect2d> bboxesTM;
    bboxesTM.reserve(20);
    std::vector<cv::Mat1b> templatesTM;
    templatesTM.reserve(20);
    std::vector<bool> boolScalesTM;
    boolScalesTM.reserve(20);   // search area scale
    bool boolTrackerTM = false; // whether tracking is successful
    getTemplateMatchingDataRight(boolTrackerTM, classIndexTMRight, bboxesTM, templatesTM, boolScalesTM, numTrackersTM);

    /* template from yolo is available */
    bool boolTrackerYolo = false;
    if (!queueYoloTemplateRight.empty())
    {
        boolTrackerYolo = true;
        // get Yolo data
        std::vector<cv::Mat1b> templatesYoloRight;
        templatesYoloRight.reserve(20); // get new data
        std::vector<cv::Rect2d> bboxesYoloRight;
        bboxesYoloRight.reserve(20); // get current frame data
        std::vector<int> classIndexesYoloRight;
        classIndexesYoloRight.reserve(100);
        getYoloDataRight(templatesYoloRight, bboxesYoloRight, classIndexesYoloRight); // get new frame
        if (!boolScalesTM.empty())
        {
            boolScalesTM.clear(); // clear all elements of scales
        }
        // combine Yolo and TM data, and update latest data
        combineYoloTMData(classIndexesYoloRight, classIndexTMRight, templatesYoloRight, templatesTM, bboxesYoloRight, bboxesTM,
            updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
    }
    /* template from yolo isn't available but TM tracker exist */
    else if (boolTrackerTM)
    {
        updatedTemplates = templatesTM;
        updatedBboxes = bboxesTM;
        updatedClasses = classIndexTMRight;
    }
    /* no template is available */
    else
    {
        // nothing to do
    }
    std::cout << "BoolTrackeTM :" << boolTrackerTM << ", boolTrackerYolo : " << boolTrackerYolo << std::endl;
    /*  start Template Matching Process */
    if (boolTrackerTM || boolTrackerYolo)
    {
        // for storing TM detection results
        std::vector<cv::Mat1b> updatedTemplatesTM;
        updatedTemplatesTM.reserve(20);
        std::vector<cv::Rect2d> updatedBboxesTM;
        updatedBboxesTM.reserve(20);
        std::vector<int> updatedClassesTM;
        updatedClassesTM.reserve(100);
        std::vector<bool> updatedSearchScales;
        updatedSearchScales.reserve(20); // if scale is Yolo or TM
        /* template mamtching */
        std::cout << "processTM start" << std::endl;
        processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);
        std::cout << "processTM finish" << std::endl;
        std::unique_lock<std::mutex> lock(mtxTarget);
        if (!updatedBboxesTM.empty())
        {
            queueTMBboxRight.push(updatedBboxesTM);       // push roi
            queueTargetBboxesRight.push(updatedBboxesTM); // push roi to target
            posSaver.push_back(updatedBboxes);            // save current position to the vector
        }
        if (!updatedTemplatesTM.empty())
        {
            queueTMTemplateRight.push(updatedTemplatesTM); // push template image
        }
        if (!updatedClassesTM.empty())
        {
            queueTMClassIndexRight.push(updatedClassesTM);
            queueTargetClassIndexesRight.push(updatedClassesTM);
            classSaver.push_back(updatedClassesTM); // save current class to the saver
        }
        if (!updatedSearchScales.empty())
        {
            queueTMScalesRight.push(updatedSearchScales);
        }
        push2YoloDataRight(updatedBboxesTM, updatedClassesTM); // push latest data to queueYoloBboes and queueYoloClassIndexes
        /* if yolo data is avilable -> send signal to target predict to change labels*/
        if (boolTrackerYolo)
        {
            queueLabelUpdateRight.push(true);
        }
    }
    else // no template or bbox -> nothing to do
    {
        if (!classIndexTMRight.empty())
        {
            std::vector<cv::Rect2d> temp{};
            queueTMClassIndexRight.push(classIndexTMRight);
            queueTargetClassIndexesRight.push(classIndexTMRight);
            classSaver.push_back(classIndexTMRight); // save current class to the saver
            std::unique_lock<std::mutex> lock(mtxTarget);
            queueTargetClassIndexesRight.push(classIndexTMRight);
            push2YoloDataRight(temp, classIndexTMRight); // push latest data to queueYoloBboes and queueYoloClassIndexes
        }
        // nothing to do
    }
}

void getTemplateMatchingDataLeft(bool& boolTrackerTM, std::vector<int>& classIndexTMLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, int& numTrackersTM)
{
    if (!queueTMClassIndexLeft.empty())
    {
        classIndexTMLeft = queueTMClassIndexLeft.front();
        numTrackersTM = classIndexTMLeft.size();
        queueTMClassIndexLeft.pop();
    }
    if (!queueTMBboxLeft.empty() && !queueTMTemplateLeft.empty() && !queueTMScalesLeft.empty())
    {
        std::cout << "previous TM tracker is available" << std::endl;
        boolTrackerTM = true;

        bboxesTM = queueTMBboxLeft.front();
        templatesTM = queueTMTemplateLeft.front();
        boolScalesTM = queueTMScalesLeft.front();

        queueTMBboxLeft.pop();
        queueTMTemplateLeft.pop();
        queueTMScalesLeft.pop();
    }
    else
    {
        boolTrackerTM = false;
    }
}

void getTemplateMatchingDataRight(bool& boolTrackerTM, std::vector<int>& classIndexTMRight, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, int& numTrackersTM)
{
    if (!queueTMClassIndexRight.empty())
    {
        classIndexTMRight = queueTMClassIndexRight.front();
        numTrackersTM = classIndexTMRight.size();
        queueTMClassIndexRight.pop();
    }
    if (!queueTMBboxRight.empty() && !queueTMTemplateRight.empty() && queueTMScalesRight.empty())
    {
        boolTrackerTM = true;

        bboxesTM = queueTMBboxRight.front();
        templatesTM = queueTMTemplateRight.front();
        boolScalesTM = queueTMScalesRight.front();

        queueTMBboxRight.pop();
        queueTMTemplateRight.pop();
        queueTMScalesRight.pop();
    }
    else
    {
        boolTrackerTM = false;
    }
}

void getYoloDataLeft(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
{
    std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
    newTemplates = queueYoloTemplateLeft.front();
    newBboxes = queueYoloBboxLeft.front();
    newClassIndexes = queueYoloClassIndexLeft.front();
    queueYoloTemplateLeft.pop();
    // queueYoloBboxLeft.pop();
    // queueYoloClassIndexLeft.pop();
}

void getYoloDataRight(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
{
    std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
    newTemplates = queueYoloTemplateRight.front();
    newBboxes = queueYoloBboxRight.front();
    newClassIndexes = queueYoloClassIndexRight.front();
    queueYoloTemplateRight.pop();
    // queueYoloBboxRight.pop();
    // queueYoloClassIndexRight.pop();
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
    for (const int classIndex : classIndexesYoloLeft)
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
    for (const int classIndex : updatedClasses)
    {
        std::cout << classIndex << " ";
    }
    std::cout << std::endl;
    for (const int classIndexTM : updatedClasses)
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
                leftRoi = (int)(matchLoc.x + leftSearch);
                topRoi = (int)(matchLoc.y + topSearch);
                rightRoi = (int)(leftRoi + templateImg.cols);
                bottomRoi = (int)(topRoi + templateImg.rows);
                cv::Rect2d roiTemplate(matchLoc.x, matchLoc.y, templateImg.cols, templateImg.rows);
                // update template
                newTemplate = img(roiTemplate);
                // update roi
                roi.x = leftRoi;
                roi.y = topRoi;
                roi.width = rightRoi - leftRoi;
                roi.height = bottomRoi - topRoi;

                // update information
                updatedBboxesTM.push_back(roi);
                updatedTemplatesTM.push_back(newTemplate);
                updatedClassesTM.push_back(classIndexTM);
                updatedSearchScales.push_back(true);
                counterTracker++;
            }
            /* doesn't meet matching criteria */
            else
            {
                updatedClassesTM.push_back(-1); // tracking fault
                counterTracker++;
            }
        }
        /* template doesn't exist -> don't do template matching */
        else
        {
            updatedClassesTM.push_back(classIndexTM);
        }
    }
}

void push2YoloDataLeft(std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM)
{
    std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
    if (!queueYoloBboxLeft.empty())
    {
        while (!queueYoloTemplateLeft.empty())
        {
            queueYoloBboxLeft.pop();
        }
    }
    if (!queueYoloClassIndexLeft.empty())
    {
        while (!queueYoloClassIndexLeft.empty())
        {
            queueYoloClassIndexLeft.pop();
        }
    }
    if (!updatedBboxesTM.empty())
    {
        queueYoloBboxLeft.push(updatedBboxesTM);
    }
    if (!updatedClassesTM.empty())
    {
        queueYoloClassIndexLeft.push(updatedClassesTM);
    }
}

void push2YoloDataRight(std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM)
{
    std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
    if (!queueYoloBboxRight.empty())
    {
        while (!queueYoloTemplateRight.empty())
        {
            queueYoloBboxRight.pop();
        }
    }
    if (!queueYoloClassIndexRight.empty())
    {
        while (!queueYoloClassIndexRight.empty())
        {
            queueYoloClassIndexRight.pop();
        }
    }
    if (!updatedBboxesTM.empty())
    {
        queueYoloBboxRight.push(updatedBboxesTM);
    }
    if (!updatedClassesTM.empty())
    {
        queueYoloClassIndexRight.push(updatedClassesTM);
    }
}

/* 3d positioning and trajectory prediction */

void targetPredict()
{
    /*
     *   First option to match objects in 2 imgs ::
     *   get latest data from dataLeft and dataRight : ok
     *  sort latest data in x value and also sort class labels : ok
     *  check each imgs number of objects
     *
     *  in each class labels :
     *  if number is same -> matche respectively
     *  else -> choose std::min(numLeft,numRight) objects: in left img -> from (numLeft - minVal) to numLeft, in right img -> from 0 to numLeft-minVal
     *  calculate 3d position
     *  save calculated 3d positions to data3D :[timesteps,numObjects,{frameIndex,centerX,centerY}]
     *  if data3D.size() >=3 : calculate trajectory and get target positions
     *
     *  Second option to match objects in 2 imgs :: This is chosen
     *   predict trajectory in 2d -> get coeeficients for linea regression and curve fitting
     *   calculate metrics = (coefficients RMSE in CurveFitting) + ( coefficients RMSE in LinearRegression )
     *   get min_ij(RMSE) -> get i-th object in left img and j-th object in right img
     *  convert {Cam} to {RobotBase}
     *  calculate Robot TCP to target objects
     */
    std::vector<std::vector<std::vector<int>>> dataLeft, dataRight; // [num Objects, time_steps, each time position ] : {frameIndex,centerX,centerY}
    std::vector<std::vector<int>> classesLeft, classesRight;        //[timsteps,num Objects]
    std::vector<std::vector<std::vector<int>>> targets3DHistory;    // for storing target postions : [number of target predictions ,
    /* while loop from here */
    if (queueTMBboxLeft.empty() || queueTMBboxRight.empty())
    {
        std::cout << " Target Prediction idling" << std::endl;
        /* get rid of imgs until yolo inference started */
        while (queueLabelUpdateLeft.empty() && queueLabelUpdateRight.empty())
        {
            if (!queueLabelUpdateLeft.empty() || !queueLabelUpdateRight.empty())
            {
                break;
            }
            std::cout << "target prediction wait for TM data" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    else
    {
        int counterFinish = 0; // counter before finish
        while (true)           // continue until finish
        {
            if (queueTargetClassIndexesLeft.empty() || queueTargetClassIndexesRight.empty())
            {
                if (counterFinish > 10)
                {
                    break;
                }
                counterFinish++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            /* get latest class labels -> exclude -1 data */
            std::vector<int> classesLatestLeft, classesLatestRight;
            // std::vector<std::vector<int>> dataLatestLeft, dataLatestRight;
            getLatestClass(classesLeft, classesLatestLeft);
            getLatestClass(classesRight, classesLatestRight);
            /* organize tracking data */
            updateTrackingData(dataLeft, dataRight, classesLeft, classesRight);
            /* get 3D position -> matching data from left to right */
            if (!dataLeft.empty() || !dataRight.empty())
            {
                /* for each data predict trajectory in 2-dimension and get coefficients
                /* data is store both in Left and Right */
                std::vector<std::vector<float>> coefficientsXLeft, coefficientsYLeft, coefficientsXRight, coefficientsYRight; // storage for coefficients in fitting in X and Y ::same size with number of objects detected : [numObjects,codfficients]
                trajectoryPredict2D(dataLeft, coefficientsXLeft, coefficientsYLeft, classesLatestLeft);
                trajectoryPredict2D(dataRight, coefficientsXRight, coefficientsYRight, classesLatestRight);
                /* match -> calculate metrics and matches */
                std::vector<std::vector<std::vector<std::vector<int>>>> datasFor3D; // [num matched Objects, 2 (=left + right), time_steps, each time position ] : {frameIndex,centerX,centerY}
                /* trajectory prediction has been done in both 2 imgs -> start object matching */
                /* at least there is one prediction data -> matching is possible */
                dataMatching(coefficientsXLeft, coefficientsXRight, coefficientsYLeft, coefficientsYRight, classesLatestLeft, classesLatestRight, dataLeft, dataRight, datasFor3D);
                /* calculate 3d position based on matching process--> predict target position */
                if (!datasFor3D.empty())
                {
                    std::vector<std::vector<int>> targets3D;
                    predict3DTargets(datasFor3D, targets3D);
                    targets3DHistory.push_back(targets3D);
                }
                /* convert target position from cameraLeft to robotBase Coordinates */

                /* push data to queueTargetPositions */
            }
        }
        /* check storage */
        std::cout << "targets predictions ::" << std::endl;
        for (int i = 0; i < targets3DHistory.size(); i++)
        {
            std::cout << i << "-th iteration" << std::endl;
            for (int j = 0; j < targets3DHistory[i].size(); j++)
            {
                std::cout << j << "-th objects target prediction ::" << std::endl;
                std::cout << "targetFrame: " << targets3DHistory[i][j][0] << ", xTarget : " << targets3DHistory[i][j][1] << ", xyarget : " << targets3DHistory[i][j][2] << ", dapthTarget : " << targets3DHistory[i][j][3] << std::endl;
            }
        }
    }
}

/* synchronize tracking data with template matching data */
/* labels will increase or keep
 *  get update information from queueTMLabelUpdate -> update label
 *  and when label becomes -1, delete all data of the label
 */
void updateTrackingData(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight, std::vector<std::vector<int>>& classesLeft, std::vector<std::vector<int>>& classesRight)
{
    int frameIndex;
    std::vector<int> classesCurrentLeft, classesCurrentRight; // latest classes
    std::vector<cv::Rect2d> bboxesLeft, bboxesRight;          // latest datas
    bool boolUpdateLabelLeft = false;
    bool boolUpdateLabelRight = false;
    /* get tracking data fromt TM */
    bool ret = getTargetData(frameIndex, classesCurrentLeft, classesCurrentRight,
        bboxesLeft, bboxesRight, boolUpdateLabelLeft, boolUpdateLabelRight);
    /* 2 imgs data available */
    if (ret)
    {
        std::vector<std::vector<int>> dataCurrentLeft, dataCurrentRight;
        /* Left data */
        if (!bboxesLeft.empty())
        {
            convertRoi2Center(classesCurrentLeft, bboxesLeft, frameIndex, dataCurrentLeft); // dataCurrentLeft is latest data : [frameIndex,xCenter,yCenter]
        }
        /* Right Data */
        if (bboxesRight.empty())
        {
            convertRoi2Center(classesCurrentRight, bboxesRight, frameIndex, dataCurrentRight); // dataCurrentRight is latest data : [frameIndex,xCenter,yCenter]
        }
        /* compare data with past data */
        std::vector<int> updatedClassesLeft, updatedClassesRight; // updated classes
        /* past Left data exist */
        /* if get updateLabel signal, change labels */
        if (!dataLeft.empty())
        {
            /* compare class indexes and match which object is the same */
            /* labels should be synchronized with TM labels */
            adjustData(classesLeft, dataLeft, dataCurrentLeft, classesCurrentLeft, updatedClassesLeft, boolUpdateLabelLeft); // classesLeft and dataLeft is storage, dataCurrentLeft is latest data, and updatedClassesLeft is new class list
            if (boolUpdateLabelLeft)
            {
                classesLeft.push_back(updatedClassesLeft);
            }
        }
        /* Left data doesn't exist */
        else
        {
            /* add new data */
            dataLeft = { dataCurrentLeft };
            classesLeft = { classesCurrentLeft };
        }
        /* past Right data exist */
        if (!dataRight.empty())
        {
            /* compare class indexesand match which object is the same */
            /* labels should be synchronized with TM labels */
            adjustData(classesRight, dataRight, dataCurrentRight, classesCurrentRight, updatedClassesRight, boolUpdateLabelRight);
            if (boolUpdateLabelRight)
            {
                classesRight.push_back(updatedClassesRight);
            }
        }
        /* Right data doesn't exist */
        else
        {
            /* add new data */
            dataRight = { dataCurrentRight };
            classesRight = { classesCurrentRight };
        }
    }
    /* neither data is available */
    else
    {
        /* Nothing to do */
    }
}

bool getTargetData(int& frameIndex, std::vector<int>& classesLeft, std::vector<int>& classesRight, std::vector<cv::Rect2d>& bboxesLeft,
    std::vector<cv::Rect2d>& bboxesRight, bool& boolUpdateLabelLeft, bool& boolUpdateLabelRight)
{
    std::unique_lock<std::mutex> lock(mtxTarget); // Lock the mutex
    /* both data is available*/
    bool ret = false;
    if (!queueTargetFrameIndex.empty())
    {
        frameIndex = queueTargetFrameIndex.front();
        queueTargetFrameIndex.pop();
    }
    if (!queueTargetBboxesLeft.empty())
    {
        classesLeft = queueTargetClassIndexesLeft.front();
        bboxesLeft = queueTargetBboxesLeft.front();
        queueTargetClassIndexesLeft.pop();
        queueTargetBboxesLeft.pop();
        ret = true;
    }
    if (!queueTargetBboxesRight.empty())
    {
        classesRight = queueTargetClassIndexesRight.front();
        bboxesRight = queueTargetBboxesRight.front();
        queueTargetClassIndexesRight.pop();
        queueTargetBboxesRight.pop();
        ret = true;
    }
    if (!queueLabelUpdateLeft.empty())
    {
        boolUpdateLabelLeft = true;
    }
    if (!queueLabelUpdateRight.empty())
    {
        boolUpdateLabelRight = true;
    }
    return ret;
}

void convertRoi2Center(std::vector<int>& classes, std::vector<cv::Rect2d>& bboxes, const int& frameIndex, std::vector<std::vector<int>>& tempData)
{
    int counterRoi = 0;
    for (const int classIndex : classes)
    {
        /* bbox is available */
        if (classIndex != -1)
        {
            int centerX = (int)(bboxes[counterRoi].x + bboxes[counterRoi].width / 2);
            int centerY = (int)(bboxes[counterRoi].y + bboxes[counterRoi].height / 2);
            tempData.push_back({ frameIndex, centerX, centerY });
            counterRoi++;
        }
        /* bbox isn't available */
        else
        {
            /* nothing to do */
        }
    }
}

void adjustData(std::vector<std::vector<int>>& classes, std::vector<std::vector<std::vector<int>>>& data,
    std::vector<std::vector<int>>& dataCurrent, std::vector<int>& classesCurrent,
    std::vector<int>& updatedClasses, bool& boolUpdateLabel)
{
    int counterPastClass = 0;                      // for counting past data
    int counterPastTracker = 0;                    // for counting past class
    int counterCurrentTracker = 0;                 // for counting current data
    std::vector<int> classesPast = classes.back(); // get last data from storage so classesLeft or classesRight : [time_steps, classIndexes]
    int numPastClass = classesPast.size();         // number of classes
    /* update label and add new label data and delete data if lost */
    if (boolUpdateLabel)
    {
        /* check every classes */
        for (const int classIndexCurrent : classesCurrent)
        {
            /* within already classes */
            if (counterPastClass<numPastClass)
            {
                /* new position data found-> add sequence data */
                if (classIndexCurrent != -1)
                {
                    /* match data -> add new data to existed data, data here */
                    if (classIndexCurrent == classesPast[counterPastClass])
                    {
                        data[counterPastTracker].push_back(dataCurrent[counterCurrentTracker]); // add new data in the back of existed data
                        updatedClasses.push_back(classIndexCurrent);                            // update class list
                        counterPastClass++;
                        counterPastTracker++;
                        counterCurrentTracker++;
                    }
                    /*other label -> this is awkward */
                    else
                    {
                        std::cout << "class label is different this is awkawrd. check data exchange of Yolo and Template matching" << std::endl;
                    }
                }
                /* lost data */
                else
                {
                    /* tracker lost-> delete existed data */
                    if (classesPast[counterPastClass] != -1)
                    {
                        data.erase(data.begin() + counterPastTracker); // erase data
                        updatedClasses.push_back(-1);
                        counterPastClass++;
                    }
                    /* already lost */
                    else
                    {
                        updatedClasses.push_back(-1);
                        counterPastClass++;
                    }
                }
            }
            /* new tracker */
            else
            {
                /* new tracker */
                if (classIndexCurrent != -1)
                {
                    data.push_back({ dataCurrent[counterCurrentTracker] });
                    updatedClasses.push_back(classIndexCurrent); // add new class
                    counterCurrentTracker++;
                }
                /* if new tracker class is -1, this is awkward */
                else
                {
                    std::cout << "althou new tracker, class label is -1. check code and modify" << std::endl;
                    updatedClasses.push_back(-1);
                }
            }
        }
    }
    /* template matching data update -> only add data if tracker is successful */
    else
    {
        /* check every classes */
        for (const int classIndexCurrent : classesCurrent)
        {
            /* new position data found-> add sequence data */
            if (classIndexCurrent != -1)
            {
                /* match data -> add new data to existed data, data here */
                if (classIndexCurrent == classesPast[counterPastClass])
                {
                    data[counterPastTracker].push_back(dataCurrent[counterCurrentTracker]); // add new data in the back of existed data
                    counterPastTracker++;
                    counterCurrentTracker++;
                    counterPastClass++;
                }
                /* label is different. but this is awkward because if label is -1 template matching didn't start */
                else
                {
                    std::cout << "class label :: current class " << classIndexCurrent << " and past class : " << classesPast[counterPastClass] << ". check previous code. this is exception" << std::endl;
                    counterPastClass++;
                    counterPastTracker++;
                }
            }
            /* tracking was failed */
            else
            {
                /* tracking was failed, but until yolo detection don't delete data */
                if (classesPast[counterPastClass] != -1)
                {
                    counterPastClass++;
                    counterPastTracker++;
                }
                /* tracking was already failed -> only add counterPastClass. there is no sequence data */
                else
                {
                    counterPastClass++;
                }
            }
        }
    }
}

void getLatestData(std::vector<std::vector<std::vector<int>>>& data, std::vector<std::vector<int>>& dataLatest)
{
    for (int i = 0; i < data.size(); i++)
    {
        dataLatest.push_back(data[i].back());
    }
}

void getLatestClass(std::vector<std::vector<int>>& classes, std::vector<int>& classesLatest)
{
    /* get latest classes but exclude class -1 */
    classesLatest = classes.back();
    /* delete -1 index */
    classesLatest.erase(std::remove(classesLatest.begin(), classesLatest.end(), -1), classesLatest.end());
}
bool sortByCenterX(const std::vector<int>& a, const std::vector<int>& b)
{
    /*
     * Sort in ascending order based on centerX
     */
    return a[1] < b[1];
}

void sortData(std::vector<std::vector<int>>& dataLatest, std::vector<int>& classesLatest)
{
    // Create an index array to remember the original order
    std::vector<size_t> index(dataLatest.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        index[i] = i;
    }
    // Sort data1 based on centerX values and apply the same order to data2
    std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
        { return dataLatest[a][1] < dataLatest[b][1]; });

    std::vector<std::vector<int>> sortedDataLatest(dataLatest.size());
    std::vector<int> sortedClassesLatest(classesLatest.size());

    for (size_t i = 0; i < index.size(); ++i)
    {
        sortedDataLatest[i] = dataLatest[index[i]];
        sortedClassesLatest[i] = classesLatest[index[i]];
    }

    dataLatest = sortedDataLatest;
    classesLatest = sortedClassesLatest;
}

void linearRegression(const std::vector<std::vector<int>>& data, std::vector<float>& result_x)
{
    /*
     * linear regression
     * y = ax + b
     * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
     * b = mean_y - a*mean_x
     * Args:
     *   data(std::vector<std::vector<int>>&) : {{time,x,y,z},...}
     *   result_x(std::vector<float>&) : vector for saving result x
     *   result_x(std::vector<float>&) : vector for saving result z
     */
    const int NUM_POINTS_FOR_REGRESSION = 3;
    float sumt = 0, sumx = 0, sumtx = 0, sumtt = 0; // for calculating coefficients
    float mean_t, mean_x;
    int length = data.size(); // length of data

    for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
    {
        sumt += data[length - i][0];
        sumx += data[length - i][1];
        sumtx += data[length - i][0] * data[length - i][1];
        sumtt += data[length - i][0] * data[length - i][0];
    }
    std::cout << "Linear regression" << std::endl;
    mean_t = static_cast<float>(sumt) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
    mean_x = static_cast<float>(sumx) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
    float slope_x, intercept_x;
    if (std::abs(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t) > 0.0001)
    {
        slope_x = (sumtx - NUM_POINTS_FOR_REGRESSION * mean_t * mean_x) / (sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
        intercept_x = mean_x - slope_x * mean_t;
    }
    else
    {
        slope_x = 0;
        intercept_x = 0;
    }
    result_x = { slope_x, intercept_x };
    std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n"
        << std::endl;
}

void linearRegressionZ(const std::vector<std::vector<int>>& data, std::vector<float>& result_z)
{
    /*
     * linear regression
     * y = ax + b
     * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
     * b = mean_y - a*mean_x
     * Args:
     *   data(std::vector<std::vector<int>>&) : {{time,x,y,z},...}
     *   result_x(std::vector<float>&) : vector for saving result x
     *   result_x(std::vector<float>&) : vector for saving result z
     */
    const int NUM_POINTS_FOR_REGRESSION = 3;
    float sumt = 0, sumz = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
    float mean_t, mean_z;
    int length = data.size(); // length of data

    for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
    {
        sumt += data[length - i][0];
        sumz += data[length - i][3];
        sumtt += data[length - i][0] * data[length - i][0];
        sumtz += data[length - i][0] * data[length - i][3];
    }
    std::cout << "Linear regression" << std::endl;
    mean_t = static_cast<float>(sumt) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
    mean_z = static_cast<float>(sumz) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
    float slope_z, intercept_z;
    if (std::abs(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t) > 0.0001)
    {
        slope_z = (sumtz - NUM_POINTS_FOR_REGRESSION * mean_t * mean_z) / (sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
        intercept_z = mean_z - slope_z * mean_t;
    }
    else
    {
        slope_z = 0;
        intercept_z = 0;
    }
    result_z = { slope_z, intercept_z };
    std::cout << "\n\nZ :: The best fit value of curve is : z = " << slope_z << " t + " << intercept_z << ".\n\n"
        << std::endl;
}

void curveFitting(const std::vector<std::vector<int>>& data, std::vector<float>& result)
{
    /*
     * curve fitting with parabora
     * y = c*x^2+d*x+e
     *
     * Args:
     *   data(std::vector<std::vector<int>>) : {{time,x,y,z},...}
     *   result(std::vector<float>&) : vector for saving result
     */

     // argments analysis
    int length = data.size(); // length of data

    float time1, time2, time3;
    time1 = static_cast<float>(data[length - 3][0]);
    time2 = static_cast<float>(data[length - 2][0]);
    time3 = static_cast<float>(data[length - 1][0]);
    float det = time1 * time2 * (time1 - time2) + time1 * time3 * (time3 - time1) + time2 * time3 * (time2 - time3); // det
    float c, d, e;
    if (det == 0)
    {
        c = 0;
        d = 0;
        e = 0;
    }
    else
    {
        float coef11 = (time2 - time3) / det;
        float coef12 = (time3 - time1) / det;
        float coef13 = (time1 - time2) / det;
        float coef21 = (time3 * time3 - time2 * time2) / det;
        float coef22 = (time1 * time1 - time3 * time3) / det;
        float coef23 = (time2 * time2 - time1 * time1) / det;
        float coef31 = time2 * time3 * (time2 - time3) / det;
        float coef32 = time1 * time3 * (time3 - time1) / det;
        float coef33 = time1 * time2 * (time1 - time2) / det;
        // coefficients of parabola
        c = coef11 * data[length - 3][2] + coef12 * data[length - 2][2] + coef13 * data[length - 1][2];
        d = coef21 * data[length - 3][2] + coef22 * data[length - 2][2] + coef23 * data[length - 1][2];
        e = coef31 * data[length - 3][2] + coef32 * data[length - 2][2] + coef33 * data[length - 1][2];
    }

    result = { c, d, e };

    std::cout << "y = " << c << "x^2 + " << d << "x + " << e << std::endl;
}

void trajectoryPredict2D(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<float>>& coefficientsX, std::vector<std::vector<float>>& coefficientsY, std::vector<int>& classesLatest)
{
    int counterData = 0;
    for (const std::vector<std::vector<int>>& data : dataLeft)
    {
        /* get 3 and more time-step datas -> can predict trajectory */
        if (data.size() >= 3)
        {
            /* get latest 3 time-step data */
            std::vector<std::vector<int>> tempData;
            std::vector<float> coefX, coefY;
            // Use reverse iterators to access the last three elements
            auto rbegin = data.rbegin(); // Iterator to the last element
            auto rend = data.rend();     // Iterator one past the end
            /* Here data is latest to old -> but not have bad effect to trajectory prediction */
            for (auto it = rbegin; it != rend && std::distance(rend, it) < 3; ++it)
            {
                const std::vector<int>& element = *it;
                tempData.push_back(element);
            }
            /* trajectory prediction in X and Y */
            linearRegression(tempData, coefX);
            curveFitting(tempData, coefY);
            coefficientsX.push_back(coefX);
            coefficientsY.push_back(coefY);
        }
        /* get less than 3 data -> can't predict trajectory -> x : have to make the size equal to classesLatest
         *  -> add specific value to coefficientsX and coefficientsY, not change the size of classesLatest for maintaining size consisitency between dataLeft and data Right
         */
        else
        {
            coefficientsX.push_back({ 0.0, 0.0 });
            coefficientsY.push_back({ 0.0, 0.0, 0.0 });
            // classesLatest.erase(classesLatest.begin() + counterData); //erase class
            // counterData++;
            /* can't predict trajectory */
        }
    }
}

float calculateME(std::vector<float>& coefXLeft, std::vector<float>& coefYLeft, std::vector<float>& coefXRight, std::vector<float>& coefYRight)
{
    float me = 0.0; // mean error
    for (int i = 0; i < coefYLeft.size(); i++)
    {
        me = me + (coefYLeft[i] - coefYRight[i]);
    }
    me = me + coefXLeft[0] - coefXRight[0];
    return me;
}

void dataMatching(std::vector<std::vector<float>>& coefficientsXLeft, std::vector<std::vector<float>>& coefficientsXRight,
    std::vector<std::vector<float>>& coefficientsYLeft, std::vector<std::vector<float>>& coefficientsYRight,
    std::vector<int>& classesLatestLeft, std::vector<int>& classesLatestRight,
    std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight,
    std::vector<std::vector<std::vector<std::vector<int>>>>& dataFor3D)
{
    float minVal = 20;
    int minIndexRight;
    if (!coefficientsXLeft.empty() && !coefficientsXRight.empty())
    {
        /* calculate metrics based on left img data */
        for (int i = 0; i < coefficientsXLeft.size(); i++)
        {
            /* deal with moving objects -> at least one coefficient should be more than 0 */
            if (coefficientsYLeft[i][0] != 0)
            {
                for (int j = 0; j < coefficientsXRight.size(); j++)
                {
                    /* deal with moving objects -> at least one coefficient should be more than 0 */
                    if (coefficientsYRight[i][0] != 0)
                    {
                        /* if class label is same */
                        if (classesLatestLeft[i] == classesLatestRight[j])
                        {
                            /* calculate metrics */
                            float me = calculateME(coefficientsXLeft[i], coefficientsYLeft[i], coefficientsXRight[j], coefficientsXRight[j]);
                            /* minimum value is updated */
                            if (me < minVal)
                            {
                                minVal = me;
                                minIndexRight = j; // most reliable matching index in Right img
                            }
                        }
                        /* maybe fixed objects detected */
                        else
                        {
                            /* ignore */
                        }
                    }
                }
                /* matcing object found */
                if (minVal < 20)
                {
                    dataFor3D.push_back({ dataLeft[i], dataRight[minIndexRight] }); // match objects and push_back to dataFor3D
                }
            }
            /* maybe fixed objects detected */
            else
            {
                /* ignore */
            }
        }
    }
}

void predict3DTargets(std::vector<std::vector<std::vector<std::vector<int>>>>& datasFor3D, std::vector<std::vector<int>>& targets3D)
{
    int indexL, indexR, xLeft, xRight, yLeft, yRight;
    float fX = cameraMatrix.at<double>(0, 0);
    float fY = cameraMatrix.at<double>(1, 1);
    float fSkew = cameraMatrix.at<double>(0, 1);
    float oX = cameraMatrix.at<double>(0, 2);
    float oY = cameraMatrix.at<double>(1, 2);
    /* iteration of calculating 3d position for each matched objects */
    for (std::vector<std::vector<std::vector<int>>>& dataFor3D : datasFor3D)
    {
        std::vector<std::vector<int>> dataL = dataFor3D[0];
        std::vector<std::vector<int>> dataR = dataFor3D[1];
        /* get 3 and more time-step datas -> calculate 3D position */
        int numDataL = dataL.size();
        int numDataR = dataR.size();
        std::vector<std::vector<int>> data3D; //[mm]
        // calculate 3D position
        int counter = 0; // counter for counting matching frame index
        int counterIteration = 0;
        bool boolPredict = false; // if 3 datas are available
        while (counterIteration < std::min(numDataL, numDataR))
        {
            counterIteration++;
            if (counter > 3)
            {
                boolPredict = true;
                break;
            }
            indexL = dataL[numDataL - counter][0];
            indexR = dataR[numDataR - counter][0];
            if (indexL == indexR)
            {
                xLeft = dataL[numDataL - counter][1];
                xRight = dataR[numDataR - counter][1];
                yLeft = dataL[numDataL - counter][2];
                yRight = dataR[numDataR - counter][2];
                int disparity = (int)(xLeft - xRight);
                int X = (int)(BASELINE / disparity) * (xLeft - oX - (fSkew / fY) * (yLeft - oY));
                int Y = (int)(BASELINE * (fX / fY) * (yLeft - oY) / disparity);
                int Z = (int)(fX * BASELINE / disparity);
                data3D.push_back({ indexL, X, Y, Z });
                counter++;
            }
        }
        if (boolPredict)
        {
            /* trajectoryPrediction */
            std::vector<float> coefX, coefY, coefZ;
            linearRegression(data3D, coefX);
            linearRegressionZ(data3D, coefZ);
            curveFitting(data3D, coefY);
            /* objects move */
            if (coefZ[0] < 0) // moving forward to camera
            {
                int frameTarget = (int)((TARGET_DEPTH - coefZ[1]) / coefZ[0]);
                int xTarget = (int)(coefX[0] * frameTarget + coefX[1]);
                int yTarget = (int)(coefY[0] * frameTarget * frameTarget + coefY[1] * frameTarget + coefY[2]);
                targets3D.push_back({ frameTarget, xTarget, yTarget, TARGET_DEPTH }); // push_back target position
                std::cout << "target is : ( frameTarget :  " << frameTarget << ", xTarget : " << xTarget << ", yTarget : " << yTarget << ", depthTarget : " << TARGET_DEPTH << std::endl;
            }
        }
    }
}

/* read imgs */
void pushFrame(std::array<cv::Mat1b, 2>& srcs, const int frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    // std::cout << "push imgs" << std::endl;
    // cv::Mat1b undistortedImgL, undistortedImgR;
    // cv::undistort(srcs[0], undistortedImgL, cameraMatrix, distCoeffs);
    // cv::undistort(srcs[1], undistortedImgR, cameraMatrix, distCoeffs);
    // std::array<cv::Mat1b, 2> undistortedImgs = { undistortedImgL,undistortedImgR };
    // queueFrame.push(undistortedImgs);
    queueFrame.push(srcs);
    queueFrameIndex.push(frameIndex);
}

/*
 * main function
 */
int main()
{
    // �J�����̃p�����[�^�ݒ�
    const unsigned int imgWidth = 320;  // �摜�̕�
    const unsigned int imgHeight = 320; // �摜�̍���
    const unsigned int frameRate = FPS; // �t���[�����[�g
    const unsigned int expTime = 1500;  // 792; //�I������
    const int imgGain = 18;
    const bool isBinning = true;
    const int capMode = 0;                        // 0:���m�N���C1:�J���[
    const std::string leaderCamID = "30957651";   // ���[�_�[�J����
    const std::string followerCamID = "30958851"; // �t�H�����J����

#if SYNC_CAMERAS
    std::array<Ximea, 2> cams = { Ximea(imgWidth, imgHeight, frameRate, leaderCamID, expTime, isBinning, false), Ximea(imgWidth, imgHeight, frameRate, followerCamID, expTime, isBinning, true) };
#else
    std::array<Ximea, 2> cams = { Ximea(imgWidth, imgHeight, frameRate, leaderCamID, expTime, isBinning, 0), Ximea(imgWidth, imgHeight, frameRate, followerCamID, expTime, isBinning, 0) };
#endif // SYNC_CAMERAS
    // �Q�C���ݒ�
    cams[0].SetGain(imgGain);
    cams[1].SetGain(imgGain);

    // �摜�\��
    spsc_queue<dispData_t> queDisp;
    dispData_t dispData;
    std::atomic<bool> isSaveImage = false;
    auto dispInfoPtr = std::make_unique<DispInfo>(queDisp, imgWidth, imgHeight, true);
    std::thread dispThread(std::ref(*dispInfoPtr), std::ref(isSaveImage));

    dispData.srcs[0] = cv::Mat1b::zeros(imgHeight, imgWidth);
    dispData.srcs[1] = cv::Mat1b::zeros(imgHeight, imgWidth);

    // �摜�ۑ��ݒ�
    saveImages_t saveImages;
    saveImages.resize(maxSaveImageNum);
    for (int i_i = 0; i_i < saveImages.size(); i_i++)
    {
        for (int i_s = 0; i_s <= 1; i_s++)
        {
            saveImages[i_i][i_s] = cv::Mat1b::zeros(imgHeight, imgWidth);
        }
    }
    int saveCount = 0;
    std::array<cv::Mat1b, 2> srcs = { cv::Mat1b::zeros(imgHeight, imgWidth), cv::Mat1b::zeros(imgHeight, imgWidth) }; // �擾�摜

    // multi thread code
    std::thread threadYolo(yoloDetect);
    // std::cout << "start template matching" << std::endl;
    // std::thread threadTemplateMatching(templateMatching);
    // std::thread threadTargetPredict(targetPredict);

    // ���C������
    for (int fCount = 0;; fCount++)
    {
        for (int i_i = 0; i_i < cams.size(); i_i++)
        {
            srcs[i_i] = cams[i_i].GetNextImageOcvMat(); // �摜�擾
        }
        // get images and frame index
        // std::cout << fCount << std::endl;
        pushFrame(srcs, fCount);

        // ��ʕ\��
        if (fCount % 10 == 0)
        {
            srcs[0].copyTo(dispData.srcs[0]);
            srcs[1].copyTo(dispData.srcs[1]);
            dispData.frameCount = fCount;
            queDisp.push(dispData); // this is where 2images data is stored
        }

        // �摜�ۑ�
        if (isSaveImage && saveCount < saveImages.size())
        {
            srcs[0].copyTo(saveImages[saveCount][0]);
            srcs[1].copyTo(saveImages[saveCount][1]);
            saveCount++;
        }

        // �I��
        if (dispInfoPtr->isTerminate())
        {
            // comDspacePtr->finishProc(); //dSPACE�Ƃ̒ʐM�X���b�h�I��
            break;
        }
    }

    // �摜�o��
    if (saveCount > 0)
    {
        time_t now = time(nullptr);
        const tm* lt = localtime(&now);
        // "�N����-���ԕ��b"�Ƃ����f�B���N�g��������
        const std::string dirName_left = std::to_string(lt->tm_year + 1900) + std::to_string(lt->tm_mon + 1) + std::to_string(lt->tm_mday) + "-" + std::to_string(lt->tm_hour) + std::to_string(lt->tm_min) + std::to_string(lt->tm_sec) + "-left-calibration";
        const std::string dirName_right = std::to_string(lt->tm_year + 1900) + std::to_string(lt->tm_mon + 1) + std::to_string(lt->tm_mday) + "-" + std::to_string(lt->tm_hour) + std::to_string(lt->tm_min) + std::to_string(lt->tm_sec) + "-right-calibration";
        const std::string rootDir = "imgs";
        struct stat st;
        if (stat(rootDir.c_str(), &st) != 0)
        {
            _mkdir(rootDir.c_str());
        }
        const std::string saveDir_left = rootDir + "/" + dirName_left;
        const std::string saveDir_right = rootDir + "/" + dirName_right;
        _mkdir(saveDir_right.c_str());
        _mkdir(saveDir_left.c_str());
        std::cout << "Outputing " << saveCount << "images to " << saveDir_left << std::endl;
        std::cout << "Outputing " << saveCount << "images to " << saveDir_right << std::endl;
        // progressbar bar(saveCount);
        // bar.set_opening_bracket_char("CAM: [");
        for (int i_s = 0; i_s < saveCount; i_s++)
        {
            cv::Mat1b concatImage;
            cv::Mat1b followImg;
            // cv::flip(saveImages[i_s][1], followImg, -1);
            // cv::hconcat(saveImages[i_s][0], followImg, concatImage);
            cv::imwrite(saveDir_left + "/" + std::to_string(i_s) + ".png", saveImages[i_s][0]);
            cv::imwrite(saveDir_right + "/" + std::to_string(i_s) + ".png", saveImages[i_s][1]);
            // bar.update();
        }
    }
    std::cout << "Finish main proc" << std::endl;
    dispThread.join();
    threadYolo.join();
    // threadTemplateMatching.join();
    // threadTargetPredict.join();
    // sendThread.join();
    // recvThread.join();
    return 0;
}
