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
cv::Mat createROI(cv::Point2d& center, const cv::Mat& src, const cv::Rect& roiBase) {
	return src((roiBase + cv::Point(center)) & cv::Rect(cv::Point(0, 0), src.size()));
}

std::mutex mtxImg,mtxYoloLeft,mtxYoloRight,mtxTarget; // define mutex

//camera : constant setting
const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
//template matching constant value setting
int match_method; //match method in template matching
int max_Trackbar = 5; //track bar
const char* image_window = "Source Imgage";
const char* result_window = "Result window";

const float scaleXTM = 1.2; //search area scale compared to roi
const float scaleYTM = 1.2;
const float scaleXYolo = 2.0;
const float scaleYYolo = 2.0;
const float matchingThreshold = 0.5; //matching threshold
/* 3d positioning by stereo camera */
const int BASELINE = 280; // distance between 2 cameras
//std::vector<std::vector<float>> cameraMatrix{ {179,0,160},{0,179,160},{0,0,1} }; //camera matrix from camera calibration 

/* revise here based on camera calibration */

const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    179, 0, 160,  // fx: focal length in x, cx: principal point x
    0, 179, 160,  // fy: focal length in y, cy: principal point y
    0, 0, 1     // 1: scaling factor
    );
cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 1,1,1,1,1);

/* revise here based on camera calibration */

const int TARGET_DEPTH = 400; //catching point is 40 cm away from camera position 
/*matching method of template Matching
* const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM CCOEFF \n 5: TM CCOEFF NORMED"; 2 is not so good
*/

const int matchingMethod = cv::TM_SQDIFF_NORMED; //TM_SQDIFF_NORMED is good for small template

//queue definition
std::queue<std::array<cv::Mat1b, 2>> queueFrame;                     // queue for frame
std::queue<int> queueFrameIndex;                            // queue for frame index

//left cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;                 // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft; // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;     // queue for templateMatching bbox
std::queue<std::vector<int>> queueYoloClassIndexLeft; //queue for class index
std::queue<std::vector<int>> queueTMClassIndexLeft; //queue for class index
std::queue<std::vector<bool>> queueTMScalesLeft; //queue for search area scale

//right cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;                 // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight; // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;     // queue for TM bbox
std::queue<std::vector<int>> queueYoloClassIndexRight; //queue for class index
std::queue<std::vector<int>> queueTMClassIndexRight; //queue for class index
std::queue<std::vector<bool>> queueTMScalesRight; //queue for search area scale

//3D positioning ~ trajectory prediction
std::queue<int> queueTargetFrameIndex; //TM estimation frame
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft; //bboxes from template matching for predict objects' trajectory
std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; //bboxes from template matching for predict objects' trajectory
std::queue<std::vector<int>> queueTargetClassIndexesLeft; //class from template matching for maintain consistency
std::queue<std::vector<int>> queueTargetClassIndexesRight; //class from template matching for maintain consistency
 
//declare function
/* Yolo */
void checkYoloTemplateQueueLeft();
void checkYoloTemplateQueueRight();
void checkYoloBboxQueueLeft();
void checkYoloBboxQueueRight();
void checkYoloClassIndexLeft();
void checkYoloClassIndexRight();
/* utility */
void checkStorage(std::vector<std::vector<cv::Rect2d>>&);
void checkClassStorage(std::vector<std::vector<int>>&);
/* get img */
void getImagesFromQueueYolo(std::array<cv::Mat1b, 2>&, int&); //get image from queue
void getImagesFromQueueTM(std::array<cv::Mat1b, 2>&, int&);
/* template matching */
void templateMatching(); //void* //proto definition
void templateMatchingForLeft(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&,std::vector<std::vector<int>>&);//, int, int, float, const int);
void templateMatchingForRight(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&, std::vector<std::vector<int>>&);// , int, int, float, const int);
void combineYoloTMData(std::vector<int>&, std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, 
                        std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&, const int&);
void processTM(std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<bool>&, std::vector<cv::Rect2d>&, cv::Mat1b&,
                std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&);
/* access yolo data */
void getYoloDataLeft(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,std::vector<int>&);
void getYoloDataRight(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,std::vector<int>&);
void push2YoloDataRight(std::vector<cv::Rect2d>&, std::vector<int>&);
void push2YoloDataLeft(std::vector<cv::Rect2d>&, std::vector<int>&);
/* read img */
void pushFrame(std::array<cv::Mat1b, 2>&, const int);
/* Target Prediction */
void targetPredict();
void updateTrackingData(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&, std::vector<std::vector<int>>&);
bool getTargetData(int&, std::vector<int>&, std::vector<int>&, std::vector<cv::Rect2d>&, std::vector<cv::Rect2d>&);
void convertRoi2Center(std::vector<int>&, std::vector<cv::Rect2d>&, const int&, std::vector<std::vector<int>>&);
void adjustData(std::vector<std::vector<int>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&, std::vector<int>&);
void getLatestData(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&);
void getLatestClass(std::vector<std::vector<int>>&, std::vector<int>&);
bool sortByCenterX(const std::vector<int>&, const std::vector<int>&);
void sortData(std::vector<std::vector<int>>&, std::vector<int>&);
/* trajectory prediction */
void curveFitting(const std::vector<std::vector<int>>&, std::vector<float>&); //curve fitting
void linearRegression(const std::vector<std::vector<int>>&, std::vector<float>&); //linear regression
void linearRegressionZ(const std::vector<std::vector<int>>&, std::vector<float>&);
void trajectoryPredict2D(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<int>&);
float calculateME(std::vector<float>&, std::vector<float>&, std::vector<float>&, std::vector<float>&);
void dataMatching(std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&,
    std::vector<int>&, std::vector<int>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<std::vector<std::vector<int>>>>&);
void predict3DTargets(std::vector<std::vector<std::vector<std::vector<int>>>>&, std::vector<std::vector<int>>&);

/*  YOLO class definition  */
class YOLODetect {
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;
    std::mutex mtx_; // define mutex

    std::string yolofilePath = "best_nano.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloSize = 320;
    const float iouThreshold = 0.45;
    const float confThreshold = 0.3;
    const float iouThresholdIdentity = 0.25; //for maitainig consistency of tracking 

public:
    //constructor for YOLODetect
    YOLODetect() {
        // set device
        if (torch::cuda::is_available()) {
            // device = new torch::Device(devicetype, 0);
            device = new torch::Device(torch::kCUDA);
            std::cout << "set cuda" << std::endl;
        }
        else {
            device = new torch::Device(torch::kCPU);
            std::cout << "set CPU" << std::endl;
        }
        // read param
        mdl = torch::jit::load(yolofilePath, *device);
        mdl.eval();
    };
    ~YOLODetect() { delete device; }; //Deconstructor

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

        //std::cout << imgTensor.sizes() << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMLeft; //for limiting detection area
        std::vector<int> classIndexesTMLeft; //candidate class from TM
        getYoloDataLeft(bboxesCandidateTMLeft, classIndexesTMLeft);//get latest data
        std::cout << "bboxesCandidateTMLeft size : " << bboxesCandidateTMLeft.size() << ", classIndexesTMLeft size : " << classIndexesTMLeft.size() << std::endl;
        /* inference */
        torch::Tensor preds = mdl.forward({ imgTensor }).toTensor(); //preds shape : [1,6,2100]

        /* postProcess */
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,2100) -> (1,2100,6)
        std::cout << "preds size : " << preds.sizes() << std::endl;
        std::vector<torch::Tensor> detectedBoxes0Left, detectedBoxes1Left; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Left, detectedBoxes1Left, confThreshold, iouThreshold);

        std::cout << "BBOX for Ball : " << detectedBoxes0Left.size() << " BBOX for BOX : " << detectedBoxes1Left.size() << std::endl;
        std::vector<cv::Rect2d> roiBallLeft, roiBoxLeft;
        std::vector<int> classBall, classBox;
        /* Roi Setting : take care of dealing with TM data */
        /* ROI and class index management */
        roiSetting(detectedBoxes0Left, roiBallLeft, classBall, 0, bboxesCandidateTMLeft,classIndexesTMLeft);
        roiSetting(detectedBoxes1Left, roiBoxLeft, classBox, 1, bboxesCandidateTMLeft, classIndexesTMLeft);

        /* push and save data */
        push2QueueLeft(roiBallLeft, roiBoxLeft, classBall, classBox, frame, posSaver, classSaver);

        //drawing bbox
        //ball
        //drawRectangle(frame, roiBall, 0);
        //drawRectangle(frame, roiBox, 1);
        //cv::imshow("detection", frame);
        //cv::waitKey(0);
    };

    void detectRight(cv::Mat1b& frame, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver,std::vector<std::vector<int>>& classSaver) {
        /* inference by YOLO
         *  Args:
         *      frame : img
         *      posSaver : storage for saving detected position
         *      queueYoloTemplate : queue for pushing detected img
         *      queueYoloBbox : queue for pushing detected roi, or if available, get candidate position,
         *      queueClassIndex : queue for pushing detected
         */
         //bbox0.resize(0);
         //bbox1.resize(0);
         //clsidx.resize(0);
         //predscore.resize(0);

        /* preprocess img */
        torch::Tensor imgTensor;
        preprocessImg(frame, imgTensor);

        std::cout << imgTensor.sizes() << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMRight; //for limiting detection area
        std::vector<int> classIndexesTMRight; //candidate class from TM
        auto startInference = std::chrono::high_resolution_clock::now();
        getYoloDataRight(bboxesCandidateTMRight, classIndexesTMRight);//get latest data
        auto stopInference = std::chrono::high_resolution_clock::now();
        auto durationInference = std::chrono::duration_cast<std::chrono::milliseconds>(stopInference - startInference);
        std::cout << " Time taken by YOLO inference : " << durationInference.count() << " milliseconds" << std::endl;
        /* inference */
        torch::Tensor preds = mdl.forward({ imgTensor }).toTensor();
        //preds shape : [1,6,2100]

        /* postProcess*/
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,8400) -> (1,8400,6)
        std::vector<torch::Tensor> detectedBoxes0Right, detectedBoxes1Right; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Right, detectedBoxes1Right, confThreshold, iouThreshold);

        std::vector<cv::Rect2d> roiBallRight, roiBoxRight;
        std::vector<int> classBallRight, classBoxRight;
        /* Roi Setting : take care of dealing with TM data */
        /* ROI and class index management */
        auto startRoi = std::chrono::high_resolution_clock::now();
        roiSetting(detectedBoxes0Right, roiBallRight, classBallRight, 0, bboxesCandidateTMRight, classIndexesTMRight);
        roiSetting(detectedBoxes1Right, roiBoxRight, classBoxRight, 1, bboxesCandidateTMRight, classIndexesTMRight);
        auto stopRoi = std::chrono::high_resolution_clock::now();
        auto durationRoi = std::chrono::duration_cast<std::chrono::milliseconds>(stopRoi - startRoi);
        std::cout << " Time taken by YOLO roiSetting : " << durationRoi.count() << " milliseconds" << std::endl;

        /* push and save data */
        auto startQueue = std::chrono::high_resolution_clock::now();
        push2QueueRight(roiBallRight, roiBoxRight, classBallRight, classBoxRight, frame, posSaver, classSaver);
        auto stopQueue = std::chrono::high_resolution_clock::now();
        auto durationQueue = std::chrono::duration_cast<std::chrono::milliseconds>(stopQueue - startQueue);
        std::cout << " Time taken by YOLO pushing : " << durationQueue.count() << " milliseconds" << std::endl;
        
        //drawing bbox
        //ball
        //drawRectangle(frame, roiBall, 0);
        //drawRectangle(frame, roiBox, 1);
        //cv::imshow("detection", frame);
        //cv::waitKey(0);
    };

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        cv::Size yolosize(yoloSize, yoloSize);

        // run
        cv::Mat yoloimg; //define yolo img type
        cv::resize(frame, yoloimg, yolosize);
        cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); //vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 }); //Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat); //convert to float type
        imgTensor = imgTensor.div(255); //normalization
        imgTensor = imgTensor.unsqueeze(0); //expand dims for Convolutional layer (height,width,1)
        imgTensor = imgTensor.to(*device); //transport data to GPU
    }

    void non_max_suppression2(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxes0, std::vector<torch::Tensor>& detectedBoxes1, float confThreshold, float iouThreshold) {
        /* non max suppression : remove overlapped bbox
        * Args:
        *   prediction : (1,2100,,6)
        * Return:
        *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
        */

        //initialization
        detectedBoxes0.resize(0);
        detectedBoxes1.resize(0);

        torch::Tensor xc0 = prediction.select(2, 4) > confThreshold; //get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor xc1 = prediction.select(2, 5) > confThreshold; //get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"

        //torch::nonzero(xc0[i]) returns a index of non-zero -> (n,1) n: number of "true" in xc0[i]
        //.select(1,0): this extracts the first column of the tensor returned by "torch::nonzero" -> extracts indices where xc0[i] is "True"
        //index_select(0,...):select the row along dim=0 where xc0[i] is "true"
        torch::Tensor x0 = prediction.index_select(1, torch::nonzero(xc0[0]).select(1, 0)); //box, x0.shape : (1,n,6) : n: number of candidates
        torch::Tensor x1 = prediction.index_select(1, torch::nonzero(xc1[0]).select(1, 0)); //ball x1.shape : (1,m,6) : m: number of candidates

        x0 = x0.index_select(1, x0.select(2, 4).argsort(1, true).squeeze()); //ball : sorted in descending order
        x1 = x1.index_select(1, x1.select(2, 5).argsort(1, true).squeeze()); //box : sorted in descending order

        x0 = x0.squeeze(); //(1,n,6) -> (n,6) 
        x1 = x1.squeeze(); //(1,m,6) -> (m,6) 

        //std::cout << "x0:" << x0 << std::endl;
        //std::cout << "x1:" << x1 << std::endl;
        //std::cout << "start ball iou calculation" << std::endl;
        //std::cout << "x0.size(0);" << x0.size(0) << std::endl;
        //std::cout << "x1.size(0):" << x1.size(0) << std::endl;

        //ball : non-max suppression
        if (x0.size(0) != 0)
        {
            torch::Tensor bbox0Top = xywh2xyxy(x0[0].slice(0, 0, 4));
            detectedBoxes0.push_back(bbox0Top.cpu());
            //for every candidates
            if (x0.size(0) >= 2)
            {
                //auto start = std::chrono::high_resolution_clock::now();
                nms(x0, detectedBoxes0, iouThreshold); // exclude overlapped bbox : 20 milliseconds
                //auto stop = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                //std::cout << "Time taken by nms: " << duration.count() << " milliseconds" << std::endl;
            }
        }
        //std::cout << "start box iou calculation" << std::endl;
        //std::cout << "iouThreshold" << iouThreshold << std::endl;
        //box
        if (x1.size(0) != 0)
        {
            torch::Tensor bbox1Top = xywh2xyxy(x1[0].slice(0, 0, 4));
            //std::cout << "bbox1Top:" << bbox1Top.sizes() << std::endl;
            detectedBoxes1.push_back(bbox1Top.cpu());
            if (x1.size(0) >= 2)
            {
                //auto start = std::chrono::high_resolution_clock::now();
                nms(x1, detectedBoxes1, iouThreshold); // exclude overlapped bbox : 20 milliseconds
                //auto stop = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                //std::cout << "Time taken by nms: " << duration.count() << " milliseconds" << std::endl;
            }
        }
    }

    torch::Tensor xywh2xyxy(torch::Tensor& x) {
        torch::Tensor y = x.clone();
        y[0] = x[0] - x[2] / 2; //left
        y[1] = x[1] - x[3] / 2; //top
        y[2] = x[0] + x[2] / 2; //right
        y[3] = x[1] + x[3] / 2; //bottom
        return y;
    }

    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float iouThreshold)
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
            box = xywh2xyxy(x[i].slice(0, 0, 4)); //(left,top,width,height) -> (left,top,right,bottom)

            bool addBox = true; // if save bbox as a new detection

            for (const torch::Tensor savedBox : detectedBoxes)
            {
                float iou = calculateIoU(box, savedBox); //calculate IoU

                if (iou > iouThreshold)
                {
                    addBox = false;
                    continue; //next iteration
                }
            }

            if (addBox)
            {
                detectedBoxes.push_back(box.cpu());
            }
        }
    }

    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2) {
        float left = std::max(box1[0].item<float>(), box2[0].item<float>());
        float top = std::max(box1[1].item<float>(), box2[1].item<float>());
        float right = std::min(box1[2].item<float>(), box2[2].item<float>());
        float bottom = std::min(box1[3].item<float>(), box2[3].item<float>());

        if (left < right && top < bottom) {
            float intersection = (right - left) * (bottom - top);
            float area1 = ((box1[2] - box1[0]) * (box1[3] - box1[1])).item<float>();
            float area2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])).item<float>();
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void roiSetting(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& updatedBboxes, std::vector<int>& updatedClassIndexes, int candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM)
    {
        /*
        * Get current data before YOLO inference started.
        * First : Compare YOLO detection and TM detection
        * Second : if match : return new templates in the same order with TM
        * Third : if not match : adapt as a new templates and add after TM data
        * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
        */

        /* detected by Yolo */
        if (detectedBoxes.size() != 0) 
        {
            /* some trackers exist */
            if (classIndexesTM.size() != 0) 
            {
                std::cout << "template matching succeeded" << std::endl;
                /* constant setting */
                int numBboxes = detectedBoxes.size(); //num of detection
                std::vector<cv::Rect2d> bboxesYolo; // for storing cv::Rect2d

                /* start comparison Yolo and TM data */
                comparisonTMYolo(detectedBoxes, classIndexesTM, candidateIndex, bboxesCandidate, numBboxes,bboxesYolo,updatedClassIndexes, updatedBboxes);
                /* finish comparison */

                /* deal with new trackers */
                int numNewDetection = bboxesYolo.size(); //number of new detections
                for (int i = 0; i < numNewDetection; i++)
                {
                    updatedBboxes.push_back(bboxesYolo[i]);
                    updatedClassIndexes.push_back(candidateIndex);
                }
            }
            /* Mo TM tracker exist */
            else 
            {
                std::cout << "No TM tracker exist " << std::endl;
                int numBboxes = detectedBoxes.size(); //num of detection
                int left, top, right, bottom; //score0 : ball , score1 : box
                cv::Rect2d roi;

                /* convert torch::Tensor to cv::Rect2d */
                std::vector<cv::Rect2d> bboxesYolo;
                for (int i = 0; i < numBboxes; ++i)
                {
                    float expandrate[2] = { (float)frameWidth / (float)yoloSize, (float)frameHeight / (float)yoloSize }; // resize bbox to fit original img size
                    //std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
                    left = (int)(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
                    top = (int)(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
                    right = (int)(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
                    bottom = (int)(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
                    updatedBboxes.push_back(cv::Rect2d(left, top, (right - left), (bottom - top)));
                    updatedClassIndexes.push_back(candidateIndex);
                }
            }

        }
        /* No object detected in Yolo -> return -1 class indexes */
        else 
        {
            /* some TM trackers exist */
            if (classIndexesTM.size() != 0) 
            {
                std::cout << "tracker exists, but Yolo detection has failed" << std::endl;
                for (int i = 0; i < classIndexesTM.size(); i++)
                {
                    updatedClassIndexes.push_back(-1); //push_back tracker was not found
                }
            }
            /* No TM tracker */
            else
            {
                std::cout << "No Detection , no tracker" << std::endl;
                /* No detection ,no trackers -> Nothing to do */
            }
        }
    }
    
    void roiSettingRight(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& updatedBboxes, std::vector<int>& updatedClassIndexes, int candidateIndex, std::vector<cv::Rect2d>& bboxesCandidate, std::vector<int>& classIndexesTM)
    {
        /*
        * Get current data before YOLO inference started.
        * First : Compare YOLO detection and TM detection
        * Second : if match : return new templates in the same order with TM
        * Third : if not match : adapt as a new templates and add after TM data
        * Fourth : return all class indexes including -1 (not tracked one) for maintainig data consistency
        */

        /* detected in Yolo */
        if (detectedBoxes.size() != 0)
        {
            /* some trackers exist */
            if (classIndexesTM.size() != 0)
            {
                /* constant setting */
                int numBboxes = detectedBoxes.size(); //num of detection
                std::vector<cv::Rect2d> bboxesYolo;// for storing cv::Rect2d

                /* start comparison Yolo and TM data */
                comparisonTMYolo(detectedBoxes, classIndexesTM, candidateIndex, bboxesCandidate, numBboxes, bboxesYolo, updatedClassIndexes, updatedBboxes);
                /* finish comparison */

                /* deal with new trackers */
                int numNewDetection = bboxesYolo.size(); //number of new detections
                for (int i = 0; i < numNewDetection; i++)
                {
                    updatedBboxes.push_back(bboxesYolo[i]);
                    updatedClassIndexes.push_back(candidateIndex);
                }
            }
            /* Mo TM tracker exist */
            else
            {
                int numBboxes = detectedBoxes.size(); //num of detection
                int left, top, right, bottom; //score0 : ball , score1 : box
                cv::Rect2d roi;

                bool boolCurrentPosition = false; //if current position is available
                //convert torch::Tensor to cv::Rect2d
                std::vector<cv::Rect2d> bboxesYolo;
                for (int i = 0; i < numBboxes; ++i)
                {
                    float expandrate[2] = { (float)frameWidth / (float)yoloSize, (float)frameHeight / (float)yoloSize }; // resize bbox to fit original img size
                    //std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
                    left = (int)(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
                    top = (int)(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
                    right = (int)(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
                    bottom = (int)(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
                    updatedBboxes.push_back(cv::Rect2d(left, top, (right - left), (bottom - top)));
                    updatedClassIndexes.push_back(candidateIndex);
                }
            }

        }
        /* No object detected in Yolo -> return -1 class indexes */
        else
        {
            /* some TM trackers exist */
            if (classIndexesTM.size() != 0)
            {
                for (int i = 0; i < classIndexesTM.size(); i++)
                {
                    updatedClassIndexes.push_back(-1); //push_back tracker was not found
                }
            }
            /* No TM tracker */
            else
            {
                /* No detection ,no trackers -> Nothing to do */
            }
        }
    }
    

    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2) {
        float left = std::max(box1.x, box2.x);
        float top = std::max(box1.y, box2.y);
        float right = std::min((box1.x + box1.width), (box2.x + box2.width));
        float bottom = std::min((box1.y + box1.height), (box2.y + box2.height));

        if (left < right && top < bottom) {
            float intersection = (right - left) * (bottom - top);
            float area1 = box1.width * box1.height;
            float area2 = box2.width * box2.height;
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void comparisonTMYolo(std::vector<torch::Tensor>& detectedBoxes,std::vector<int>& classIndexesTM,int& candidateIndex,std::vector<cv::Rect2d>& bboxesCandidate, 
                            int& numBboxes, std::vector<cv::Rect2d>& bboxesYolo, std::vector<int>& updatedClassIndexes,std::vector<cv::Rect2d>& updatedBboxes)
    {
        /* constant setting */
        int left, top, right, bottom; //score0 : ball , score1 : box
        cv::Rect2d roi; //for updated Roi
        bool boolCurrentPosition = false; //if current position is available

        /* convert torch::Tensor to cv::Rect2d */
        for (int i = 0; i < numBboxes; ++i)
        {
            float expandrate[2] = { (float)frameWidth / (float)yoloSize, (float)frameHeight / (float)yoloSize }; // resize bbox to fit original img size
            //std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
            left = (int)(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
            top = (int)(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
            right = (int)(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
            bottom = (int)(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
            bboxesYolo.push_back(cv::Rect2d(left, top, (right - left), (bottom - top)));
        }

        /*  compare detected bbox and TM bbox  */
        float max = iouThresholdIdentity; //set max value as threshold for lessening process volume
        int counterCandidateTM = 0;
        bool boolIdentity = false; //if found the tracker
        int indexMatch = 0; //index match
        std::vector<cv::Rect2d> newDetections; //for storing newDetection
        cv::Rect2d bboxTemp; //temporal bbox storage

        /* start comparison */
        /* if found same things : push_back detected template and classIndex, else: make sure that push_back only -1 */
        for (const int classIndex : classIndexesTM)
        {
            /* TM tracker exist */
            if (classIndex != -1)
            {
                /* Tracker labels match */
                if (candidateIndex == classIndex)
                {
                    boolIdentity = false;
                    for (int counterCandidateYolo = 0; counterCandidateYolo < numBboxes; counterCandidateYolo)
                    {
                        float iou = calculateIoU_Rect2d(bboxesCandidate[counterCandidateTM], bboxesYolo[counterCandidateYolo]);
                        if (iou >= max) //found similar bbox
                        {
                            max = iou;
                            bboxTemp = bboxesYolo[counterCandidateYolo]; //save top iou candidate
                            indexMatch = counterCandidateYolo;
                            boolIdentity = true;
                        }
                    }
                    /* find matched tracker */
                    if (boolIdentity)
                    {
                        std::cout << "TM and Yolo Tracker matched!" << std::endl;
                        //add data
                        updatedClassIndexes.push_back(candidateIndex);
                        updatedBboxes.push_back(bboxTemp);
                        //delete candidate bbox
                        bboxesYolo.erase(bboxesYolo.begin() + indexMatch);
                    }
                    /* not found matched tracker -> return classIndex -1 to updatedClassIndexes */
                    else
                    {
                        std::cout << "TM and Yolo Tracker didn't match" << std::endl;
                        updatedClassIndexes.push_back(-1);
                    }
                }
                /* other labels : nothing to do */
                else
                {
                    std::cout << "other objects" << std::endl;
                    /* Null */
                }
            }
            /* templateMatching Tracking was fault -> return False */
            else
            {
                std::cout << "template matching tracking was failed" << std::endl;
                updatedClassIndexes.push_back(-1);
            }
        }
    }

    void push2QueueLeft(std::vector<cv::Rect2d>& roiBallLeft, std::vector<cv::Rect2d>& roiBoxLeft, std::vector<int> classBall, std::vector<int>& classBox, cv::Mat1b& frame,std::vector<std::vector<cv::Rect2d>>& posSaver,std::vector<std::vector<int>>& classSaver)
    {
        /*
        * push detection data to queueuLeft
        */

        /* both objects detected */
        if (roiBallLeft.size() != 0 && roiBoxLeft.size() != 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBallLeft.size(); i++)
            {
                updatedRoi.push_back(roiBallLeft[i]);
                updatedTemplates.push_back(frame(roiBallLeft[i]));
            }
            for (int i = 0; i < roiBoxLeft.size(); i++)
            {
                updatedRoi.push_back(roiBoxLeft[i]);
                updatedTemplates.push_back(frame(roiBoxLeft[i]));
                updatedClassIndexes.push_back(1);
            }
            /* update class indexes */
            for (int i = 0; i < classBall.size(); i++)
            {
                updatedClassIndexes.push_back(classBall[i]);
            }
            for (int i = 0; i < classBox.size(); i++)
            {
                updatedClassIndexes.push_back(classBox[i]);
            }

            //save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            //push detected data
            queueYoloBboxLeft.push(updatedRoi);
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
            //std::cout << "Both are detected" << std::endl;
        }
        /*  only ball bbox detected  */
        else if (roiBallLeft.size() != 0 && roiBoxLeft.size() == 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBallLeft.size(); i++)
            {
                updatedRoi.push_back(roiBallLeft[i]);
                updatedTemplates.push_back(frame(roiBallLeft[i]));
            }
            /* update class indexes */
            for (int i = 0; i < classBall.size(); i++)
            {
                updatedClassIndexes.push_back(classBall[i]);
            }
            //save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            //push detected data
            queueYoloBboxLeft.push(updatedRoi);
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
        }
        /*  only box bbox detected  */
        else if (roiBallLeft.size() == 0 && roiBoxLeft.size() != 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBoxLeft.size(); i++)
            {
                updatedRoi.push_back(roiBoxLeft[i]);
                updatedTemplates.push_back(frame(roiBoxLeft[i]));
                updatedClassIndexes.push_back(1);
            }
            /* update class indexes */
            for (int i = 0; i < classBox.size(); i++)
            {
                updatedClassIndexes.push_back(classBox[i]);
            }

            /* save detected data */ 
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            /* push detected data */
            queueYoloBboxLeft.push(updatedRoi);
            queueYoloTemplateLeft.push(updatedTemplates);
            queueYoloClassIndexLeft.push(updatedClassIndexes);
        }
        /* no object detected */
        else
        {
            std::vector<int> updatedClassIndexes;
            if (classBall.size() != 0)
            {
                /* update ball class indexes */
                for (int i = 0; i < classBall.size(); i++)
                {
                    updatedClassIndexes.push_back(classBall[i]);
                }
            }
            if (classBox.size() != 0)
            {
                /* update box class indexes */
                for (int i = 0; i < classBox.size(); i++)
                {
                    updatedClassIndexes.push_back(classBox[i]);
                }
            }
            queueYoloClassIndexLeft.push(updatedClassIndexes);
        }
    }

    void push2QueueRight(std::vector<cv::Rect2d>& roiBall, std::vector<cv::Rect2d>& roiBox, std::vector<int> classBall, std::vector<int>& classBox, cv::Mat1b& frame, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
    {
        /*
        * push detection data to queueuRight
        */

        /* both objects detected */
        if (roiBall.size() != 0 && roiBox.size() != 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBall.size(); i++)
            {
                updatedRoi.push_back(roiBall[i]);
                updatedTemplates.push_back(frame(roiBall[i]));
            }
            for (int i = 0; i < roiBox.size(); i++)
            {
                updatedRoi.push_back(roiBox[i]);
                updatedTemplates.push_back(frame(roiBox[i]));
                updatedClassIndexes.push_back(1);
            }
            /* update class indexes */
            for (int i = 0; i < classBall.size(); i++)
            {
                updatedClassIndexes.push_back(classBall[i]);
            }
            for (int i = 0; i < classBox.size(); i++)
            {
                updatedClassIndexes.push_back(classBox[i]);
            }

            //save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            //push detected data
            queueYoloBboxRight.push(updatedRoi);
            queueYoloTemplateRight.push(updatedTemplates);
            queueYoloClassIndexRight.push(updatedClassIndexes);
            //std::cout << "Both are detected" << std::endl;
        }
        /*  only ball bbox detected  */
        else if (roiBall.size() != 0 && roiBox.size() == 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBall.size(); i++)
            {
                updatedRoi.push_back(roiBall[i]);
                updatedTemplates.push_back(frame(roiBall[i]));
            }
            /* update class indexes */
            for (int i = 0; i < classBall.size(); i++)
            {
                updatedClassIndexes.push_back(classBall[i]);
            }
            //save detected data
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            //push detected data
            queueYoloBboxRight.push(updatedRoi);
            queueYoloTemplateRight.push(updatedTemplates);
            queueYoloClassIndexRight.push(updatedClassIndexes);
        }
        /*  only box bbox detected  */
        else if (roiBall.size() == 0 && roiBox.size() != 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat1b> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            /* update bbox and templates */
            for (int i = 0; i < roiBox.size(); i++)
            {
                updatedRoi.push_back(roiBox[i]);
                updatedTemplates.push_back(frame(roiBox[i]));
                updatedClassIndexes.push_back(1);
            }
            /* update class indexes */
            for (int i = 0; i < classBox.size(); i++)
            {
                updatedClassIndexes.push_back(classBox[i]);
            }

            /* save detected data */
            posSaver.push_back(updatedRoi);
            classSaver.push_back(updatedClassIndexes);
            /* push detected data */
            queueYoloBboxRight.push(updatedRoi);
            queueYoloTemplateRight.push(updatedTemplates);
            queueYoloClassIndexRight.push(updatedClassIndexes);
        }
        /* no object detected */
        else
        {
            std::vector<int> updatedClassIndexes;
            if (classBall.size() != 0)
            {
                /* update ball class indexes */
                for (int i = 0; i < classBall.size(); i++)
                {
                    updatedClassIndexes.push_back(classBall[i]);
                }
            }
            if (classBox.size() != 0)
            {
                /* update box class indexes */
                for (int i = 0; i < classBox.size(); i++)
                {
                    updatedClassIndexes.push_back(classBox[i]);
                }
            }
            queueYoloClassIndexRight.push(updatedClassIndexes);
        }
    }

    void drawRectangle(cv::Mat1b frame, std::vector<cv::Rect2d>& ROI, int index)
    {
        if (ROI.size() != 0)
        {
            if (index == 0) //ball
            {
                for (int k = 0; k < ROI.size(); k++)
                {
                    cv::rectangle(frame, cv::Point((int)ROI[k].x, (int)ROI[k].y), cv::Point((int)(ROI[k].x + ROI[k].width), (int)(ROI[k].y + ROI[k].height)), cv::Scalar(255, 0, 0), 3);
                }

            }
            if (index == 1) //box
            {
                for (int k = 0; k < ROI.size(); k++)
                {
                    cv::rectangle(frame, cv::Point((int)ROI[k].x, (int)ROI[k].y), cv::Point((int)(ROI[k].x + ROI[k].width), (int)(ROI[k].y + ROI[k].height)), cv::Scalar(0, 255, 255), 3);
                }
            }
        }
    }

    void getYoloDataLeft(std::vector<cv::Rect2d>& bboxes,std::vector<int>& classes)
    {
        std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
        if (!queueYoloBboxLeft.empty())
        {
            std::cout << "Left Img : Yolo bbox available from TM " << std::endl;
            bboxes = queueYoloBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            for (int i = 0; i < bboxes.size(); i++)
            {
                std::cout << "BBOX :" << bboxes[i].x<<","<<bboxes[i].y<<","<<bboxes[i].width<<","<<bboxes[i].height << std::endl;
            }

            queueYoloBboxLeft.pop(); //remove yolo bbox
        }
        if (!queueYoloClassIndexLeft.empty())
        {
            std::cout << "Left Img : Yolo Class available from TM" << std::endl;
            classes = queueYoloClassIndexLeft.front(); //get current tracking status
            queueYoloClassIndexLeft.pop();
        }
    }

    void getYoloDataRight(std::vector<cv::Rect2d>& bboxes,std::vector<int>& classes)
    {
        std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
        if (!queueYoloBboxRight.empty())
        {
            std::cout << "Right Img : Yolo bbox available from TM " << std::endl;
            bboxes = queueYoloBboxRight.front(); // get new yolodata : {{x,y.width,height},...}
            queueYoloBboxRight.pop(); //remove yolo bbox
        }
        if (!queueYoloClassIndexRight.empty())
        {
            std::cout << "Right Img : Yolo Class available from TM" << std::endl;
            classes = queueYoloClassIndexLeft.front(); // get current tracking status
            queueYoloClassIndexRight.pop();
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
    YOLODetect yolodetector;

    //vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    std::vector<std::vector<int>> classSaverYoloLeft;
    std::vector<std::vector<cv::Rect2d>> posSaverYoloRight;
    std::vector<std::vector<int>> classSaverYoloRight;
    std::array<cv::Mat1b, 2> imgs;
    int frameIndex;
    int countIteration = 1;
    //for 1img
    /* sleep for 2 seconds */
    std::this_thread::sleep_for(std::chrono::seconds(2));

    while (!queueFrame.empty()) //continue until finish
    {
        std::cout << " -- " << countIteration << " -- " << std::endl;
        // get img from queue
        auto start = std::chrono::high_resolution_clock::now();
        getImagesFromQueueYolo(imgs, frameIndex);
        /* start yolo detection */
        std::cout << "yolo : input img size : " << imgs[LEFT_CAMERA].rows << "," << imgs[LEFT_CAMERA].cols << std::endl;
        yolodetector.detectLeft(imgs[LEFT_CAMERA], frameIndex, posSaverYoloLeft,classSaverYoloLeft);
        yolodetector.detectRight(imgs[RIGHT_CAMERA], frameIndex, posSaverYoloRight,classSaverYoloRight);
        //std::thread threadLeftYolo(&YOLODetect::detectLeft, &yolodetector,std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(posSaverYoloLeft));
        //std::thread threadRightYolo(&YOLODetect::detectRight, &yolodetector,std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(posSaverYoloRight));
        //wait for each thread finishing
        //threadLeftYolo.join();
        //threadRightYolo.join();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO inference : " << duration.count() << " milliseconds" << std::endl;
        countIteration++;
    }

    //check code consistency
    /*
    std::cout << "Check Template Queue" << std::endl;
    std::cout << "Left" << std::endl;
    checkYoloTemplateQueueLeft();
    std::cout << "Right" << std::endl;
    checkYoloTemplateQueueRight();
    std::cout << "Check bbox queue" << std::endl;
    std::cout << "Left" << std::endl;
    checkYoloBboxQueueLeft();
    std::cout << "Right" << std::endl;
    checkYoloBboxQueueRight();
    std::cout << "check yolo class index" << std::endl;
    std::cout << "Left" << std::endl;
    checkYoloClassIndexLeft();
    std::cout << "Right" << std::endl;
    checkYoloClassIndexRight();
    */
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


void getImagesFromQueueYolo(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    imgs = queueFrame.front();
    frameIndex = queueFrameIndex.front();
    // remove frame from queue
    queueFrame.pop();
    queueFrameIndex.pop();
}

void checkYoloTemplateQueueLeft()
{
    std::vector<cv::Mat1b> templateImgs;
    int count = 1;
    while (!queueYoloTemplateLeft.empty())
    {
        std::cout << count << "-th iteration of queueYoloTemplateImgs" << std::endl;
        templateImgs = queueYoloTemplateLeft.front();
        queueYoloTemplateLeft.pop();
        std::cout << "queueYolotemplateImgs detected : " << templateImgs.size() << std::endl;
        for (int i = 0; i < templateImgs.size(); i++)
        {
            std::cout << "template img size : " << templateImgs[i].size() << std::endl;
        }
        count++;
    }
}

void checkYoloTemplateQueueRight()
{
    std::vector<cv::Mat1b> templateImgs;
    int count = 1;
    while (!queueYoloTemplateRight.empty())
    {
        std::cout << count << "-th iteration of queueYoloTemplateImgs" << std::endl;
        templateImgs = queueYoloTemplateRight.front();
        queueYoloTemplateRight.pop();
        std::cout << "queueYolotemplateImgs detected : " << templateImgs.size() << std::endl;
        for (int i = 0; i < templateImgs.size(); i++)
        {
            std::cout << "template img size : " << templateImgs[i].size() << std::endl;
        }
        count++;
    }
}

void checkYoloBboxQueueLeft()
{
    int count = 1;
    std::vector<cv::Rect2d> bboxes;
    while (!queueYoloBboxLeft.empty())
    {
        std::cout << count << "-th iteration of queueYoloBboxes" << std::endl;
        bboxes = queueYoloBboxLeft.front();
        queueYoloBboxLeft.pop();
        for (int i = 0; i < bboxes.size(); i++)
        {
            std::cout << "bbox : " << bboxes[i].x << "," << bboxes[i].y << "," << bboxes[i].width << "," << bboxes[i].height << "," << std::endl;
        }
        count++;
    }
}

void checkYoloBboxQueueRight()
{
    int count = 1;
    std::vector<cv::Rect2d> bboxes;
    while (!queueYoloBboxRight.empty())
    {
        std::cout << count << "-th iteration of queueYoloBboxes" << std::endl;
        bboxes = queueYoloBboxRight.front();
        queueYoloBboxRight.pop();
        for (int i = 0; i < bboxes.size(); i++)
        {
            std::cout << "bbox : " << bboxes[i].x << "," << bboxes[i].y << "," << bboxes[i].width << "," << bboxes[i].height << "," << std::endl;
        }
        count++;
    }
}

void checkYoloClassIndexLeft()
{
    std::vector<int> indexes;
    int count = 1;
    while (!queueYoloClassIndexLeft.empty())
    {
        std::cout << count << "-th iteration of queueClassIndex" << std::endl;
        indexes = queueYoloClassIndexLeft.front();
        queueYoloClassIndexLeft.pop();
        for (int i = 0; i < indexes.size(); i++)
        {
            std::cout << "class : " << indexes << " ";
        }
        std::cout << std::endl;
        count++;
    }
}

void checkYoloClassIndexRight()
{
    std::vector<int> indexes;
    int count = 1;
    while (!queueYoloClassIndexRight.empty())
    {
        std::cout << count << "-th iteration of queueClassIndex" << std::endl;
        indexes = queueYoloClassIndexRight.front();
        queueYoloClassIndexRight.pop();
        for (int i = 0; i < indexes.size(); i++)
        {
            std::cout << "class : " << indexes << " ";
        }
        std::cout << std::endl;
        count++;
    }
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

/*
* template matching thread function definition
*/

void templateMatching() //void*
{
    /* Template Matching
    * return most similar region
    */

    //vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    std::vector<std::vector<int>> classSaverTMLeft;
    std::vector<std::vector<cv::Rect2d>> posSaverTMRight;
    std::vector<std::vector<int>> classSaverTMRight;

    int countIteration = 0;
    /* sleep for 2 seconds */
    //std::this_thread::sleep_for(std::chrono::seconds(30));
    // for each img iterations
    while (!queueFrame.empty()) //continue until finish
    {
        countIteration++;
        std::cout << " -- " << countIteration << " -- " << std::endl;
        // get img from queue
        std::array<cv::Mat1b, 2> imgs;
        int frameIndex;
        //get images from queue
        auto start = std::chrono::high_resolution_clock::now();
        getImagesFromQueueTM(imgs, frameIndex);

        std::cout << "start 2 imgs template matching : " << std::endl;
        //get templateImg from both camera
        if (!queueTMTemplateLeft.empty() && !queueTMTemplateRight.empty())
        {
            /* get images */
            std::vector<cv::Mat1b> templateImgsLeft = queueTMTemplateLeft.front();
            std::vector<cv::Mat1b> templateImgsRight = queueTMTemplateRight.front();
            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTMTemplateLeft.pop();
            queueTMTemplateRight.pop();
            /* start threads */
            std::thread threadLeftTM(templateMatchingForLeft, std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(templateImgsLeft), std::ref(posSaverTMLeft), std::ref(classSaverTMLeft));
            std::thread threadRightTM(templateMatchingForRight, std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(templateImgsRight), std::ref(posSaverTMRight), std::ref(classSaverTMRight));
            /* wait for each thrad finish */
            threadLeftTM.join();
            threadRightTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        }
        /* get templateImg from only left camera */
        else if (!queueTMTemplateLeft.empty() && queueTMTemplateRight.empty())
        {
            /* get images */
            std::vector<cv::Mat1b> templateImgsLeft = queueTMTemplateLeft.front();
            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTMTemplateLeft.pop();
            /* start threads */
            std::thread threadLeftTM(templateMatchingForLeft, std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(templateImgsLeft), std::ref(posSaverTMLeft), std::ref(classSaverTMLeft));
            threadLeftTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        }
        /* get templateImg from only right camera */
        else if (queueTMTemplateLeft.empty() && !queueTMTemplateRight.empty())
        {
            /* get images */
            std::vector<cv::Mat1b> templateImgsRight = queueTMTemplateRight.front();
            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTMTemplateRight.pop();
            /* start threads */
            std::thread threadRightTM(templateMatchingForRight, std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(templateImgsRight), std::ref(posSaverTMRight),std::ref(classSaverTMRight));
            threadRightTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        }
        // can't get template images
        else
        {
            continue;
        }
    }

    //check data
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

void getImagesFromQueueTM(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    imgs = queueFrame.front();
    frameIndex = queueFrameIndex.front();
    queueTargetFrameIndex.push(frameIndex); 
    // remove frame from queue
    queueFrame.pop();
    queueFrameIndex.pop();
}

/*  Template Matching Function  */

/*
*  Template Matching :: Left
*/

void templateMatchingForLeft(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, std::vector<std::vector<cv::Rect2d>>& posSaver,std::vector<std::vector<int>>& classSaver)
{
    //for updating templates
    std::vector<cv::Rect2d> updatedBboxes;
    std::vector<cv::Mat1b> updatedTemplates;
    std::vector<int> updatedClasses;

    //get Template Matching data
    std::vector<int> classIndexTMLeft = queueTMClassIndexLeft.front(); //get class index
    int numTrackersTM = classIndexTMLeft.size();
    std::vector<cv::Rect2d> bboxesTM;
    std::vector<cv::Mat1b> templatesTM;
    std::vector<bool> boolScalesTM; //search area scale
    bool boolTrackerTM = false; //whether tracking is successful
    /* Template Matching bbox available */
    if (!queueTMBboxLeft.empty())
    {
        bboxesTM = queueTMBboxLeft.front();
        templatesTM = queueTMTemplateLeft.front();
        boolScalesTM = queueTMScalesLeft.front();
        queueTMBboxLeft.pop();
        queueTMTemplateLeft.pop();
        queueTMScalesLeft.pop();
        boolTrackerTM = true;
    }

    //template from yolo is available
    bool boolTrackerYolo = false;
    /* Yolo Template is availble */
    if (!queueYoloTemplateLeft.empty())
    {
        boolTrackerYolo = true;
        boolScalesTM.clear(); //clear all elements of scales
        //get Yolo data
        std::vector<cv::Mat1b> templatesYoloLeft; // get new data
        std::vector<cv::Rect2d> bboxesYoloLeft;        // get current frame data
        std::vector<int> classIndexesYoloLeft;
        getYoloDataLeft(templatesYoloLeft, bboxesYoloLeft, classIndexesYoloLeft);//get new frame
        //combine Yolo and TM data, and update latest data
        combineYoloTMData(classIndexesYoloLeft, classIndexTMLeft, templatesYoloLeft, templatesTM, bboxesYoloLeft, bboxesTM,
                            updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
    }
    /* template from yolo isn't available */
    else
    {
        /* TM tracker is available */
        if (boolTrackerTM)
        {
            updatedTemplates = templatesTM;
            updatedBboxes = bboxesTM;
            updatedClasses = classIndexTMLeft;
        }
        /* Tracking is being failed */
        else
        {
            //nothing to do
        }
    }

    /* so far prepare latest template, bbox, class index if available.If all tracker was failed nothing to do from here */

    /*  Template Matching Process */
    if (boolTrackerTM || boolTrackerYolo)
    {
        std::cout << "template matching process has started" << std::endl;
        int counterTracker = 0;
        //for storing TM detection results
        std::vector<cv::Mat1b> updatedTemplatesTM;
        std::vector<cv::Rect2d> updatedBboxesTM;
        std::vector<int> updatedClassesTM;
        std::vector<bool> updatedSearchScales; //if scale is Yolo or TM
        //template matching
        processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);

        /*
        if (counter == 99)
        {
            //draw rectangle
            cv::Mat img_display;
            img.copyTo(img_display);
            cv::rectangle(img_display, cv::Point(leftRoi, topRoi), cv::Point(rightRoi, bottomRoi), cv::Scalar::all(20), 2, 8, 0);
            cv::rectangle(result, cv::Point(leftRoi, topRoi), cv::Point(rightRoi, bottomRoi), cv::Scalar::all(20), 2, 8, 0);
            cv::imshow(image_window, img_display);
            cv::imshow(result_window, result);
        }
        */
        if (updatedBboxesTM.size() != 0)
        {
            queueTMBboxLeft.push(updatedBboxesTM); //push roi 
            queueTargetBboxesLeft.push(updatedBboxesTM); //push roi to target
            posSaver.push_back(updatedBboxes);//save current position to the vector
            push2YoloDataLeft(updatedBboxesTM, updatedClassesTM); //push latest data to queueYoloBboes and queueYoloClassIndexes
        }
        if (updatedTemplatesTM.size() != 0)
        {
            queueTMTemplateLeft.push(updatedTemplatesTM);// push template image
        }
        if (updatedClassesTM.size() != 0)
        {
            queueTMClassIndexLeft.push(updatedClassesTM);
            queueTargetClassIndexesLeft.push(updatedClassesTM);
            classSaver.push_back(updatedClassesTM); //save current class to the saver
        }
        if (updatedSearchScales.size() != 0)
        {
            queueTMScalesLeft.push(updatedSearchScales);
        }
    }
    else //no template or bbox -> nothing to do
    {
        //nothing to do
    }
}

/*
* Template Matching :: Right camera
*/
void templateMatchingForRight(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, std::vector<std::vector<cv::Rect2d>>& posSaver,std::vector<std::vector<int>>& classSaver)
{
    //for updating templates
    std::vector<cv::Rect2d> updatedBboxes;
    std::vector<cv::Mat1b> updatedTemplates;
    std::vector<int> updatedClasses;

    //get Template Matching data
    std::vector<int> classIndexTMRight = queueTMClassIndexRight.front(); //get class index
    std::vector<cv::Rect2d> bboxesTM;
    std::vector<cv::Mat1b> templatesTM;
    std::vector<bool> boolScalesTM; //search area scale
    int numTrackersTM = classIndexTMRight.size();
    bool boolTrackerTM = false; //whether tracking is successful
    if (!queueTMBboxRight.empty())
    {
        bboxesTM = queueTMBboxRight.front();
        templatesTM = queueTMTemplateRight.front();
        boolScalesTM = queueTMScalesRight.front();
        queueTMBboxRight.pop();
        queueTMTemplateRight.pop();
        queueTMScalesRight.pop();
        boolTrackerTM = true;
    }

    /* template from yolo is available */
    bool boolTrackerYolo = false;
    if (!queueYoloTemplateRight.empty())
    {
        boolTrackerYolo = true;
        boolScalesTM.clear(); //clear all elements of scales
        //get Yolo data
        std::vector<cv::Mat1b> templatesYoloRight; // get new data
        std::vector<cv::Rect2d> bboxesYoloRight;        // get current frame data
        std::vector<int> classIndexesYoloRight;
        getYoloDataRight(templatesYoloRight, bboxesYoloRight, classIndexesYoloRight);//get new frame

        //combine Yolo and TM data, and update latest data
        combineYoloTMData(classIndexesYoloRight, classIndexTMRight, templatesYoloRight, templatesTM, bboxesYoloRight, bboxesTM,
                    updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
        
    }
    /* template from yolo isn't available */
    else
    {
        if (boolTrackerTM)
        {
            updatedTemplates = templatesTM;
            updatedBboxes = bboxesTM;
            updatedClasses = classIndexTMRight;
        }
        else
        {
            //nothing to do
        }
    }

    /* so far prepare latest template, bbox, class index if available.If all tracker was failed nothing to do from here */

    /*  Template Matching Process */
    if (boolTrackerTM || boolTrackerYolo)
    {
        //for storing TM detection results
        std::vector<cv::Mat1b> updatedTemplatesTM;
        std::vector<cv::Rect2d> updatedBboxesTM;
        std::vector<int> updatedClassesTM;
        std::vector<bool> updatedSearchScales; //if scale is Yolo or TM
        processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);

        /*
        if (counter == 99)
        {
            //draw rectangle
            cv::Mat img_display;
            img.copyTo(img_display);
            cv::rectangle(img_display, cv::Point(leftRoi, topRoi), cv::Point(rightRoi, bottomRoi), cv::Scalar::all(20), 2, 8, 0);
            cv::rectangle(result, cv::Point(leftRoi, topRoi), cv::Point(rightRoi, bottomRoi), cv::Scalar::all(20), 2, 8, 0);
            cv::imshow(image_window, img_display);
            cv::imshow(result_window, result);
        }
        */
        if (updatedBboxesTM.size() != 0)
        {
            queueTMBboxRight.push(updatedBboxesTM); //push roi 
            queueTargetBboxesRight.push(updatedBboxesTM); //push roi to target
            posSaver.push_back(updatedBboxes);//save current position to the vector
            push2YoloDataRight(updatedBboxesTM, updatedClassesTM); //push latest data to queueYoloBboes and queueYoloClassIndexes
        }
        if (updatedTemplatesTM.size() != 0)
        {
            queueTMTemplateRight.push(updatedTemplatesTM);// push template image
        }
        if (updatedClassesTM.size() != 0)
        {
            queueTMClassIndexRight.push(updatedClassesTM);
            queueTargetClassIndexesRight.push(updatedClassesTM);
            classSaver.push_back(updatedClassesTM); //save current class to the saver
        }
        if (updatedSearchScales.size() != 0)
        {
            queueTMScalesRight.push(updatedSearchScales);
        }
    }
    else //no template or bbox -> nothing to do
    {
        //nothing to do
    }
}

void combineYoloTMData(std::vector<int>& classIndexesYoloLeft, std::vector<int>& classIndexTMLeft, std::vector<cv::Mat1b>& templatesYoloLeft,std::vector<cv::Mat1b>& templatesTM,
                        std::vector<cv::Rect2d>& bboxesYoloLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes, 
                        std::vector<int>& updatedClasses, std::vector<bool>& boolScalesTM, const int& numTrackersTM)
{
    int counterYolo = 0;
    int counterTM = 0; //for counting TM adaptations
    int counterClassTM = 0; //for counting TM class counter
    //organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
    //think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)


    /* should check carefully -> compare num of detection */
    for (const int classIndex : classIndexesYoloLeft)
    {
        /* tracker was successful */
        if (classIndex != -1) 
        {
            if (counterClassTM < numTrackersTM) //numTrackersTM : num ber of class indexes
            {
                /*update tracker*/
                if (classIndex == classIndexTMLeft[counterClassTM]) 
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesTM[counterTM]); //update bbox to TM one
                    updatedClasses.push_back(classIndex); //update class
                    boolScalesTM.push_back(true);//scale is set to TM
                    counterTM++;
                    counterYolo++;
                    counterClassTM++;
                }
                /*new tracker was made -> increase tracker*/
                else if ((classIndex != classIndexTMLeft[counterClassTM]) && (classIndexTMLeft[counterClassTM] != -1))
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]); //update bbox to YOLO's one
                    updatedClasses.push_back(classIndex); //update class to YOLO's one
                    boolScalesTM.push_back(false); //scale is set to Yolo
                    counterYolo++;

                }
                /*revive tracker*/
                else 
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]); //update bbox to YOLO's one
                    updatedClasses.push_back(classIndex); //update class to YOLO's one
                    boolScalesTM.push_back(false); //scale is set to Yolo
                    counterYolo++;
                    counterClassTM++;
                }
            }
            /*for box new tracker*/
            else
            {
                updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                updatedBboxes.push_back(bboxesYoloLeft[counterYolo]); //update bbox to YOLO's one
                updatedClasses.push_back(classIndex); //update class to YOLO's one
                boolScalesTM.push_back(false); //scale is set to Yolo
                counterYolo++;

            }
        }
        else //yolo tracker can't detect previous tracker
        {
            if (classIndexTMLeft[counterClassTM] != -1) //tracking was successful in TM : continue tracking
            {
                updatedTemplates.push_back(templatesTM[counterTM]); //update tracker to TM's one
                updatedBboxes.push_back(bboxesTM[counterTM]); //update bbox to TM's one
                updatedClasses.push_back(classIndexTMLeft[counterClassTM]);
                boolScalesTM.push_back(true);//scale is set to TM
                counterTM++;
                counterClassTM++;
            }
            else //both tracking was failed : delete tracker
            {
                counterClassTM++;
                //delete tracker
            }
        }
    }
}

void processTM(std::vector<int>& updatedClasses,std::vector<cv::Mat1b>& updatedTemplates,std::vector<bool>& boolScalesTM,std::vector<cv::Rect2d>& updatedBboxes,cv::Mat1b& img, 
               std::vector<cv::Mat1b>& updatedTemplatesTM, std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales)
{
    int counterTracker = 0;
    //get bbox from queue for limiting search area
    int leftSearch, topSearch, rightSearch, bottomSearch;
    //iterate for each tracking classes
    for (const int classIndexTM : updatedClasses)
    {
        /* template exist -> start template matching */
        if (classIndexTM != -1)
        {
            const cv::Mat1b templateImg = updatedTemplates[counterTracker];
            //std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;
            //search area setting 
            if (boolScalesTM[counterTracker]) //scale is set to TM : smaller search area
            {
                leftSearch = std::max(0, (int)(updatedBboxes[counterTracker].x - (scaleXTM - 1) * updatedBboxes[counterTracker].width / 2));
                topSearch = std::max(0, (int)(updatedBboxes[counterTracker].y - (scaleYTM - 1) * updatedBboxes[counterTracker].height / 2));
                rightSearch = std::min(img.cols, (int)(updatedBboxes[counterTracker].x + (scaleXTM + 1) * updatedBboxes[counterTracker].width / 2));
                bottomSearch = std::min(img.rows, (int)(updatedBboxes[counterTracker].y + (scaleYTM + 1) * updatedBboxes[counterTracker].height / 2));
            }
            else //scale is set to YOLO : larger search area
            {
                leftSearch = std::max(0, (int)(updatedBboxes[counterTracker].x - (scaleXYolo - 1) * updatedBboxes[counterTracker].width / 2));
                topSearch = std::max(0, (int)(updatedBboxes[counterTracker].y - (scaleYYolo - 1) * updatedBboxes[counterTracker].height / 2));
                rightSearch = std::min(img.cols, (int)(updatedBboxes[counterTracker].x + (scaleXYolo + 1) * updatedBboxes[counterTracker].width / 2));
                bottomSearch = std::min(img.rows, (int)(updatedBboxes[counterTracker].y + (scaleYYolo + 1) * updatedBboxes[counterTracker].height / 2));
            }
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            img = img(searchArea);
            //std::cout << "img search size:" << img.rows << "," << img.cols << std::endl;

            //search area in template matching
            //int result_cols = img.cols - templ.cols + 1;
            //int result_rows = img.rows - templ.rows + 1;
            cv::Mat result; //for saving template matching results
            int result_cols = img.cols - templateImg.cols + 1;
            int result_rows = img.rows - templateImg.rows + 1;

            result.create(result_rows, result_cols, CV_32FC1); //create result array for matching quality
            //std::cout << "create result" << std::endl;
            //const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
            cv::matchTemplate(img, templateImg, result, matchingMethod); // template Matching
            //std::cout << "templateMatching" << std::endl;
            cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat()); // normalize result score between 0 and 1
            //std::cout << "normalizing" << std::endl;
            double minVal; //minimum score
            double maxVal; //max score
            cv::Point minLoc; //minimum score left-top points
            cv::Point maxLoc; //max score left-top points
            cv::Point matchLoc;

            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); //In C++, we should prepare type-defined box for returns, which is usually pointer
            //std::cout << "minmaxLoc: " << maxVal << std::endl;
            if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
            {
                matchLoc = minLoc;
            }
            else
            {
                matchLoc = maxLoc;
            }

            //meet matching criteria :: setting bbox
            //std::cout << "max value : " << maxVal << std::endl;
            if (minVal <= matchingThreshold)
            {
                int leftRoi, topRoi, rightRoi, bottomRoi;
                cv::Mat1b newTemplate;
                cv::Rect2d roi;
                //std::cout << "search area limited" << std::endl;
                // 
                //convert search region coordinate to frame coordinate
                leftRoi = (int)(matchLoc.x + leftSearch);
                topRoi = (int)(matchLoc.y + topSearch);
                rightRoi = (int)(leftRoi + templateImg.cols);
                bottomRoi = (int)(topRoi + templateImg.rows);
                cv::Rect2d  roiTemplate(matchLoc.x, matchLoc.y, templateImg.cols, templateImg.rows);
                //update template
                newTemplate = img(roiTemplate);
                //update roi
                roi.x = leftRoi;
                roi.y = topRoi;
                roi.width = rightRoi - leftRoi;
                roi.height = bottomRoi - topRoi;

                //update information
                updatedBboxesTM.push_back(roi);
                updatedTemplatesTM.push_back(newTemplate);
                updatedClassesTM.push_back(classIndexTM);
                updatedSearchScales.push_back(true);
                counterTracker++;
                //cv::imwrite("templateImgRight.jpg", newTemplate);
                //std::cout << "roi : " << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
                //std::cout << "newTemplate size :" << newTemplate.rows << "," << newTemplate.cols << std::endl;
                //std::cout << "push img" << std::endl;

            }
            /* doesn't meet matching criteria */
            else
            {
                updatedClassesTM.push_back(-1);//tracking fault
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
        

void getYoloDataLeft(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
{
    std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
    newTemplates = queueYoloTemplateLeft.front();
    newBboxes = queueYoloBboxLeft.front();
    newClassIndexes = queueYoloClassIndexLeft.front();
    queueYoloTemplateLeft.pop();
    queueYoloBboxLeft.pop();
    queueYoloClassIndexLeft.pop();
}

void getYoloDataRight(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
{
    std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
    newTemplates = queueYoloTemplateRight.front();
    newBboxes = queueYoloBboxRight.front();
    newClassIndexes = queueYoloClassIndexRight.front();
    queueYoloTemplateRight.pop();
    queueYoloBboxRight.pop();
    queueYoloClassIndexRight.pop();
}

void push2YoloDataLeft(std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM)
{
    std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
    queueYoloBboxLeft.push(updatedBboxesTM);
    queueYoloClassIndexLeft.push(updatedClassesTM);
}

void push2YoloDataRight(std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM)
{
    std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
    queueYoloBboxRight.push(updatedBboxesTM);
    queueYoloClassIndexRight.push(updatedClassesTM);
}

/* read imgs */
void pushFrame(std::array<cv::Mat1b,2>& srcs, const int frameIndex)
{
    std::unique_lock<std::mutex> lock(mtxImg); // Lock the mutex
    //std::cout << "push imgs" << std::endl;
    cv::Mat1b undistortedImgL, undistortedImgR;
    cv::undistort(srcs[0], undistortedImgL, cameraMatrix, distCoeffs);
    cv::undistort(srcs[1], undistortedImgR, cameraMatrix, distCoeffs);
    std::array<cv::Mat1b, 2> undistortedImgs = { undistortedImgL,undistortedImgR };
    queueFrame.push(undistortedImgs);
    queueFrameIndex.push(frameIndex);
}

/* 3d positioning and trajectory prediction */

void targetPredict()
{
    std::vector<std::vector<std::vector<int>>> dataLeft, dataRight; // [num Objects, time_steps, each time position ] : {frameIndex,centerX,centerY}
    std::vector<std::vector<int>> classesLeft, classesRight;
    /* while loop from here */
    /* organize tracking data */
    updateTrackingData(dataLeft,dataRight,classesLeft,classesRight);
    /* get 3D position -> matching data from left to right */
    std::vector<int> classesLatestLeft, classesLatestRight;
    std::vector<std::vector<int>> dataLatestLeft, dataLatestRight;
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
    /* get latest class labels -> exclude -1 data */
    getLatestClass(classesLeft, classesLatestLeft);
    getLatestClass(classesRight, classesLatestRight);
    /* for each data predict trajectory in 2-dimension and get coefficients */
    std::vector<std::vector<float>> coefficientsXLeft, coefficientsYLeft, coefficientsXRight, coefficientsYRight; //storage for coefficients in fitting in X and Y ::same size with number of objects detected : [numObjects,codfficients]
    trajectoryPredict2D(dataLeft, coefficientsXLeft, coefficientsYLeft, classesLatestLeft);
    trajectoryPredict2D(dataRight, coefficientsXRight, coefficientsYRight, classesLatestRight);
    /* match -> calculate metrics and matches */
    std::vector<std::vector<std::vector<std::vector<int>>>> datasFor3D; // [num matched Objects, 2 (=left + right), time_steps, each time position ] : {frameIndex,centerX,centerY}
    /* trajectory prediction has been done in both 2 imgs -> start object matching */
    /* at least there is one prediction data -> matching is possible */
    dataMatching(coefficientsXLeft, coefficientsXRight, coefficientsYLeft, coefficientsYRight,classesLatestLeft, classesLatestRight, dataLeft, dataRight, datasFor3D);
    /* calculate 3d position based on matching process--> predict target position */
    if (datasFor3D.size() != 0)
    {
        std::vector<std::vector<int>> targets3D;
        predict3DTargets(datasFor3D, targets3D);
    }
    /* convert target position from cameraLeft to robotBase Coordinates */

    /* push data to queueTargetPositions */
}

/* organize tracking data */
void updateTrackingData(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight, std::vector<std::vector<int>>& classesLeft, std::vector<std::vector<int>>& classesRight)
{
    int frameIndex;
    std::vector<int> classesCurrentLeft, classesCurrentRight;
    std::vector<cv::Rect2d> bboxesLeft, bboxesRight;
    /* get tracking data fromt TM */
    bool ret = getTargetData(frameIndex, classesCurrentLeft, classesCurrentRight, bboxesLeft, bboxesRight);
    /* 2 imgs data available */
    if (ret)
    {
        std::vector<std::vector<int>> dataCurrentLeft, dataCurrentRight;
        /* Left data */
        convertRoi2Center(classesCurrentLeft,bboxesLeft,frameIndex,dataCurrentLeft);
        /* Right Data */
        convertRoi2Center(classesCurrentRight, bboxesRight, frameIndex, dataCurrentRight);
        /* compare data with past data */
        std::vector<int> updatedClassesLeft, updatedClassesRight; // updated classes
        /* past Left data exist */
        if (dataLeft.size() != 0)
        {
            /* compare class indexesand match which object is the same */
            /* labels should be synchronized with TM labels */
            adjustData(classesLeft, dataLeft, dataCurrentLeft, updatedClassesLeft);
            classesLeft.push_back(updatedClassesLeft);
        }
        /* Left data doesn't exist */
        else
        {
            /* add new data */
            dataLeft = { dataCurrentLeft };
            classesLeft = { classesCurrentLeft };
        }
        /* past Right data exist */
        if (dataRight.size() != 0)
        {
            /* compare class indexesand match which object is the same */
            /* labels should be synchronized with TM labels */
            adjustData(classesRight, dataRight, dataCurrentRight, updatedClassesRight);
            classesRight.push_back(updatedClassesRight);
        }
        /* Left data doesn't exist */
        else
        {
            /* add new data */
            dataRight = { dataCurrentRight };
            classesRight = { classesCurrentRight };
        }
    }
    /* 2 imgs data isn't available */
    else
    {
        /* Nothing to do */
    }
}

bool getTargetData(int& frameIndex,std::vector<int>& classesLeft,std::vector<int>& classesRight,std::vector<cv::Rect2d>& bboxesLeft,std::vector<cv::Rect2d>& bboxesRight)
{
    std::unique_lock<std::mutex> lock(mtxTarget); // Lock the mutex
    /* both data is available*/
    bool ret;
    if (!queueTargetBboxesLeft.empty() && !queueTargetBboxesRight.empty())
    {
        frameIndex = queueTargetFrameIndex.front();
        classesLeft = queueTargetClassIndexesLeft.front();
        classesRight = queueTargetClassIndexesRight.front();
        bboxesLeft = queueTargetBboxesLeft.front();
        bboxesRight = queueTargetBboxesRight.front();
        queueTargetFrameIndex.pop();
        queueTargetClassIndexesLeft.pop();
        queueTargetClassIndexesRight.pop();
        queueTargetBboxesLeft.pop();
        queueTargetBboxesRight.pop();
        ret = true;
    }
    /* both 2 data can't be available */
    else
    {
        if (!queueTargetBboxesLeft.empty())
        {
            queueTargetBboxesLeft.pop();
        }
        if (!queueTargetBboxesRight.empty())
        {
            queueTargetBboxesRight.pop();
        }
        if (!queueTargetClassIndexesLeft.empty())
        {
            queueTargetClassIndexesLeft.pop();
        }
        if (!queueTargetClassIndexesRight.empty())
        {
            queueTargetClassIndexesRight.pop();
        }
        if (!queueTargetFrameIndex.empty())
        {
            queueTargetFrameIndex.pop();
        }
        ret = false;
    }
    return ret;
}

void convertRoi2Center(std::vector<int>& classes, std::vector<cv::Rect2d>& bboxes,const int& frameIndex, std::vector<std::vector<int>>& tempData)
{
    int counterRoi = 0;
    for (const int classIndex : classes)
    {
        /* bbox is available */
        if (classIndex != -1)
        {
            int centerX = (int)(bboxes[counterRoi].x + bboxes[counterRoi].width / 2);
            int centerY = (int)(bboxes[counterRoi].y + bboxes[counterRoi].height / 2);
            tempData.push_back({ frameIndex,centerX,centerY });
            counterRoi++;
        }
        /* bbox isn't available */
        else
        {
            /* nothing to do */
        }
    }
}

void adjustData(std::vector<std::vector<int>>& classes, std::vector<std::vector<std::vector<int>>>& data, std::vector<std::vector<int>>& dataCurrent, std::vector<int>& updatedClasses)
{
    int counterPastData = 0; //for counting past data
    int counterPastClass = 0; //for counting past class
    int counterCurrentData = 0; //for counting current data
    std::vector<int> classesPast = classes.back(); //get last data : [time_steps, classIndexes]
    int numData = classesPast.size(); //number of classes
    for (const int classIndex : classesPast)
    {
        /* new bbox exist */
        if (classIndex != -1)
        {
            /* within already detected objects */
            if (counterPastData < numData)
            {
                /* match index -> add new data to the list */
                /* counterPastClass and counterPastData counterCurrentData ++ */
                if (classIndex == classesPast[counterPastClass])
                {
                    data[counterPastData].push_back(dataCurrent[counterCurrentData]); //add current data
                    updatedClasses.push_back(classIndex);
                    counterPastData++;
                    counterPastClass++;
                    counterCurrentData++;
                }
                /* doesn't match index -> newly detected objects -> add new data */
                /* counterPastData counterCurrentData ++ */
                else if (classIndex != classesPast[counterPastClass])
                {
                    std::vector<std::vector<int>> newDetection{ dataCurrent[counterCurrentData] };
                    data.insert(data.begin() + counterPastData, newDetection); //add new data
                    updatedClasses.push_back(classIndex); //add new class label
                    counterPastData++;
                    counterCurrentData++;
                }
            }
            /* out of alread detected objects -> newly detected objects */
            /* doesn't match index -> newly detected objects -> add new data */
            /* counterPastData counterCurrentData ++ */
            else
            {
                std::vector<std::vector<int>> newDetection{ dataCurrent[counterCurrentData] };
                data.insert(data.begin() + counterPastData, newDetection); //add new data
                updatedClasses.push_back(classIndex); //add new class label
                counterPastData++;
                counterCurrentData++;
            }
        }
        /* lose tracker */
        else
        {
            /* tracking was successfule so far -> delete past data */
            /* counterPastDataconstant, counterPastClass++ */
            if (classesPast[counterPastClass] != -1)
            {
                data.erase(data.begin() + counterPastData); //delete data
                updatedClasses.push_back(classIndex); //update class index
                counterPastClass++;
            }
            /* already lost -> update class index */
            else
            {
                updatedClasses.push_back(classIndex);
                counterPastClass++;
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
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }
    // Sort data1 based on centerX values and apply the same order to data2
    std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
        {
            return dataLatest[a][1] < dataLatest[b][1];
        });


    std::vector<std::vector<int>> sortedDataLatest(dataLatest.size());
    std::vector<int> sortedClassesLatest(classesLatest.size());

    for (size_t i = 0; i < index.size(); ++i) {
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
    float sumt = 0, sumx = 0, sumz = 0, sumtx = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
    float mean_t, mean_x, mean_z;
    int length = data.size(); //length of data


    for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
    {
        sumt += data[length - i][0];
        sumx += data[length - i][1];
        sumtx += data[length - i][0] * data[length - i][1];
        sumtt += data[length - i][0] * data[length - i][0];
    }
    std::cout << "Linear regression" << std::endl;
    mean_t = (float)sumt / (float)NUM_POINTS_FOR_REGRESSION;
    mean_x = (float)sumx / (float)NUM_POINTS_FOR_REGRESSION;
    float slope_x, intercept_x;
    if ((sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t) > 0.0001)
    {
        slope_x = (float)(sumtx - NUM_POINTS_FOR_REGRESSION * mean_t * mean_x) / (float)(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
        intercept_x = mean_x - slope_x * mean_t;
    }
    else
    {
        slope_x = 0;
        intercept_x = 0;
    }
    result_x = { slope_x,intercept_x };
    std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n" << std::endl;
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
    float sumt = 0,  sumz = 0,  sumtt = 0, sumtz = 0; // for calculating coefficients
    float mean_t,  mean_z;
    int length = data.size(); //length of data


    for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
    {
        sumt += data[length - i][0];
        sumz += data[length - i][3];
        sumtt += data[length - i][0] * data[length - i][0];
        sumtz += data[length - i][0] * data[length - i][3];
    }
    std::cout << "Linear regression" << std::endl;
    mean_t = (float)sumt / (float)NUM_POINTS_FOR_REGRESSION;
    mean_z = (float)sumz / (float)NUM_POINTS_FOR_REGRESSION;
    float slope_z = (float)(sumtz - NUM_POINTS_FOR_REGRESSION * mean_t * mean_z) / (float)(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
    float intercept_z = mean_z - slope_z * mean_t;

    result_z = { slope_z,intercept_z };
    std::cout << "\n\nZ :: The best fit value of curve is : z = " << slope_z << " t + " << intercept_z << ".\n\n" << std::endl;
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

    //argments analysis
    int length = data.size(); //length of data

    float det, coef11, coef12, coef13, coef21, coef22, coef23, coef31, coef32, coef33; //coefficients matrix and det
    float time1, time2, time3;
    time1 = float(data[length - 3][0]);
    time2 = float(data[length - 2][0]);
    time3 = float(data[length - 1][0]);
    det = time1 * time2 * (time1 - time2) + time1 * time3 * (time3 - time1) + time2 * time3 * (time2 - time3); //det
    float c, d, e;
    if (det == 0)
    {
        c = 0; d = 0; e = 0;
    }
    else
    {
        coef11 = (time2 - time3) / det;
        coef12 = (time3 - time1) / det;
        coef13 = (time1 - time2) / det;
        coef21 = (time3 * time3 - time2 * time2) / det;
        coef22 = (time1 * time1 - time3 * time3) / det;
        coef23 = (time2 * time2 - time1 * time1) / det;
        coef31 = time2 * time3 * (time2 - time3) / det;
        coef32 = time1 * time3 * (time3 - time1) / det;
        coef33 = time1 * time2 * (time1 - time2) / det;
        // coefficients of parabola
        c = coef11 * data[length - 3][2] + coef12 * data[length - 2][2] + coef13 * data[length - 1][2];
        d = coef21 * data[length - 3][2] + coef22 * data[length - 2][2] + coef23 * data[length - 1][2];
        e = coef31 * data[length - 3][2] + coef32 * data[length - 2][2] + coef33 * data[length - 1][2];
    }

    result = { c,d,e };


    std::cout << "y = " << c << "x^2 + " << d << "x + " << e << std::endl;
}

void trajectoryPredict2D(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<float>>& coefficientsX, std::vector<std::vector<float>>& coefficientsY,std::vector<int>& classesLatest)
{
    int counterData = 0;
    for (const std::vector<std::vector<int>> data : dataLeft)
    {
        /* get 3 and more time-step datas -> can predict trajectory */
        if (data.size() >= 3)
        {
            /* get latest 3 time-step data */
            std::vector<std::vector<int>> tempData;
            std::vector<float> coefX, coefY;
            // Use reverse iterators to access the last three elements
            auto rbegin = data.rbegin();  // Iterator to the last element
            auto rend = data.rend();      // Iterator one past the end
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
            coefficientsX.push_back({ 0.0,0.0 });
            coefficientsY.push_back({ 0.0,0.0,0.0 });
            //classesLatest.erase(classesLatest.begin() + counterData); //erase class
            counterData++;
            /* can't predict trajectory */
        }
    }
}

float calculateME(std::vector<float>& coefXLeft, std::vector<float>& coefYLeft, std::vector<float>& coefXRight, std::vector<float>& coefYRight)
{
    float me = 0.0; //mean error
    for (int i = 0; i < coefYLeft.size(); i++)
    {
        me = me + (coefYLeft[i] - coefYRight[i]);
    }
    me = me + coefXLeft[0] - coefXLeft[0];
    return me;
}

void dataMatching(std::vector<std::vector<float>>& coefficientsXLeft, std::vector<std::vector<float>>& coefficientsXRight, std::vector<std::vector<float>>& coefficientsYLeft, std::vector<std::vector<float>>& coefficientsYRight,
                  std::vector<int>& classesLatestLeft, std::vector<int>& classesLatestRight, std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight,
                  std::vector<std::vector<std::vector<std::vector<int>>>>& dataFor3D)
{
    float minVal = 20;
    int minIndexRight;
    if (coefficientsXLeft.size() != 0 && coefficientsXRight.size() != 0)
    {
        /* calculate metrics based on left img data */
        for (int i = 0; i < coefficientsXLeft.size(); i++)
        {
            /* deal with moving objects -> at least one coefficient should be more than 0 */
            if (coefficientsYLeft[i][0] != 0 || coefficientsYLeft[i][1] != 0 || coefficientsYLeft[i][2] != 0)
            {
                for (int j = 0; j < coefficientsXRight.size(); j++)
                {
                    /* deal with moving objects -> at least one coefficient should be more than 0 */
                    if (coefficientsYRight[i][0] != 0 || coefficientsYRight[i][1] != 0 || coefficientsYRight[i][2] != 0)
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
                                minIndexRight = j; //most reliable matching index in Right img
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
                    dataFor3D.push_back({ dataLeft[i],dataRight[minIndexRight] }); //match objects and push_back to dataFor3D
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

void predict3DTargets(std::vector<std::vector<std::vector<std::vector<int>>>>& datasFor3D,std::vector<std::vector<int>>& targets3D)
{
    int index, xLeft, xRight, yLeft, yRight;
    float fX = cameraMatrix.at<double>(0,0);
    float fY = cameraMatrix.at<double>(1, 1);
    float fSkew = cameraMatrix.at<double>(0, 1);
    float oX = cameraMatrix.at<double>(0, 2);
    float oY = cameraMatrix.at<double>(1, 2);
    /* iteration of calculating 3d position for each matched objects */
    for (std::vector<std::vector<std::vector<int>>> dataFor3D : datasFor3D)
    {
        std::vector<std::vector<int>> dataL = dataFor3D[0];
        std::vector<std::vector<int>> dataR = dataFor3D[1];
        /* get 3 and more time-step datas -> calculate 3D position */
        int numDataL = dataL.size();
        int numDataR = dataR.size();
        std::vector<std::vector<int>> data3D; //[mm]
        //calculate 3D position
        for (int i = 3; i > 0; i++)
        {
            index = dataL[numDataL - i][0];
            xLeft = dataL[numDataL - i][1];
            xRight = dataR[numDataR - i][1];
            yLeft = dataL[numDataL - i][2];
            yRight = dataR[numDataR - i][2];
            int disparity = (int)(xLeft - xRight);
            int X = (int)(BASELINE / disparity) * (xLeft - oX - (fSkew / fY) * (yLeft - oY));
            int Y = (int)(BASELINE * (fX / fY) * (yLeft - oY) / disparity);
            int Z = (int)(fX * BASELINE / disparity);
            data3D.push_back({ index,X,Y,Z });
        }
        /* trajectoryPrediction */
        std::vector<float> coefX, coefY, coefZ;
        linearRegression(data3D, coefX);
        linearRegressionZ(data3D, coefZ);
        curveFitting(data3D, coefY);
        /* objects move */
        if (coefZ[0] != 0)
        {
            int frameTarget = (int)((TARGET_DEPTH - coefZ[1]) / coefZ[0]);
            int xTarget = (int)(coefX[0] * frameTarget + coefX[1]);
            int yTarget = (int)(coefY[0] * frameTarget * frameTarget + coefY[1] * frameTarget + coefY[2]);
            targets3D.push_back({ frameTarget,xTarget,yTarget,TARGET_DEPTH }); //push_back target position 
            std::cout << "target is : ( frameTarget :  " << frameTarget << ", xTarget : " << xTarget << ", yTarget : " << yTarget << ", depthTarget : " << TARGET_DEPTH << std::endl;
        }

    }
}

/*
* main function
*/
int main()
{
	//カメラのパラメータ設定
	const unsigned int imgWidth = 320; //画像の幅
	const unsigned int imgHeight = 320; //画像の高さ
	const unsigned int frameRate = 10; //フレームレート
	const unsigned int expTime = 1500;// 792; //露光時間
	const int imgGain = 18;
	const bool isBinning = true;
	const int capMode = 1; //0:モノクロ，1:カラー
	const std::string leaderCamID = "30957651";		//リーダーカメラ
	const std::string followerCamID = "30958851"; 	//フォロワカメラ

#if SYNC_CAMERAS
	std::array<Ximea, 2> cams = { Ximea(imgWidth, imgHeight, frameRate, leaderCamID, expTime, isBinning, false), Ximea(imgWidth, imgHeight, frameRate, followerCamID, expTime, isBinning, true) };
#else
	std::array<Ximea, 2> cams = { Ximea(imgWidth, imgHeight, frameRate, leaderCamID, expTime, isBinning, 0), Ximea(imgWidth, imgHeight, frameRate, followerCamID, expTime, isBinning,0) };
#endif //SYNC_CAMERAS
	//ゲイン設定
	cams[0].SetGain(imgGain);
	cams[1].SetGain(18);

	//画像表示
	spsc_queue<dispData_t> queDisp;
	dispData_t dispData;
	std::atomic<bool> isSaveImage = false;
	auto dispInfoPtr = std::make_unique<DispInfo>(queDisp, imgWidth, imgHeight, true);
	std::thread dispThread(std::ref(*dispInfoPtr), std::ref(isSaveImage));

	dispData.srcs[0] = cv::Mat1b::zeros(imgHeight, imgWidth);
	dispData.srcs[1] = cv::Mat1b::zeros(imgHeight, imgWidth);

	//画像保存設定
	saveImages_t saveImages;
	saveImages.resize(maxSaveImageNum);
	for (int i_i = 0; i_i < saveImages.size(); i_i++) {
		for (int i_s = 0; i_s <= 1; i_s++) {
			saveImages[i_i][i_s] = cv::Mat1b::zeros(imgHeight, imgWidth);
		}
	}
	int saveCount = 0;
	std::array<cv::Mat1b, 2> srcs = { cv::Mat1b::zeros(imgHeight, imgWidth),cv::Mat1b::zeros(imgHeight, imgWidth) }; //取得画像

    // multi thread code
    std::thread threadYolo(yoloDetect);
    std::cout << "start template matching" << std::endl;
    std::thread threadTemplateMatching(templateMatching);


	//メイン処理
	for (int fCount = 0;; fCount++)
    {
		for (int i_i = 0; i_i < cams.size(); i_i++)
        {
			srcs[i_i] = cams[i_i].GetNextImageOcvMat(); //画像取得
		}
        //get images and frame index
        pushFrame(srcs, fCount);
		

		//画面表示
		if (fCount % 10 == 0) {
			srcs[0].copyTo(dispData.srcs[0]);
			srcs[1].copyTo(dispData.srcs[1]);
			dispData.frameCount = fCount;
			queDisp.push(dispData); //this is where 2images data is stored
		}

		//画像保存
		if (isSaveImage && saveCount < saveImages.size()) {
			srcs[0].copyTo(saveImages[saveCount][0]);
			srcs[1].copyTo(saveImages[saveCount][1]);
			saveCount++;
		}

		//終了
		if (dispInfoPtr->isTerminate()) {
			//comDspacePtr->finishProc(); //dSPACEとの通信スレッド終了
			break;
		}
	}
    threadYolo.join();
    threadTemplateMatching.join();
	//std::cout << " get imgs size" << std::endl;
    
	/*
    while (!queueFrame.empty())
	{
		std::array<cv::Mat1b, 2> imgs = queueFrame.front();
		int index = queueFrameIndex.front();
		queueFrame.pop();
		queueFrameIndex.pop();
		std::cout << index << "-th imgs :: 2 frame size :: " << imgs[0].size() << "," << imgs[1].size() << std::endl;
	}
    */
    

	//画像出力
	if (saveCount > 0) {
		time_t now = time(nullptr);
		const tm* lt = localtime(&now);
		// "年月日-時間分秒"というディレクトリをつくる
		const std::string dirName_left = std::to_string(lt->tm_year + 1900) + std::to_string(lt->tm_mon + 1) + std::to_string(lt->tm_mday) + "-" + std::to_string(lt->tm_hour) + std::to_string(lt->tm_min) + std::to_string(lt->tm_sec) + "-left-calibration";
		const std::string dirName_right = std::to_string(lt->tm_year + 1900) + std::to_string(lt->tm_mon + 1) + std::to_string(lt->tm_mday) + "-" + std::to_string(lt->tm_hour) + std::to_string(lt->tm_min) + std::to_string(lt->tm_sec) + "-right-calibration";
		const std::string rootDir = "imgs";
		struct stat st;
		if (stat(rootDir.c_str(), &st) != 0) {
			_mkdir(rootDir.c_str());
		}
		const std::string saveDir_left = rootDir + "/" + dirName_left;
		const std::string saveDir_right = rootDir + "/" + dirName_right;
		_mkdir(saveDir_right.c_str());
		_mkdir(saveDir_left.c_str());
		std::cout << "Outputing " << saveCount << "images to " << saveDir_left << std::endl;
		std::cout << "Outputing " << saveCount << "images to " << saveDir_right << std::endl;
		//progressbar bar(saveCount);
		//bar.set_opening_bracket_char("CAM: [");
		for (int i_s = 0; i_s < saveCount; i_s++) {
			cv::Mat1b concatImage;
			cv::Mat1b followImg;
			//cv::flip(saveImages[i_s][1], followImg, -1);
			//cv::hconcat(saveImages[i_s][0], followImg, concatImage);
			cv::imwrite(saveDir_left + "/" + std::to_string(i_s) + ".png", saveImages[i_s][0]);
			cv::imwrite(saveDir_right + "/" + std::to_string(i_s) + ".png", saveImages[i_s][1]);
			//bar.update();
		}
	}

	std::cout << "Finish main proc" << std::endl;
	dispThread.join();
	//sendThread.join();
	//recvThread.join();
	return 0;
}
