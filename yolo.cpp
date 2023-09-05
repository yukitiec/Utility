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

std::mutex mtxImg,mtxYoloLeft,mtxYoloRight; // define mutex

//camera : constant setting
const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
//template matching constant value setting
int match_method; //match method in template matching
int max_Trackbar = 5; //track bar
const char* image_window = "Source Imgage";
const char* result_window = "Result window";

const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
const float scaleXTM = 1.2; //search area scale compared to roi
const float scaleYTM = 1.2;
const float scaleXYolo = 2.0;
const float scaleYYolo = 2.0;
const float matchingThreshold = 0.5; //matching threshold
/*matching method of template Matching
* const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM CCOEFF \n 5: TM CCOEFF NORMED"; 2 is not so good
*/

const int matchingMethod = cv::TM_SQDIFF_NORMED; //TM_SQDIFF_NORMED is good for small template

//queue definition
std::queue<std::array<cv::Mat1b, 2>> queueFrame;                     // queue for frame
std::queue<int> queueFrameIndex;                            // queue for frame index
std::queue<int> queueTMFrame; //templateMatching estimation frame

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

//declare function
void getImagesFromQueue(std::array<cv::Mat1b, 2>&, int&);
void checkYoloTemplateQueueLeft();
void checkYoloTemplateQueueRight();
void checkYoloBboxQueueLeft();
void checkYoloBboxQueueRight();
void checkYoloClassIndexLeft();
void checkYoloClassIndexRight();
void checkStorage(std::vector<std::vector<cv::Rect2d>>&);
void checkClassStorage(std::vector<std::vector<int>>&);
void templateMatching(); //void* //proto definition
void getImagesFromQueue(std::array<cv::Mat1b, 2>&, int&); //get image from queue
void templateMatchingForLeft(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&,std::vector<std::vector<int>>&);//, int, int, float, const int);
void templateMatchingForRight(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&, std::vector<std::vector<int>>&);// , int, int, float, const int);
void combineYoloTMData(std::vector<int>&, std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, 
                        std::vector<cv::Rect2d>&, std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&, const int&);
void processTM(std::vector<int>&, std::vector<cv::Mat1b>&, std::vector<bool>&, std::vector<cv::Rect2d>&, cv::Mat1b&,
                std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&, std::vector<int>&, std::vector<bool>&);
void getYoloDataLeft(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,std::vector<int>&);
void getYoloDataRight(std::vector<cv::Mat1b>&, std::vector<cv::Rect2d>&,std::vector<int>&);
void pushFrame(std::array<cv::Mat1b, 2>&, const int);

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
    const float confThreshold = 0.45;
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
        /* inference */
        torch::Tensor preds = mdl.forward({ imgTensor }).toTensor();
        //std::cout << "preds size : " << preds.sizes() << std::endl;
        //std::cout << "preds type : " << typeid(preds).name() << std::endl;
        //preds shape : [1,6,2100]

        /* postProcess*/
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,8400) -> (1,8400,6)
        //std::cout << "preds size : " << preds.size() << std::endl;
        std::vector<torch::Tensor> detectedBoxes0Left, detectedBoxes1Left; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Left, detectedBoxes1Left, confThreshold, iouThreshold);

        //std::cout << "BBOX for Ball : " << detectedBoxes0.size() << " BBOX for BOX : " << detectedBoxes1.size() << std::endl;
        std::vector<cv::Rect2d> roiBallLeft, roiBoxLeft;
        std::vector<int> classBall, classBox;
        /* Roi Setting : take care of dealing with TM data */
        //ball
        //std::cout << "Ball detection :: " << std::endl;
        /* ROI and class index management */
        roiSetting(detectedBoxes0Left, roiBallLeft, classBall, 0, bboxesCandidateTMLeft,classIndexesTMLeft);
        //box
        //std::cout << "Box detection :: " << std::endl;
        roiSetting(detectedBoxes1Left, roiBoxLeft, classBox, 1, bboxesCandidateTMLeft, classIndexesTMLeft);
        //std::cout << "frame size :" << frame.rows << "," << frame.cols << std::endl;
        // both bbox detected

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

        //std::cout << imgTensor.sizes() << std::endl;
        /* get latest data */
        std::vector<cv::Rect2d> bboxesCandidateTMRight; //for limiting detection area
        std::vector<int> classIndexesTMRight; //candidate class from TM
        getYoloDataRight(bboxesCandidateTMRight, classIndexesTMRight);//get latest data
        /* inference */
        torch::Tensor preds = mdl.forward({ imgTensor }).toTensor();
        //std::cout << "preds size : " << preds.sizes() << std::endl;
        //std::cout << "preds type : " << typeid(preds).name() << std::endl;
        //preds shape : [1,6,2100]

        /* postProcess*/
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,8400) -> (1,8400,6)
        //std::cout << "preds size : " << preds.size() << std::endl;
        std::vector<torch::Tensor> detectedBoxes0Right, detectedBoxes1Right; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0Right, detectedBoxes1Right, confThreshold, iouThreshold);

        //std::cout << "BBOX for Ball : " << detectedBoxes0.size() << " BBOX for BOX : " << detectedBoxes1.size() << std::endl;
        std::vector<cv::Rect2d> roiBallRight, roiBoxRight;
        std::vector<int> classBallRight, classBoxRight;
        /* Roi Setting : take care of dealing with TM data */
        //ball
        //std::cout << "Ball detection :: " << std::endl;
        /* ROI and class index management */
        roiSetting(detectedBoxes0Right, roiBallRight, classBallRight, 0, bboxesCandidateTMRight, classIndexesTMRight);
        //box
        //std::cout << "Box detection :: " << std::endl;
        roiSetting(detectedBoxes1Right, roiBoxRight, classBoxRight, 1, bboxesCandidateTMRight, classIndexesTMRight);
        //std::cout << "frame size :" << frame.rows << "," << frame.cols << std::endl;
        // both bbox detected

        /* push and save data */
        push2QueueRight(roiBallRight, roiBoxRight, classBallRight, classBoxRight, frame, posSaver, classSaver);
        
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
                    break; //next iteration
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
                /* constant setting */
                int numBboxes = detectedBoxes.size(); //num of detection
                std::vector<cv::Rect2d> bboxesYolo;// for storing cv::Rect2d

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
            /* TM tracker exist*/
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
                        //add data
                        updatedClassIndexes.push_back(candidateIndex);
                        updatedBboxes.push_back(bboxTemp);
                        //delete candidate bbox
                        bboxesYolo.erase(bboxesYolo.begin() + indexMatch);
                    }
                    /* not found matched tracker -> return classIndex -1 to updatedClassIndexes */
                    else
                    {
                        updatedClassIndexes.push_back(-1);
                    }
                }
                /* other labels : nothing to do */
                else
                {
                    /* Null */
                }
            }
            /* templateMatching Tracking was fault -> return False */
            else
            {
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
            bboxes = queueYoloBboxLeft.front(); // get new yolodata : {{x,y.width,height},...}
            queueYoloBboxLeft.pop(); //remove yolo bbox
        }
        if (!queueYoloClassIndexLeft.empty())
        {
            classes = queueYoloClassIndexLeft.front(); //get current tracking status
            queueYoloClassIndexLeft.pop();
        }
    }

    void getYoloDataRight(std::vector<cv::Rect2d>& bboxes,std::vector<int>& classes)
    {
        std::unique_lock<std::mutex> lock(mtxYoloRight); // Lock the mutex
        if (!queueYoloBboxRight.empty())
        {
            bboxes = queueYoloBboxRight.front(); // get new yolodata : {{x,y.width,height},...}
            queueYoloBboxRight.pop(); //remove yolo bbox
        }
        if (!queueYoloClassIndexRight.empty())
        {
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
        getImagesFromQueue(imgs, frameIndex);
        /* start yolo detection */
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


void getImagesFromQueue(std::array<cv::Mat1b, 2>& imgs, int& frameIndex)
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
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // for each img iteration
    while (!queueFrame.empty()) //continue until finish
    {
        countIteration++;
        std::cout << " -- " << countIteration << " -- " << std::endl;
        // get img from queue
        std::array<cv::Mat1b, 2> imgs;
        int frameIndex;
        //get images from queue
        auto start = std::chrono::high_resolution_clock::now();
        getImagesFromQueue(imgs, frameIndex);

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
            break;
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
    std::vector<cv::Rect2d> bboxesTM;
    std::vector<cv::Mat1b> templatesTM;
    std::vector<bool> boolScalesTM; //search area scale
    int numTrackersTM = classIndexTMLeft.size();
    bool boolTrackerTM = false; //whether tracking is successful
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
    //template from yolo isn't available
    else
    {
        if (boolTrackerTM)
        {
            updatedTemplates = templatesTM;
            updatedBboxes = bboxesTM;
            updatedClasses = classIndexTMLeft;
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
        
        queueTMBboxLeft.push(updatedBboxesTM); //push roi 
        queueTMTemplateLeft.push(updatedTemplatesTM);// push template image
        queueTMClassIndexLeft.push(updatedClassesTM);
        queueTMScalesLeft.push(updatedSearchScales);
        posSaver.push_back(updatedBboxes);//save current position to the vector
        classSaver.push_back(updatedClassesTM); //save current class to the saver
        push2YoloDataLeft(updatedBboxesTM, updatedClassesTM); //push latest data to queueYoloBboes and queueYoloClassIndexes
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
        queueTMBboxRight.push(updatedBboxesTM); //push roi 
        queueTMTemplateRight.push(updatedTemplatesTM);// push template image
        queueTMClassIndexRight.push(updatedClassesTM);
        queueTMScalesRight.push(updatedSearchScales);
        posSaver.push_back(updatedBboxes);//save current position to the vector
        classSaver.push_back(updatedClassesTM); //save current class to the saver
        push2YoloDataRight(updatedBboxesTM, updatedClassesTM); //push latest data to queueYoloBboes and queueYoloClassIndexes
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
    for (const int classIndex : classIndexesYoloLeft)
    {
        if (classIndex != -1) //tracker was successful
        {
            if (counterClassTM < numTrackersTM)
            {
                if (classIndex == classIndexTMLeft[counterClassTM]) // update tracker
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesTM[counterTM]); //update bbox to TM one
                    updatedClasses.push_back(classIndex); //update class
                    boolScalesTM.push_back(true);//scale is set to TM
                    counterTM++;
                    counterYolo++;
                    counterClassTM++;
                }
                else if (classIndexTMLeft[counterTM] != -1)//new tracker was made -> increase tracker
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]); //update bbox to YOLO's one
                    updatedClasses.push_back(classIndex); //update class to YOLO's one
                    boolScalesTM.push_back(false); //scale is set to Yolo
                    counterYolo++;

                }
                else //revive tracker
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); //update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]); //update bbox to YOLO's one
                    updatedClasses.push_back(classIndex); //update class to YOLO's one
                    boolScalesTM.push_back(false); //scale is set to Yolo
                    counterYolo++;
                    counterClassTM++;
                }
            }
            //for box new tracker
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
    queueFrame.push(srcs);
    queueFrameIndex.push(frameIndex);
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
	for (int fCount = 0;; fCount++) {
		for (int i_i = 0; i_i < cams.size(); i_i++) {
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
            threadYolo.join();
            threadTemplateMatching.join();
			break;
		}
	}
	std::cout << " get imgs size" << std::endl;
	while (!queueFrame.empty())
	{
		std::array<cv::Mat1b, 2> imgs = queueFrame.front();
		int index = queueFrameIndex.front();
		queueFrame.pop();
		queueFrameIndex.pop();
		std::cout << index << "-th imgs :: 2 frame size :: " << imgs[0].size() << "," << imgs[1].size() << std::endl;
	}

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
