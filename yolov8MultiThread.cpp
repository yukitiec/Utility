#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <thread>
#include <memory>
#include <mutex>
#include <queue>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui/highgui.hpp" //ìÆâÊÇï\é¶Ç∑ÇÈç€Ç…ïKóv

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>

std::queue<cv::Mat> queueFrame;                     // queue for frame
std::queue<int> queueFrameIndex;                            // queue for frame index
std::queue<std::vector<cv::Mat>> queueYoloTemplate;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBbox;                 // queue for yolo bbox
std::queue<std::vector<int>> queueClassIndex; //queue for yolo detection class index

class YOLODetect {
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;
    std::mutex mtx_; // define mutex

    std::string yolofilePath = "best.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloSize = 320;
    const float iouThreshold = 0.45;
    const float confThreshold = 0.3;

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

    void detect(cv::Mat frame, std::vector<std::vector<cv::Rect2d>>& posSaver) {
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

        //measure processing time
        auto start = std::chrono::high_resolution_clock::now();

        cv::Size yolosize(yoloSize, yoloSize);

        // run
        cv::Mat yoloimg; //define yolo img type
        cv::resize(frame, yoloimg, yolosize);
        //cv::cvtColor(yoloimg, yoloimg, cv::COLOR_GRAY2RGB);
        torch::Tensor imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); //vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 }); //Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat); //convert to float type
        imgTensor = imgTensor.div(255); //normalization
        imgTensor = imgTensor.unsqueeze(0); //expand dims for Convolutional layer (height,width,1)
        imgTensor = imgTensor.to(*device); //transport data to GPU
        //std::cout << "Device: " << *device << std::endl;
        //std::cout << "start prediction" << std::endl;
        //torch::Tensor preds = mdl.forward({ imgTensor }).toTuple()->elements()[0].toTensor(); //inference
        torch::Tensor preds = mdl.forward({ imgTensor }).toTensor();
        //torch::Tensor preds = torch::tensor(torch::ArrayRef<float>(pred));
        //std::cout << "preds size : " << preds << std::endl;
        //std::cout << "preds type : " << typeid(preds).name() << std::endl;
        //preds shape : [1,6,2100]

        //non_max_suppression2(preds, 0.4, 0.5);
        //preds = preds.squeeze(0);
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,8400) -> (1,8400,6)
        //std::cout << "preds size : " << preds << std::endl;
        std::vector<torch::Tensor> detectedBoxes0, detectedBoxes1; //(n,6),(m,6)
        non_max_suppression2(preds, detectedBoxes0, detectedBoxes1, confThreshold, iouThreshold);

        //std::cout << "BBOX for Ball : " << detectedBoxes0.size() << " BBOX for BOX : " << detectedBoxes1.size() << std::endl;
        std::vector<cv::Rect2d> roiBall, roiBox;

        //ball
        //std::cout << "Ball detection :: " << std::endl;
        roiSetting(detectedBoxes0, roiBall,0);
        //box
        //std::cout << "Box detection :: " << std::endl;
        roiSetting(detectedBoxes1, roiBox,1);
        //std::cout << "frame size :" << frame.rows << "," << frame.cols << std::endl;
        // both bbox detected
        if (roiBall.size() != 0 && roiBox.size() != 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            for (int i = 0; i < roiBall.size(); i++)
            {
                updatedRoi.push_back(roiBall[i]);
                updatedTemplates.push_back(frame(roiBall[i]));
                updatedClassIndexes.push_back(0);
                
            }
            for (int i = 0; i < roiBox.size(); i++)
            {
                updatedRoi.push_back(roiBox[i]);
                updatedTemplates.push_back(frame(roiBox[i]));
                updatedClassIndexes.push_back(1);
            }
            //save detected data
            posSaver.push_back(updatedRoi);
            //push detected data
            queueYoloBbox.push(updatedRoi);
            queueYoloTemplate.push(updatedTemplates);
            queueClassIndex.push(updatedClassIndexes);
            std::cout << "Both are detected" << std::endl;
        }
        //only ball bbox detected
        else if (roiBall.size() != 0 && roiBox.size() == 0)
        {
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            for (int i = 0; i < roiBall.size(); i++)
            {
                updatedRoi.push_back(roiBall[i]);
                updatedTemplates.push_back(frame(roiBall[i]));
                updatedClassIndexes.push_back(0);
            }
            //save detected data
            posSaver.push_back(updatedRoi);
            //push detected data
            queueYoloBbox.push(updatedRoi);
            queueYoloTemplate.push(updatedTemplates);
            queueClassIndex.push(updatedClassIndexes);
            std::cout << "only ball detected " << std::endl;
        }
        //only box bbox detected
        else if (roiBall.size() == 0 && roiBox.size() != 0)
        {
            //std::cout << "Only box detected" << std::endl;
            std::vector<cv::Rect2d> updatedRoi;
            std::vector<cv::Mat> updatedTemplates;
            std::vector<int> updatedClassIndexes;
            for (int i = 0; i < roiBox.size(); i++)
            {
                updatedRoi.push_back(roiBox[i]);
                //std::cout << "roiBox[i]::" << roiBox[i].x << "," << roiBox[i].y << "," << roiBox[i].width << "," << roiBox[i].height << std::endl;
                //std::cout << "frame(roiBox[i]).size():" << frame(roiBox[i]).rows << "," << frame(roiBox[i]).cols << std::endl;
                updatedTemplates.push_back(frame(roiBox[i]));
                updatedClassIndexes.push_back(1);
            }
            //std::cout << "push back all detection to vector" << std::endl;
            //save detected data
            posSaver.push_back(updatedRoi);
            //push detected data
            queueYoloBbox.push(updatedRoi);
            queueYoloTemplate.push(updatedTemplates);
            queueClassIndex.push(updatedClassIndexes);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        //drawing bbox
        //ball
        //drawRectangle(frame, roiBall, 0);
        //drawRectangle(frame, roiBox, 1);
        //cv::imshow("detection", frame);
        //cv::waitKey(0);
    };

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
                nms(x0, detectedBoxes0, iouThreshold); // exclude overlapped bbox
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
                nms(x1, detectedBoxes1, iouThreshold); // exclude overlapped bbox
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

    void roiSetting(std::vector<torch::Tensor>& detectedBoxes, std::vector<cv::Rect2d>& ROI,int candidateIndex)
    {
        if (detectedBoxes.size() == 0)
        {
            return;
        }
        else
        {
            int numBoxes = detectedBoxes.size();
            int left, top, right, bottom; //score0 : ball , score1 : box
            cv::Rect2d roi;

            bool boolCurrentPosition = false; //if current position is available
            std::vector<cv::Rect2d> bboxesCandidate; //for limiting detection area 
            //criteria for detected bbox's reliability
            if (!queueYoloBbox.empty())
            {
                getYolobbox(bboxesCandidate); //get bboxes from queueYoloBbox : this is from templateMatching tracking, so latest position
                boolCurrentPosition = true;
            }

            for (int i = 0; i < numBoxes; ++i)
            {
                float expandrate[2] = { (float)frameWidth / (float)yoloSize, (float)frameHeight / (float)yoloSize }; // resize bbox to fit original img size
                //std::cout << "expandRate :" << expandrate[0] << "," << expandrate[1] << std::endl;
                left = (int)(detectedBoxes[i][0].item().toFloat() * expandrate[0]);
                top = (int)(detectedBoxes[i][1].item().toFloat() * expandrate[1]);
                right = (int)(detectedBoxes[i][2].item().toFloat() * expandrate[0]);
                bottom = (int)(detectedBoxes[i][3].item().toFloat() * expandrate[1]);
                if (boolCurrentPosition) //limit detected area
                {
                    float max = 0.;
                    std::vector<int> classIndexes = queueClassIndex.front();
                    queueClassIndex.pop();
                    int countCandidates = 0;
                    for (const cv::Rect2d candidate : bboxesCandidate)
                    {
                        if (candidateIndex == classIndexes[countCandidates])
                        {
                            float iou = calculateIoU_Rect2d(candidate, cv::Rect2d(left, top, (right - left), (bottom - top)));
                            if (max < iou)
                            {
                                max = iou;
                            }
                        }
                        countCandidates++;
                    }
                    //different things detected
                    if (max < iouThreshold)
                    {
                        break;
                    }
                    //same things seemed to be detected
                    else
                    {
                        roi.x = left;
                        roi.y = top;
                        roi.width = right - left;
                        roi.height = bottom - top;
                        if (roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 && (roi.x + roi.width) < frameWidth && (roi.y + roi.height) < frameHeight)
                        {
                            ROI.push_back(roi);
                            //std::cout << "bbox : " << roi.x << "," << roi.y << "," << (roi.x + roi.width) << "," << (roi.y + roi.height) << std::endl;
                        }
                    }
                }
                else //not available current position
                {
                    roi.x = left;
                    roi.y = top;
                    roi.width = right - left;
                    roi.height = bottom - top;
                    if (roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 && (roi.x + roi.width) < frameWidth && (roi.y + roi.height) < frameHeight)
                    {
                        ROI.push_back(roi);
                        //std::cout << "bbox : " << roi.x << "," << roi.y << "," << (roi.x + roi.width) << "," << (roi.y + roi.height) << std::endl;
                    }
                }
            }
        }
    }

    float calculateIoU_Rect2d(const cv::Rect2d& box1, const cv::Rect2d& box2) {
        float left = std::max(box1.x, box2.x);
        float top = std::max(box1.y, box2.y);
        float right = std::min((box1.x+box1.width), (box2.x+box2.width));
        float bottom = std::min((box1.y+box1.height), (box2.y+box2.height));

        if (left < right && top < bottom) {
            float intersection = (right - left) * (bottom - top);
            float area1 = box1.width * box1.height;
            float area2 = box2.width * box2.height;
            float unionArea = area1 + area2 - intersection;

            return intersection / unionArea;
        }

        return 0.0f; // No overlap
    }

    void drawRectangle(cv::Mat frame, std::vector<cv::Rect2d>& ROI, int index)
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

    void getYolobbox(std::vector<cv::Rect2d>& bboxes)
    {
        std::unique_lock<std::mutex> lock(mtx_); // Lock the mutex
        bboxes = queueYoloBbox.front(); // get new yolodata : {{x,y.width,height},...}
        queueYoloBbox.pop(); //remove yolo bbox
    }
};

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
    std::vector<std::vector<cv::Rect2d>> posSaverYolo;
    int counter = 0;
    while (!queueFrame.empty()) //continue until finish
    {
        std::cout << " -- " << counter << " -- " << std::endl;
        // get img from queue
        cv::Mat img = queueFrame.front();
        int frameIndex = queueFrameIndex.front();
        // remove frame from queue
        queueFrame.pop();
        queueFrameIndex.pop();

        //yolo detection
        yolodetector.detect(img, posSaverYolo);
        counter++;

    }
    //check code consistency
    std::vector<cv::Mat> templateImgs;
    int count = 1;
    while (!queueYoloTemplate.empty())
    {
        std::cout << count << "-th iteration of queueYoloTemplateImgs" << std::endl;
        templateImgs = queueYoloTemplate.front();
        queueYoloTemplate.pop();
        std::cout << "queueYolotemplateImgs detected : " << templateImgs.size() << std::endl;
        for (int i = 0; i < templateImgs.size(); i++)
        {
            std::cout << "template img size : "<<templateImgs[i].size() << std::endl;
        }
        count++;
    }
    count = 1;
    std::vector<cv::Rect2d> bboxes;
    while (!queueYoloBbox.empty())
    {
        std::cout << count << "-th iteration of queueYoloBboxes" << std::endl;
        bboxes = queueYoloBbox.front();
        queueYoloBbox.pop();
        for (int i = 0; i < bboxes.size(); i++)
        {
            std::cout << "bbox : " << bboxes[i].x<<","<< bboxes[i].y << ","<< bboxes[i].width << ","<< bboxes[i].height << "," << std::endl;
        }
        count++;
    }
    std::vector<int> indexes;
    count = 1;
    while (!queueClassIndex.empty())
    {
        std::cout << count << "-th iteration of queueClassIndex" << std::endl;
        indexes = queueClassIndex.front();
        queueClassIndex.pop();
        for (int i = 0; i < indexes.size(); i++)
        {
            std::cout << "class : " << indexes<<" ";
        }
        std::cout << std::endl;
        count++;
    }
    count = 1;
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

int main()
{
    cv::Mat img = cv::imread("imgs/0028.jpg");
    std::cout << img.size() << std::endl;

    for (int i = 0; i < 30; i++)
    {
        queueFrame.push(img);
        queueFrameIndex.push(i);
    }
    // multi thread code
    std::thread threadYolo(yoloDetect);

    threadYolo.join();

    return 0;
}

