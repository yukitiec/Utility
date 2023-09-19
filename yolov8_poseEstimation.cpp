#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <opencv2/opencv.hpp>

/*  YOLO class definition  */
class YOLOPose {
private:
    torch::jit::script::Module mdl;
    torch::DeviceType devicetype;
    torch::Device* device;

    std::string yolofilePath = "yolov8s-pose.torchscript";
    int frameWidth = 320;
    int frameHeight = 320;
    const int yoloWidth = 320;
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth,yoloHeight };
    const float IoUThreshold = 0.45;
    const float ConfThreshold = 0.45;
    const float IoUThresholdIdentity = 0.25; //for maitainig consistency of tracking 
    /* initialize function */
    void initializeDevice()
    {
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
    //constructor for YOLODetect
    YOLOPose() {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLOPose() { delete device; }; //Deconstructor

    void detect(cv::Mat1b& frame,int& counter)//, const int frameIndex, std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver)
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
        //std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
        //std::cout << "finish preprocess" << std::endl;

        //std::cout << imgTensor.sizes() << std::endl;
        /* inference */
        torch::Tensor preds;
        /* wrap to disable grad calculation */
        {
            torch::NoGradGuard no_grad;
            preds = mdl.forward({ imgTensor }).toTensor(); //preds shape : [1,6,2100]
        }
        //std::cout << "finish inference" << std::endl;
        preds = preds.permute({ 0,2,1 }); //change order : (1,6,8400) -> (1,8400,6)
        //std::cout << "preds size:" << preds.sizes() << std::endl;
        //std::cout << "preds size : " << preds << std::endl;
        std::vector<torch::Tensor> detectedBoxesHuman; //(n,56)
        /*detect human */
        nonMaxSuppressionHuman(preds, detectedBoxesHuman, ConfThreshold, IoUThreshold);
        /* get keypoints from detectedBboxesHuman -> shoulder,elbow,wrist */
        std::vector<std::vector<std::vector<int>>> keyPoints; //vector for storing keypoints
        /* if human detected, extract keypoints */
        if (!detectedBoxesHuman.empty())
        {
            std::cout << "Human detected!" << std::endl;
            keyPointsExtractor(detectedBoxesHuman, keyPoints, ConfThreshold);
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
            //std::cout << "frame size:" << frame.cols << "," << frame.rows << std::endl;
            /* draw keypoints in the frame */
            drawCircle(frame, keyPoints,counter);
        }
        std::string save_path = "video/poseEstimation" + std::to_string(counter) + ".jpg";
        cv::imwrite(save_path, frame);
        
    }

    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor)
    {
        // run
        cv::Mat yoloimg; //define yolo img type
        cv::cvtColor(frame, yoloimg, cv::COLOR_GRAY2RGB);
        cv::resize(yoloimg, yoloimg, YOLOSize);
        imgTensor = torch::from_blob(yoloimg.data, { yoloimg.rows, yoloimg.cols, 3 }, torch::kByte); //vector to tensor
        imgTensor = imgTensor.permute({ 2, 0, 1 }); //Convert shape from (H,W,C) -> (C,H,W)
        imgTensor = imgTensor.toType(torch::kFloat); //convert to float type
        imgTensor = imgTensor.div(255); //normalization
        imgTensor = imgTensor.unsqueeze(0); //(1,3,320,320)
        imgTensor = imgTensor.to(*device); //transport data to GPU
    }

    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold)
    {
        /* non max suppression : remove overlapped bbox
        * Args:
        *   prediction : (1,2100,,6)
        * Return:
        *   detectedbox0,detectedboxs1 : (n,6), (m,6), number of candidate
        */

        torch::Tensor xc = prediction.select(2, 4) > confThreshold; //get dimenseion 2, and 5th element of prediction : score of ball :: xc is "True" or "False"
        torch::Tensor x = prediction.index_select(1, torch::nonzero(xc[0]).select(1, 0)); //box, x0.shape : (1,n,6) : n: number of candidates
        x = x.index_select(1, x.select(2, 4).argsort(1, true).squeeze()); //ball : sorted in descending order
        x = x.squeeze(); //(1,n,56) -> (n,56) 
        if (x.size(0) != 0)
        {
            /* 1 dimension */
            if (x.dim() == 1)
            {
                //std::cout << "top defined" << std::endl;
                detectedBoxesHuman.push_back(x.cpu());
            }
            /* more than 2 dimensions */
            else
            {
                //std::cout << "top defined" << std::endl;
                detectedBoxesHuman.push_back(x[0].cpu());
                //std::cout << "push back top data" << std::endl;
                //for every candidates
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
        y[0] = x[0] - x[2] / 2; //left
        y[1] = x[1] - x[3] / 2; //top
        y[2] = x[0] + x[2] / 2; //right
        y[3] = x[1] + x[3] / 2; //bottom
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
                float iou = calculateIoU(box, savedBox); //calculate IoU
                /* same bbox : already found -> nod add */
                if (iou > iouThreshold)
                {
                    addBox = false;
                    break; //next iteration
                }
            }
            /* new tracker */
            if (addBox)
            {
                detectedBoxes.push_back(x[i].cpu());
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
                    keyPointsTemp.push_back({ static_cast<int>((frameWidth / yoloWidth) * detectedBboxesHuman[i][3 * j + 7 - 2].item<int>()),static_cast<int>((frameHeight / yoloHeight) * detectedBboxesHuman[i][3 * j + 7 - 1].item<int>()) }); /*(xCenter,yCenter)*/
                }
                else
                {
                    keyPointsTemp.push_back({ -1,-1 });
                }
            }
            keyPoints.push_back(keyPointsTemp);
        }
    }
    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI,int& counter)
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
        std::string save_path = "video/poseEstimation"+ std::to_string(counter) + ".jpg";
        cv::imwrite(save_path, frame);
    }
};

int main()
{
    YOLOPose yoloPoseEstimator; //construct
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
    if (!capture.isOpened()) {
        //error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 1;
    while (true) {
        // Read the next frame
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        cv::Mat1b frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        auto start = std::chrono::high_resolution_clock::now();
        yoloPoseEstimator.detect(frameGray,counter);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
        counter++;
    }

    return 0;
}
