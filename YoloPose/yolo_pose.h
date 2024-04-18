#pragma once

#ifndef YOLO_POSE_H
#define YOLO_POSE_H

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

    std::string yolofilePath = "yolov8m-pose.torchscript";
    int frameWidth = 320; //original image size
    int frameHeight = 320;
    const int yoloWidth = 320; //Yolo input size
    const int yoloHeight = 320;
    const cv::Size YOLOSize{ yoloWidth,yoloHeight };
    const float IoUThreshold = 0.45; //IoU threshold for deleting detected bbox
    const float ConfThreshold = 0.45; //Confidence threshold for detection
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
    };

    void loadModel()
    {
        // read param
        mdl = torch::jit::load(yolofilePath, *device);
        mdl.to(*device);
        mdl.eval();
        std::cout << "load model" << std::endl;
    };

public:
    //constructor for YOLODetect
    YOLOPose() {
        initializeDevice();
        loadModel();
        std::cout << "YOLO construtor has finished!" << std::endl;
    };
    ~YOLOPose() { delete device; }; //Deconstructor

    //main :: posSaver {left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist}
    void detect(cv::Mat1b& frame, int& counter, std::vector<std::vector<std::vector<std::vector<int>>>>& posSaver);

    //preprocess img
    void preprocessImg(cv::Mat1b& frame, torch::Tensor& imgTensor);

    //delete duplicated bbox
    void nonMaxSuppressionHuman(torch::Tensor& prediction, std::vector<torch::Tensor>& detectedBoxesHuman, float confThreshold, float iouThreshold);

    //center to ROI 
    torch::Tensor xywh2xyxy(torch::Tensor& x);

    //non-max suppression
    void nms(torch::Tensor& x, std::vector<torch::Tensor>& detectedBoxes, float& iouThreshold);

    //IoU
    float calculateIoU(const torch::Tensor& box1, const torch::Tensor& box2);

    //extract keypoints
    void keyPointsExtractor(std::vector<torch::Tensor>& detectedBboxesHuman, std::vector<std::vector<std::vector<int>>>& keyPoints, const int& ConfThreshold);

    //draw keypoints in image
    void drawCircle(cv::Mat1b& frame, std::vector<std::vector<std::vector<int>>>& ROI, int& counter);
};

#endif