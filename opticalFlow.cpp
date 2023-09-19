// opticalFlow.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

const std::string filename = "video/yolotest.mp4";
const bool save = true;
bool boolSparse = false;
bool boolGray = true;
std::string methodDenseOpticalFlow = "farneback"; //"lucasKanade_dense","rlof"

void sparseOpticalFlow(cv::VideoCapture& capture,cv::Mat& old_gray,std::vector<cv::Point2f>& p0,std::vector<cv::Point2f>& p1,cv::Mat& mask, std::vector<cv::Scalar>& colors)
{
    int counter = 1;
    while (true) {
        // Read new frame
        cv::Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        // Calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, cv::Size(7, 7), 2, criteria);
        std::vector<cv::Point2f> good_new;

        // Visualization part
        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // Draw the tracks
                cv::line(mask, p1[i], p0[i], colors[i], 2);
                cv::circle(frame, p1[i], 5, colors[i], -1);
            }
        }

        // Display the demo
        cv::Mat img;
        add(frame, mask, img);
        if (save) {
            std::string save_path = "video/imgs/opticalFlow" + std::to_string(counter) + ".jpg";
            cv::imwrite(save_path, img);
        }
        imshow("flow", img);
        int keyboard = cv::waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Update the previous frame and previous points
        old_gray = frame_gray.clone();
        p0 = good_new;
        counter++;
    }
}

template <typename Method, typename... Args>
void denseOpticalFlow(cv::VideoCapture& capture, cv::Mat& initFrame, Method method, Args&&... args)
{
    cv::Mat prvs = initFrame;
    int counter = 1;
    while (true) {
        // Read the next frame
        cv::Mat frame2, next;
        capture >> frame2;
        if (frame2.empty())
            break;

        // Preprocessing for exact method
        if (boolGray)
        {
            cv::cvtColor(frame2, next, cv::COLOR_BGR2GRAY);
        }
        else
        {
            next = frame2;
        }
        // Calculate Optical Flow
        cv::Mat flow(prvs.size(), CV_32FC2);
        //define opticalflow method
        method(prvs, next, flow, std::forward<Args>(args)...);

        // Visualization part
        cv::Mat flow_parts[2];
        split(flow, flow_parts);

        // Convert the algorithm's output into Polar coordinates
        cv::Mat magnitude, angle, magn_norm;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, cv::NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        // Build hsv image
        cv::Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);

        // Display the results
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
        if (save) {
            std::string save_path = "video/imgs/frame_" + std::to_string(counter) + ".jpg";
            imwrite(save_path, bgr);
        }
        imshow("frame", frame2);
        imshow("flow", bgr);
        int keyboard = cv::waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Update the previous frame
        prvs = next;
        counter++;
    }
}

int main()
{
    // Read the video 
    cv::VideoCapture capture(filename);
    if (!capture.isOpened()) {
        //error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }

    // Create random colors
    std::vector<cv::Scalar> colors;
    cv::RNG rng;
    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r, g, b));
    }

    cv::Mat old_frame, old_gray;
    std::vector<cv::Point2f> p0, p1;

    // Read first frame and find corners in it
    capture >> old_frame;
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

    /* sparse optical flow */
    if (boolSparse)
    {
        sparseOpticalFlow(capture, old_gray, p0, p1,mask, colors);
    }
    /* dense optical flow */
    else
    {
        /*
        if (methodDenseOpticalFlow == "lucasKanade_dense")
        {
            denseOpticalFlow(capture, old_gray, cv::optflow::calcOpticalFlowSparseToDense, 8, 128, 0.05f, true, 500.0f, 1.5f);
        }
        */
        if (methodDenseOpticalFlow == "farneback")
        {
            denseOpticalFlow(capture,old_gray, cv::calcOpticalFlowFarneback, 0.5,3, 15, 3, 5, 1.2, 0);
        }
        else if (methodDenseOpticalFlow == "rlof") {
            denseOpticalFlow(
                capture,old_gray, cv::optflow::calcOpticalFlowDenseRLOF, cv::Ptr<cv::optflow::RLOFOpticalFlowParameter>(), 1.f, cv::Size(6, 6),
                cv::optflow::InterpolationType::INTERP_EPIC, 128, 0.05f, 999.0f,
                15, 100, true, 500.0f, 1.5f, false); // default OpenCV params
        }
    }
}

