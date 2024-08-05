#include "stdafx.h"
#include "kcftracker.h"

const int Scale = 2;
const float scale_search = 1.5;
const bool bool_crop = true;


bool HOG = false;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

int main() {
    cv::Rect search;
    // declares all required variables
    cv::Rect roi;
    cv::Mat frame;
    cv::Mat gray;
    // create a tracker object
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create(); //library-based
    // Create KCFTracker object
    //KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB); //customized
    // set input video
    std::string video = "ball_box_0119_left.mp4";
    cv::VideoCapture cap(video);
    
    // perform the tracking processG
    std::printf("Start the tracking process, press ESC to quit.\n");
    int counter = 0;
    for (;; ) {
        // get frame from the video]
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        // stop the program if no more images
        if (frame.rows == 0 || frame.cols == 0)
            break;
        //preprocess
        cv::resize(frame, frame, cv::Size((int)(frame.rows / Scale), (int)(frame.cols / Scale)));
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        //std::cout << "resize image" << std::endl;
        if (counter == 0) //select ROI
            roi = selectROI("select ROI", frame);
       
        //cropp image for high-speed
        int left_search = std::min(frame.cols,std::max(0, (int)(roi.x - roi.width * (scale_search / 2))));
        int top_search = std::min(frame.rows, std::max(0, (int)(roi.y - roi.height * (scale_search / 2))));
        int right_search = std::min(frame.cols, std::max(left_search, (int)(roi.x + roi.width * (1 + scale_search / 2))));
        int bottom_search = std::min(frame.rows, std::max(top_search, (int)(roi.y + roi.height * (1 + scale_search / 2))));
        cv::Mat croppedFrame;
        if (bool_crop && right_search > left_search && bottom_search > top_search)
        {
            croppedFrame = frame(cv::Rect(left_search, top_search, (right_search - left_search), (bottom_search - top_search)));
            roi.x -= left_search; //world -> local
            roi.y -= top_search; //world -> local
        }
        else
            croppedFrame = frame.clone();
        
        // First frame, give the groundtruth to the tracker
        if (counter == 0) {
            //customize-based
            //tracker.init(roi, croppedFrame);
            
            //library-based
            tracker->init(croppedFrame, roi);
            bool success = tracker->update(croppedFrame, roi);
        }
        // Update
        else {
            //customize-based
            //roi = tracker.update(gray);
            
            //library-based
            bool success = tracker->update(croppedFrame, roi);
        }
        //time measuring & post process
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        int fps = (int)(1000000 / duration.count());
        if (bool_crop && croppedFrame.cols < frame.cols && croppedFrame.rows < frame.rows)
        {
            roi.x += left_search;
            roi.y += top_search;
        }
        // draw the tracked object
        cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2, 1);
        // Display the FPS on the frame
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // show image with the tracked object
        cv::imshow("tracker", frame);
        counter++;
        //quit on ESC button
        if (cv::waitKey(1) == 27)break;
    }
    return 0;
}