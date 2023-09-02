// templateMatching_MultiThread.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
//opencv
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
//multi thread
#include <mutex>
#include <thread>
#include <queue>

cv::Mat img; // overall img
cv::Mat templ; //template img
cv::Mat result; //result img

std::queue<cv::Mat> queue_frame;                     // queue for frame
std::queue<int> queue_frame_index;                            // queue for frame index
std::queue<std::vector<int>> queue_yolo_template;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<int>> queue_yolo_bbox;                 // queue for yolo bbox
std::queue<cv::Mat> queue_templateMatching_template; // queue for templateMatching template img : for real cv::Mat
std::queue<cv::Rect2d> queue_templateMatching_bbox;     // queue for templateMatching bbox

int match_method; //match method in template matching
int max_Trackbar = 5; //track bar
const char* image_window = "Source Imgage";
const char* result_window = "Result window";

void templateMatching(); //void* //proto definition



void templateMatching() //void*
{
    /* Template Matching
    * return most similar region
    */
    int scaleX = 2; //search area scale compared to roi
    int scaleY = 2;
    float matchingThreshold = 0.5; //matching threshold
    /*matching method of template Matching
    * const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
    */
    const int matchingMethod = cv::TM_SQDIFF_NORMED;
    int counter = 0;
    while (!queue_frame.empty()) //continue until finish
    {
        std::cout << " -- " << counter << " -- " << std::endl;
        // get img from queue
        cv::Mat img = queue_frame.front();
        int frameIndex = queue_frame_index.front();
        // remove frame from queue
        queue_frame.pop();
        queue_frame_index.pop();


        cv::Mat templateImg; //template image
        bool boolSearch = false; //if search area limited
        //get templateImg from queue
        if (!queue_templateMatching_template.empty())
        {
            templateImg = queue_templateMatching_template.front();
            std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queue_templateMatching_template.pop();
        }
        //get bbox from queue for limiting search area
        int leftSearch, topSearch, rightSearch, bottomSearch;
        if (!queue_templateMatching_bbox.empty())
        {
            cv::Rect2d bbox = queue_templateMatching_bbox.front(); // bbox : (left,top,width,height)
            queue_templateMatching_bbox.pop(); // remove bbox

            //search area setting
            leftSearch = std::max(0, (int)(bbox.x - (scaleX - 1) * bbox.width / 2));
            topSearch = std::max(0, (int)(bbox.y - (scaleY - 1) * bbox.height / 2));
            rightSearch = std::min(img.cols, (int)(bbox.x + (scaleX + 1) * bbox.width / 2));
            bottomSearch = std::min(img.rows, (int)(bbox.y + (scaleY + 1) * bbox.height / 2));
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            boolSearch = true;
            //crop image
            img = img(searchArea);
            std::cout << "img search size:" << img.rows << "," << img.cols << std::endl;
            if (counter == 99)
            {
                cv::imshow("cropped img", img);
            }
        }
        //std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;

        //search area in template matching
        //int result_cols = img.cols - templ.cols + 1;
        //int result_rows = img.rows - templ.rows + 1;
        cv::Mat result; //for saving template matching results
        int result_cols = img.cols - templateImg.cols + 1;
        int result_rows = img.rows - templateImg.rows + 1;

        result.create(result_rows, result_cols, CV_32FC1); //create result array for matching quality
        std::cout << "create result" << std::endl;
        //const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
        cv::matchTemplate(img, templateImg, result, matchingMethod); // template Matching
        std::cout << "templateMatching" << std::endl;
        cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat()); // normalize result score between 0 and 1
        std::cout << "normalizing" << std::endl;
        double minVal; //minimum score
        double maxVal; //max score
        cv::Point minLoc; //minimum score left-top points
        cv::Point maxLoc; //max score left-top points
        cv::Point matchLoc;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); //In C++, we should prepare type-defined box for returns, which is usually pointer
        std::cout << "minmaxLoc: " << maxVal << std::endl;
        if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
        {
            matchLoc = minLoc;
        }
        else
        {
            matchLoc = maxLoc;
        }

        //setting bbox
        if (maxVal >= matchingThreshold)
        {
            int leftRoi, topRoi, rightRoi, bottomRoi;
            cv::Mat newTemplate;
            cv::Rect2d roi;
            if (boolSearch)
            {
                std::cout << "search area limited" << std::endl;
                leftRoi = (int)(matchLoc.x + leftSearch);
                topRoi = (int)(matchLoc.y + topSearch);
                rightRoi = (int)(leftRoi + templateImg.cols);
                bottomRoi = (int)(topRoi + templateImg.rows);
                cv::Rect2d  roiTemplate(matchLoc.x, matchLoc.y, templateImg.cols, templateImg.rows);
                newTemplate = img(roiTemplate);
            }
            else
            {
                leftRoi = matchLoc.x;
                topRoi = matchLoc.y;
                rightRoi = leftRoi + templateImg.cols;
                bottomRoi = topRoi + templateImg.rows;
                cv::Rect2d roiTemplate(leftRoi, topRoi, templateImg.cols, templateImg.rows);
                newTemplate = img(roiTemplate);
            }
            roi.x = leftRoi;
            roi.y = topRoi;
            roi.width = rightRoi - leftRoi;
            roi.height = bottomRoi - topRoi;

            queue_templateMatching_bbox.push(roi); //push roi 
            queue_templateMatching_template.push(newTemplate);// push template image
            std::cout << "roi : " << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
            std::cout << "newTemplate size :" << newTemplate.rows << "," << newTemplate.cols << std::endl;
            std::cout << "push img" << std::endl;
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
        }
        counter++;
    }    
}

int main()
{
    img = cv::imread("mario.jfif");
    templ = cv::imread("kuri1.png");
    std::cout << "img size" << img.size().height << std::endl;
    std::cout << "template size" << templ.size() << std::endl;
    cv::namedWindow(image_window, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(result_window, cv::WINDOW_AUTOSIZE);

    for (int i = 0; i < 100; i++)
    {
        queue_frame.push(img);
        queue_frame_index.push(i);
    }
    queue_templateMatching_template.push(templ);

    std::thread threadTemplateMatching(templateMatching);
    threadTemplateMatching.join();
    
    cv::waitKey(0);
    return EXIT_SUCCESS;
}