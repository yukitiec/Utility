// singleTemplateMatching.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"

cv::Mat img; // overall img
cv::Mat templ; //template img
cv::Mat result; //result img

int match_method; //match method in template matching
int max_Trackbar = 5; //track bar

const char* image_window = "Source Imgage";
const char* result_window = "Result window";

void MatchingMethod(int, const cv::Mat&, const cv::Mat&, cv::Rect2d&); //void* //proto definition

int main()
{

    img = cv::imread("mario.jfif");
    templ = cv::imread("kuri1.png");
    std::cout << "img size" << img.size().height << std::endl;
    std::cout << "template size" << templ.size() << std::endl;

    cv::Rect2d search;
    search.x = img.size().width/2;
    search.y = img.size().height/2;
    search.width = img.size().width/2;
    search.height = img.size().height/2;
    

    
    if (img.empty() || templ.empty())
    {
        std::cout << "Can't read one of the img" << std::endl;
        return EXIT_FAILURE;
    }

    cv::namedWindow(image_window, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(result_window, cv::WINDOW_AUTOSIZE);

    const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
    //if check different methods, change MatchingMethod args :: const cv::Mat& -> void*
    //cv::createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod); //TM CCORR is not so good

    MatchingMethod(0,img,templ,search);

    cv::waitKey(0);
    return EXIT_SUCCESS;
}

void MatchingMethod(int, const cv::Mat& img,const cv::Mat& templ, cv::Rect2d &search) //void*
{
    /* Template Matching
    * return most similar region
    */
    cv::Mat img_display;
    cv::Rect2d roi; //define roi :[x,y,width,height]
    int left, right, top, bottom; // for search area
    int left_bbox, right_bbox, top_bbox, bottom_bbox; // for updated bbox
    img.copyTo(img_display);

    //search area
    left = search.x;
    left = std::max(left, 0);
    top = search.y;
    top = std::max(top, 0);
    right = left + search.width;
    right = std::min(right, img.size().height);
    bottom = top + search.height;
    bottom = std::min(bottom, img.size().height);

    cv::Mat img_search = img(search); //set roi
    std::cout << "img search size:" << img_search.size().width << "," << img_search.size().height << std::endl;
    cv::imshow("cropped img", img_search);
    
    
    //search area in template matching
    //int result_cols = img.cols - templ.cols + 1;
    //int result_rows = img.rows - templ.rows + 1;
    int result_cols = img_search.cols - templ.cols + 1;
    int result_rows = img_search.rows - templ.rows + 1;

    result.create(result_rows, result_cols, CV_32FC1); //create result array for matching quality

    //cv::matchTemplate(img, templ, result, match_method); // template Matching
    cv::matchTemplate(img_search, templ, result, match_method); // template Matching

    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat()); // normalize result score between 0 and 1

    double minVal; //minimum score
    double maxVal; //max score
    cv::Point minLoc; //minimum score left-top points
    cv::Point maxLoc; //max score left-top points
    cv::Point matchLoc;

    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); //In C++, we should prepare type-defined box for returns, which is usually pointer
    if (match_method == cv::TM_SQDIFF || match_method == cv::TM_SQDIFF_NORMED)
    {
        matchLoc = minLoc;
    }
    else
    {
        matchLoc = maxLoc;
    }

    //
    left_bbox = matchLoc.x + left;
    top_bbox = matchLoc.y + top;
    right_bbox = left_bbox + templ.cols;
    if (right_bbox > img.size().width)
    {
        right_bbox = img.size().width;
    }
    bottom_bbox = top_bbox + templ.rows;
    if (bottom_bbox > img.size().height)
    {
        bottom_bbox = img.size().height;
    }
    //draw rectangle into result images
    //cv::rectangle(img_display, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(125), 2, 8, 0);
    //cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(125), 2, 8, 0);
    cv::rectangle(img_display, cv::Point(left_bbox,top_bbox), cv::Point(right_bbox, bottom_bbox), cv::Scalar::all(20), 2, 8, 0);
    cv::rectangle(result, cv::Point(left_bbox, top_bbox), cv::Point(right_bbox, bottom_bbox), cv::Scalar::all(20), 2, 8, 0);

    //set roi
    //roi.x = matchLoc.x;
    //roi.y = matchLoc.y;
    //roi.width = templ.cols;
    //roi.height = templ.rows;
    roi.x = left_bbox;
    roi.y = top_bbox;
    roi.width = right_bbox-left_bbox;
    roi.height = bottom_bbox - top_bbox;

    std::cout << "Roi is : " << "("<< roi.x<<","<< roi.y<<","<< roi.width<<","<< roi.height<<")" << std::endl;

    cv::imshow(image_window, img_display);
    cv::imshow(result_window, result);

    return;
}

