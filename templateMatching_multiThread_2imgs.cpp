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


std::queue<std::array<cv::Mat1b,2>> queueFrame;                     // queue for frame
std::queue<int> queueFrameIndex;                            // queue for frame index
std::queue<int> queueTemplateMatchingFrame; //templateMatching estimation frame

//left cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;                 // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTemplateMatchingTemplateLeft; // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTemplateMatchingBboxLeft;     // queue for templateMatching bbox
std::queue<std::vector<int>> queueClassIndexLeft; //queue for class index

//right cam
std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;                 // queue for yolo bbox
std::queue<std::vector<cv::Mat1b>> queueTemplateMatchingTemplateRight; // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<cv::Rect2d>> queueTemplateMatchingBboxRight;     // queue for templateMatching bbox
std::queue<std::vector<int>> queueClassIndexRight; //queue for class index

int match_method; //match method in template matching
int max_Trackbar = 5; //track bar
const char* image_window = "Source Imgage";
const char* result_window = "Result window";

const int LEFT_CAMERA = 0;
const int RIGHT_CAMERA = 1;
const int scaleX = 1.2; //search area scale compared to roi
const int scaleY = 1.2;
const float matchingThreshold = 0.5; //matching threshold
/*matching method of template Matching
* const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM CCOEFF \n 5: TM CCOEFF NORMED"; 2 is not so good
*/

const int matchingMethod = cv::TM_SQDIFF_NORMED; //TM_SQDIFF_NORMED is good for small template

void templateMatching(); //void* //proto definition
void getImagesFromQueue(std::array<cv::Mat1b, 2>&, int&); //get image from queue
void checkData(std::vector<std::vector<cv::Rect2d>>&); //check data
void templateMatchingForLeft(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&);//, int, int, float, const int);
void templateMatchingForRight(cv::Mat1b&, const int, std::vector<cv::Mat1b>&, std::vector<std::vector<cv::Rect2d>>&);// , int, int, float, const int);


void templateMatching() //void*
{
    /* Template Matching
    * return most similar region
    */

    //vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverLeft;
    std::vector<std::vector<cv::Rect2d>> posSaverRight;
    
    int countIteration = 0;

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
        if (!queueTemplateMatchingTemplateLeft.empty() && !queueTemplateMatchingTemplateRight.empty())
        {
            std::vector<cv::Mat1b> templateImgsLeft = queueTemplateMatchingTemplateLeft.front();
            std::vector<cv::Mat1b> templateImgsRight = queueTemplateMatchingTemplateRight.front();

            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTemplateMatchingTemplateLeft.pop();
            queueTemplateMatchingTemplateRight.pop();
            std::thread threadLeftTM(templateMatchingForLeft, std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(templateImgsLeft), std::ref(posSaverLeft));
            std::thread threadRightTM(templateMatchingForRight, std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(templateImgsRight), std::ref(posSaverRight));
            //wait for each thrad finish
            threadLeftTM.join();
            threadRightTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        }
        //get templateImg from only left camera
        else if (!queueTemplateMatchingTemplateLeft.empty() && queueTemplateMatchingTemplateRight.empty())
        {
            std::vector<cv::Mat1b> templateImgsLeft = queueTemplateMatchingTemplateLeft.front();
            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTemplateMatchingTemplateLeft.pop();
            std::thread threadLeftTM(templateMatchingForLeft, std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(templateImgsLeft), std::ref(posSaverLeft));
            threadLeftTM.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;
        }
        //get templateImg from only right camera
        else if (queueTemplateMatchingTemplateLeft.empty() && !queueTemplateMatchingTemplateRight.empty())
        {
            std::vector<cv::Mat1b> templateImgsRight = queueTemplateMatchingTemplateRight.front();
            //std::cout << "template img size : " << templateImg.rows << "," << templateImg.cols << std::endl;
            queueTemplateMatchingTemplateRight.pop();
            std::thread threadRightTM(templateMatchingForRight, std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(templateImgsRight), std::ref(posSaverRight));
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
    std::cout << "left posSaver: " << std::endl;
    checkData(posSaverLeft);
    std::cout << "right posSaver: " << std::endl;
    checkData(posSaverRight);
    //check frameIndex
    std::cout << " [ queueTemplateMatchingFrame ]" << std::endl;
    while (!queueTemplateMatchingFrame.empty())
    {
        int frameIndex = queueTemplateMatchingFrame.front();
        queueTemplateMatchingFrame.pop();
        std::cout << "frame index :" << frameIndex << std::endl;
    }

}

void getImagesFromQueue(std::array<cv::Mat1b,2>& imgs, int& frameIndex)
{
    imgs = queueFrame.front();
    frameIndex = queueFrameIndex.front();
    // remove frame from queue
    queueFrame.pop();
    queueFrameIndex.pop();
}

void checkData(std::vector<std::vector<cv::Rect2d>>& posSaver)
{
    for (int i = 0; i < posSaver.size(); i++)
    {
        std::cout << " ---- " << i << "-th iteration ---- " << std::endl;
        for (int j = 0; j < posSaver[i].size(); j++)
        {
            std::cout << "bbox : " << posSaver[i][j].x << "," << posSaver[i][j].y << "," << posSaver[i][j].width << "," << posSaver[i][j].height << std::endl;
        }
    }
}

void templateMatchingForLeft(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs,std::vector<std::vector<cv::Rect2d>>& posSaver)
{
    bool boolSearch = false; //if search area limited
    std::vector<int> classIndexLeft = queueClassIndexLeft.front(); //get class index
    //get bbox from queue for limiting search area
    int leftSearch, topSearch, rightSearch, bottomSearch;
    std::vector<cv::Rect2d> bboxes;
    if (!queueTemplateMatchingBboxLeft.empty())
    {
        bboxes = queueTemplateMatchingBboxLeft.front(); // bbox : (left,top,width,height)
        queueTemplateMatchingBboxLeft.pop(); // remove bbox
        boolSearch = true;
    }

    int counterBboxes = 0;
    std::vector<cv::Mat1b> updatedTemplates;
    std::vector<cv::Rect2d> updatedBboxes;
    std::vector<int> updatedClassIndex;
    //iterate for each template imgs
    for (const cv::Mat1b templateImg : templateImgs)
    {
        //std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;

        if (boolSearch)
        {
            //search area setting
            leftSearch = std::max(0, (int)(bboxes[counterBboxes].x - (scaleX - 1) * bboxes[counterBboxes].width / 2));
            topSearch = std::max(0, (int)(bboxes[counterBboxes].y - (scaleY - 1) * bboxes[counterBboxes].height / 2));
            rightSearch = std::min(img.cols, (int)(bboxes[counterBboxes].x + (scaleX + 1) * bboxes[counterBboxes].width / 2));
            bottomSearch = std::min(img.rows, (int)(bboxes[counterBboxes].y + (scaleY + 1) * bboxes[counterBboxes].height / 2));
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            img = img(searchArea);
            //std::cout << "img search size:" << img.rows << "," << img.cols << std::endl;
        }

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
            if (boolSearch)
            {
                //std::cout << "search area limited" << std::endl;
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

            updatedBboxes.push_back(roi);
            updatedTemplates.push_back(newTemplate);
            updatedClassIndex.push_back(classIndexLeft[counterBboxes]);
            counterBboxes++;
            cv::imwrite("templateImgLeft.jpg", newTemplate);
            //std::cout << "roi : " << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
            //std::cout << "newTemplate size :" << newTemplate.rows << "," << newTemplate.cols << std::endl;
            //std::cout << "push img" << std::endl;

        }
        //doesn't meet matching criteria
        else
        {
            counterBboxes++;
            break;
        }
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
    }
    queueTemplateMatchingBboxLeft.push(updatedBboxes); //push roi 
    queueTemplateMatchingTemplateLeft.push(updatedTemplates);// push template image
    queueClassIndexRight.push(updatedClassIndex); //update class index
    queueTemplateMatchingFrame.push(frameIndex); //push template matching successful index
    posSaver.push_back(updatedBboxes);//save current position to the vector
}

void templateMatchingForRight(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs, std::vector<std::vector<cv::Rect2d>>& posSaver)
{
    bool boolSearch = false; //if search area limited
    std::vector<int> classIndexRight = queueClassIndexRight.front(); //get class index

    //get bbox from queue for limiting search area
    int leftSearch, topSearch, rightSearch, bottomSearch;
    std::vector<cv::Rect2d> bboxes;
    if (!queueTemplateMatchingBboxRight.empty())
    {
        bboxes = queueTemplateMatchingBboxRight.front(); // bbox : (left,top,width,height)
        queueTemplateMatchingBboxRight.pop(); // remove bbox
        boolSearch = true;
    }

    int counterBboxes = 0;
    std::vector<cv::Mat1b> updatedTemplates;
    std::vector<cv::Rect2d> updatedBboxes;
    std::vector<int> updatedClassIndex;
    //iterate for each template imgs
    for (const cv::Mat1b templateImg : templateImgs)
    {
        //std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;

        if (boolSearch)
        {
            //search area setting
            leftSearch = std::max(0, (int)(bboxes[counterBboxes].x - (scaleX - 1) * bboxes[counterBboxes].width / 2));
            topSearch = std::max(0, (int)(bboxes[counterBboxes].y - (scaleY - 1) * bboxes[counterBboxes].height / 2));
            rightSearch = std::min(img.cols, (int)(bboxes[counterBboxes].x + (scaleX + 1) * bboxes[counterBboxes].width / 2));
            bottomSearch = std::min(img.rows, (int)(bboxes[counterBboxes].y + (scaleY + 1) * bboxes[counterBboxes].height / 2));
            cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
            img = img(searchArea);
            //std::cout << "img search size:" << img.rows << "," << img.cols << std::endl;
        }

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
            if (boolSearch)
            {
                //std::cout << "search area limited" << std::endl;
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

            updatedBboxes.push_back(roi);
            updatedTemplates.push_back(newTemplate);
            updatedClassIndex.push_back(classIndexRight[counterBboxes]);
            counterBboxes++;
            cv::imwrite("templateImgRight.jpg", newTemplate);
            //std::cout << "roi : " << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << std::endl;
            //std::cout << "newTemplate size :" << newTemplate.rows << "," << newTemplate.cols << std::endl;
            //std::cout << "push img" << std::endl;

        }
        //doesn't meet matching criteria
        else
        {
            counterBboxes++;
            break;
        }
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
    }
    queueTemplateMatchingBboxRight.push(updatedBboxes); //push roi 
    queueTemplateMatchingTemplateRight.push(updatedTemplates);// push template image
    queueClassIndexRight.push(updatedClassIndex);
    posSaver.push_back(updatedBboxes);//save current position to the vector
}

int main()
{
    std::string rootDir = "imgs";
    std::string imgPath = rootDir + "/" + "0018.jpg";
    std::string template1Path = rootDir + "/" + "template1.jpg";
    std::string template2Path = rootDir + "/" + "template2.jpg";
    cv::Mat1b img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat1b templ1 = cv::imread(template1Path, cv::IMREAD_GRAYSCALE);
    cv::Mat1b templ2 = cv::imread(template2Path, cv::IMREAD_GRAYSCALE);
    cv::imwrite("img.jpg", img);
    cv::imwrite("template1.jpg", templ1);
    cv::imwrite("template2.jpg", templ2);
    std::cout << "img size: " << img.rows<<","<<img.cols << std::endl;
    std::cout << "template1 size: " << templ1.rows<< "," << templ1.cols << std::endl;
    std::cout << "template2 size: " << templ2.rows << "," << templ2.cols << std::endl;
    //cv::namedWindow(image_window, cv::WINDOW_AUTOSIZE);
    //cv::namedWindow(result_window, cv::WINDOW_AUTOSIZE);

    for (int i = 0; i < 10; i++)
    {
        std::array<cv::Mat1b, 2> imgs = { img,img };
        queueFrame.push(imgs);
        queueFrameIndex.push(i);
    }
    std::cout << "finish pushing imgs" << std::endl;
    std::vector<cv::Mat1b> templateImg1{ templ1 };
    std::vector<cv::Mat1b> templateImg2{ templ2 };
    cv::imwrite("template1beforeQueue.jpg", templateImg1[0]);
    cv::imwrite("template2beforeQueue.jpg", templateImg2[0]);
    queueTemplateMatchingTemplateLeft.push(templateImg1);
    queueTemplateMatchingTemplateRight.push(templateImg2);
    std::vector<int> classIndexleft{ 0 };
    std::vector<int> classIndexRight{ 1 };
    queueClassIndexLeft.push(classIndexleft);
    queueClassIndexRight.push(classIndexRight);
    std::cout << "finish pushing class index" << std::endl;

    std::cout << "start template matching" << std::endl;
    std::thread threadTemplateMatching(templateMatching);
    threadTemplateMatching.join();

    cv::waitKey(0);
    return EXIT_SUCCESS;
}