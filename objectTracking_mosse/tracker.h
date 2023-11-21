#pragma once

/*
* tracker.h
*   - MOT class
*   - template matching
*/

#ifndef TRACKER_H
#define TRACKER_H

#include "stdafx.h"
#include "global_parameters.h"
// tracker
extern const bool boolMOSSE;

// queue definition
extern std::queue<cv::Mat1b> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

// left cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateLeft; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxLeft;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateLeft;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxLeft;      // queue for templateMatching bbox
extern std::queue<std::vector<int>> queueYoloClassIndexLeft;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexLeft;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesLeft;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateLeft;                    // for updating labels of sequence data
//std::queue<int> queueNumLabels;                           // current labels number -> for maintaining label number consistency
extern std::queue<bool> queueStartYolo; //if new Yolo inference can start

// right cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
extern std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

// 3D positioning ~ trajectory prediction
extern std::queue<int> queueTargetFrameIndex;                      // TM estimation frame
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

class TemplateMatching
{
private:
    // template matching constant value setting
    const float scaleXTM = 1.5; // search area scale compared to roi
    const float scaleYTM = 1.5;
    const float scaleXYolo = 2.0;
    const float scaleYYolo = 2.0;
    const float matchingThreshold = 0.2;             // matching threshold
    const int MATCHINGMETHOD = cv::TM_SQDIFF_NORMED; // //cv::TM_SQDIFF_NORMED -> unique background, cv::TM_CCOEFF_NORMED, cv::TM_CCORR_NORMED -> patterned background // TM_SQDIFF_NORMED is good for small template
    const float MoveThreshold = 0.0;                 // move threshold of objects

public:

    //constructor
    TemplateMatching()
    {
        std::cout << "construtor of template matching" << std::endl;
    }

    /* Template Matching :: Left */
    void templateMatchingForLeft(cv::Mat1b& img, const int frameIndex, std::vector<cv::Mat1b>& templateImgs,
        std::vector<std::vector<cv::Rect2d>>& posSaver, std::vector<std::vector<int>>& classSaver, std::vector<int>& detectedFrame, std::vector<int>& detectedFrameClass)
    {
        // for updating templates
        std::vector<cv::Rect2d> updatedBboxes;
        updatedBboxes.reserve(30);
        std::vector<cv::Mat1b> updatedTemplates;
        updatedTemplates.reserve(30);
        std::vector<int> updatedClasses;
        updatedClasses.reserve(300);

        // get Template Matching data
        std::vector<int> classIndexTMLeft;
        classIndexTMLeft.reserve(30);
        int numTrackersTM = 0;
        std::vector<cv::Rect2d> bboxesTM;
        bboxesTM.reserve(30);
        std::vector<cv::Mat1b> templatesTM;
        templatesTM.reserve(30);
        std::vector<bool> boolScalesTM;
        boolScalesTM.reserve(30);   // search area scale
        bool boolTrackerTM = false; // whether tracking is successful

        /* Template Matching bbox available */
        getTemplateMatchingDataLeft(boolTrackerTM, classIndexTMLeft, bboxesTM, templatesTM, boolScalesTM, numTrackersTM);

        // template from yolo is available
        bool boolTrackerYolo = false;
        /* Yolo Template is availble */
        if (!queueYoloTemplateLeft.empty())
        {
            /* get Yolo data and update Template matchin data */
            organizeData(boolScalesTM, boolTrackerYolo, classIndexTMLeft, templatesTM, bboxesTM, updatedTemplates, updatedBboxes, updatedClasses, numTrackersTM);
        }
        /* template from yolo isn't available but TM tracker exist */
        else if (boolTrackerTM)
        {
            updatedTemplates = templatesTM;
            updatedBboxes = bboxesTM;
            updatedClasses = classIndexTMLeft;
        }
        /* no template is available */
        else
        {
            // nothing to do
        }
        std::cout << "BoolTrackerTM :" << boolTrackerTM << ", boolTrackerYolo : " << boolTrackerYolo << std::endl;
        /*  Template Matching Process */
        if (boolTrackerTM || boolTrackerYolo)
        {
            //std::cout << "template matching process has started" << std::endl;
            int counterTracker = 0;
            int numClasses = updatedClasses.size();
            int numTrackers = updatedTemplates.size();
            //std::cout << "number of classes = " << numClasses << ", number of templates = " << numTrackers << ", number of bboxes = "<<updatedBboxes.size()<<", number of scales="<<boolScalesTM.size()<<std::endl;
            // Initialization for storing TM detection results
            // templates
            int rows = 1;
            int cols = 1;
            int defaultValue = 0;
            std::vector<cv::Mat1b> updatedTemplatesTM(numTrackers, cv::Mat1b(rows, cols, defaultValue));
            // bbox
            cv::Rect2d defaultROI(-1, -1, -1, -1);
            std::vector<cv::Rect2d> updatedBboxesTM(numTrackers, defaultROI);
            // class labels
            std::vector<int> updatedClassesTM(numClasses, -1);
            // scales
            bool defaultBool = false;
            std::vector<bool> updatedSearchScales(numTrackers, defaultBool);
            // finish initialization

            // template matching
            //std::cout << "processTM start" << std::endl;
            processTM(updatedClasses, updatedTemplates, boolScalesTM, updatedBboxes, img, updatedTemplatesTM, updatedBboxesTM, updatedClassesTM, updatedSearchScales);
            //std::cout << "processTM finish" << std::endl;
            if (!updatedSearchScales.empty())
            {
                if (!queueTMScalesLeft.empty()) queueTMScalesLeft.pop(); // pop before push
                queueTMScalesLeft.push(updatedSearchScales);
            }
            if (!updatedBboxesTM.empty())
            {
                if (!queueTMBboxLeft.empty()) queueTMBboxLeft.pop();
                queueTMBboxLeft.push(updatedBboxesTM); // push roi
                posSaver.push_back(updatedBboxesTM);     // save current position to the vector
                //sequence data
                if (!queueTargetFrameIndex.empty()) queueTargetFrameIndex.pop();
                if (!queueTargetClassIndexesLeft.empty()) queueTargetClassIndexesLeft.pop();
                if (!queueTargetBboxesLeft.empty()) queueTargetBboxesLeft.pop();
                queueTargetFrameIndex.push(frameIndex);
                queueTargetClassIndexesLeft.push(updatedClassesTM);
                queueTargetBboxesLeft.push(updatedBboxesTM);
            }
            if (!updatedTemplatesTM.empty())
            {
                if (!queueTMTemplateLeft.empty()) queueTMTemplateLeft.pop();
                queueTMTemplateLeft.push(updatedTemplatesTM); // push template image
            }
            if (!updatedClassesTM.empty())
            {
                if (!queueTMClassIndexLeft.empty()) queueTMClassIndexLeft.pop();
                queueTMClassIndexLeft.push(updatedClassesTM);
                classSaver.push_back(updatedClassesTM); // save current class to the saver
            }
            /* if yolo data is avilable -> send signal to target predict to change labels*/
            if (boolTrackerYolo)
            {
                queueLabelUpdateLeft.push(true);
                if (!queueStartYolo.empty()) queueStartYolo.pop();
                queueStartYolo.push(true);
                std::cout << "push true to queueStartYolo :: TM" << std::endl;
            }
            else
            {
                queueLabelUpdateLeft.push(false);
            }
            detectedFrame.push_back(frameIndex);
            detectedFrameClass.push_back(frameIndex);
        }
        else // no template or bbox -> nothing to do
        {
            if (!classIndexTMLeft.empty())
            {
                if (!queueTMClassIndexLeft.empty()) queueTMClassIndexLeft.pop();
                queueTMClassIndexLeft.push(classIndexTMLeft);
                classSaver.push_back(classIndexTMLeft); // save current class to the saver
                detectedFrameClass.push_back(frameIndex);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                // nothing to do
            }
        }
    }

    void getTemplateMatchingDataLeft(bool& boolTrackerTM, std::vector<int>& classIndexTMLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& templatesTM, std::vector<bool>& boolScalesTM, int& numTrackersTM)
    {
        if (!queueTMClassIndexLeft.empty())
        {
            classIndexTMLeft = queueTMClassIndexLeft.front();
            numTrackersTM = classIndexTMLeft.size();
        }
        if (!queueTMBboxLeft.empty() && !queueTMTemplateLeft.empty() && !queueTMScalesLeft.empty())
        {
            //std::cout << "previous TM tracker is available" << std::endl;
            boolTrackerTM = true;

            bboxesTM = queueTMBboxLeft.front();
            templatesTM = queueTMTemplateLeft.front();
            boolScalesTM = queueTMScalesLeft.front();
        }
        else
        {
            boolTrackerTM = false;
        }
    }

    void organizeData(std::vector<bool>& boolScalesTM, bool& boolTrackerYolo, std::vector<int>& classIndexTMLeft, std::vector<cv::Mat1b>& templatesTM,
        std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes, std::vector<int>& updatedClasses, int& numTrackersTM)
    {
        std::unique_lock<std::mutex> lock(mtxYoloLeft); // Lock the mutex
        //std::cout << "TM :: Yolo data is available" << std::endl;
        boolTrackerYolo = true;
        if (!boolScalesTM.empty())
        {
            boolScalesTM.clear(); // clear all elements of scales
        }
        // get Yolo data
        std::vector<cv::Mat1b> templatesYoloLeft;
        templatesYoloLeft.reserve(10); // get new data
        std::vector<cv::Rect2d> bboxesYoloLeft;
        bboxesYoloLeft.reserve(10); // get current frame data
        std::vector<int> classIndexesYoloLeft;
        classIndexesYoloLeft.reserve(150);
        getYoloDataLeft(templatesYoloLeft, bboxesYoloLeft, classIndexesYoloLeft); // get new frame
        // combine Yolo and TM data, and update latest data
        combineYoloTMData(classIndexesYoloLeft, classIndexTMLeft, templatesYoloLeft, templatesTM, bboxesYoloLeft, bboxesTM, updatedTemplates, updatedBboxes, updatedClasses, boolScalesTM, numTrackersTM);
    }

    void getYoloDataLeft(std::vector<cv::Mat1b>& newTemplates, std::vector<cv::Rect2d>& newBboxes, std::vector<int>& newClassIndexes)
    {
        newTemplates = queueYoloTemplateLeft.front();
        newBboxes = queueYoloBboxLeft.front();
        newClassIndexes = queueYoloClassIndexLeft.front();
        queueYoloTemplateLeft.pop();
        queueYoloBboxLeft.pop();
        queueYoloClassIndexLeft.pop();
        std::cout << "TM get data from Yolo :: class label :: ";
        for (const int& classIndex : newClassIndexes)
        {
            std::cout << classIndex << " ";
        }
        std::cout << std::endl;
    }

    void combineYoloTMData(std::vector<int>& classIndexesYoloLeft, std::vector<int>& classIndexTMLeft, std::vector<cv::Mat1b>& templatesYoloLeft, std::vector<cv::Mat1b>& templatesTM,
        std::vector<cv::Rect2d>& bboxesYoloLeft, std::vector<cv::Rect2d>& bboxesTM, std::vector<cv::Mat1b>& updatedTemplates, std::vector<cv::Rect2d>& updatedBboxes,
        std::vector<int>& updatedClasses, std::vector<bool>& boolScalesTM, const int& numTrackersTM)
    {
        int counterYolo = 0;
        int counterTM = 0;      // for counting TM adaptations
        int counterClassTM = 0; // for counting TM class counter
        // organize current situation : determine if tracker is updated with Yolo or TM, and is deleted
        // think about tracker continuity : tracker survival : (not Yolo Tracker) and (not TM tracker)

        /* should check carefully -> compare num of detection */
        for (const int& classIndex : classIndexesYoloLeft)
        {
            /* after 2nd time from YOLO data */
            if (numTrackersTM != 0)
            {
                //std::cout << "TM tracker already exist" << std::endl;
                /* first numTrackersTM is existed Templates -> if same label, update else, unless tracking was successful lost tracker */
                if (counterClassTM < numTrackersTM) // numTrackersTM : num ber of class indexes
                {
                    /*update tracker*/
                    if (classIndex != -1)
                    {
                        /* if classIndex != -1, update tracker. and if tracker of TM is successful, search aream can be limited */
                        if (classIndex == classIndexTMLeft[counterClassTM])
                        {
                            updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                            updatedBboxes.push_back(bboxesTM[counterTM]);               // update bbox to TM one
                            updatedClasses.push_back(classIndex);                       // update class
                            boolScalesTM.push_back(true);                               // scale is set to TM
                            counterTM++;
                            counterYolo++;
                            counterClassTM++;
                        }
                        /* trakcer of TM was failed */
                        else
                        {
                            updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                            updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                            updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                            boolScalesTM.push_back(false);                              // scale is set to Yolo
                            counterYolo++;
                            counterClassTM++;
                        }
                    }
                    /* tracker not found in YOLO */
                    else
                    {
                        /* template matching was successful -> keep tracking */
                        if (classIndexTMLeft[counterClassTM] != -1)
                        {
                            updatedTemplates.push_back(templatesTM[counterTM]); // update tracker to TM's one
                            updatedBboxes.push_back(bboxesTM[counterTM]);       // update bbox to TM's one
                            updatedClasses.push_back(classIndexTMLeft[counterClassTM]);
                            boolScalesTM.push_back(true); // scale is set to TM
                            counterTM++;
                            counterClassTM++;
                        }
                        /* both tracking was failed -> lost */
                        else
                        {
                            updatedClasses.push_back(classIndex);
                            counterClassTM++;
                        }
                    }
                }
                /* new tracker -> add new templates * maybe in this case all calss labels should be positive, not -1 */
                else
                {
                    if (classIndex != -1)
                    {
                        updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                        updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                        updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                        boolScalesTM.push_back(false);                              // scale is set to Yolo
                        counterYolo++;
                    }
                    /* this is for exception, but prepare for emergency*/
                    else
                    {
                        std::cout << "this is exception:: even if new tracker, class label is -1. Should revise code " << std::endl;
                        updatedClasses.push_back(classIndex);
                    }
                }
            }
            /* for the first time from YOLO data */
            else
            {
                std::cout << "first time of TM" << std::endl;
                /* tracker was successful */
                if (classIndex != -1)
                {
                    updatedTemplates.push_back(templatesYoloLeft[counterYolo]); // update template to YOLO's one
                    updatedBboxes.push_back(bboxesYoloLeft[counterYolo]);       // update bbox to YOLO's one
                    updatedClasses.push_back(classIndex);                       // update class to YOLO's one
                    boolScalesTM.push_back(false);                              // scale is set to Yolo
                    counterYolo++;
                }
                /* tracker was not found in YOLO */
                else
                {
                    updatedClasses.push_back(classIndex);
                }
            }
        }
    }

    void processTM(std::vector<int>& updatedClasses, std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, cv::Mat1b& img,
        std::vector<cv::Mat1b>& updatedTemplatesTM, std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales)
    {
        int counterTracker = 0; // counter for number of tracker
        int counter = 0;        // counter for all classes
        // iterate for each tracking classes
        std::cout << "check updateClasses: ";
        for (const int& classIndex : updatedClasses)
        {
            std::cout << classIndex << " ";
        }
        std::cout << std::endl;
        std::vector<std::thread> threadTM; // prepare threads
        for (const int& classIndexTM : updatedClasses)
        {
            if (classIndexTM != -1)
            {
                // push data to thread
                threadTM.emplace_back(&TemplateMatching::matchingTemplate, this,classIndexTM, counterTracker, counter,
                    std::ref(updatedTemplates), std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(img),
                    std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales));
                // std::cout << counter << "-th thread has started" << std::endl;
                counterTracker += 1;
                counter += 1;
            }
            else
            {
                // push data to thread
                //threadTM.emplace_back(matchingTemplate, classIndexTM, counterTracker, counter,
                //    std::ref(updatedTemplates), std::ref(boolScalesTM), std::ref(updatedBboxes), std::ref(img),
                //    std::ref(updatedTemplatesTM), std::ref(updatedBboxesTM), std::ref(updatedClassesTM), std::ref(updatedSearchScales));
                counter += 1;
            }
        }
        int counterThread = 0;
        if (!threadTM.empty())
        {
            for (std::thread& thread : threadTM)
            {
                thread.join();
                counterThread++;
            }
            std::cout << counterThread << " threads have finished!" << std::endl;
        }
        else
        {
            std::cout << "no thread has started" << std::endl;
        }
        // organize data -> if templates not updated -> erase data
        int counterCheck = 0;
        while (counterCheck < updatedBboxesTM.size())
        {
            // not updated
            if (updatedBboxesTM[counterCheck].width == -1)
            {
                updatedTemplatesTM.erase(updatedTemplatesTM.begin() + counterCheck);
                updatedBboxesTM.erase(updatedBboxesTM.begin() + counterCheck);
                updatedSearchScales.erase(updatedSearchScales.begin() + counterCheck);
            }
            else
            {
                counterCheck++;
            }
        }
    }

    void matchingTemplate(const int classIndexTM, int counterTracker, int counter,
        std::vector<cv::Mat1b>& updatedTemplates, std::vector<bool>& boolScalesTM, std::vector<cv::Rect2d>& updatedBboxes, cv::Mat1b& img,
        std::vector<cv::Mat1b>& updatedTemplatesTM, std::vector<cv::Rect2d>& updatedBboxesTM, std::vector<int>& updatedClassesTM, std::vector<bool>& updatedSearchScales)
    {
        /* template exist -> start template matching */
        //std::cout << "thread : start matchingTemplate" << std::endl;
        //std::cout << "boolScalesTM : ";
        //for (int i = 0; i < boolScalesTM.size(); i++)
        //    std::cout << boolScalesTM[i] << " ";
        //std::cout << std::endl;
        //std::cout << "counterTracker=" << counterTracker << ", counter=" << counter << std::endl;
        // get bbox from queue for limiting search area
        int leftSearch, topSearch, rightSearch, bottomSearch;
        const cv::Mat1b& templateImg = updatedTemplates[counterTracker]; // for saving memory, using reference data
        // std::cout<<"template img size:" << templateImg.rows << "," << templateImg.cols << std::endl;
        // search area setting
        //std::cout << "scale of TM : " << boolScalesTM[counterTracker] << std::endl;
        if (boolScalesTM[counterTracker]) // scale is set to TM : smaller search area
        {
            leftSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].x - (scaleXTM - 1) * updatedBboxes[counterTracker].width / 2));
            topSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].y - (scaleYTM - 1) * updatedBboxes[counterTracker].height / 2));
            rightSearch = std::min(img.cols, static_cast<int>(updatedBboxes[counterTracker].x + (scaleXTM + 1) * updatedBboxes[counterTracker].width / 2));
            bottomSearch = std::min(img.rows, static_cast<int>(updatedBboxes[counterTracker].y + (scaleYTM + 1) * updatedBboxes[counterTracker].height / 2));
        }
        else // scale is set to YOLO : larger search area
        {
            leftSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].x - (scaleXYolo - 1) * updatedBboxes[counterTracker].width / 2));
            topSearch = std::max(0, static_cast<int>(updatedBboxes[counterTracker].y - (scaleYYolo - 1) * updatedBboxes[counterTracker].height / 2));
            rightSearch = std::min(img.cols, static_cast<int>(updatedBboxes[counterTracker].x + (scaleXYolo + 1) * updatedBboxes[counterTracker].width / 2));
            bottomSearch = std::min(img.rows, static_cast<int>(updatedBboxes[counterTracker].y + (scaleYYolo + 1) * updatedBboxes[counterTracker].height / 2));
        }
        cv::Rect2d searchArea(leftSearch, topSearch, (rightSearch - leftSearch), (bottomSearch - topSearch));
        //std::cout << "img size : width = " << img.cols << ", height = " << img.rows << std::endl;
        //std::cout << "croppdeImg size: left=" << searchArea.x << ", top=" << searchArea.y << ", width=" << searchArea.width << ", height=" << searchArea.height << std::endl;
        cv::Mat1b croppedImg = img.clone();
        croppedImg = croppedImg(searchArea); // crop img
        // MOSSE Tracker
        if (boolMOSSE)
        {
            //to be done
        }
        // Template Matching
        else
        {
            cv::Mat result; // for saving template matching results
            int result_cols = croppedImg.cols - templateImg.cols + 1;
            int result_rows = croppedImg.rows - templateImg.rows + 1;
            //std::cout << "result_cols :" << result_cols << ", result_rows:" << result_rows << std::endl;
            // template seems to go out of frame
            if (result_cols <= 0 || result_rows <= 0)
            {
                //std::cout << "template seems to go out from frame" << std::endl;
            }
            else
            {
                //std::cout << "croppedImg :: left=" << leftSearch << ", top=" << topSearch << ", right=" << rightSearch << ", bottom=" << bottomSearch << std::endl;
                result.create(result_rows, result_cols, CV_32FC1); // create result array for matching quality+
                // const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED"; 2 is not so good
                cv::matchTemplate(croppedImg, templateImg, result, MATCHINGMETHOD); // template Matching
                //std::cout << "finish matchTemplate" << std::endl;
                double minVal;    // minimum score
                double maxVal;    // max score
                cv::Point minLoc; // minimum score left-top points
                cv::Point maxLoc; // max score left-top points
                cv::Point matchLoc;

                cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat()); // In C++, we should prepare type-defined box for returns, which is usually pointer
                std::cout << "matching template :: score :: minVal = " << minVal << ", maxVal=" << maxVal << std::endl;
                // std::cout << "minmaxLoc: " << maxVal << std::endl;

                // meet matching criteria :: setting bbox
                // std::cout << "max value : " << maxVal << std::endl;
                /* find matching object */
                if (maxVal >= matchingThreshold)
                {
                    if (MATCHINGMETHOD == cv::TM_SQDIFF || MATCHINGMETHOD == cv::TM_SQDIFF_NORMED)
                    {
                        matchLoc = minLoc;
                    }
                    else
                    {
                        matchLoc = maxLoc;
                    }
                    int leftRoi, topRoi, rightRoi, bottomRoi;
                    cv::Mat1b newTemplate;
                    cv::Rect2d roi;
                    leftRoi = std::max(0, static_cast<int>(matchLoc.x + leftSearch));
                    topRoi = std::max(0, static_cast<int>(matchLoc.y + topSearch));
                    rightRoi = std::min(img.cols, static_cast<int>(leftRoi + templateImg.cols));
                    bottomRoi = std::min(img.rows, static_cast<int>(topRoi + templateImg.rows));
                    // update roi
                    roi.x = leftRoi;
                    roi.y = topRoi;
                    roi.width = rightRoi - leftRoi;
                    roi.height = bottomRoi - topRoi;
                    /* moving constraints */
                    if (std::pow((updatedBboxes[counterTracker].x - roi.x), 2) + std::pow((updatedBboxes[counterTracker].y - roi.y), 2) >= MoveThreshold)
                    {
                        // update template image
                        newTemplate = img(roi);
                        //std::cout << "new ROI : left=" << roi.x << ", top=" << roi.y << ", width=" << roi.width << ", height=" << roi.height << std::endl;
                        //std::cout << "new template size :: width = " << newTemplate.cols << ", height = " << newTemplate.rows << std::endl;

                        // update information
                        updatedBboxesTM.at(counterTracker) = roi;
                        updatedTemplatesTM.at(counterTracker) = newTemplate;
                        updatedClassesTM.at(counter) = classIndexTM; // only class labels updated with counter index
                        updatedSearchScales.at(counterTracker) = true;
                        std::cout << "  ------- succeed in tracking with TM ------------------ " << std::endl;
                    }
                    /* not moving */
                    else
                    {
                        // class label -> default value
                    }
                }
                /* doesn't meet matching criteria */
                else
                {
                    // class label -> default value
                }
            }
        }
    }

};

#endif
