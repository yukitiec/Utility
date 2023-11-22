#include "stdafx.h"
#include "yolo.h"
#include "tracker.h"
#include "utility.h"
#include "sequence.h"
#include "prediction.h"
#include "global_parameters.h"

// queue definition
extern std::queue<cv::Mat1b> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerYolo_right;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_left;
extern std::queue<std::vector<cv::Ptr<cv::mytracker::TrackerMOSSE>>> queueTrackerMOSSE_right;

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


// declare function
/* yolo detection */
void yoloDetect();
/* template matching */
void templateMatching();   
/* add data to storage */
void sequence();

/* Yolo thread function definition */
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
    YOLODetect yolodetectorLeft;
    Utility utYolo;
    //std::cout << "yolo initialization has finished" << std::endl;
    /* initialization */
    if (!queueYoloBboxLeft.empty())
    {
        //std::cout << "queueYoloBboxLeft isn't empty" << std::endl;
        while (!queueYoloBboxLeft.empty())
        {
            queueYoloBboxLeft.pop();
        }
    }
    if (!queueYoloTemplateLeft.empty())
    {
        //std::cout << "queueYoloTemplateLeft isn't empty" << std::endl;
        while (!queueYoloTemplateLeft.empty())
        {
            queueYoloTemplateLeft.pop();
        }
    }
    if (!queueYoloClassIndexLeft.empty())
    {
        //std::cout << "queueYoloClassIndexesLeft isn't empty" << std::endl;
        while (!queueYoloClassIndexLeft.empty())
        {
            queueYoloClassIndexLeft.pop();
        }
    }
    // vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    posSaverYoloLeft.reserve(300);
    std::vector<int> detectedFrame;
    detectedFrame.reserve(300);
    std::vector<int> detectedFrameClass;
    detectedFrame.reserve(300);
    std::vector<std::vector<int>> classSaverYoloLeft;
    classSaverYoloLeft.reserve(300);
    int frameIndex;
    int countIteration = 0;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat1b img;
        int frameIndex;
        bool boolImgs = utYolo.getImagesFromQueueYolo(img, frameIndex);
        if (!boolImgs)
        {
            if (counterFinish > 10)
            {
                break;
            }
            // No more frames in the queue, exit the loop
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::cout << "wait for image pushed" << std::endl;
            counterFinish++;
            continue;
        }
        std::cout << " YOLO -- " << countIteration << " -- " << std::endl;

        /*start yolo detection */
        yolodetectorLeft.detectLeft(img, frameIndex, posSaverYoloLeft, classSaverYoloLeft, detectedFrame, detectedFrameClass, countIteration);
        // std::thread threadLeftYolo(&YOLODetect::detectLeft, &yolodetectorLeft,std::ref(imgs[LEFT_CAMERA]), frameIndex, std::ref(posSaverYoloLeft),std::ref(classSaverYoloLeft));
        // std::thread threadRightYolo(&YOLODetect::detectRight, &yolodetectorRight,std::ref(imgs[RIGHT_CAMERA]), frameIndex, std::ref(posSaverYoloRight),std::ref(classSaverYoloRight));
        // wait for each thread finishing
        // threadLeftYolo.join();
        // threadRightYolo.join();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
        countIteration++;
    }
    /* check data */
    std::cout << "position saver : Yolo : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverYoloLeft size:" << posSaverYoloLeft.size() << ", detectedFrame size:" << detectedFrame.size() << std::endl;
    utYolo.checkStorage(posSaverYoloLeft, detectedFrame);
    std::cout << " : Left : " << std::endl;
    std::cout << "classSaverYoloLeft size:" << classSaverYoloLeft.size() << ", detectedFrameClass size:" << detectedFrameClass.size() << std::endl;
    utYolo.checkClassStorage(classSaverYoloLeft, detectedFrameClass);
}


/* template matching thread function definition */
void templateMatching() // void*
{
    /* Template Matching
     * return most similar region
     */

    //constructor
    TemplateMatching tm;
    Utility utTM;

     // vector for saving position
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    posSaverTMLeft.reserve(2000);
    std::vector<std::vector<int>> classSaverTMLeft;
    classSaverTMLeft.reserve(2000);
    std::vector<int> detectedFrame;
    detectedFrame.reserve(300);
    std::vector<int> detectedFrameClass;
    detectedFrame.reserve(300);
    //initialization
    while (!queueTrackerYolo_left.empty())
        queueTrackerYolo_left.pop();
    while (!queueTrackerYolo_right.empty())
        queueTrackerYolo_right.pop();
    while (!queueTrackerMOSSE_left.empty())
        queueTrackerMOSSE_left.pop();
    while (!queueTrackerMOSSE_right.empty())
        queueTrackerMOSSE_right.pop();
    int countIteration = 0;
    int counterFinish = 0;
    /* sleep for 2 seconds */
    // std::this_thread::sleep_for(std::chrono::seconds(30));
    //  for each img iterations
    int counterStart = 0;
    while (counterStart < 2)
    {
        if (!queueYoloBboxLeft.empty())
        {
            while (true)
            {
                if (queueYoloBboxLeft.empty())
                {
                    counterStart += 1;
                    break;
                }
            }
        }
    }
    std::cout << "start template matching" << std::endl;
    while (true) // continue until finish
    {
        countIteration++;
        std::cout << " -- " << countIteration << " -- " << std::endl;
        // get img from queue
        cv::Mat1b img;
        int frameIndex;
        bool boolImgs = utTM.getImagesFromQueueTM(img, frameIndex);
        //std::cout << "get imgs" << std::endl;
        if (!boolImgs)
        {
            if (counterFinish == 10)
            {
                break;
            }
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else
        {
            counterFinish = 0; // reset
        }
        std::vector<cv::Mat1b> templateImgsLeft;
        templateImgsLeft.reserve(100);
        bool boolLeft = false;
        /*start template matching process */
        auto start = std::chrono::high_resolution_clock::now();
        tm.templateMatchingForLeft(img, frameIndex, templateImgsLeft, posSaverTMLeft, classSaverTMLeft, detectedFrame, detectedFrameClass);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by template matching: " << duration.count() << " milliseconds" << std::endl;
    }
    // check data
    std::cout << "position saver : TM : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverTMLeft size:" << posSaverTMLeft.size() << ", detectedFrame size:" << detectedFrame.size() << std::endl;
    utTM.checkStorageTM(posSaverTMLeft, detectedFrame);
    std::cout << "Class saver : TM : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "classSaverTMLeft size:" << classSaverTMLeft.size() << ", detectedFrameClass size:" << detectedFrameClass.size() << std::endl;
    utTM.checkClassStorageTM(classSaverTMLeft, detectedFrameClass);
}

void sequence()
{
    Sequence seq; //sequential data
    Utility utSeq; //check data

    std::vector<std::vector<std::vector<int>>> seqData; //storage for sequential data
    std::vector<std::vector<int>> seqClasses; // storage for sequential classes

    while (true)
    {
        if (!queueTargetBboxesLeft.empty()) break;
        //std::cout << "wait for target data" << std::endl;
    }
    std::cout << "start saving sequential data" << std::endl;
    int counterIteration = 0;
    int counterFinish = 0;
    while (true) // continue until finish
    {
        counterIteration++;
        //std::cout << "get imgs" << std::endl;
        if (queueFrame.empty() && queueTargetBboxesLeft.empty())
        {
            if (counterFinish == 10) break;
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
            continue;
        }
        else
        {
            /* new detection data available */
            if (!queueTargetBboxesLeft.empty())
            {
                auto start = std::chrono::high_resolution_clock::now();
                seq.updateData(seqData, seqClasses);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << " Time taken by save sequential data : " << duration.count() << " milliseconds" << std::endl;
            }
            /* no data available */
            else
            {
                //nothing to do
            }
        }
    }
    utSeq.checkSeqData(seqData, seqClasses);


}

/* main function */
int main()
{
    /* video inference */
    const std::string filename = "test300fps.mp4";
    cv::VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 0;

    //constructor
    Utility utMain; //utility class

    // multi thread code
    std::thread threadYolo(yoloDetect);
    std::cout << "start Yolo thread" << std::endl;
    std::thread threadTemplateMatching(templateMatching);
    std::cout << "start template matching thread" << std::endl;
    std::thread threadRemoveImg(&Utility::removeFrame,&utMain);
    std::cout << "remove frame has started" << std::endl;
    std::thread threadSeq(sequence);

    while (true)
    {
        // Read the next frame
        cv::Mat frame;
        capture >> frame;
        counter++;
        if (frame.empty())
            break;
        cv::Mat1b frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_RGB2GRAY);
        // cv::Mat1b frameGray;
        //  cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        utMain.pushFrame(frameGray, counter);
    }

    // std::thread threadTargetPredict(targetPredict);
    threadYolo.join();
    threadTemplateMatching.join();
    threadRemoveImg.join();
    threadSeq.join();
    return 0;
}
