#include "stdafx.h"
#include "yolo_batch.h"
#include "tracker.h"
#include "utility.h"
#include "sequence.h"
#include "prediction.h"
#include "global_parameters.h"
#include "triangulation.h"
#include "matching.h"
#include "mosse.h"

//saveFile
extern const std::string file_yolo_bbox_left;
extern const std::string file_yolo_class_left;
extern const std::string file_tm_bbox_left;
extern const std::string file_tm_class_left;
extern const std::string file_seq_bbox_left;
extern const std::string file_seq_class_left;
extern const std::string file_yolo_bbox_right;
extern const std::string file_yolo_class_right;
extern const std::string file_tm_bbox_right;
extern const std::string file_tm_class_right;
extern const std::string file_seq_bbox_right;
extern const std::string file_seq_class_right;

// camera : constant setting
extern const int LEFT_CAMERA;
extern const int RIGHT_CAMERA;

//Yolo signals
extern std::queue<bool> queueYolo_tracker2seq_left, queueYolo_tracker2seq_right;
extern std::queue<bool> queueYolo_seq2tri_left, queueYolo_seq2tri_right;

// tracker
extern const bool boolMOSSE;
extern const double threshold_mosse;

// queue definition
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame; // queue for frame
extern std::queue<int> queueFrameIndex;  // queue for frame index

//mosse
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
extern std::queue<bool> queueStartYolo_left; //if new Yolo inference can start
extern std::queue<bool> queueStartYolo_right; //if new Yolo inference can start

// right cam
extern std::queue<std::vector<cv::Mat1b>> queueYoloTemplateRight; // queue for yolo template : for real cv::Mat type
extern std::queue<std::vector<cv::Rect2d>> queueYoloBboxRight;    // queue for yolo bbox
extern std::queue<std::vector<cv::Mat1b>> queueTMTemplateRight;   // queue for templateMatching template img : for real cv::Mat
extern std::queue<std::vector<cv::Rect2d>> queueTMBboxRight;      // queue for TM bbox
extern std::queue<std::vector<int>> queueYoloClassIndexRight;     // queue for class index
extern std::queue<std::vector<int>> queueTMClassIndexRight;       // queue for class index
extern std::queue<std::vector<bool>> queueTMScalesRight;          // queue for search area scale
extern std::queue<bool> queueLabelUpdateRight;                    // for updating labels of sequence data

// for saving sequence data
extern std::vector<std::vector<std::vector<int>>> seqData_left, seqData_right; //storage for sequential data
extern std::queue<int> queueTargetFrameIndex_left;                      // TM estimation frame
extern std::queue<int> queueTargetFrameIndex_right;
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesLeft;  // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<cv::Rect2d>> queueTargetBboxesRight; // bboxes from template matching for predict objects' trajectory
extern std::queue<std::vector<int>> queueTargetClassIndexesLeft;   // class from template matching for maintain consistency
extern std::queue<std::vector<int>> queueTargetClassIndexesRight;  // class from template matching for maintain consistency

//for matching 
extern std::queue<std::vector<int>> queueUpdateLabels_left;
extern std::queue<std::vector<int>> queueUpdatedLabels_right;

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
    //YOLODetect yolodetectorLeft;
    YOLODetect_batch yolo;
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
    //left
    std::vector<std::vector<cv::Rect2d>> posSaverYoloLeft;
    posSaverYoloLeft.reserve(300);
    std::vector<std::vector<int>> classSaverYoloLeft;
    classSaverYoloLeft.reserve(300);
    //right
    std::vector<std::vector<cv::Rect2d>> posSaverYoloRight;
    posSaverYoloLeft.reserve(300);
    std::vector<std::vector<int>> classSaverYoloRight;
    classSaverYoloLeft.reserve(300);
    //detectedFrame
    //left
    std::vector<int> detectedFrameLeft;
    detectedFrameLeft.reserve(300);
    std::vector<int> detectedFrameClassLeft;
    detectedFrameClassLeft.reserve(300);
    //right
    std::vector<int> detectedFrameRight;
    detectedFrameRight.reserve(300);
    std::vector<int> detectedFrameClassRight;
    detectedFrameClassRight.reserve(300);
    int frameIndex;
    int countIteration = 0;
    /* while queueFrame is empty wait until img is provided */
    int counterFinish = 0; // counter before finish
    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::array<cv::Mat1b, 2> frames;
        int frameIndex;
        bool boolImgs = utYolo.getImagesFromQueueYolo(frames, frameIndex);
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
        //concatenate 2 imgs horizontally
        cv::Mat1b concatFrame;
        cv::hconcat(frames[LEFT_CAMERA], frames[RIGHT_CAMERA], concatFrame);
        std::cout << " YOLO -- " << countIteration << " -- " << std::endl;

        /*start yolo detection */
        yolo.detect(concatFrame, frameIndex, posSaverYoloLeft, posSaverYoloRight, classSaverYoloLeft, classSaverYoloRight,
            detectedFrameLeft, detectedFrameRight, detectedFrameClassLeft, detectedFrameClassRight, countIteration);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
        countIteration++;
    }
    /* check data */
    std::cout << "position saver : Yolo : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverYoloLeft size:" << posSaverYoloLeft.size() << ", detectedFrame size:" << detectedFrameLeft.size() << std::endl;
    utYolo.checkStorage(posSaverYoloLeft, detectedFrameLeft, file_yolo_bbox_left);
    std::cout << "classSaverYoloLeft size:" << classSaverYoloLeft.size() << ", detectedFrameClass size:" << detectedFrameClassLeft.size() << std::endl;
    utYolo.checkClassStorage(classSaverYoloLeft, detectedFrameClassLeft, file_yolo_class_left);
    std::cout << " : Right : " << std::endl;
    std::cout << "posSaverYoloRight size:" << posSaverYoloRight.size() << ", detectedFrame size:" << detectedFrameRight.size() << std::endl;
    utYolo.checkStorage(posSaverYoloRight, detectedFrameRight, file_yolo_bbox_right);
    std::cout << "classSaverYoloLeft size:" << classSaverYoloRight.size() << ", detectedFrameClass size:" << detectedFrameClassRight.size() << std::endl;
    utYolo.checkClassStorage(classSaverYoloRight, detectedFrameClassRight, file_yolo_class_right);
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
    //left
    std::vector<std::vector<cv::Rect2d>> posSaverTMLeft;
    posSaverTMLeft.reserve(2000);
    std::vector<std::vector<int>> classSaverTMLeft;
    classSaverTMLeft.reserve(2000);
    //right
    std::vector<std::vector<cv::Rect2d>> posSaverTMRight;
    posSaverTMLeft.reserve(2000);
    std::vector<std::vector<int>> classSaverTMRight;
    classSaverTMLeft.reserve(2000);
    //detected frame
    //left
    std::vector<int> detectedFrameLeft;
    detectedFrameLeft.reserve(300);
    std::vector<int> detectedFrameClassLeft;
    detectedFrameLeft.reserve(300);
    //right
    std::vector<int> detectedFrameRight;
    detectedFrameRight.reserve(300);
    std::vector<int> detectedFrameClassRight;
    detectedFrameRight.reserve(300);
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
        std::array<cv::Mat1b, 2> frames;
        int frameIndex;
        bool boolImgs = utTM.getImagesFromQueueTM(frames, frameIndex);
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
        counterFinish = 0; // reset
        cv::Mat1b frame_left = frames[LEFT_CAMERA];
        cv::Mat1b frame_right = frames[RIGHT_CAMERA];
        std::vector<cv::Mat1b> templateImgsLeft;
        templateImgsLeft.reserve(100);
        bool boolLeft = false;
        /*start template matching process */
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_left(&TemplateMatching::templateMatching, tm, std::ref(frame_left), std::ref(frameIndex), std::ref(posSaverTMLeft), std::ref(classSaverTMLeft), std::ref(detectedFrameLeft), std::ref(detectedFrameClassLeft),
            std::ref(queueTMClassIndexLeft), std::ref(queueTMBboxLeft), std::ref(queueTMTemplateLeft), std::ref(queueTrackerMOSSE_left), std::ref(queueTMScalesLeft),
            std::ref(queueYoloClassIndexLeft), std::ref(queueYoloBboxLeft), std::ref(queueYoloTemplateLeft), std::ref(queueTrackerYolo_left),std::ref(queueStartYolo_left),
            std::ref(queueTargetFrameIndex_left), std::ref(queueTargetClassIndexesLeft), std::ref(queueTargetBboxesLeft),std::ref(queueYolo_tracker2seq_left));
        std::thread thread_right(&TemplateMatching::templateMatching, tm, std::ref(frame_right), std::ref(frameIndex), std::ref(posSaverTMRight), std::ref(classSaverTMRight), std::ref(detectedFrameRight), std::ref(detectedFrameClassRight),
            std::ref(queueTMClassIndexRight), std::ref(queueTMBboxRight), std::ref(queueTMTemplateRight), std::ref(queueTrackerMOSSE_right), std::ref(queueTMScalesRight),
            std::ref(queueYoloClassIndexRight), std::ref(queueYoloBboxRight), std::ref(queueYoloTemplateRight), std::ref(queueTrackerYolo_right), std::ref(queueStartYolo_right),
            std::ref(queueTargetFrameIndex_right), std::ref(queueTargetClassIndexesRight), std::ref(queueTargetBboxesRight), std::ref(queueYolo_tracker2seq_right));
        thread_left.join();
        thread_right.join();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time taken by template matching: " << duration.count() << " milliseconds" << std::endl;
    }
    // check data
    std::cout << "position saver : TM : " << std::endl;
    std::cout << " : Left : " << std::endl;
    std::cout << "posSaverTMLeft size:" << posSaverTMLeft.size() << ", detectedFrame size:" << detectedFrameLeft.size() << std::endl;
    utTM.checkStorageTM(posSaverTMLeft, detectedFrameLeft, file_tm_bbox_left);
    std::cout << "Class saver : TM : " << std::endl;
    std::cout << "classSaverTMLeft size:" << classSaverTMLeft.size() << ", detectedFrameClass size:" << detectedFrameClassLeft.size() << std::endl;
    utTM.checkClassStorageTM(classSaverTMLeft, detectedFrameClassLeft, file_tm_class_left);
    std::cout << " : Right : " << std::endl;
    std::cout << "posSaverTMRight size:" << posSaverTMRight.size() << ", detectedFrame size:" << detectedFrameRight.size() << std::endl;
    utTM.checkStorageTM(posSaverTMRight, detectedFrameRight, file_tm_bbox_right);
    std::cout << "Class saver : TM : " << std::endl;
    std::cout << "classSaverTMRight size:" << classSaverTMRight.size() << ", detectedFrameClass size:" << detectedFrameClassRight.size() << std::endl;
    utTM.checkClassStorageTM(classSaverTMRight, detectedFrameClassRight, file_tm_class_right);
}

void sequence()
{
    Sequence seq; //sequential data
    Utility utSeq; //check data

    std::vector<std::vector<int>> seqClasses_left, seqClasses_right; // storage for sequential classes

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
                std::thread thread_left(&Sequence::updateData, seq, std::ref(seqData_left), std::ref(seqClasses_left), std::ref(queueTargetFrameIndex_left),
                    std::ref(queueTargetClassIndexesLeft), std::ref(queueTargetBboxesLeft), std::ref(queueUpdateLabels_left), std::ref(queueYolo_tracker2seq_left), std::ref(queueYolo_seq2tri_left));
                std::thread thread_right(&Sequence::updateData, seq, std::ref(seqData_right), std::ref(seqClasses_right), std::ref(queueTargetFrameIndex_right),
                    std::ref(queueTargetClassIndexesRight), std::ref(queueTargetBboxesRight), std::ref(queueUpdateLabels_right), std::ref(queueYolo_tracker2seq_right), std::ref(queueYolo_seq2tri_right));
                thread_left.join();
                thread_right.join();
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
    std::cout << "sequential data" << std::endl;
    std::cout << "LEFT" << std::endl;
    utSeq.checkSeqData(seqData_left, seqClasses_left, file_seq_bbox_left, file_seq_class_left);
    std::cout << "RIGHT" << std::endl;
    utSeq.checkSeqData(seqData_right, seqClasses_right, file_seq_bbox_right, file_seq_class_right);
}

/* main function */
int main()
{
    /* video inference */
    //constructor 
    Utility ut;
    Triangulation tri;
    Prediction predict;

    /* video inference */;
    cv::VideoCapture capture_left(filename_left);
    if (!capture_left.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open left file!" << std::endl;
        return 0;
    }
    cv::VideoCapture capture_right(filename_right);
    if (!capture_right.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open right file!" << std::endl;
        return 0;
    }
    int counter = 0;

    // multi thread code
    std::thread threadYolo(yoloDetect);
    std::cout << "start Yolo thread" << std::endl;
    std::thread threadTemplateMatching(templateMatching);
    std::cout << "start template matching thread" << std::endl;
    std::thread threadRemoveImg(&Utility::removeFrame, ut);
    std::cout << "remove frame has started" << std::endl;
    std::thread threadSeq(sequence);
    std::thread threadTri(&Triangulation::main, tri);
    std::cout << "start triangulation thread" << std::endl;
    //std::thread threadPred(&Prediction::main, predict);
    //std::cout << "start prediction thread" << std::endl;


    while (true)
    {
        // Read the next frame
        cv::Mat frame_left, frame_right;
        capture_left >> frame_left;
        capture_right >> frame_right;
        counter++;
        if (frame_left.empty() || frame_right.empty())
            break;
        cv::Mat1b frameGray_left, frameGray_right;
        cv::cvtColor(frame_left, frameGray_left, cv::COLOR_RGB2GRAY);
        cv::cvtColor(frame_right, frameGray_right, cv::COLOR_RGB2GRAY);
        std::array<cv::Mat1b, 2> frames = { frameGray_left,frameGray_right };
        // cv::Mat1b frameGray;
        //  cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
        ut.pushFrame(frames, counter);
    }

    // std::thread threadTargetPredict(targetPredict);
    threadYolo.join();
    threadTemplateMatching.join();
    threadRemoveImg.join();
    threadSeq.join();
    threadTri.join();
    //threadPred.join();
    return 0;
}
