#include "stdafx.h"
#include "yolopose.h"
#include "yolopose_batch.h"
#include "opticalflow.h"
#include "utility.h"
#include "triangulation.h"
#include "global_parameters.h"

extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
extern std::queue<int> queueFrameIndex;
/* left */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_left;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_left;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_left;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_left;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_left; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
/* right */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch_right;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueYoloSearchRoi_right;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch_right;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2i>>> queueOFSearchRoi_right;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove_right; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]
/*3D position*/
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_left;
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueTriangulation_right;

/* constant valude definition */
extern const std::string filename_left;
extern const std::string filename_right;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
extern const bool boolBatch;
extern const int LEFT;
extern const int RIGHT;
extern const std::string methodDenseOpticalFlow; //"lucasKanade_dense","rlof"
extern const float qualityCorner;
/* roi setting */
extern const int roiWidthOF;
extern const int roiHeightOF;
extern const int roiWidthYolo;
extern const int roiHeightYolo;
extern const int MoveThreshold;
extern const float epsironMove;
/* dense optical flow skip rate */
extern const int skipPixel;
/*if exchange template of Yolo */
extern const bool boolChange;

/* save date */
extern const std::string file_yolo_left;
extern const std::string file_yolo_right;
extern const std::string file_of_left;
extern const std::string file_of_right;

/* Declaration of function */
void yolo();
void denseOpticalFlow();



void yolo()
{
    /* constructor of YOLOPoseEstimator */
    //if (boolBatch) 
    YOLOPoseBatch yolo;
    //else YOLOPose yolo_left, yolo_right;
    Utility utyolo;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    posSaver_left.reserve(300);
    posSaver_right.reserve(300);
    if (queueFrame.empty())
    {
        while (queueFrame.empty())
        {
            if (!queueFrame.empty())
            {
                break;
            }
            //std::cout << "wait for images" << std::endl;
        }
    }
    /* frame is available */
    else
    {
        int counter = 1;
        int counterFinish = 0;
        while (true)
        {
            if (counterFinish == 10)
            {
                break;
            }
            /* frame can't be available */
            if (queueFrame.empty())
            {
                counterFinish++;
                /* waiting */
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            /* frame available -> start yolo pose estimation */
            else
            {
                counterFinish = 0;
                std::array<cv::Mat1b, 2> frames;
                int frameIndex;
                auto start = std::chrono::high_resolution_clock::now();
                utyolo.getImages(frames, frameIndex);
                //if (boolBatch)
                //{
                cv::Mat1b concatFrame;
                //std::cout << "frames[LEFT]:" << frames[LEFT].rows << "," << frames[LEFT].cols << ", frames[RIGHT]:" << frames[RIGHT].rows << "," << frames[RIGHT].cols << std::endl;
                if (frames[LEFT].rows > 0 && frames[RIGHT].rows > 0)
                {
                    cv::hconcat(frames[LEFT], frames[RIGHT], concatFrame);//concatenate 2 imgs horizontally
                    yolo.detect(concatFrame, frameIndex, counter, posSaver_left, posSaver_right, queueYoloOldImgSearch_left, queueYoloSearchRoi_left, queueYoloOldImgSearch_right, queueYoloSearchRoi_right);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                    std::cout << "Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
                }
            }
        }
    }
    std::cout << "YOLO" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    std::cout << "posSaver_left size=" << posSaver_left.size() << std::endl;
    utyolo.saveYolo(posSaver_left,file_yolo_left);
    std::cout << "*** RIGHT ***" << std::endl;
    std::cout << "posSaver_right size=" << posSaver_right.size() << std::endl;
    utyolo.saveYolo(posSaver_right,file_yolo_right);
}

void denseOpticalFlow()
{
    /* construction of class */
    OpticalFlow of;
    Utility utof;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_left; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver_right; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}
    posSaver_left.reserve(2000);
    posSaver_right.reserve(2000);

    int counterStart = 0;
    while (true)
    {
        if (counterStart == 2) break;
        if (!queueYoloOldImgSearch_left.empty() || !queueYoloOldImgSearch_right.empty())
        {
            counterStart++;
            int countIteration = 1;
            while (!queueYoloOldImgSearch_left.empty() || !queueYoloOldImgSearch_right.empty())
            {
                std::cout << countIteration << ":: remove yolo data ::" << std::endl;
                countIteration++;
                if (!queueYoloOldImgSearch_left.empty()) queueYoloOldImgSearch_left.pop();
                if (!queueYoloOldImgSearch_right.empty()) queueYoloOldImgSearch_right.pop();
            }
            while (!queueYoloSearchRoi_left.empty() || !queueYoloSearchRoi_right.empty())
            {
                if (!queueYoloSearchRoi_left.empty()) queueYoloSearchRoi_left.pop();
                if (!queueYoloSearchRoi_right.empty()) queueYoloSearchRoi_right.pop();
            }
        }
        if (!queueFrame.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    /* frame is available */
    int counter = 1;
    int counterFinish = 0;
    while (true)
    {
        if (counterFinish == 10)
        {
            break;
        }
        if (queueFrame.empty())
        {
            counterFinish++;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        /* frame available */
        else
        {
            std::cout << "start opticalflow tracking" << std::endl;
            /* get images from queue */
            std::array<cv::Mat1b,2> frames;
            int frameIndex;
            auto start = std::chrono::high_resolution_clock::now();
            utof.getImages(frames, frameIndex);
            cv::Mat1b frame_left = frames[0];
            cv::Mat1b frame_right = frames[1];
            std::thread thread_OF_left(&OpticalFlow::main, &of, std::ref(frame_left), std::ref(frameIndex), std::ref(posSaver_left), std::ref(queueYoloOldImgSearch_left), std::ref(queueYoloSearchRoi_left), std::ref(queueOFOldImgSearch_left), std::ref(queueOFSearchRoi_left),std::ref(queuePreviousMove_left),std::ref(queueTriangulation_left));
            std::thread thread_OF_right(&OpticalFlow::main, &of, std::ref(frame_right), std::ref(frameIndex), std::ref(posSaver_right), std::ref(queueYoloOldImgSearch_right), std::ref(queueYoloSearchRoi_right), std::ref(queueOFOldImgSearch_right), std::ref(queueOFSearchRoi_right), std::ref(queuePreviousMove_right), std::ref(queueTriangulation_right));
            std::cout << "both OF threads have started" << std::endl;
            thread_OF_left.join();
            thread_OF_right.join();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "***** posSaver_left size=" << posSaver_left.size() << ", posSaver_right size=" << posSaver_right.size() << "********" << std::endl;
            std::cout << " Time taken by OpticalFlow : " << duration.count() << " milliseconds" << std::endl;
        }
    }
    std::cout << "Optical Flow" << std::endl;
    std::cout << "*** LEFT ***" << std::endl;
    utof.save(posSaver_left,file_of_left);
    std::cout << "*** RIGHT ***" << std::endl;
    utof.save(posSaver_right,file_of_right);
}

int main()
{
    /* image inference */
    /*
    cv::Mat img = cv::imread("video/0019.jpg");
    cv::Mat1b imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    std::cout << img.size()<< std::endl;
    int counter = 1;
    yoloPoseEstimator.detect(imgGray,counter);
    */
    //constructor 
    Utility ut;
    Triangulation tri;

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
    /* start multiThread */
    std::thread threadYolo(yolo);
    std::thread threadOF(denseOpticalFlow);
    std::thread threadRemoveFrame(&Utility::removeFrame, ut);
    std::thread thread3d(&Triangulation::main, tri);
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
        ut.pushImg(frames, counter);
    }
    threadYolo.join();
    threadOF.join();
    threadRemoveFrame.join();
    thread3d.join();

    return 0;
}
