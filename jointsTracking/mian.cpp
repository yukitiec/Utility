
#include "stdafx.h"
#include "yolopose.h"
#include "opticalflow.h"
#include "utility.h"
#include "global_parameters.h"

extern std::queue<cv::Mat1b> queueFrame;
extern std::queue<int> queueFrameIndex;
/* yolo and optical flow */
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueYoloOldImgSearch;      // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2d>>> queueYoloSearchRoi;        // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Mat1b>>> queueOFOldImgSearch;        // queue for old image for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<cv::Rect2d>>> queueOFSearchRoi;          // queue for search roi for optical flow. vector size is [num human,6]
extern std::queue<std::vector<std::vector<std::vector<float>>>> queuePreviousMove; // queue for saving previous ROI movement : [num human,6 joints, 2D movements]

/* constant valude definition */
extern const std::string filename;
extern const bool save;
extern const bool boolSparse;
extern const bool boolGray;
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

/* Declaration of function */
void yolo();
void getImages(cv::Mat1b&, int&);
void denseOpticalFlow();

void pushImg(cv::Mat1b&, int&);
void removeFrame();


void yolo()
{
    /* constructor of YOLOPoseEstimator */
    YOLOPose yoloPoseEstimator;
    Utility utyolo;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,joints,element] :{frameIndex,xCenter,yCenter}
    if (queueFrame.empty())
    {
        while (queueFrame.empty())
        {
            if (!queueFrame.empty())
            {
                break;
            }
            std::cout << "wait for images" << std::endl;
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
                cv::Mat1b frame;
                int frameIndex;
                auto start = std::chrono::high_resolution_clock::now();
                utyolo.getImages(frame, frameIndex);
                yoloPoseEstimator.detect(frame, frameIndex, counter, posSaver);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout << " Time taken by YOLO detection : " << duration.count() << " milliseconds" << std::endl;
            }
        }
    }
    utyolo.saveYolo(posSaver);
}

void denseOpticalFlow()
{
    /* construction of class */
    OpticalFlow of;
    Utility utof;
    /* prepare storage */
    std::vector<std::vector<std::vector<std::vector<int>>>> posSaver; //[sequence,numHuman,numJoints,position] :{frameIndex,xCenter,yCenter}

    int counterStart = 0;
    while (true)
    {
        if (counterStart == 2)
        {
            break;
        }
        if (!queueYoloOldImgSearch.empty())
        {
            counterStart++;
            int countIteration = 1;
            while (!queueYoloOldImgSearch.empty())
            {
                std::cout << countIteration << " :: remove yolo data" << std::endl;
                countIteration++;
                queueYoloOldImgSearch.pop();
            }
            while (!queueYoloSearchRoi.empty())
            {
                queueYoloSearchRoi.pop();
            }
        }
        cv::Mat1b frame;
        int frameIndex;
        if (!queueFrame.empty())
        {
            // getImages(frame, frameIndex);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::cout << "wait for yolo detection" << std::endl;
        }
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
            std::cout << "start optical flow tracking" << std::endl;
            /* get images from queue */
            cv::Mat1b frame;
            int frameIndex;
            auto start = std::chrono::high_resolution_clock::now();
            utof.getImages(frame, frameIndex);
            /* optical flow process for each joints */
            std::vector<std::vector<cv::Mat1b>> previousImg;           //[number of human,0~6,cv::Mat1b]
            std::vector<std::vector<cv::Rect2d>> searchRoi;            //[number of human,6,cv::Rect2d], if tracker was failed, roi.x == -1
            std::vector<std::vector<std::vector<float>>> previousMove; // [number of human,6,movement in x and y] ROI movement of each joint
            of.getPreviousData(previousImg, searchRoi, previousMove);
            std::cout << "finish getting previous data " << std::endl;
            /* start optical flow process */
            /* for every human */
            std::vector<std::vector<cv::Mat1b>> updatedImgHuman;
            std::vector<std::vector<cv::Rect2d>> updatedSearchRoiHuman;
            std::vector<std::vector<std::vector<float>>> updatedMoveDists;
            std::vector<std::vector<std::vector<int>>> updatedPositionsHuman;
            for (int i = 0; i < searchRoi.size(); i++)
            {

                /* for every joints */
                std::vector<cv::Mat1b> updatedImgJoints;
                std::vector<cv::Rect2d> updatedSearchRoi;
                std::vector<std::vector<float>> moveJoints; // roi movement
                std::vector<std::vector<int>> updatedPositions;
                std::vector<int> updatedPosLeftShoulder, updatedPosRightShoulder, updatedPosLeftElbow, updatedPosRightElbow, updatedPosLeftWrist, updatedPosRightWrist;
                cv::Mat1b updatedImgLeftShoulder, updatedImgRightShoulder, updatedImgLeftElbow, updatedImgRightElbow, updatedImgLeftWrist, updatedImgRightWrist;
                cv::Rect2d updatedSearchRoiLeftShoulder, updatedSearchRoiRightShoulder, updatedSearchRoiLeftElbow, updatedSearchRoiRightElbow, updatedSearchRoiLeftWrist, updatedSearchRoiRightWrist;
                std::vector<float> moveLS, moveRS, moveLE, moveRE, moveLW, moveRW;
                bool boolLeftShoulder = false;
                bool boolRightShoulder = false;
                bool boolLeftElbow = false;
                bool boolRightElbow = false;
                bool boolLeftWrist = false;
                bool boolRightWrist = false;
                std::vector<std::thread> threadJoints;
                /* start optical flow process for each joints */
                /* left shoulder */
                int counterTracker = 0;
                if (searchRoi[i][0].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][0]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftShoulder), std::ref(updatedSearchRoiLeftShoulder), std::ref(moveLS), std::ref(updatedPosLeftShoulder));
                    boolLeftShoulder = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftShoulder.x = -1;
                    updatedSearchRoiLeftShoulder.y = -1;
                    updatedSearchRoiLeftShoulder.width = -1;
                    updatedSearchRoiLeftShoulder.height = -1;
                    updatedPosLeftShoulder = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right shoulder */
                if (searchRoi[i][1].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][1]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightShoulder), std::ref(updatedSearchRoiRightShoulder), std::ref(moveRS), std::ref(updatedPosRightShoulder));
                    boolRightShoulder = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightShoulder.x = -1;
                    updatedSearchRoiRightShoulder.y = -1;
                    updatedSearchRoiRightShoulder.width = -1;
                    updatedSearchRoiRightShoulder.height = -1;
                    updatedPosRightShoulder = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* left elbow */
                if (searchRoi[i][2].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][2]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftElbow), std::ref(updatedSearchRoiLeftElbow), std::ref(moveLE), std::ref(updatedPosLeftElbow));
                    boolLeftElbow = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftElbow.x = -1;
                    updatedSearchRoiLeftElbow.y = -1;
                    updatedSearchRoiLeftElbow.width = -1;
                    updatedSearchRoiLeftElbow.height = -1;
                    updatedPosLeftElbow = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right elbow */
                if (searchRoi[i][3].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][3]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightElbow), std::ref(updatedSearchRoiRightElbow), std::ref(moveRE), std::ref(updatedPosRightElbow));
                    boolRightElbow = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightElbow.x = -1;
                    updatedSearchRoiRightElbow.y = -1;
                    updatedSearchRoiRightElbow.width = -1;
                    updatedSearchRoiRightElbow.height = -1;
                    updatedPosRightElbow = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* left wrist */
                if (searchRoi[i][4].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][4]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgLeftWrist), std::ref(updatedSearchRoiLeftWrist), std::ref(moveLW), std::ref(updatedPosLeftWrist));
                    boolLeftWrist = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiLeftWrist.x = -1;
                    updatedSearchRoiLeftWrist.y = -1;
                    updatedSearchRoiLeftWrist.width = -1;
                    updatedSearchRoiLeftWrist.height = -1;
                    updatedPosLeftWrist = std::vector<int>{ frameIndex, -1, -1 };
                }
                /* right wrist */
                if (searchRoi[i][5].x > 0)
                {
                    threadJoints.emplace_back(&OpticalFlow::opticalFlow, &of, std::ref(frame), std::ref(frameIndex),
                        std::ref(previousImg[i][counterTracker]), std::ref(searchRoi[i][5]), std::ref(previousMove[i][counterTracker]),
                        std::ref(updatedImgRightWrist), std::ref(updatedSearchRoiRightWrist), std::ref(moveRW), std::ref(updatedPosRightWrist));
                    boolRightWrist = true;
                    counterTracker++;
                }
                else
                {
                    updatedSearchRoiRightWrist.x = -1;
                    updatedSearchRoiRightWrist.y = -1;
                    updatedSearchRoiRightWrist.width = -1;
                    updatedSearchRoiRightWrist.height = -1;
                    updatedPosRightWrist = std::vector<int>{ frameIndex, -1, -1 };
                }
                std::cout << "all threads have started" << std::endl;
                /* wait for all thread has finished */
                int counterThread = 0;
                if (!threadJoints.empty())
                {
                    for (std::thread& thread : threadJoints)
                    {
                        thread.join();
                        counterThread++;
                    }
                    std::cout << counterThread << " threads have finished!" << std::endl;
                }
                else
                {
                    std::cout << "no thread has started" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                /* combine all data and push data to queue */
                /* search roi */
                updatedSearchRoi.push_back(updatedSearchRoiLeftShoulder);
                updatedSearchRoi.push_back(updatedSearchRoiRightShoulder);
                updatedSearchRoi.push_back(updatedSearchRoiLeftElbow);
                updatedSearchRoi.push_back(updatedSearchRoiRightElbow);
                updatedSearchRoi.push_back(updatedSearchRoiLeftWrist);
                updatedSearchRoi.push_back(updatedSearchRoiRightWrist);
                updatedPositions.push_back(updatedPosLeftShoulder);
                updatedPositions.push_back(updatedPosRightShoulder);
                updatedPositions.push_back(updatedPosLeftElbow);
                updatedPositions.push_back(updatedPosRightElbow);
                updatedPositions.push_back(updatedPosLeftWrist);
                updatedPositions.push_back(updatedPosRightWrist);
                /* updated img */
                /* left shoulder*/
                if (updatedSearchRoi[0].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftShoulder);
                    moveJoints.push_back(moveLS);
                }
                /* right shoulder*/
                if (updatedSearchRoi[1].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightShoulder);
                    moveJoints.push_back(moveRS);
                }
                /*left elbow*/
                if (updatedSearchRoi[2].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftElbow);
                    moveJoints.push_back(moveLE);
                }
                /*right elbow */
                if (updatedSearchRoi[3].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightElbow);
                    moveJoints.push_back(moveRE);
                }
                /* left wrist*/
                if (updatedSearchRoi[4].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgLeftWrist);
                    moveJoints.push_back(moveLW);
                }
                /*right wrist*/
                if (updatedSearchRoi[5].x > 0)
                {
                    updatedImgJoints.push_back(updatedImgRightWrist);
                    moveJoints.push_back(moveRW);
                }
                /* combine all data for one human */
                updatedSearchRoiHuman.push_back(updatedSearchRoi);
                updatedPositionsHuman.push_back(updatedPositions);
                if (!updatedImgJoints.empty())
                {
                    updatedImgHuman.push_back(updatedImgJoints);
                    updatedMoveDists.push_back(moveJoints);
                }
            }
            /* push updated data to queue */
            queueOFSearchRoi.push(updatedSearchRoiHuman);
            if (!updatedImgHuman.empty())
            {
                queueOFOldImgSearch.push(updatedImgHuman);
                queuePreviousMove.push(updatedMoveDists);
            }
            /* arrange posSaver */
            if (!posSaver.empty())
            {
                std::vector<std::vector<std::vector<int>>> all; //all human data
                // for each human
                for (int i = 0; i < updatedPositionsHuman.size();i++)
                {
                    std::vector<std::vector<int>> tempHuman;
                    /* same human */
                    if (posSaver[posSaver.size() - 1].size() > i)
                    {
                        // for each joint
                        for (int j = 0; j < updatedPositionsHuman[i].size();j++)
                        {
                            // detected
                            if (updatedPositionsHuman[i][j][1] != -1)
                            {
                                tempHuman.push_back(updatedPositionsHuman[i][j]);
                            }
                            // not detected
                            else
                            {
                                // already detected
                                if (posSaver[posSaver.size() - 1][i][j][1] != -1)
                                {
                                    tempHuman.push_back(posSaver[posSaver.size() - 1][i][j]); //adapt last detection
                                }
                                // not detected yet
                                else
                                {
                                    tempHuman.push_back(updatedPositionsHuman[i][j]); //-1
                                }
                            }
                        }
                    }
                    //new human
                    else
                    {
                        tempHuman = updatedPositionsHuman[i];
                    }
                    all.push_back(tempHuman); //push human data
                }
                posSaver.push_back(all);
            }
            // first detection
            else
            {
                posSaver.push_back(updatedPositionsHuman);
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << " Time taken by OpticalFlow : " << duration.count() << " milliseconds" << std::endl;
        }
    }
    utof.saveOF(posSaver);
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

    /* video inference */
    const std::string filename = "yolotest.mp4";
    cv::VideoCapture capture(filename);
    if (!capture.isOpened())
    {
        // error in opening the video input
        std::cerr << "Unable to open file!" << std::endl;
        return 0;
    }
    int counter = 0;
    /* start multiThread */
    std::thread threadYolo(yolo);
    std::thread threadOF(denseOpticalFlow);
    std::thread threadRemoveFrame(&Utility::removeFrame,ut);
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
        ut.pushImg(frameGray, counter);
    }
    threadYolo.join();
    threadOF.join();
    threadRemoveFrame.join();

    return 0;
}
