// multiThreadPractice.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <queue>

std::mutex mtx_; // define mutex
uint32_t count_;

/*
 * template for multi treading process for real time tracking
 *
 *
 *
 */

std::queue<std::vector<int>> queue_frame;                     // queue for frame
std::queue<int> queue_frame_index;                            // queue for frame index
std::queue<std::vector<int>> queue_yolo_template;             // queue for yolo template : for real cv::Mat type
std::queue<std::vector<int>> queue_yolo_bbox;                 // queue for yolo bbox
std::queue<std::vector<int>> queue_templateMatching_template; // queue for templateMatching template img : for real cv::Mat
std::queue<std::vector<int>> queue_templateMatching_bbox;     // queue for templateMatching bbox

void templateMatching(); //, std::queue<std::vector<int>>&, std::queue<int>&, std::queue<std::vector<int>>&, std::queue<std::vector<int>>&, std::queue<std::vector<int>>&, std::queue<std::vector<int>>&);
void yolo();      // std::queue<std::vector<int>>&, std::queue<int>&, std::queue<std::vector<int>>&, std::queue<std::vector<int>>&);
void popFrame(std::vector<int>&, int&);
void getYolobbox(std::vector<int>&);
void pushYolobbox(std::vector<int>&);
void processYolo(std::queue<std::vector<int>>&, std::queue<int>&, std::vector<std::vector<int>>&, int, int);
void showData(std::vector<std::vector<int>>&);

void templateMatching()
{
    /*multi threading :: template matching for real time tracking
     * Assumption : in this function, after first yolo detection has succeeded
     *
     * Args:
     *
     *   data : {frame_index, center_x, center_y}
     *	queue_frame : input img frame :: img
     *	queue_frame_index : input img frame index
     *	queue_yolo_bbox : queue for storing yolo estimation data :: {frame index, bbox}
     *	queue_yolo_template : queue for yolo template img
     *	queue_templateMatching_bbox : queue for storing templateMatching estimating data :: {frame_index, bbox}
     *	queue_templateMatching_template : queue gor templateMatching template img
     *
     */

    std::vector<std::vector<int>> trackingData; // queue for saving tracking data
    trackingData.reserve(10000);                 // reserve large data

    int counter_yolo = 0;
    int frame_idx = 0;
    std::this_thread::sleep_for(std::chrono::seconds(2)); // wait for 10 sec until camera starts
    // frame data is there
    while (!queue_frame.empty())
    {
        std::vector<int> newFrame; // get new data
        int frameIndex;        // get current frame data
        popFrame(newFrame, frameIndex);//get new frame
        //std::cout << "queue_yolo_template.empty() : " << queue_yolo_template.empty() << std::endl;
        // if yolo new data avilable
        if (!queue_yolo_template.empty())
        {
            std::vector<int> bboxYolo = queue_yolo_bbox.front();            // get new yolodata -> for new template
            std::vector<int> templateImgYolo = queue_yolo_template.front(); // get new yolo template img
            queue_yolo_bbox.pop();                                           // delete yolo bbox : num data yolo 0
            queue_yolo_template.pop();                                       // delete yolo template : num data yolo 0

            // update roi and start new estimation
            //  tracking of template matching was successful
            if (!queue_templateMatching_bbox.empty())
            {
                //update queue_yolo_bbox to latest position
                //queue_yolo_bbox.pop(); 
                pushYolobbox(trackingData.back());// pass latest data  : num data yolo 1 {frame_index, center_x, center_y}

                std::vector<int> bboxTemplateMatching = queue_templateMatching_bbox.front(); // get latest data
                int centerX = (int)((bboxTemplateMatching[1] + bboxTemplateMatching[3]) / 2);
                int centerY = (int)((bboxTemplateMatching[2] + bboxTemplateMatching[4]) / 2);
                queue_templateMatching_bbox.pop();                // delete templateMatching data
                queue_templateMatching_template.pop();            // delte templateMatching template img (because template img is updated by yolo)

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                /*
                 * start new template matching process : use template img yolo and limit search area with {center_x,center_y}
                 */
                std::vector<int> newBbox;
                newBbox.push_back(frameIndex);
                for (int i = 0; i < newFrame.size(); i++)
                {
                    newBbox.push_back(newFrame[i]);
                }
                queue_templateMatching_bbox.push(newBbox);
                queue_templateMatching_template.push(newFrame);
                //add latest data to trackingData
                int center_x = (int)((newBbox[1] + newBbox[3]) / 2);
                int center_y = (int)((newBbox[2] + newBbox[4]) / 2);
                trackingData.push_back({ frameIndex, center_x, center_y });
                //counter_yolo++;
            }
            // template matching tracking was failed -> restart based on yolo detection
            else
            {
                int centerX = (int)((bboxYolo[0] + bboxYolo[2]) / 2);
                int centerY = (int)((bboxYolo[1] + bboxYolo[3]) / 2);

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                /*
                 * start new template matching proces : use templateImg_yolo and limit search area with bbox_yolo
                 */
                std::vector<int> newBbox;
                newBbox.push_back(frameIndex);
                for (int i = 0; i < newFrame.size(); i++)
                {
                    newBbox.push_back(newFrame[i]);
                }
                queue_templateMatching_bbox.push(newBbox);
                queue_templateMatching_template.push(newFrame);
                //add latest data to trackingData
                int center_x = (int)((newBbox[1] + newBbox[3]) / 2);
                int center_y = (int)((newBbox[2] + newBbox[4]) / 2);
                trackingData.push_back({ frameIndex,centerX,centerY });
                //counter_yolo++;
            }
        }
        // can't get yolo data
        else
        {
            // tracking of template matching was successful
            if (!queue_templateMatching_bbox.empty())
            {
                //update queue_yolo_bbox to latest position
                if (!queue_yolo_bbox.empty())
                {
                    queue_yolo_bbox.pop();
                }
                queue_yolo_bbox.push(trackingData.back()); // pass latest data  : num data yolo 1 {frame_index, center_x, center_y}
                std::vector<int> bbox_templateMatching = queue_templateMatching_bbox.front(); // get latest data
                std::vector<int> templateImg = queue_templateMatching_template.front(); // get latest template img
                queue_templateMatching_bbox.pop();                // delete templateMatching data
                queue_templateMatching_template.pop();            // delte templateMatching template img (because template img is updated by yolo)

                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                /*
                 * start new template matching process : use templateImg  and limit search area with {center_x,center_y}
                 */
                std::vector<int> newBbox;
                newBbox.push_back(frameIndex);
                for (int i = 0; i < newFrame.size(); i++)
                {
                    newBbox.push_back(newFrame[i]);
                }
                queue_templateMatching_bbox.push(newBbox);
                queue_templateMatching_template.push(newFrame);
                //add latest data to trackingData
                int center_x = (int)((newBbox[1] + newBbox[3]) / 2);
                int center_y = (int)((newBbox[2] + newBbox[4]) / 2);
                trackingData.push_back({ frameIndex, center_x, center_y });
            }
            // template matching tracking was failed -> restart based on yolo detection
            else
            {
                //no process
            }
        }
        std::cout << "Template Matching current frame :: " << frameIndex << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        //std::cout << "img available? : " << queue_frame.empty() << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "template matching :: " << std::endl;
    std::cout << "tracking data : " << trackingData.size() << " , " << trackingData[0].size() << std::endl;

}

void yolo()
{
    /*multi threading :: template matching for real time tracking
     * Assumption : in this function, after first yolo detection has succeeded
     *
     * Args:
     *	queue_frame : input img frame :: img
     *	queue_frame_index : input img frame index
     *	queue_yolo_bbox : queue for storing yolo estimation data :: {frame index, bbox}
     *	queue_yolo_template : queue for yolo template img
     *
     */

    std::vector<std::vector<int>> trackingData; // queue for saving tracking data
    trackingData.reserve(1000);                 // reserve large data
    int counter_yolo = 0;
    int frame_idx = 0;
    std::this_thread::sleep_for(std::chrono::seconds(2)); // wait for 10 sec until camera starts
    // frame data is there
    while (!queue_frame.empty())
    {
        // if tracking was successful -> limit the detection area
        if (!queue_yolo_bbox.empty())
        {
            std::vector<int> bbox; // get new yolodata
            getYolobbox(bbox);
            int x_candidate = bbox[1];                       // candidate center
            int y_candidate = bbox[2];                       // candidate center
            // start new yolo process
            processYolo(queue_frame, queue_frame_index, trackingData, x_candidate, y_candidate);
        }
        // if tracking was failed or tracking hasn't started
        else
        {
            // start new yolo process
            processYolo(queue_frame, queue_frame_index, trackingData, -1, -1);
        }
    }
    std::cout << "YOLO :: " << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "tracking data size : " << trackingData.size() << " , " << trackingData[0].size() << std::endl;
    //showData(trackingData);

}

void processYolo(std::queue<std::vector<int>>& frameQueue, std::queue<int>& frameIndexQueue, std::vector<std::vector<int>>& trackingData, int x_candidate = -1, int y_candidate = -1)
{
    std::vector<int> newFrame; // get new data
    int frameIndex;        // get current frame data
    popFrame(newFrame, frameIndex);
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait for 10 msec
    /*
     * yolo inference, and compare the results to (x_candidate, y_candidate)
     */
    int centerX = (int)((newFrame[0] + newFrame[2]) / 2);
    int centerY = (int)((newFrame[1] + newFrame[3]) / 2);
    //std::unique_lock<std::mutex> lock(mtx_);
    queue_yolo_bbox.push(newFrame);
    queue_yolo_template.push(newFrame);
    trackingData.push_back({ frameIndex,centerX,centerY });
    std::cout << "YOLO :: current frame index is " << frameIndex << std::endl;
}

void pushYolobbox(std::vector<int>& data)
{
    std::unique_lock<std::mutex> lock(mtx_); // Lock the mutex
    queue_yolo_bbox.push(data);
}

void getYolobbox(std::vector<int>& bbox)
{
    std::unique_lock<std::mutex> lock(mtx_); // Lock the mutex
    bbox = queue_yolo_bbox.front(); // get new yolodata
}

void popFrame(std::vector<int>& newFrame, int& frameIndex)
{
    std::unique_lock<std::mutex> lock(mtx_); // Lock the mutex
    newFrame = queue_frame.front();
    frameIndex = queue_frame_index.front();

    // Access the queue safely
    queue_frame.pop();
    queue_frame_index.pop();
}

void showData(std::vector<std::vector<int>>& data)
{
    //std::unique_lock<std::mutex> lock(mtx_);
    for (int j = 0; j < data.size(); j++)
    {
        std::cout << j << " :: " << "{ ";
        for (int k = 0; k < data.size(); k++)
        {
            std::cout << data[j][k] << " ";
        }
        std::cout << "}" << std::endl;
    }
}

int main()
{
    // define class

    // multi thread code
    std::thread thread_yolo(yolo);                               //,queue_frame,queue_frame_index, queue_yolo_bbox,queue_yolo_template);
    std::thread thread_templateMatching(templateMatching); // , queue_frame, queue_frame_index, queue_yolo_bbox, queue_yolo_template, queue_templateMatching_bbox, queue_templateMatching_template);

    std::vector<int> img;
    int index;
    for (int i = 1; i < 101; i++)
    {
        img = { i, i, i, i }; // read image
        index = i;    // get frame index
        std::cout << i << std::endl;
        queue_frame.push({ i,i,i,i });            // push img to queue
        queue_frame_index.push(i);    // push img index to queue
    }
   
    thread_yolo.join();
    thread_templateMatching.join();

    return 0;
}
