#pragma once

#ifndef PREDICTION_H
#define PREDICTION_H

#include "stdafx.h";
#include "global_parameters.h";


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

class Prediction
{
public:
    Prediction()
    {
        std::cout << "construct Prediction class" << std::endl;
    }

    void linearRegression(const std::vector<std::vector<int>>& data, std::vector<float>& result_x)
    {
        /*
         * linear regression
         * y = ax + b
         * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
         * b = mean_y - a*mean_x
         * Args:
         *   data(std::vector<std::vector<int>>&) : {{time,x,y,z},...}
         *   result_x(std::vector<float>&) : vector for saving result x
         *   result_x(std::vector<float>&) : vector for saving result z
         */
        const int NUM_POINTS_FOR_REGRESSION = 3;
        float sumt = 0, sumx = 0, sumtx = 0, sumtt = 0; // for calculating coefficients
        float mean_t, mean_x;
        int length = data.size(); // length of data

        for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
        {
            sumt += data[length - i][0];
            sumx += data[length - i][1];
            sumtx += data[length - i][0] * data[length - i][1];
            sumtt += data[length - i][0] * data[length - i][0];
        }
        std::cout << "Linear regression" << std::endl;
        mean_t = static_cast<float>(sumt) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
        mean_x = static_cast<float>(sumx) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
        float slope_x, intercept_x;
        if (std::abs(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t) > 0.0001)
        {
            slope_x = (sumtx - NUM_POINTS_FOR_REGRESSION * mean_t * mean_x) / (sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
            intercept_x = mean_x - slope_x * mean_t;
        }
        else
        {
            slope_x = 0;
            intercept_x = 0;
        }
        result_x = { slope_x, intercept_x };
        std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n"
            << std::endl;
    }

    void linearRegressionZ(const std::vector<std::vector<int>>& data, std::vector<float>& result_z)
    {
        /*
         * linear regression
         * y = ax + b
         * a = (sigma(xy)-n*mean_x*mean_y)/(sigma(x^2)-n*mean_x^2)
         * b = mean_y - a*mean_x
         * Args:
         *   data(std::vector<std::vector<int>>&) : {{time,x,y,z},...}
         *   result_x(std::vector<float>&) : vector for saving result x
         *   result_x(std::vector<float>&) : vector for saving result z
         */
        const int NUM_POINTS_FOR_REGRESSION = 3;
        float sumt = 0, sumz = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
        float mean_t, mean_z;
        int length = data.size(); // length of data

        for (int i = 1; i < NUM_POINTS_FOR_REGRESSION + 1; i++)
        {
            sumt += data[length - i][0];
            sumz += data[length - i][3];
            sumtt += data[length - i][0] * data[length - i][0];
            sumtz += data[length - i][0] * data[length - i][3];
        }
        std::cout << "Linear regression" << std::endl;
        mean_t = static_cast<float>(sumt) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
        mean_z = static_cast<float>(sumz) / static_cast<float>(NUM_POINTS_FOR_REGRESSION);
        float slope_z, intercept_z;
        if (std::abs(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t) > 0.0001)
        {
            slope_z = (sumtz - NUM_POINTS_FOR_REGRESSION * mean_t * mean_z) / (sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
            intercept_z = mean_z - slope_z * mean_t;
        }
        else
        {
            slope_z = 0;
            intercept_z = 0;
        }
        result_z = { slope_z, intercept_z };
        std::cout << "\n\nZ :: The best fit value of curve is : z = " << slope_z << " t + " << intercept_z << ".\n\n"
            << std::endl;
    }

    void curveFitting(const std::vector<std::vector<int>>& data, std::vector<float>& result)
    {
        /*
         * curve fitting with parabora
         * y = c*x^2+d*x+e
         *
         * Args:
         *   data(std::vector<std::vector<int>>) : {{time,x,y,z},...}
         *   result(std::vector<float>&) : vector for saving result
         */

         // argments analysis
        int length = data.size(); // length of data

        float time1, time2, time3;
        time1 = static_cast<float>(data[length - 3][0]);
        time2 = static_cast<float>(data[length - 2][0]);
        time3 = static_cast<float>(data[length - 1][0]);
        float det = time1 * time2 * (time1 - time2) + time1 * time3 * (time3 - time1) + time2 * time3 * (time2 - time3); // det
        float c, d, e;
        if (det == 0)
        {
            c = 0;
            d = 0;
            e = 0;
        }
        else
        {
            float coef11 = (time2 - time3) / det;
            float coef12 = (time3 - time1) / det;
            float coef13 = (time1 - time2) / det;
            float coef21 = (time3 * time3 - time2 * time2) / det;
            float coef22 = (time1 * time1 - time3 * time3) / det;
            float coef23 = (time2 * time2 - time1 * time1) / det;
            float coef31 = time2 * time3 * (time2 - time3) / det;
            float coef32 = time1 * time3 * (time3 - time1) / det;
            float coef33 = time1 * time2 * (time1 - time2) / det;
            // coefficients of parabola
            c = coef11 * data[length - 3][2] + coef12 * data[length - 2][2] + coef13 * data[length - 1][2];
            d = coef21 * data[length - 3][2] + coef22 * data[length - 2][2] + coef23 * data[length - 1][2];
            e = coef31 * data[length - 3][2] + coef32 * data[length - 2][2] + coef33 * data[length - 1][2];
        }

        result = { c, d, e };

        std::cout << "y = " << c << "x^2 + " << d << "x + " << e << std::endl;
    }

    void trajectoryPredict2D(std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<float>>& coefficientsX, std::vector<std::vector<float>>& coefficientsY, std::vector<int>& classesLatest)
    {
        int counterData = 0;
        for (const std::vector<std::vector<int>>& data : dataLeft)
        {
            /* get 3 and more time-step datas -> can predict trajectory */
            if (data.size() >= 3)
            {
                /* get latest 3 time-step data */
                std::vector<std::vector<int>> tempData;
                std::vector<float> coefX, coefY;
                // Use reverse iterators to access the last three elements
                auto rbegin = data.rbegin(); // Iterator to the last element
                auto rend = data.rend();     // Iterator one past the end
                /* Here data is latest to old -> but not have bad effect to trajectory prediction */
                for (auto it = rbegin; it != rend && std::distance(rend, it) < 3; ++it)
                {
                    const std::vector<int>& element = *it;
                    tempData.push_back(element);
                }
                /* trajectory prediction in X and Y */
                linearRegression(tempData, coefX);
                curveFitting(tempData, coefY);
                coefficientsX.push_back(coefX);
                coefficientsY.push_back(coefY);
            }
            /* get less than 3 data -> can't predict trajectory -> x : have to make the size equal to classesLatest
             *  -> add specific value to coefficientsX and coefficientsY, not change the size of classesLatest for maintaining size consisitency between dataLeft and data Right
             */
            else
            {
                coefficientsX.push_back({ 0.0, 0.0 });
                coefficientsY.push_back({ 0.0, 0.0, 0.0 });
                // classesLatest.erase(classesLatest.begin() + counterData); //erase class
                // counterData++;
                /* can't predict trajectory */
            }
        }
    }

    float calculateME(std::vector<float>& coefXLeft, std::vector<float>& coefYLeft, std::vector<float>& coefXRight, std::vector<float>& coefYRight)
    {
        float me = 0.0; // mean error
        for (int i = 0; i < coefYLeft.size(); i++)
        {
            me = me + (coefYLeft[i] - coefYRight[i]);
        }
        me = me + coefXLeft[0] - coefXRight[0];
        return me;
    }

    void dataMatching(std::vector<std::vector<float>>& coefficientsXLeft, std::vector<std::vector<float>>& coefficientsXRight,
        std::vector<std::vector<float>>& coefficientsYLeft, std::vector<std::vector<float>>& coefficientsYRight,
        std::vector<int>& classesLatestLeft, std::vector<int>& classesLatestRight,
        std::vector<std::vector<std::vector<int>>>& dataLeft, std::vector<std::vector<std::vector<int>>>& dataRight,
        std::vector<std::vector<std::vector<std::vector<int>>>>& dataFor3D)
    {
        float minVal = 20;
        int minIndexRight;
        if (!coefficientsXLeft.empty() && !coefficientsXRight.empty())
        {
            /* calculate metrics based on left img data */
            for (int i = 0; i < coefficientsXLeft.size(); i++)
            {
                /* deal with moving objects -> at least one coefficient should be more than 0 */
                if (coefficientsYLeft[i][0] != 0)
                {
                    for (int j = 0; j < coefficientsXRight.size(); j++)
                    {
                        /* deal with moving objects -> at least one coefficient should be more than 0 */
                        if (coefficientsYRight[i][0] != 0)
                        {
                            /* if class label is same */
                            if (classesLatestLeft[i] == classesLatestRight[j])
                            {
                                /* calculate metrics */
                                float me = calculateME(coefficientsXLeft[i], coefficientsYLeft[i], coefficientsXRight[j], coefficientsXRight[j]);
                                /* minimum value is updated */
                                if (me < minVal)
                                {
                                    minVal = me;
                                    minIndexRight = j; // most reliable matching index in Right img
                                }
                            }
                            /* maybe fixed objects detected */
                            else
                            {
                                /* ignore */
                            }
                        }
                    }
                    /* matcing object found */
                    if (minVal < 20)
                    {
                        dataFor3D.push_back({ dataLeft[i], dataRight[minIndexRight] }); // match objects and push_back to dataFor3D
                    }
                }
                /* maybe fixed objects detected */
                else
                {
                    /* ignore */
                }
            }
        }
    }

    void predict3DTargets(std::vector<std::vector<std::vector<std::vector<int>>>>& datasFor3D, std::vector<std::vector<int>>& targets3D)
    {
        int indexL, indexR, xLeft, xRight, yLeft, yRight;
        float fX = cameraMatrix.at<double>(0, 0);
        float fY = cameraMatrix.at<double>(1, 1);
        float fSkew = cameraMatrix.at<double>(0, 1);
        float oX = cameraMatrix.at<double>(0, 2);
        float oY = cameraMatrix.at<double>(1, 2);
        /* iteration of calculating 3d position for each matched objects */
        for (std::vector<std::vector<std::vector<int>>>& dataFor3D : datasFor3D)
        {
            std::vector<std::vector<int>> dataL = dataFor3D[0];
            std::vector<std::vector<int>> dataR = dataFor3D[1];
            /* get 3 and more time-step datas -> calculate 3D position */
            int numDataL = dataL.size();
            int numDataR = dataR.size();
            std::vector<std::vector<int>> data3D; //[mm]
            // calculate 3D position
            int counter = 0; // counter for counting matching frame index
            int counterIteration = 0;
            bool boolPredict = false; // if 3 datas are available
            while (counterIteration < std::min(numDataL, numDataR))
            {
                counterIteration++;
                if (counter > 3)
                {
                    boolPredict = true;
                    break;
                }
                indexL = dataL[numDataL - counter][0];
                indexR = dataR[numDataR - counter][0];
                if (indexL == indexR)
                {
                    xLeft = dataL[numDataL - counter][1];
                    xRight = dataR[numDataR - counter][1];
                    yLeft = dataL[numDataL - counter][2];
                    yRight = dataR[numDataR - counter][2];
                    int disparity = (int)(xLeft - xRight);
                    int X = (int)(BASELINE / disparity) * (xLeft - oX - (fSkew / fY) * (yLeft - oY));
                    int Y = (int)(BASELINE * (fX / fY) * (yLeft - oY) / disparity);
                    int Z = (int)(fX * BASELINE / disparity);
                    data3D.push_back({ indexL, X, Y, Z });
                    counter++;
                }
            }
            if (boolPredict)
            {
                /* trajectoryPrediction */
                std::vector<float> coefX, coefY, coefZ;
                linearRegression(data3D, coefX);
                linearRegressionZ(data3D, coefZ);
                curveFitting(data3D, coefY);
                /* objects move */
                if (coefZ[0] < 0) // moving forward to camera
                {
                    int frameTarget = (int)((TARGET_DEPTH - coefZ[1]) / coefZ[0]);
                    int xTarget = (int)(coefX[0] * frameTarget + coefX[1]);
                    int yTarget = (int)(coefY[0] * frameTarget * frameTarget + coefY[1] * frameTarget + coefY[2]);
                    targets3D.push_back({ frameTarget, xTarget, yTarget, TARGET_DEPTH }); // push_back target position
                    std::cout << "target is : ( frameTarget :  " << frameTarget << ", xTarget : " << xTarget << ", yTarget : " << yTarget << ", depthTarget : " << TARGET_DEPTH << std::endl;
                }
            }
        }
    }

};

#endif

