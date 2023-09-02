// trajectory_prediction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>


const int NUM_POINTS_FOR_REGRESSION = 3;
const int TARGET_DEPTH = 6;

//define trajectory regression coefficients
std::vector<float> coef_x;
std::vector<float> coef_y;
std::vector<float> coef_z;
// target position
std::vector<std::vector<int>> position_target;
//  of function
// Decleration of function
void curveFitting(const std::vector<std::vector<int>>&,std::vector<float>&); //curve fitting
void linearRegression(const std::vector<std::vector<int>>&,std::vector<float>&, std::vector<float>&); //linear regression

int main()
{


    std::vector<std::vector<int>> positions;
    positions.reserve(300); //reserve capacity for large inputs
    std::vector<std::vector<int>> p{ { 1,1,34,20 },{ 2,3,41,18 },{ 3,5,46,16 },{ 4,7,49,14 } };

    //int length = sizeof(data) / sizeof(data[0]); //if want to know the number of all elements, sizeof(data)/sizeof(data[0][0]) is good
    for (int i = 0; i < 4; i++)
    {
        positions.push_back(p[i]);
        // more than 2 is necessary for regression
        if (positions.size() >= 3)
        {
           linearRegression(positions,coef_x,coef_z);
           std::cout << "coefficients for x regression : ";
           for (int i = 0; i < coef_x.size(); i++)
           {
               std::cout << coef_x[i] << " ";
           }
           std::cout << std::endl;
           std::cout << "coefficients for z regression : ";
           for (int i = 0; i < coef_z.size(); i++)
           {
               std::cout << coef_z[i] << " ";
           }
           std::cout << std::endl;
           curveFitting(positions,coef_y);
           std::cout << "coefficients for y regression : ";
           for (int i = 0; i < coef_y.size(); i++)
           {
               std::cout << coef_y[i] << " ";
           }
           std::cout << std::endl;
           //calculate target position  
           if (coef_z[0] != 0.0)
           {
               int frame_target = (int)((TARGET_DEPTH - coef_z[1]) / coef_z[0]);
               int x_target = (int)(coef_x[0] * frame_target + coef_x[1]);
               int y_target = (int)(coef_y[0] * frame_target * frame_target + coef_y[1] * frame_target + coef_y[2]);
               std::vector<int> target_temp{ frame_target,x_target,y_target,TARGET_DEPTH };
               position_target.push_back(target_temp);

               std::cout << "target is : ( ";
               for (int i = 0; i < target_temp.size(); i++)
               {
                   std::cout << target_temp[i] << ",";
               }
               std::cout << " )" << std::endl;
           }
        }  
    }
}

void linearRegression(const std::vector<std::vector<int>>& data,std::vector<float>& result_x,std::vector<float>& result_z)
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
    float sumt = 0, sumx = 0, sumz = 0, sumtx = 0, sumtt = 0, sumtz = 0; // for calculating coefficients
    float mean_t, mean_x,mean_z;
    int length = data.size(); //length of data
    

    for (int i = 1; i < NUM_POINTS_FOR_REGRESSION+1; i++)
    {
        sumt += data[length-i][0];
        sumx += data[length-i][1];
        sumz += data[length - i][3];
        sumtx += data[length-i][0] * data[length-i][1];
        sumtt += data[length-i][0] * data[length-i][0];
        sumtz += data[length - i][0] * data[length - i][3];
    }
    std::cout << "Linear regression" << std::endl;
    mean_t = (float)sumt / (float)NUM_POINTS_FOR_REGRESSION;
    mean_x = (float)sumx / (float)NUM_POINTS_FOR_REGRESSION;
    mean_z = (float)sumz / (float)NUM_POINTS_FOR_REGRESSION;
    float slope_x = (float)(sumtx - NUM_POINTS_FOR_REGRESSION * mean_t * mean_x) / (float)(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
    float intercept_x = mean_x - slope_x * mean_t;
    float slope_z = (float)(sumtz - NUM_POINTS_FOR_REGRESSION * mean_t * mean_z) / (float)(sumtt - NUM_POINTS_FOR_REGRESSION * mean_t * mean_t);
    float intercept_z = mean_z - slope_z * mean_t;

    result_x = { slope_x,intercept_x };
    result_z = { slope_z,intercept_z };
    std::cout << "\n\nX :: The best fit value of curve is : x = " << slope_x << " t + " << intercept_x << ".\n\n" << std::endl;
    std::cout << "\n\nZ :: The best fit value of curve is : z = " << slope_z << " t + " << intercept_z << ".\n\n" << std::endl;
}


void curveFitting(const std::vector<std::vector<int>>& data,std::vector<float>& result)
{
    /*
    * curve fitting with parabora
    * y = c*x^2+d*x+e
    * 
    * Args:
    *   data(std::vector<std::vector<int>>) : {{time,x,y,z},...}
    *   result(std::vector<float>&) : vector for saving result 
    */

    //argments analysis
    int length = data.size(); //length of data

    float det, coef11, coef12, coef13, coef21, coef22, coef23, coef31, coef32, coef33; //coefficients matrix and det
    float time1, time2, time3;
    time1 = float(data[length-3][0]);
    time2 = float(data[length-2][0]);
    time3 = float(data[length-1][0]);
    det = time1 * time2 * (time1 - time2) + time1 * time3 * (time3 - time1) + time2 * time3 * (time2 - time3); //det
    float c, d, e;
    if (det == 0)
    {
        c = 0; d = 0; e = 0;
    }
    else
    {
        coef11 = (time2 - time3) / det;
        coef12 = (time3 - time1) / det;
        coef13 = (time1 - time2) / det;
        coef21 = (time3 * time3 - time2 * time2) / det;
        coef22 = (time1 * time1 - time3 * time3) / det;
        coef23 = (time2 * time2 - time1 * time1) / det;
        coef31 = time2 * time3 * (time2 - time3) / det;
        coef32 = time1 * time3 * (time3 - time1) / det;
        coef33 = time1 * time2 * (time1 - time2) / det;
        // coefficients of parabola
        c = coef11 * data[length-3][2] + coef12 * data[length - 2][2] + coef13 * data[length - 1][2];
        d = coef21 * data[length - 3][2] + coef22 * data[length - 2][2] + coef23 * data[length - 1][2];
        e = coef31 * data[length - 3][2] + coef32 * data[length - 2][2] + coef33 * data[length - 1][2];
    }

    result = { c,d,e };
    

    std::cout << "y = " << c << "x^2 + " << d << "x + " << e << std::endl;
}
