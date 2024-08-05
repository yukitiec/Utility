#include "stdafx.h"
#include "ik.h"

int main()
{
    const double pi = 3.14159265358979323846;
    IK ur_ik;
    std::vector<double> joints_cur{ -0.10,-0.18,0.71,-1.1,-1.6 };
    //std::vector<double> joints_cur{ 0,0,0,0,0,0 };
    std::vector<double> joints{ 131.1/180*pi,-107.16/180*pi,4.43/180*pi,-81.9/180*pi,37.2/180*pi,50.71/180*pi };
    std::vector<double> pose_target{-0.109532,-0.196164,0.812061,-1.25234,-1.63126,2.04242}, joints_target;
    double pose_matrix[4][4];
    /*
    ur_ik.fk_best(joints, pose_matrix, pose_target);
    std::cout << "pose_matrix2=" << std::endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << pose_matrix[i][j] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "pose2=";
    for (double angle : pose_target)
        std::cout << " " << angle;
    std::cout << std::endl;*/
    auto start = std::chrono::steady_clock::now();
    bool bool_success = ur_ik.ik(pose_target, joints_cur, joints_target);//9.2 usec, 5usec
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (end - start);

    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    std::cout << "success=" << bool_success << std::endl;
    if (bool_success) {
        std::cout << "joints_target=";
        for (double joint : joints_target)
            std::cout << " " << joint;
        std::cout << std::endl;
        std::cout << "det(J)=" << ur_ik.det_j() << std::endl;
        ur_ik.fk_best(joints_target, pose_matrix, pose_target);
        std::cout << "pose=";
        for (double angle : pose_target)
            std::cout << " " << angle;
    }


}
