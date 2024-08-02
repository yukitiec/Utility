// analysis_eyeToHand.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "stdafx.h"
#include "triangulation.h"
#include "chess2camera.h"

//chessboard pose in tcp coordinate.
const cv::Mat H_tcp2chess = (cv::Mat_<double>(4, 4) <<
    -1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 3.0,
    0.0, 0.0, 0.0, 1.0);

std::vector<std::vector<double>> readCSV(std::string& file_name) {
    /**
    * @brief read CSV file and save data into a vector.
    * @param[in] file path
    */

    // Create a 2D vector to store the CSV data
    std::vector<std::vector<double>> data;

    // Open the CSV file
    std::ifstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return std::vector<std::vector<double>>{};
    }

    std::string line;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Split the line by commas and store the values in the row vector
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value)); // Convert the string to double
        }

        // Add the row vector to the data vector
        data.push_back(row);
    }

    // Close the file
    file.close();

    // Output the data to verify
    for (const auto& row : data) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return data;
}

cv::Mat createHomogeneousMatrix(std::vector<double>& pose) {
    /**
    * @brief create homogeneous transformation matrix from std::vector<double> which is (x,y,z,nx,ny,nz)
    */
    const double PI = 3.14159265358979323846;
    // Create rotation vector
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5] );
    //normalize between 0 and pi
    double norm_rvec = cv::norm(rvec);
    if (norm_rvec > PI) {//0<=theta<=PI
        double theta = std::fmod(norm_rvec, 2.0 * PI);
        if (theta <= PI) {
            rvec = theta / norm_rvec * rvec;
        }
        else if (theta > PI) {//PI<theta<=2*PI
            theta = theta - 2.0 * PI;
            rvec = theta / norm_rvec * rvec;
        }
    }

    // Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // Create translation vector
    cv::Mat t = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);

    // Create homogeneous transformation matrix
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3))); // Copy rotation matrix to the upper-left 3x3 part
    t.copyTo(T(cv::Rect(3, 0, 1, 3))); // Copy translation vector to the first 3 elements of the last column

    return T;
}


int main()
{

    std::string rootDir = "C:/Users/kawaw/cpp/eyeToHand/eyeToHand/data/camera_calibration";
    std::string file_intrinsic_left = "camera0_intrinsics.dat";
    file_intrinsic_left = "/" + file_intrinsic_left;
    file_intrinsic_left = rootDir + file_intrinsic_left;
    std::string file_intrinsic_right = "camera1_intrinsics.dat";
    file_intrinsic_right = "/" + file_intrinsic_right;
    file_intrinsic_right = rootDir + file_intrinsic_right;
    std::string file_extrinsic_left = "camera0_rot_trans.dat";
    file_extrinsic_left = "/" + file_extrinsic_left;
    file_extrinsic_left = rootDir + file_extrinsic_left;
    std::string file_extrinsic_right = "camera1_rot_trans.dat";
    file_extrinsic_right = "/" + file_extrinsic_right;
    file_extrinsic_right = rootDir + file_extrinsic_right;
    
    //load camera parameters
    Triangulation tri(file_intrinsic_left, file_intrinsic_right, file_extrinsic_left, file_extrinsic_right);//Triangulation

    //construct chess2camera
    Chess2camera chess2camera(rootDir);
    
    //load robot 3D pose.
    std::string rootData = "C:/Users/kawaw/cpp/eyeToHand/eyeToHand/data/eyeHand";
    std::string rootJoints = rootData + "/csv";
    std::string file_joints = "joints.csv";
    file_joints = "/" + file_joints;
    file_joints = rootJoints + file_joints;
    std::vector<std::vector<double>> pose_robot = readCSV(file_joints);//(n_data,{x,y,z,nx,ny,nz}) -> homogeneous matrix
    int num_pose = pose_robot.size();
    std::vector<cv::Mat> Hs_base2tcp(num_pose);
    //get transform matrix.
    for (int idx = 0; idx < num_pose; idx++) {
        pose_robot[idx][0] *= 1000.0;//x [m] -> [mm]
        pose_robot[idx][1] *= 1000.0;//y [m] -> [mm]
        pose_robot[idx][2] *= 1000.0;//z [m] -> [mm]
        Hs_base2tcp[idx] = createHomogeneousMatrix(pose_robot[idx]);
    }

    //convert from tcp (tool center position) to chessboard coordinates.
    std::vector<cv::Mat> Hs_base2chess(num_pose);
    for (int idx = 0; idx < num_pose; idx++)
        Hs_base2chess[idx] = Hs_base2tcp[idx] * H_tcp2chess;


    //2.corner detection and show display to check whether we'll reverse the points.
    chess2camera.main(rootData);
    //convert translation and rotation vector to homogeneous matrix
    std::vector<cv::Mat> Hs_camera2chess(num_pose);
    for (int idx = 0; idx < num_pose; idx++) {
        std::vector<double> pose(6);
        for (int j = 0; j < 3; j++)
            pose[j] = chess2camera.tvecs_left[idx].at<double>(j);
        for (int j = 0; j < 3; j++)
            pose[j+3] = chess2camera.rvecs_left[idx].at<double>(j);
        Hs_camera2chess[idx] = createHomogeneousMatrix(pose);
    }

    //cv::calibrateHandEye() -> get transformation matrix.
    std::vector<cv::Mat> R_base2chess(num_pose), t_base2chess(num_pose), R_chess2camera(num_pose), t_chess2camera(num_pose);
    cv::Mat R_base2cam, t_base2cam,H_chess2camera;
    for (int idx = 0; idx < num_pose; idx++) {
        //chess pose in the base frame.
        R_base2chess[idx] = Hs_base2chess[idx](cv::Rect(0, 0, 3, 3));
        t_base2chess[idx] = Hs_base2chess[idx](cv::Rect(3, 0, 1, 3));
        //invert H_camera2chess
        cv::invert(Hs_camera2chess[idx], H_chess2camera);
        R_chess2camera[idx] = H_chess2camera(cv::Rect(0, 0, 3, 3));
        t_chess2camera[idx] = H_chess2camera(cv::Rect(3, 0, 1, 3));
    }

    //calibrateHandEye
    cv::calibrateHandEye(R_base2chess, t_base2chess, R_chess2camera, t_chess2camera, R_base2cam, t_base2cam);
    cv::Mat H_base2cam = cv::Mat::zeros(4, 4, CV_64F);//camera pose in a robot base frame.
    H_base2cam.at<double>(3, 3) = 1.0;
    R_base2cam.copyTo(H_base2cam(cv::Rect(0, 0, 3, 3)));
    t_base2cam.copyTo(H_base2cam(cv::Rect(3, 0, 1, 3)));

    //save transformation matrix
    // Open a CSV file for writing
    std::ofstream file_transform("transform_camera2base.csv");
    if (!file_transform.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return -1;
    }
    // Write the matrix data to the CSV file
    for (int row = 0; row < H_base2cam.rows; ++row) {
        for (int col = 0; col < H_base2cam.cols; ++col) {
            file_transform << H_base2cam.at<double>(row, col);
            if (col < H_base2cam.cols - 1) {
                file_transform << ",";  // Separate values with commas
            }
        }
        file_transform << "\n";  // New line after each row
    }
    file_transform.close();

    //// Invert H_camera2chess
    //cv::Mat H_camera2chess_inv;
    //std::vector<cv::Mat> Hs_camera2base(num_pose); //transform points in a camera coordinate to one in a base coordinate,
    //for (int idx = 0; idx < num_pose; idx++) {
    //    cv::invert(Hs_camera2chess[idx], H_camera2chess_inv);
    //    // Compute H_camera2base
    //    H_base2cam = Hs_base2chess[idx] * H_camera2chess_inv;
    //}
    
    //Evaluation
    //triangulate points of chessboard corners
    std::vector<cv::Point3d> points_camera;//points in a camera frame.
    tri.cal3D(chess2camera.points_left, chess2camera.points_right, 0, points_camera);//dlt
    for (int i = 0; i < points_camera.size(); i++) {
        std::cout << "point-" << i << " :: x=" << points_camera[i].x << " mm, y=" << points_camera[i].y << ", z=" << points_camera[i].z << " mm" << std::endl;
    }
    //transform from a camera frame to the robot base frame.
    cv::Mat P,P_base,P_base_inv,H_cam2base,P_diff;
    cv::invert(H_base2cam, H_cam2base);
    for (int idx = 0; idx < points_camera.size(); idx++) {
        P = (cv::Mat_<double>(4, 1) <<
            points_camera[idx].x,
            points_camera[idx].y,
            points_camera[idx].z,
            1);  // Homogeneous coordinates
        P_base = H_base2cam * P;
        P_base_inv = H_cam2base * P;
        std::cout << "estimated points=" << P_base << std::endl;
        std::cout << "true points=" << Hs_base2chess[idx](cv::Rect(3, 0, 1, 4));
        std::cout << "estimated points (inversed)=" << P_base_inv << std::endl;
        P_diff = P_base- Hs_base2chess[idx](cv::Rect(3, 0, 1, 4));
        double norm_diff = cv::norm(P_diff);
        std::cout << "error=" << norm_diff << " mm" << std::endl;
    }
}