#pragma once

#include "stdafx.h"
#include "triangulation.h"

class Chess2camera {
private:
    std::string rootDir;
    std::string file_intrinsic_left;
    std::string file_intrinsic_right;
    std::string file_extrinsic_left;
    std::string file_extrinsic_right;
    Triangulation tri;

    // Define the supported image file extensions
    const std::vector<std::string> imageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff" };

public:
    //params
    int type_process, width, height;
    double size;
    std::vector<cv::Mat> imgs_left, imgs_right;
    std::vector<cv::Point2d> points_left, points_right;
    std::vector<cv::Mat> tvecs_left, rvecs_left, tvecs_right, rvecs_right;//rotation vector and translation vector.
    //params

    Chess2camera(const std::string& rootDir)
        : rootDir(rootDir),
        file_intrinsic_left(rootDir + "/camera0_intrinsics.dat"),
        file_intrinsic_right(rootDir + "/camera1_intrinsics.dat"),
        file_extrinsic_left(rootDir + "/camera0_rot_trans.dat"),
        file_extrinsic_right(rootDir + "/camera1_rot_trans.dat"),
        tri(file_intrinsic_left, file_intrinsic_right, file_extrinsic_left, file_extrinsic_right)
    {
        std::cout << "construct Chess2camera class" << std::endl;
    }

    /**
    * @brief get homogeneous matrix from chessboard to camera by cv::solvePnP()
    * @param[in] rootData path to the root directory having images.
    */
    void main(std::string& rootData);

    /**
    * @brief get chessboard corners and camera pose from cv::solvePnP()
    * @param[in] rootUndistort, rootDraw : path to the root directory
    * @param[in] imgs images to check
    * @oaram[out] points original points of the chessboard
    * @param[out] rvecs,tvecs rotation and translation vector
    * @param[in] bool_left whether images are from left.
    */
    void getPose(std::string& rootUndistort, std::string& rootDraw, std::vector<cv::Mat>& imgs, std::vector<cv::Point2d>& points, 
        std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs, bool bool_left);

    /**
    * @brief make a directory.
    */
    void makeDir(std::filesystem::path& dirPath);

    /**
    * @brief check image extension.
    */
    bool hasImageExtension(const std::string& filename);

    /**
    * @brief get images from a directory.
    */
    std::vector<cv::Mat> getImages(std::string& rootImgs);

    /**
    * @brief draw projected 3D points on a chessboard.
    */
    cv::Mat draw3DPoints(cv::Mat& image, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point3f>& axis, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs);
};