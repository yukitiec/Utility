#include "stdafx.h"


void makeDir(std::filesystem::path& dirPath) {
    /**
    * @brief make a directory.
    */

    try {
        if (std::filesystem::create_directory(dirPath)) {
            std::cout << "Directory created successfully: " << dirPath << std::endl;
        }
        else {
            std::cout << "Directory already exists or could not be created: " << dirPath << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

int main()
{
    //make directory for saving data
    std::filesystem::path path_root = "result";
    std::filesystem::path path_img = "result/img";
    std::filesystem::path path_img_left = "result/img/left";
    std::filesystem::path path_img_right = "result/img/right";
    std::filesystem::path path_undistort_left = "result/img/left/undistort";
    std::filesystem::path path_undistort_right = "result/img/right/undistort";
    std::filesystem::path path_csv = "result/csv";
    makeDir(path_root);
    makeDir(path_img);
    makeDir(path_img_left);
    makeDir(path_img_right);
    makeDir(path_undistort_left);
    makeDir(path_undistort_right);
    makeDir(path_csv);

    //directory
    std::string rootDir = "result";
    std::string imgDir_left = "result/img/left";
    std::string imgDir_right = "result/img/right";
    std::string undistortDir_left = "result/img/undistort/left";
    std::string undistortDir_right = "result/img/undistort/right";
    std::string csvDir = "result/csv";

}
