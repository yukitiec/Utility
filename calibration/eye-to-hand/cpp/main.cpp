// analysis_eyeToHand.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "stdafx.h"
#include "triangulation.h"

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

// Define the supported image file extensions
const std::vector<std::string> imageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff" };

bool hasImageExtension(const std::string& filename) {
    for (const auto& ext : imageExtensions) {
        if (filename.size() >= ext.size() &&
            filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<cv::Mat> getImages(std::string& rootImgs) {

    /**
    * @brief get images from a designated directory.
    * @param[in] rootImgs : root path for a directory.
    * @return images storage. 
    */

    std::vector<std::string> imageFiles;

    // Iterate over the files in the directory
    for (const auto& entry : std::filesystem::directory_iterator(rootImgs)) {
        if (entry.is_regular_file() && hasImageExtension(entry.path().string())) {
            imageFiles.push_back(entry.path().string());
        }
    }

    // get images and save in std::vector<cv::Mat>
    std::vector<cv::Mat> imgs;
    for (const auto& imageFile : imageFiles) {
        cv::Mat image = cv::imread(imageFile);
        imgs.push_back(image);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imageFile << std::endl;
            continue;
        }
    }

    return imgs;
}


int main()
{
	std::string rootDir = "C:/Users/kawaw/cpp/eyeToHand_calibration/eyeToHand_calibration/camera_calibration";
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

	//load robot 3D pose.
    std::string rootData = "C:/Users/kawaw/cpp/eyeToHand_calibration/eyeToHand_calibration/eyeHand";
    std::string rootJoints = rootData + "/csv";
    std::string file_joints = "joints.csv";
    file_joints = "/" + file_joints;
    file_joints = rootJoints + file_joints;
    std::vector<std::vector<double>> pose_robot = readCSV(file_joints);

	//1,get chessboard information
	int type_process, width, height;
    double size;
	std::cout << ":: Chessboard information :: \n width=";
	std::cin >> width;
	std::cout << "height=";
	std::cin >> height;
	std::cout << "size=";
	std::cin >> size;
	std::cout << std::endl;

	//2.corner detection and show display to check whether we'll reverse the points.
    //get images in a directory.
    std::string rootImgs = rootData + "/img";
    std::string rootImgs_left = rootImgs + "/left";
    std::string rootImgs_right = rootImgs + "/right";
    
    //get images
    std::vector<cv::Mat> imgs_left, imgs_right;
    imgs_left = getImages(rootImgs_left);
    imgs_right = getImages(rootImgs_right);

    //get corners and display. And check whether original points is correct.
    for (cv::Mat& image : imgs_left) {
        cv::imshow("images", image);
        cv::waitKey(1000);
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Find chessboard corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, cv::Size(width, height), corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            // Refine corner locations
            cv::cornerSubPix(gray, corners, cv::Size(7, 7), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 50, 0.03));

            // Draw circles on detected corners
            for (const auto& corner : corners) {
                cv::circle(image, corner, 5, cv::Scalar(0, 0, 255), -1);
            }

            // Create 3D coordinates
            std::vector<cv::Point3d> objectPoints;
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    objectPoints.emplace_back(j * size, i * size, 0.0);
                }
            }

            // Output the 2D and 3D points
            std::cout << "2D Points:" << std::endl;
            for (const auto& corner : corners) {
                std::cout << corner << std::endl;
            }

            std::cout << "3D Points:" << std::endl;
            for (const auto& point : objectPoints) {
                std::cout << point << std::endl;
            }
            // Show the image with detected corners
            cv::imshow("Detected Corners", image);
            cv::waitKey(1000);
        }
        else {
            std::cerr << "Chessboard corners not found!" << std::endl;
        }
    }
	//3.calculate H_Camera2Chess with cv::solvePnP -> rotation vector and translation.
	//save in the storage


	//cv::calibrateHandEye() -> get transformation matrix.

}