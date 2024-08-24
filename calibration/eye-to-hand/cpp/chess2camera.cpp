#include "chess2camera.h"

void Chess2camera::main(std::string& rootData) {

    //1,get chessboard information
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
    std::string rootUndistort_left = rootImgs_left + "/undistort/";
    std::string rootUndistort_right = rootImgs_right + "/undistort/";
    std::string rootDraw_left = rootUndistort_left + "draw/";
    std::string rootDraw_right = rootUndistort_right + "draw/";

    //make directory
    std::filesystem::path path_undistort_left = rootUndistort_left;
    std::filesystem::path path_undistort_right = rootUndistort_right;
    std::filesystem::path path_draw_left = rootDraw_left;
    std::filesystem::path path_draw_right = rootDraw_right;

    makeDir(path_undistort_left);
    makeDir(path_undistort_right);
    makeDir(path_draw_left);
    makeDir(path_draw_right);

    //get images
    imgs_left = getImages(rootImgs_left);
    imgs_right = getImages(rootImgs_right);

    //initialize
    int num_imgs = imgs_left.size();
    //original points
    points_left = std::vector<cv::Point2d>(num_imgs, cv::Point2d(0, 0));
    points_right = std::vector<cv::Point2d>(num_imgs, cv::Point2d(0, 0));
    //rotation vector and translation vector.
    tvecs_left = std::vector<cv::Mat>(num_imgs);
    rvecs_left = std::vector<cv::Mat>(num_imgs); 
    tvecs_right = std::vector<cv::Mat>(num_imgs);
    rvecs_right = std::vector<cv::Mat>(num_imgs);

    //3.get corners and display. And check whether original points is correct. -> get camera pose against chessboards
    //left
    getPose(rootUndistort_left, rootDraw_left, imgs_left, points_left, rvecs_left, tvecs_left,true);
    //right
    getPose(rootUndistort_right, rootDraw_right, imgs_right, points_right, rvecs_right, tvecs_right,false);    
}

void Chess2camera::getPose(std::string& rootUndistort, std::string& rootDraw,
    std::vector<cv::Mat>& imgs,std::vector<cv::Point2d>& points,std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
    bool bool_left) {
    //variables
    cv::Mat rvec, tvec;
    cv::Mat undistortedImage;
    std::ostringstream filenameStream;
    std::string filename;
    int idx_image = 1;

    std::string winName = "Detected Corners";
    cv::namedWindow(winName);
    for (cv::Mat& image : imgs) {
        //undistort image.
        if (bool_left)
            cv::undistort(image, undistortedImage, tri.cameraMatrix_left, tri.distCoeffs_left);
        else
            cv::undistort(image, undistortedImage, tri.cameraMatrix_right, tri.distCoeffs_right);

        //initialize
        filenameStream.str("");
        filenameStream.clear();
        filenameStream << rootUndistort << std::setw(2) << std::setfill('0') << idx_image << ".png";
        filename = filenameStream.str();
        cv::imwrite(filename, undistortedImage);//save images

        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(undistortedImage, gray, cv::COLOR_BGR2GRAY);

        // Find chessboard corners
        std::vector<cv::Point2f> corners, corners_tmp,corners_tmp2;
        bool found = cv::findChessboardCorners(gray, cv::Size(width, height), corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            // Refine corner locations
            cv::cornerSubPix(gray, corners, cv::Size(7, 7), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.00001));

            // Draw circles on detected corners
            int counter_corner = 0; int bool_reverse = 0;//not reverse.
            cv::Mat image_draw = undistortedImage.clone();
            for (const auto& corner : corners) {
                if (counter_corner == 0)//first corner
                    cv::circle(image_draw, corner, 5, cv::Scalar(255, 0, 0), -1);//B
                else//others
                    cv::circle(image_draw, corner, 5, cv::Scalar(0, 0, std::abs((counter_corner * 20) % 255)), -1);//R
                counter_corner++;
            }

            // Show the image with detected corners
            //cv::imwrite("detected.png", image_draw);
            cv::imshow(winName, image_draw);
            cv::waitKey(10);
            std::cout << "Will you reverse corners order? 0:Okay, 1:reverse corners, 2:change direction, 3:reverse and change direction :: " << std::endl;
            std::cin >> bool_reverse;

            // Reverse the vector
            corners_tmp = corners;
            while (true) {
                if (bool_reverse == 0) break;
                corners_tmp = corners;
                //reverse
                if (bool_reverse == 1) {
                    std::reverse(corners_tmp.begin(), corners_tmp.end());
                    image_draw = undistortedImage.clone();
                    counter_corner = 0;
                    for (const auto& corner : corners_tmp) {
                        if (counter_corner == 0)//first corner
                            cv::circle(image_draw, corner, 5, cv::Scalar(255, 0, 0), -1);//B
                        else//others
                            cv::circle(image_draw, corner, 5, cv::Scalar(0, 0, std::abs((255 - counter_corner * 50) % 255)), -1);//R
                        counter_corner++;
                    }
                }
                else if (bool_reverse == 2) {//other direction
                    int idx = 0; int max_idx;
                    for (int row = 0; row < height; row++) {//for each row
                        max_idx = (row + 1) * width - 1;//max index
                        for (int col = 0; col < width; col++) {//for each column
                            idx = row * width + col;
                            corners_tmp[idx] = corners[max_idx - col];
                        }
                    }
                    //save
                    image_draw = undistortedImage.clone();
                    counter_corner = 0;
                    for (const auto& corner : corners_tmp) {
                        if (counter_corner == 0)//first corner
                            cv::circle(image_draw, corner, 5, cv::Scalar(255, 0, 0), -1);//B
                        else//others
                            cv::circle(image_draw, corner, 5, cv::Scalar(0, 0, std::abs((counter_corner * 50) % 255)), -1);//R
                        counter_corner++;
                    }
                }
                else if (bool_reverse == 3) {//other direction and reverse
                    corners_tmp2 = corners;
                    std::reverse(corners_tmp2.begin(), corners_tmp2.end());
                    int idx = 0; int max_idx;
                    for (int row = 0; row < height; row++) {//for each row
                        max_idx = row * width - 1;//max index
                        for (int col = 0; col < width; col++) {//for each column
                            idx = row * width + col;
                            corners_tmp[idx] = corners_tmp2[max_idx - col];
                        }
                    }
                    //save
                    image_draw = undistortedImage.clone();
                    counter_corner = 0;
                    for (const auto& corner : corners_tmp) {
                        if (counter_corner == 0)//first corner
                            cv::circle(image_draw, corner, 5, cv::Scalar(255, 0, 0), -1);//B
                        else//others
                            cv::circle(image_draw, corner, 5, cv::Scalar(0, 0, std::abs((255 - counter_corner * 50) % 255)), -1);//R
                        counter_corner++;
                    }
                }
                //cv::imwrite("detected.png", image_draw);
                cv::imshow(winName, image_draw);
                cv::waitKey(10);
                std::cout << "Is this Okay? 0:Okay, 1:reverse corners, 2:change direction, 3:reverse and change direction :: " << std::endl;
                std::cin >> bool_reverse;
            }
            //close windows
            //cv::destroyWindow("Detected Corners");

            //save original points
            corners = corners_tmp;
            points[idx_image-1] = corners[0];

            // Create 3D coordinates
            std::vector<cv::Point3d> objectPoints;
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    objectPoints.emplace_back(j * size, i * size, 0.0);
                }
            }

            // Create axis points
            std::vector<cv::Point3f> axis = {
                {(float)size, 0, 0},
                {0, (float)size, 0},
                {0, 0, (float)size}
            };

            //3.calculate H_Chess2camera with cv::solvePnP -> rotation vector and translation.
            //calculate camera pose against chessboard.
            cv::solvePnP(objectPoints, corners, tri.cameraMatrix_left, tri.distCoeffs_left, rvec, tvec);
            //save rvec and tvec.
            rvecs[idx_image-1] = rvec;
            tvecs[idx_image-1] = tvec;

            // Draw 3D axes
            image_draw = draw3DPoints(undistortedImage, corners, axis, rvec, tvec, tri.cameraMatrix_left, tri.distCoeffs_left);
            //initialize
            filenameStream.str("");
            filenameStream.clear();
            filenameStream << rootDraw << std::setw(2) << std::setfill('0') << idx_image << ".png";
            filename = filenameStream.str();
            std::cout << "filename=" << filename << std::endl;
            cv::imwrite(filename, undistortedImage);//save images

            //rvec and tvec are what I want.
        }
        else {
            std::cerr << "Chessboard corners not found!" << std::endl;
        }
        cv::destroyWindow(winName);
        //increment
        idx_image++;
    }
}

cv::Mat Chess2camera::draw3DPoints(cv::Mat& image, const std::vector<cv::Point2f>& corners, const std::vector<cv::Point3f>& axis, const cv::Mat& rvec, const cv::Mat& tvec, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    std::vector<cv::Point2f> imgpts;
    cv::projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs, imgpts);

    // Draw lines between the projected points
    cv::line(image, corners[0], imgpts[0], cv::Scalar(0, 0, 255), 5); // x-axis in red
    cv::line(image, corners[0], imgpts[1], cv::Scalar(0, 255, 0), 5); // y-axis in green
    cv::line(image, corners[0], imgpts[2], cv::Scalar(255, 0, 0), 5); // z-axis in blue

    return image;
}

void Chess2camera::makeDir(std::filesystem::path& dirPath) {
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

bool Chess2camera::hasImageExtension(const std::string& filename) {
    for (const auto& ext : imageExtensions) {
        if (filename.size() >= ext.size() &&
            filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<cv::Mat> Chess2camera::getImages(std::string& rootImgs) {

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