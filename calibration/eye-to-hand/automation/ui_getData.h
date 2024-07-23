#pragma once


class UI_getData {
	/**
	* @brief User interface. displaying current images and robot joints angle
	*/
private:
	const std::vector<std::string> jointNames = { "base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3" };
	const std::string sign_save = "s";//save sign.
	const std::string sign_continue = "c";//continue sign.
public:
	UI_getData() {
		std::cout << "construct UI for getting images and robot tcp position data" << std::endl;
	};

	~UI_getData() {};

	/**
	* @brief move robot for chessboard is within camera field of view, and save images and robot TCP pose.
	* @param[in] frame_left,frame_right left and right frames.
	* @param[in] imgDir, csvDir image and csv directory.
	*/
	void main(cv::Mat& frame_left, cv::Mat& frame_right, std::string& imgDir_left, std::string& imgDir_right, std::string& csvDir);

	/**
	* @brief move robot according to human inputs.
	* @param[out] jointsValues incrementals of each joint
	*/
	void adjustRobot(std::vector<double>& jointValues);

};