#include "ui_getData.h"

void UI_getData::main(cv::Mat& frame_left,cv::Mat& frame_right,std::string& imgDir_left,std::string& imgDir_right, std::string& csvDir) {

    int counter;
    std::string bool_save,bool_continue;
    std::string file_path,name_file;
    std::vector<double> joints_ur,tcp_ur;

    std::cout << "input start counter :: ";
    std::cin >> counter;
    std::cout << std::endl;
    //get current robot angles.
    //joints_ur = urDI->getActualQ();
    //for (double& joint:joints_ur)
    //std::cout<<joint<<",";
    //std::cout<<std::endl;

    //incrementals of each joint angle. 
    std::vector<double> jointValues(6, 0);
    //get incremental of each joint.
    adjustRobot(jointValues);
    //move robot with moveJ()
    
    //hear whether save image and TCP pose.
    std::cout << "Will you save images and TCP pose?" << std::endl;
    std::cout << "type \'s\' if you wanna save, and otherwise type \'n\':";
    std::cin >> bool_save;

    //save data
    if (bool_save.compare(sign_save) == 0) {
        //save frame_left and frame_wright.
        name_file = std::to_string(counter) + ".png";
        name_file = "/" + name_file;
        file_path = imgDir_left + "/" + name_file;
        cv::imwrite(file_path, frame_left);
        file_path = imgDir_right + "/" + name_file;
        cv::imwrite(file_path, frame_right);

        //save joints angle
        //tcp_ur = urDI-> getActualTCP();
        name_file = std::to_string(counter) + ".csv";
        name_file = "/" + name_file;
        file_path = csvDir + name_file;
        std::ofstream file(file_path);
        for (int j = 0; j < tcp_ur.size();j++) {
            file << tcp_ur[j];
            if (j<tcp_ur.size() - 1) file << ",";
        }
        file << "\n";
        counter++;
    }

    //continue
    std::cout << "Will you continue?" << std::endl;
    std::cout << "type \'c\' if you wanna continue, and otherwise type \'q\':";
    std::cin >> bool_continue;
    //save data
    if (bool_save.compare(sign_continue) == 0) {
        std::cout << "Will you really stop?" << std::endl;
        std::cout << "type \'q\' if you wanna quit, and otherwise type \'c\':";
        std::cin >> bool_continue;
        if (bool_save.compare(sign_continue) == 0)
            break;
    }

}

void UI_getData::adjustRobot(std::vector<double>& jointValues) {

    std::cout << "Please input values for the following joints:\n";
    for (size_t i = 0; i < jointNames.size(); ++i) {
        std::cout << "Enter incremental angle for " << jointNames[i] << ": ";
        while (!(std::cin >> jointValues[i])) {
            std::cin.clear(); // clear the error flag
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // discard invalid input
            std::cout << "Invalid input. Please enter a numerical value for " << jointNames[i] << ": ";
        }
    }
    std::cout << "\nYou entered the following values:\n";
    for (size_t i = 0; i < jointNames.size(); ++i) {
        std::cout << jointNames[i] << ": " << jointValues[i] << ",";
    }
    std::cout << std::endl;
}