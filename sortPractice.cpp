#include <iostream>
#include <vector>
#include <algorithm>>
#include <opencv2/opencv.hpp>

void getLatestData(std::vector<std::vector<std::vector<int>>>&, std::vector<std::vector<int>>&);
void getLatestClass(std::vector<std::vector<int>>&, std::vector<int>&);
bool sortByCenterX(const std::vector<int>&, const std::vector<int>&);
void sortData(std::vector<std::vector<int>>&, std::vector<int>&);

int main()
{
	std::vector<std::vector<std::vector<int>>> dataLeft{ {{1,1,1}},{{1,3,3}},{{1,2,2}} };
	std::vector<std::vector<std::vector<int>>> dataRight{ {{1,1,1}},{{1,3,3}},{{1,2,2}} }; // [num Objects, time_steps, each time position ] : {frameIndex,centerX,centerY}
	std::vector<std::vector<int>> classesLeft{ {0,0,1,-1} };
	std::vector<std::vector<int>> classesRight{ {0,1,-1,0} };

	/* get latest data from dataLeft and dataRight */
	std::vector<std::vector<int>> dataLatestLeft, dataLatestRight;
	std::vector<int> classesLatestLeft, classesLatestRight;
	getLatestData(dataLeft, dataLatestLeft);
	getLatestData(dataRight, dataLatestRight);
	getLatestClass(classesLeft, classesLatestLeft);
	getLatestClass(classesRight, classesLatestRight);
	std::cout << "classesLatestLeft : " << std::endl;
	for (const int classIndex : classesLatestLeft)
	{
		std::cout <<  classIndex << " ";
	}
	std::cout << std::endl;
	/* sort dataLeft and dataRight */
	sortData(dataLatestLeft, classesLatestLeft);
	sortData(dataLatestRight, classesLatestRight);
	std::cout << "classesLatestLeft sorted : " << std::endl;
	for (const int classIndex : classesLatestLeft)
	{
		std::cout << classIndex << " ";
	}
	std::cout << std::endl;
	std::cout << "dataLatestLeft sorted : " << std::endl;
	for (int i=0;i<dataLatestLeft.size();i++)
	{
		std::cout << i << " : { ";
		for (int j = 0; j < dataLatestLeft[i].size(); j++)
		{
			std::cout << dataLatestLeft[i][j] << " ";
		}
		std::cout << "}" << std::endl;
		
	}
	std::cout << std::endl;

	std::vector<std::vector<int>> data = {
		{1, 10, 20},
		{2, 30, 40},
		{3, 50, 60},
		{4, 70, 80},
		{5, 90, 100}
	};

	std::vector<std::vector<int>> tempData;
	// Check if the vector has at least 3 elements
	if (data.size() >= 3) {
		// Use reverse iterators to access the last three elements
		auto rbegin = data.rbegin();  // Iterator to the last element
		auto rend = data.rend();      // Iterator one past the end

		for (auto it = rbegin; it != rend && std::distance(rbegin, it) < 3; ++it) {
			const std::vector<int>& element = *it;
			tempData.push_back(element);

			// Access and use the elements of the last three vectors as needed
			int index = element[0];
			int centerX = element[1];
			int centerY = element[2];

			std::cout << "Index: " << index << ", centerX: " << centerX << ", centerY: " << centerY << std::endl;
		}
		for (int i = 0; i < tempData.size(); i++)
		{
			std::cout << i << "-th data ::{ ";
			for (int j = 0; j < tempData.size(); j++)
			{
				std::cout << tempData[i][j] << " ";
			}
			std::cout << "}" << std::endl;
		}
	}
	else {
		std::cout << "The vector does not have at least 3 elements." << std::endl;
	}

	std::vector<int> a{ 1,2,3,4 };
	a.erase(a.begin() + 0);
	std::cout << " a = { ";
	for (int i = 0; i < a.size(); i++)
	{
		std::cout << a[i] << " ";
	}
	std::cout << "}" << std::endl;

	for (int i = 3; i > 0; i--)
	{
		std::cout << i << std::endl;
	}

	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
		1, 0, 2,  // fx: focal length in x, cx: principal point x
		0, 3, 4,  // fy: focal length in y, cy: principal point y
		0, 0, 1     // 1: scaling factor
		);

	std::cout << "fx :" << cameraMatrix.at<double>(0,0) << "fy : " << cameraMatrix.at<double>(1,1) << std::endl;

	return 0;



}

void getLatestData(std::vector<std::vector<std::vector<int>>>& data, std::vector<std::vector<int>>& dataLatest)
{
	for (int i = 0; i < data.size(); i++)
	{
		dataLatest.push_back(data[i].back());
	}
}

void getLatestClass(std::vector<std::vector<int>>& classes, std::vector<int>& classesLatest)
{
	/* get latest classes but exclude class -1 */
	classesLatest = classes.back();
	/* delete -1 index */
	classesLatest.erase(std::remove(classesLatest.begin(),classesLatest.end(),-1),classesLatest.end());
}
bool sortByCenterX(const std::vector<int>& a, const std::vector<int>& b) 
{
	/*
	* Sort in ascending order based on centerX 
	*/
	return a[1] < b[1]; 
}

void sortData(std::vector<std::vector<int>>& dataLatest, std::vector<int>& classesLatest)
{
	// Create an index array to remember the original order
	std::vector<size_t> index(dataLatest.size());
	for (size_t i = 0; i < index.size(); ++i) {
		index[i] = i;
	}
	// Sort data1 based on centerX values and apply the same order to data2
	std::sort(index.begin(), index.end(), [&](size_t a, size_t b) 
		{
		return dataLatest[a][1] < dataLatest[b][1];
		});

	
	std::vector<std::vector<int>> sortedDataLatest(dataLatest.size());
	std::vector<int> sortedClassesLatest(classesLatest.size());

	for (size_t i = 0; i < index.size(); ++i) {
		sortedDataLatest[i] = dataLatest[index[i]];
		sortedClassesLatest[i] = classesLatest[index[i]];
	}

	dataLatest = sortedDataLatest;
	classesLatest = sortedClassesLatest;
}