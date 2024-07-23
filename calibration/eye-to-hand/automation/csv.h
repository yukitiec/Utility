#pragma once

#include "stdafx.h"


class CSV_manipulator {
	/**
	* @brief manipulate CSV file like writing or reading.
	*/
private:

public:
	CSV_manipulator() {
		std::cout << "construct a csv manipulator class" << std::endl;
	};

	~CSV_manipulator() {};

	/**
	* @brief write data in a csv file.
	*/
	void writeCSV(const std::string& filename, const std::vector<std::vector<double>>& matrix);

	/**
	* @brief read data from csv file.
	*/
	std::vector<std::vector<double>> readCSV(const std::string& filename);

};