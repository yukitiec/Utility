#pragma once

#ifndef IK_H
#define IK_H

#include "stdafx.h"

class IK {
private:
	const double pi = 3.14159265358979323846;
	//UR geometric params
	// https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
	//rotation angle [rad]
	double alpha1 = (pi/2);
	double alpha2 = 0;
	double alpha3 = 0;
	double alpha4 = (pi/2);
	double alpha5 = -(pi/2);
	double alpha6 = 0; //nothing
	//link distance [m]
	double a1 = 0;
	double a2 = -0.4250;
	double a3 = -0.3922;
	double a4 = 0;
	double a5 = 0;
	double a6 = 0; //none
	//joint distance
	double d1 = 0.1625;
	double d2 = 0;
	double d3 = 0;
	double d4 = 0.1333;
	double d5 = 0.0997;
	double d6 = 0.0996;
	//penalty 
	const double Max_change_ = pi;
	//determinant minimum
	const double Min_det_ = 1.0;

public:
	//for inverse kinematics
	double A, B, C, D, E, F;
	double r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz;//transform matrix [m]
	double theta1_cur, theta2_cur, theta3_cur, theta4_cur, theta5_cur, theta6_cur; //current joints
	std::vector<double> theta1, theta3, theta5; //joints angle [rad]
	double theta2, theta4, theta6, Kc, Ks, theta234, s3, c3;
	int idx1, idx3, idx5; //what data is used
	bool bool1 = false; bool bool24 = false; bool bool3 = false; bool bool4 = false; bool bool5 = false; //theta_n is fixed?
	int ch1 = -1; int ch3 = -1; int ch5 = -1; //num of changing values

	//constructor
	IK() {
		std::cout << "construct Inverse Kinematics class" << std::endl;
	};

	//destructor
	~IK() {};

	//forward kinematics
	void fk(std::vector<double>& joints, double(&pose_matrix)[4][4],std::vector<double>& pose);
	//with matrix
	void fk2(std::vector<double>& joints, double(&pose_matrix)[4][4], std::vector<double>& pose);
	//with matrix
	void fk_best(std::vector<double>& joints, double(&pose_matrix)[4][4], std::vector<double>& pose);

	void T01(double& theta,double(&matrix)[4][4]);
	void T12(double& theta, double(&matrix)[4][4]);
	void T23(double& theta, double(&matrix)[4][4]);
	void T34(double& theta, double(&matrix)[4][4]);
	void T45(double& theta, double(&matrix)[4][4]);
	void T56(double& theta, double(&matrix)[4][4]);
	void T6(double(&matrix)[4][4]);

	void rot_x(double& theta, double(&matrix)[4][4]);
	void rot_y(double& theta, double(&matrix)[4][4]);
	void rot_z(double& theta, double(&matrix)[4][4]);
	void trans_x(double& x, double(&matrix)[4][4]);
	void trans_z(double& z, double(&matrix)[4][4]);
	void multiply_matrix(double(&matrix1)[4][4], double(&matrix2)[4][4], double(&matrix)[4][4]);

	//inverse kinematics
	bool ik(std::vector<double>& pose_target, std::vector<double>& joints_cur, std::vector<double>& joints_target);

	void rodrigues(std::vector<double>& tcp);
	
	void pose2matrix(std::vector<double>& tcp);

	//state1 -> theta1, C,D,E
	bool state1();

	//state5 -> calculate theta5
	bool state5();

	//state6 -> calculate theta6 and F
	void state6();

	//state3 -> calculate theta3
	bool state3();

	//state24 -> calculate theta2 and theta44
	bool state24();

	//A~F
	void calA();
	void calB();
	void calC();
	void calD();
	void calE();
	void calF();

	//theta1~theta6
	bool cal1();
	void cal2();
	bool cal3();
	void cal4();
	void cal5();
	void cal6();
	void cal234();

	//det(J)
	double det_j();

	//check angle
	void checkAngle(std::vector<double>& theta);

	void checkAngle_scalar(double& theta);
};

#endif