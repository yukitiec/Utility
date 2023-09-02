// robotControl.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <thread>
#include <chrono>

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <ur_rtde/rtde_receive_interface.h>

#pragma comment(lib, "rtde.lib")

int main()
{
	const std::string URIP = "169.254.52.200";
	namespace UR = ur_rtde;

	std::cout << "start" << std::endl;
	UR::RTDEControlInterface urCtrl(URIP);
	std::cout << "Control" << std::endl;
	UR::RTDEIOInterface urDO(URIP);
	std::cout << "Interface" << std::endl;
	UR::RTDEReceiveInterface urDI(URIP);
	std::cout << "Receive" << std::endl;

	//Œ»Ý‚ÌŠÖßŠpŽæ“¾
	//auto curJnt = urDI.getActualQ();
	//std::cout << "current joint: " << std::endl;
	//for (auto j : curJnt) {
	//	std::cout << j << ", ";
	//}
	//std::cout << std::endl;


	std::vector<double> home_pose_jnt = { -0.15,0.76,1.02,1.8,0.13,-0.14 };
	std::vector<double> tcp_pose1_jnt = { -0.15,0.76,0.92,1.8,0.13,-0.14 };
	std::vector<double> tcp_pose2_jnt = { -0.3,0.76,0.92,1.8,0.13,-0.14 };
	std::vector<double> pose_c,pose_difference;
	
	std::vector<std::vector<double>> targets;
	pose_c = urDI.getActualTCPPose();
	pose_c[0] += 0.1;
	pose_c[2] += 0.1;
	targets.push_back(pose_c);
	pose_c[0] -= 0.1;
	pose_c[2] += 0.1;
	targets.push_back(pose_c);
	pose_c[0] -= 0.1;
	pose_c[2] -= 0.1;
	targets.push_back(pose_c);
	pose_c[0] += 0.1;
	pose_c[2] -= 0.1;
	targets.push_back(pose_c);

	/* Real time robot control to catch */
	double acceleration = 0.3; //set acceleration
	double velocity = 0.6; //set velocity
	double omega = 0.6; // angular velocity
	double Gain_vel = 700; // Gain for calculating velocity
	double Gain_angle = 60; // Gain for calculating velocity
	double Gain_vel_diff; // Gain for calculating velocity
	double Gain_angle_diff; // Gain for calculating velocity
	double velocity_tmp; //for temporary velocity
	double omega_tmp; //for temporary velocity
	double dt = 1.0 / 100; //move 500 fps, every 2msec sending signal to UR
	double diff; // variable for calculating distance to target
	std::vector<double> difference; //pose diffenrence between current and target
	double vel_norm=0.0; //velocity norm for moving to target
	double omega_norm=0.0; // angular velocity norm for moving target orientation

	// 4 seconds control
	int count_target = 0;
	std::vector<double> target = targets[count_target];
	for (unsigned int i = 0; i < 500; i++)
	{
		if (i %  150 == 149)
		{
			count_target += 1;
			target = targets[count_target];
			std::cout << "target has changed" << std::endl;
		}
		std::chrono::steady_clock::time_point t_start = urCtrl.initPeriod(); //define time
		pose_c = urDI.getActualTCPPose(); // get current pose
		// calculate distance to target
		for (int j = 0; j < pose_c.size(); j++)
		{
			diff = target[j] - pose_c[j];
			if (diff < 0.001)
			{
				diff = 0.0;
			}
			difference.push_back(diff);
			//calculate norm
			if (j < 3)
			{
				vel_norm += diff * diff;
			}
			else
			{
				omega_norm += diff * diff;
			}
			std::cout << "distance to target : " <<j<<" : "<< diff << std::endl;
		}
		vel_norm = std::pow(vel_norm, 0.5);
		omega_norm = std::pow(omega_norm, 0.5);
		Gain_vel_diff = Gain_vel * vel_norm; //calculatenorm
		velocity_tmp = ((std::exp(Gain_vel_diff) - 1) / (std::exp(Gain_vel_diff) + 1)) * omega; //calculate velocity
		Gain_angle_diff = Gain_angle * omega_norm; //calculatenorm
		omega_tmp = ((std::exp(Gain_angle_diff) - 1) / (std::exp(Gain_angle_diff) + 1)) * omega; //calculate velocity
		// calculate target pose velocity
		for (int j = 0; j < pose_c.size(); j++)
		{
			//velocity
			if (j < 3)
			{
				
				difference[j] = (velocity_tmp * difference[j] / vel_norm);
				if (difference[j] < 0.001)
				{
					difference[j] = 0.0;
				}
				
			
			}
			else
			{
				difference[j] = (omega_tmp * difference[j] / vel_norm);
				if (difference[j] < 0.001)
				{
					difference[j] = 0.0;
				}
			}
			std::cout << "velocity : " << j << " : " << difference[j]<<std::endl;
		}
		// move robot
		urCtrl.speedL(difference, acceleration, dt);
		//wait
		urCtrl.waitPeriod(t_start);
	}
	urCtrl.stopL(acceleration);

	/* simple robot control program
	for (int i = 0; i < 5; i++)
	{
		std::cout << "finish initializing" << std::endl;
		pose_c = urDI.getActualTCPPose();
		std::cout << "{ " << std::endl;
		for (int i = 0; i < pose_c.size(); i++)
		{
			std::cout << pose_c[i] << " ";
		}
		std::cout << " }" << std::endl;
		pose_c[0] += 0.1;
		//rtde_control.getInverseKinematics();
	
		urCtrl.moveL(pose_c,0.1,0.05); //moovL(target position, speed,acceleration)
		pose_c = urDI.getActualTCPPose();
		std::cout << "{ " << std::endl;
		for (int i = 0; i < pose_c.size(); i++)
		{
			std::cout << pose_c[i] << " ";
		}
		std::cout << " }" << std::endl;
		pose_c[0] -= 0.1;
		//rtde_control.getInverseKinematics();

		urCtrl.moveL(pose_c, 0.1, 0.05); //moovL(target position, speed,acceleration)
	}
	*/

	
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
