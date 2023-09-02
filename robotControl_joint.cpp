// robotControl_joint.cpp : This file contains the 'main' function. Program execution begins and ends there.
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

	std::vector<double> pose_c, pose_difference;

	std::vector<std::vector<double>> targets;
	pose_c = urDI.getActualTCPPose();
	pose_c[0] += 0.1;
	//pose_c[2] += 0.1;
	targets.push_back(pose_c);
	pose_c[0] -= 0.1;
	//pose_c[2] += 0.1;
	targets.push_back(pose_c);
	pose_c[0] += 0.1;
	//pose_c[2] -= 0.1;
	targets.push_back(pose_c);
	pose_c[0] -= 0.1;
	//pose_c[2] -= 0.1;
	targets.push_back(pose_c);
	std::cout << "initializing" << std::endl;

	/* Real time robot control to catch */
	double acceleration = 0.1; //set acceleration
	double velocity = 0.2; //set velocity
	double omega = 0.2; // angular velocity
	double Gain = 50; // Gain for calculating velocity
	double Gain_diff ; // Gain for calculating velocity
	double velocity_tmp; //for temporary velocity
	double dt = 1.0 / 10; //move 500 fps, every 2msec sending signal to UR
	double diffJ; // variable for calculating distance to target
	std::vector<double> currentJ, targetJ; // variable for current joint angles and target joint angles
	std::vector<double> differenceJ; //pose diffenrence between current and target
	double max; //max distance between target and current joint angle
	double omega_norm = 0.0; // angular velocity norm for moving target orientation

	// 4 seconds control
	int count_target = 0;
	std::vector<double> target = targets[count_target];
	if (urCtrl.getInverseKinematicsHasSolution(target))
	{
		targetJ = urCtrl.getInverseKinematics(target); //get target angle
		std::cout << "Get inverse kinematic solution : { ";
		for (int j = 0; j < targetJ.size(); j++)
		{
			std::cout << targetJ[j] << " ";
		}
		std::cout << "}" << std::endl;
	}
	std::cout << "start communication" << std::endl;
	for (unsigned int i = 0; i < 500; i++)
	{
		std::cout << "-------<<" << i << ">>---------" << std::endl;
		//std::cout << "target:" << target[0] << "," << target[1] << ","<< target[2] << ","<< target[3] << "," << target[4] << "," << target[5] << std::endl;
		if (i % 150 == 149)
		{
			count_target++;
			target = targets[count_target];
			if (urCtrl.getInverseKinematicsHasSolution(target))
			{
				targetJ = urCtrl.getInverseKinematics(target); //get target angle
				std::cout << "Get inverse kinematic solution : { ";
				for (int j = 0; j < targetJ.size(); j++)
				{
					std::cout << targetJ[j] << " ";
				}
				std::cout << "}" << std::endl;
			}
			std::cout << "////////////////////////// target has changed /////////////////////////" << std::endl;
		}
		std::chrono::steady_clock::time_point t_start = urCtrl.initPeriod(); //define time
		currentJ = urDI.getActualQ (); // get current joint 
		//std::cout << "Current Joints angle :" << currentJ[0] << "," << currentJ[1] << "," << currentJ[2] << "," << currentJ[3] << "," << currentJ[4] << "," << currentJ[5] << std::endl;
		// calculate distance to target
		//std::cout << "current Joints size:" << currentJ.size() << std::endl; // 6

		for (int j = 0; j < currentJ.size(); j++)
		{
			//std::cout << "calculaete difference" << std::endl;
			//std::cout << "targetJ[j]" << targetJ[j] << ", currentJ[j]:" << currentJ[j] << std::endl;
			diffJ = targetJ[j] - currentJ[j];
			if (diffJ < 0.001)
			{
				diffJ = 0.0;
			}
			Gain_diff = diffJ * Gain; //calculate diff
			velocity_tmp = ((std::exp(Gain_diff) - 1) / (std::exp(Gain_diff) + 1)) * omega; //calculate velocity
			if (velocity_tmp < 0.03)
			{
				velocity_tmp = 0.0;
			}
			differenceJ.push_back(velocity_tmp);
			std::cout << "velocity : " << j << " : " << velocity_tmp << std::endl;
		}

			
		// move robot
		urCtrl.speedL(differenceJ, acceleration, dt);
		//wait
		urCtrl.waitPeriod(t_start);
		
	}
	urCtrl.stopJ(acceleration);

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

