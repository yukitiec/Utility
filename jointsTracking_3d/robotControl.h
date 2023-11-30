#pragma once

#ifndef ROBOTCONTROL_H
#define ROBOTCONTROL_H

#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include "stdafx.h"

namespace UR = ur_rtde;

/* from joints to robot control */
extern std::queue<std::vector<std::vector<std::vector<int>>>> queueJointsPositions;
extern std::queue<std::array<cv::Mat1b, 2>> queueFrame;
/* for predicting human intention */
std::queue<std::vector<std::vector<int>>> queueHumanWrists;
/* updated robot motion command */
std::queue<std::vector<double>> queueRobotAvoidance;
/* notify danger */
extern std::queue<bool> queueDanger;

/*mutex*/
extern std::mutex mtxRobot;

//IP address setting
const std::string URIP = "169.254.52.200";
UR::RTDEControlInterface urCtrl(URIP);
UR::RTDEIOInterface urDO(URIP);
UR::RTDEReceiveInterface urDI(URIP);


class RobotControl
{
private:
	// constant value setting
	double dt = 1.0 / 500; //move 500 fps, every 2msec sending signal to UR
	int splitRate = 3; //divide move distance by splitRate
	double acceleration = 3.0; //set acceleration
	double velocity = 3.0; //set velocity
	double omega = 3.0; // angular velocity
	double Gain_vel = 1; // Gain for calculating velocity
	double Gain_angle = 1.0; // Gain for calculating velocity

	//avoidance action
	const double Vel_max = 1.0;
	const double dist_threshold = 0.3; //distance threshold
	const double gradient = -Vel_max / dist_threshold; //repelling velocity


public:
	RobotControl()
	{
		std::cout << "construct robot control class" << std::endl;
	}

	void main()
	{
		/* Real time robot control to catch */
		double Gain_vel_diff; // Gain for calculating velocity
		double Gain_angle_diff; // Gain for calculating velocity
		double velocity_tmp; //for temporary velocity
		double omega_tmp; //for temporary velocity
		double diff; // variable for calculating distance to target
		std::vector<double> difference; //pose diffenrence between current and target
		double difPositionNorm = 0.0; //velocity norm for moving to target
		double difAngleNorm = 0.0; // angular velocity norm for moving target orientation

		int counterIteration = 0;
		int counterFinish = 0;

		while (true)
		{
			if (!queueJointsPositions.empty()) break;
			//std::cout << "wait for target data" << std::endl;
		}
		std::cout << "start avoid action of UR" << std::endl;
		//start dif calculation thread
		std::thread threadUR(&RobotControl::moveRobot, this);
		while (true) // continue until finish
		{
			counterIteration++;
			if (queueFrame.empty() && queueJointsPositions.empty())
			{
				if (counterFinish == 10) break;
				counterFinish++;
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
				std::cout << "By finish : remain count is " << (10 - counterFinish) << std::endl;
				continue;
			}
			//detect human joint position and decide robot motion
			else
			{
				counterFinish = 0;
				std::vector<std::vector<std::vector<int>>> humans;
				//humans position updated
				if (!queueJointsPositions.empty())
				{
					//get 3D positions in robot base coordinate
					humans = queueJointsPositions.front();
					queueJointsPositions.pop();
					//send wrist position for predicting human intention
					for (std::vector<std::vector<int>>& human : humans)
					{
						if (!queueHumanWrists.empty()) queueHumanWrists.pop();
						if (human[4][0] != -1 || human[5][0] != -1) queueHumanWrists.push({ human[4],human[5] });
					}
					std::cout << "update human joints" << std::endl;
				}
				//current human position
				if (!humans.empty())
				{
					std::vector<double> distances;
					std::vector<std::vector<int>> joints_danger;
					//get robot current positions
					auto start = std::chrono::high_resolution_clock::now();
					std::vector<double> pose_c = urDI.getActualTCPPose();
					std::vector<double> velocity_c = urDI.getActualTCPSpeed();
					//human detected
					if (!humans.empty())
					{
						//for each human
						for (std::vector<std::vector<int>>& human : humans)
						{
							//for each joint, if false, {-1,-1,-1,-1} was inserted
							for (std::vector<int>& joint : human)
							{
								//detected
								if (joint[0] != -1)
								{
									calculateDistances(pose_c, joint,distances,joints_danger);
								}
							}
						}
						//if robot in dangerous zone
						if (!joints_danger.empty())
						{
							//update robot motion to avoid collision
							std::vector<double> vec_update;
							updateMotion(pose_c, velocity_c, joints_danger,distances,vec_update);
							if (!queueRobotAvoidance.empty()) queueRobotAvoidance.pop();
							if (!vec_update.empty())
							{
								queueRobotAvoidance.push(vec_update);
								std::cout << "avoidance action :: Vx=" << vec_update[0] << ", Vy=" << vec_update[1] << ", Vz=" << vec_update[2] << std::endl;
							}
						}
					}
					auto end = std::chrono::high_resolution_clock::now();
					std::chrono::duration<double> duration = end - start;
					std::cout << "Time taken: by avoidance action " << duration.count() << " seconds" << std::endl;
				}
			}
		}
	}
	
	void calculateDistances(std::vector<double>& pose_c, std::vector<int>& joint, std::vector<double>& distances, std::vector<std::vector<int>>& joints_danger)
	{
		double dist = std::pow((std::pow((pose_c[0] - joint[1]), 2) + std::pow((pose_c[1] - joint[2]), 2) + std::pow((pose_c[2] - joint[3]), 2)), 0.5);
		//if dist is smaller than threshold add data
		if (dist <= dist_threshold)
		{
			distances.push_back(dist);
			joints_danger.push_back(joint);
		}
	}
	void updateMotion(std::vector<double>& pose_c, std::vector<double>& velocity_c, std::vector<std::vector<int>>& joints_danger, std::vector<double>& distances, std::vector<double>& vec_update)
	{
		std::vector<double> vec_r2j(3,0.0); //{0.0,0.0,0.0}
		if (distances.size() == 1)
		{
			vec_r2j = std::vector<double>{ (joints_danger[0][1]- pose_c[0]),(joints_danger[0][2]- pose_c[1]),(joints_danger[0][3]- pose_c[2]) };//robot to joint
		}
		else if (distances.size() >= 2)
		{
			double sum_dist = 0;
			std::vector<double> center_danger(3,0.0);
			for (int i = 0; i < joints_danger.size(); i++)
			{
				sum_dist += distances[i]; //sum distances
				center_danger[0] += distances[i] * joints_danger[i][1]; //X
				center_danger[1] += distances[i] * joints_danger[i][2]; //Y
				center_danger[2] += distances[i] * joints_danger[i][3]; //Z
			}
			center_danger[0] /= sum_dist; center_danger[1] /= sum_dist; center_danger[2] /= sum_dist; //gravity points in danger
			vec_r2j = std::vector<double>{ (center_danger[0] - pose_c[0]),(center_danger[1] - pose_c[1]),(center_danger[2] - pose_c[2])};//robot to joint
		}
		double norm_vec_r2j = std::pow((std::pow(vec_r2j[0], 2) + std::pow(vec_r2j[1], 2) + std::pow(vec_r2j[2], 2)), 0.5); //norm of vector from robot to joint
		double norm_parallel = (velocity_c[0] * vec_r2j[0] + velocity_c[1] * vec_r2j[1] + velocity_c[2] * vec_r2j[2]) / norm_vec_r2j; //norm of parallel element of robot motion
		std::vector<double> vec_perpendicular{ (velocity_c[0] - (norm_parallel * vec_r2j[0] / norm_vec_r2j)),(velocity_c[1] - (norm_parallel * vec_r2j[1] / norm_vec_r2j)),(velocity_c[2] - (norm_parallel * vec_r2j[2] / norm_vec_r2j)) }; //perpendicular velocity of robot motion
		double vel_repell = -(gradient * distances[0] + Vel_max); //repelling velocity
		std::vector<double> vec_repell{ vel_repell * vec_r2j[0] / norm_vec_r2j,vel_repell * vec_r2j[1] / norm_vec_r2j,vel_repell * vec_r2j[2] / norm_vec_r2j }; //repelling vector
		vec_update = std::vector<double>{ (vec_perpendicular[0] + vec_repell[0]),(vec_perpendicular[1] + vec_repell[1]),(vec_perpendicular[2] + vec_repell[2]) };//robot collision-avoidance motion
	}

	void moveRobot()
	{
		std::unique_lock<std::mutex> lock(mtxRobot);
		std::vector<double> dMove;
		bool boolMove = false; //can move?
		while (true)
		{
			// get new data
			if (!queueRobotAvoidance.empty())
			{
				//auto start_l = std::chrono::high_resolution_clock::now();
				dMove = queueRobotAvoidance.front();
				dMove = { dMove[0] / splitRate,dMove[1] / splitRate,dMove[2] / splitRate,dMove[3] / splitRate,dMove[4] / splitRate,dMove[5] / splitRate };
				boolMove = true;
				queueRobotAvoidance.pop();
			}
			// move robot
			if (!dMove.empty() && boolMove)
			{
				int counter = 0;
				auto start_l = std::chrono::high_resolution_clock::now();
				while (queueRobotAvoidance.empty() && counter < 5)
				{
					urCtrl.speedL(dMove, acceleration, dt);
					counter += 1;
					//auto end_l = std::chrono::high_resolution_clock::now();
					//std::chrono::duration<double> duration = end_l - start_l;
					//std::cout << "Time taken: by speedL " << duration.count() << " seconds" << std::endl;

				}
				boolMove = false;
				auto end_l = std::chrono::high_resolution_clock::now();
				std::chrono::duration<double> duration = end_l - start_l;
				std::cout << "Time taken: by speedL " << duration.count() << " seconds" << std::endl;
			}
			// wait for next motivation
			else
			{
				//urCtrl.stopL(acceleration);
				//nothing to do
			}
			auto end = std::chrono::high_resolution_clock::now();
		}
		std::cout << "break" << std::endl;
		urCtrl.stopL(acceleration);
	}

};

#endif 