#include "ik.h"


//forward kinematics
void IK::fk(std::vector<double>& joints, double(&pose_matrix)[4][4],std::vector<double>& pose) {
	/**
	* @brief forward kinematics
	* @param[in] joints joints angle {base,shoulder,elbow,wrist1,wrist2,wrist3} [rad]
	* @param[out] pose matrix of tool center pose
	*/
	double theta1 = joints[0];
	double theta2 = joints[1];
	double theta3 = joints[2];
	double theta4 = joints[3];
	double theta5 = joints[4];
	double theta6 = joints[5];

	//rotation matrix
	r11 = (cos(theta1) * cos(theta2 + theta3 + theta4) * cos(theta5) * cos(theta6)) + (cos(theta6) * sin(theta1) * sin(theta5)) - (cos(theta1) * sin(theta2 + theta3 + theta4) * sin(theta6));
	r21 = (cos(theta2 + theta3 + theta4) * cos(theta5) * cos(theta6) * sin(theta1)) - (cos(theta1) * cos(theta6) * sin(theta5)) - (sin(theta1) * sin(theta2 + theta3 + theta4) * sin(theta6));
	r31 = (cos(theta5) * cos(theta6) * sin(theta2 + theta3 + theta4)) + (cos(theta2 + theta3 + theta4) * sin(theta6));
	r12 = -(cos(theta1) * cos(theta2 + theta3 + theta4) * cos(theta5) * sin(theta6)) - (sin(theta1) * sin(theta5) * sin(theta6)) - (cos(theta1) * cos(theta6) * sin(theta2 + theta3 + theta4));
	r22 = -(cos(theta2 + theta3 + theta4) * cos(theta5) * sin(theta1) * sin(theta6)) + (cos(theta1) * sin(theta5) * sin(theta6)) - (cos(theta6) * sin(theta1) * sin(theta2 + theta3 + theta4));
	r32 = -(cos(theta5) * sin(theta2 + theta3 + theta4) * sin(theta6)) + (cos(theta2 + theta3 + theta4) * cos(theta6));
	r13 = -(cos(theta1) * cos(theta2 + theta3 + theta4) * sin(theta5)) + (cos(theta5) * sin(theta1));
	r23 = -(cos(theta2 + theta3 + theta4) * sin(theta1) * sin(theta5)) - (cos(theta1) * cos(theta5));
	r33 = -sin(theta2 + theta3 + theta4) * sin(theta5);
	//translatiion
	px = (r13 * d6) + (cos(theta1) * (sin(theta2 + theta3 + theta4) * d5 + cos(theta2 + theta3) * a3 + cos(theta2) * a2)) + (sin(theta1) * d4);
	py = (r23 * d6) + (sin(theta1) * (sin(theta2 + theta3 + theta4) * d5 + cos(theta2 + theta3) * a3 + cos(theta2) * a2)) - (cos(theta1) * d4);
	pz = (r33 * d6) - (cos(theta2 + theta3 * theta4) * d5) + (sin(theta2 + theta3) * a3) + (sin(theta2) * a2) + d1;
	
	//rotation
	pose_matrix[0][0] = r11;
	pose_matrix[1][0] = r21;
	pose_matrix[2][0] = r31;
	pose_matrix[0][1] = r12;
	pose_matrix[1][1] = r22;
	pose_matrix[2][1] = r32;
	pose_matrix[0][2] = r13;
	pose_matrix[1][2] = r23;
	pose_matrix[2][2] = r33;
	//translation
	pose_matrix[0][3] = px;
	pose_matrix[1][3] = py;
	pose_matrix[2][3] = pz;
	//residuals
	pose_matrix[3][0] = 0;
	pose_matrix[3][1] = 0;
	pose_matrix[3][2] = 0;
	pose_matrix[3][3] = 1;

	double roll = -std::atan2(r23,r33);
	double pitch = std::asin(r13);
	double yaw = -std::atan2(r12,r11);
	pose = std::vector<double>{px, py, pz, roll, pitch, yaw};
}

void IK::fk2(std::vector<double>& joints, double(&pose_matrix)[4][4], std::vector<double>& pose) {
	/**
	* @brief forward kinematics
	* @param[in] joints joints angle {base,shoulder,elbow,wrist1,wrist2,wrist3} [rad]
	* @param[out] pose matrix of tool center pose
	*/
	double theta1 = joints[0];
	double theta2 = joints[1];
	double theta3 = joints[2];
	double theta4 = joints[3];
	double theta5 = joints[4];
	double theta6 = joints[5];
	double t01[4][4], t12[4][4], t23[4][4], t34[4][4], t45[4][4], t56[4][4];
	T01(theta1, t01);
	T12(theta2, t12);
	T23(theta3, t23);
	T34(theta4, t34);
	T45(theta5, t45);
	T56(theta6, t56);
	multiply_matrix(t01, t12, pose_matrix);
	multiply_matrix(pose_matrix, t23, pose_matrix);
	multiply_matrix(pose_matrix, t34, pose_matrix);
	multiply_matrix(pose_matrix, t45, pose_matrix);
	multiply_matrix(pose_matrix, t56, pose_matrix);
	//XYZ rotation
	double roll = -std::atan2(pose_matrix[1][2], pose_matrix[2][2]);
	double pitch = std::asin(pose_matrix[0][2]);
	double yaw = -std::atan2(pose_matrix[0][1], pose_matrix[0][0]);
	pose = std::vector<double>{ px, py, pz, roll, pitch, yaw };
}

//forward kinematics
void IK::fk_best(std::vector<double>& joints, double(&pose_matrix)[4][4], std::vector<double>& pose) {
	/**
	* @brief forward kinematics with https://repository.gatech.edu/server/api/core/bitstreams/e56759bc-92c8-43df-aa62-0dc47581459d/content
	* @param[in] joints joints angle {base,shoulder,elbow,wrist1,wrist2,wrist3} [rad]
	* @param[out] pose matrix of tool center pose
	*/
	double theta1 = joints[0];
	double theta2 = joints[1];
	double theta3 = joints[2];
	double theta4 = joints[3];
	double theta5 = joints[4];
	double theta6 = joints[5];

	double c1 = cos(theta1);
	double s1 = sin(theta1);
	double c2 = cos(theta2);
	double s2 = sin(theta2);
	double c3 = cos(theta3);
	double s3 = sin(theta3);
	double c4 = cos(theta4);
	double s4 = sin(theta4);
	double c5 = cos(theta5);
	double s5 = sin(theta5);
	double c6 = cos(theta6);
	double s6 = sin(theta6);
	double c234 = cos(theta2 + theta3 + theta4);
	double s234 = sin(theta2 + theta3 + theta4);

	r11 = c6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0) - (s6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0;
	r21 = c6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0) + s6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0);
	r31 = (s234 * c6 + c234 * s6) / 2.0 + s234 * c5 * c6 - (s234 * c6 - c234 * s6) / 2.0;
	r12 = -(c6 * ((s1 * c234 + c1 * s234) - (s1 * c234 - c1 * s234))) / 2.0 - s6 * (s1 * s5 + ((c1 * c234 - s1 * s234) * c5) / 2.0 + ((c1 * c234 + s1 * s234) * c5) / 2.0);
	r22 = c6 * ((c1 * c234 - s1 * s234) / 2.0 - (c1 * c234 + s1 * s234) / 2.0) - s6 * (((s1 * c234 + c1 * s234) * c5) / 2.0 - c1 * s5 + ((s1 * c234 - c1 * s234) * c5) / 2.0);
	r32 = (c234 * c6 + s234 * s6) / 2.0 + (c234 * c6 - s234 * s6) / 2.0 - s234 * c5 * s6;
	r13 = c5 * s1 - ((c1 * c234 - s1 * s234) * s5) / 2.0 - ((c1 * c234 + s1 * s234) * s5) / 2.0;
	r23 = -c1 * c5 - ((s1 * c234 + c1 * s234) * s5) / 2.0 + ((c1 * s234 - s1 * c234) * s5) / 2.0;
	r33 = (c234 * c5 - s234 * s5) / 2.0 - (c234 * c5 + s234 * s5) / 2.0;
	px = -(d5 * (s1 * c234 - c1 * s234)) / 2.0 + (d5 * (s1 * c234 + c1 * s234)) / 2.0 + d4 * s1 - (d6 * (c1 * c234 - s1 * s234) * s5) / 2.0 - (d6 * (c1 * c234 + s1 * s234) * s5) / 2.0 + a2 * c1 * c2 + d6 * c5 * s1 + a3 * c1 * c2 * c3 - a3 * c1 * s2 * s3;
	py = -(d5 * (c1 * c234 - s1 * s234)) / 2.0 + (d5 * (c1 * c234 + s1 * s234)) / 2.0 - d4 * c1 - (d6 * (s1 * c234 + c1 * s234) * s5) / 2.0 - (d6 * (s1 * c234 - c1 * s234) * s5) / 2.0 - d6 * c1 * c5 + a2 * c2 * s1 + a3 * c2 * c3 * s1 - a3 * s1 * s2 * s3;
	pz = d1 + (d6 * (c234 * c5 - s234 * s5)) / 2.0 + a3 * (s2 * c3 + c2 * s3) + a2 * s2 - (d6 * (c234 * c5 + s234 * s5)) / 2.0 - d5 * c234;
	//rotation
	pose_matrix[0][0] = r11;
	pose_matrix[1][0] = r21;
	pose_matrix[2][0] = r31;
	pose_matrix[0][1] = r12;
	pose_matrix[1][1] = r22;
	pose_matrix[2][1] = r32;
	pose_matrix[0][2] = r13;
	pose_matrix[1][2] = r23;
	pose_matrix[2][2] = r33;
	//translation
	pose_matrix[0][3] = px;
	pose_matrix[1][3] = py;
	pose_matrix[2][3] = pz;
	//residuals
	pose_matrix[3][0] = 0;
	pose_matrix[3][1] = 0;
	pose_matrix[3][2] = 0;
	pose_matrix[3][3] = 1;
	//double t6[4][4];
	//multiply_matrix(pose_matrix, t6, pose_matrix);
	//XYZ
	double roll = -std::atan2(r23, r33);
	double pitch = std::asin(r13);
	double yaw = -std::atan2(r12, r11);
	//roll pitch yaw
	//double roll = std::atan2(r21, r11);
	//double pitch = -std::asin(r31);
	//double yaw = std::atan2(r32, r33);
	//ZYX Å~
	//double roll = std::atan2(r32, r33);
	//double pitch = -std::asin(r31);
	//double yaw = std::atan2(r21, r11);
	//ZYZÅ@Å~
	//double roll = -std::atan2(r32, r31);
	//double pitch = std::asin(r33);
	//double yaw = std::atan2(r23, r13);
	//double roll = -std::atan2(pose_matrix[1][2], pose_matrix[2][2]);
	//double pitch = std::asin(pose_matrix[0][2]);
	//double yaw = -std::atan2(pose_matrix[0][1], pose_matrix[0][0]);
	pose = std::vector<double>{ px, py, pz, roll, pitch, yaw };
}

void IK::T01(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base0 to 1
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha1, R_alpha);
	trans_x(a1, R_a);
	rot_z(theta, R_theta);
	trans_z(d1, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T12(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base1 to 2
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha2, R_alpha);
	trans_x(a2, R_a);
	rot_z(theta, R_theta);
	trans_z(d2, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T23(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base2 to 3
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha3, R_alpha);
	trans_x(a3, R_a);
	rot_z(theta, R_theta);
	trans_z(d3, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T34(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base3 to 4
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha4, R_alpha);
	trans_x(a4, R_a);
	rot_z(theta, R_theta);
	trans_z(d4, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T45(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base4 to 5
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha5, R_alpha);
	trans_x(a5, R_a);
	rot_z(theta, R_theta);
	trans_z(d5, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T56(double& theta, double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base5 to 6
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	rot_x(alpha6, R_alpha);
	trans_x(a6, R_a);
	rot_z(theta, R_theta);
	trans_z(d6, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::T6(double(&matrix)[4][4])
{
	/**
	* @brief transformation matrix from base5 to 6
	*/
	double R_alpha[4][4], R_a[4][4], R_theta[4][4], R_d[4][4];
	double alpha = pi / 2;
	double theta = -pi / 2;
	double delta = 0.0;
	rot_x(alpha, R_alpha);
	trans_x(delta, R_a);
	rot_z(theta, R_theta);
	trans_z(delta, R_d);
	//multiply
	multiply_matrix(R_alpha, R_a, matrix);
	multiply_matrix(matrix, R_theta, matrix);
	multiply_matrix(matrix, R_d, matrix);
}

void IK::rot_x(double& theta, double(&matrix)[4][4]) {
	/**
	* @brief rotation matrix around x-axis
	*/
	matrix[0][0] = 1;
	matrix[1][0] = 0;
	matrix[2][0] = 0;
	matrix[0][1] = 0;
	matrix[1][1] = cos(theta);
	matrix[2][1] = sin(theta);
	matrix[0][2] = 0;
	matrix[1][2] = -sin(theta);
	matrix[2][2] = cos(theta);
	//translation
	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	//residuals
	matrix[3][0] = 0;
	matrix[3][1] = 0;
	matrix[3][2] = 0;
	matrix[3][3] = 1;
}

void IK::rot_y(double& theta, double(&matrix)[4][4]) {
	/**
	* @brief rotation matrix around y-axis
	*/
	matrix[0][0] = cos(theta);
	matrix[1][0] = 0;
	matrix[2][0] = -sin(theta);
	matrix[0][1] = 0;
	matrix[1][1] = 1;
	matrix[2][1] = 0;
	matrix[0][2] = sin(theta);
	matrix[1][2] = 0;
	matrix[2][2] = cos(theta);
	//translation
	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	//residuals
	matrix[3][0] = 0;
	matrix[3][1] = 0;
	matrix[3][2] = 0;
	matrix[3][3] = 1;
}

void IK::rot_z(double& theta, double(&matrix)[4][4]) {
	/**
	* @brief rotation matrix around z-axis
	*/
	matrix[0][0] = cos(theta);
	matrix[1][0] = sin(theta);
	matrix[2][0] = 0;
	matrix[0][1] = -sin(theta);
	matrix[1][1] = cos(theta);
	matrix[2][1] = 0;
	matrix[0][2] = 0;
	matrix[1][2] = 0;
	matrix[2][2] = 1;
	//translation
	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	//residuals
	matrix[3][0] = 0;
	matrix[3][1] = 0;
	matrix[3][2] = 0;
	matrix[3][3] = 1;
}

void IK::trans_x(double& x, double(&matrix)[4][4]) {
	/**
	* @brief translation matrix in x axis
	*/
	matrix[0][0] = 1;
	matrix[1][0] = 0;
	matrix[2][0] = 0;
	matrix[0][1] = 0;
	matrix[1][1] = 1;
	matrix[2][1] = 0;
	matrix[0][2] = 0;
	matrix[1][2] = 0;
	matrix[2][2] = 1;
	//translation
	matrix[0][3] = x;
	matrix[1][3] = 0;
	matrix[2][3] = 0;
	//residuals
	matrix[3][0] = 0;
	matrix[3][1] = 0;
	matrix[3][2] = 0;
	matrix[3][3] = 1;
}

void IK::trans_z(double& z, double(&matrix)[4][4]) {
	/**
	* @brief translation matrix in z axis
	*/
	matrix[0][0] = 1;
	matrix[1][0] = 0;
	matrix[2][0] = 0;
	matrix[0][1] = 0;
	matrix[1][1] = 1;
	matrix[2][1] = 0;
	matrix[0][2] = 0;
	matrix[1][2] = 0;
	matrix[2][2] = 1;
	//translation
	matrix[0][3] = 0;
	matrix[1][3] = 0;
	matrix[2][3] = z;
	//residuals
	matrix[3][0] = 0;
	matrix[3][1] = 0;
	matrix[3][2] = 0;
	matrix[3][3] = 1;
}

void IK::multiply_matrix(double(&matrix1)[4][4], double(&matrix2)[4][4], double(&matrix)[4][4])
{
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 4; k++) {
				matrix[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
}

bool IK::ik(std::vector<double>& pose_target, std::vector<double>& joints_cur,std::vector<double>& joints_target) {
	/**
	* @brief Inverse Kinematics
	* @param[in] pose_target target tcp pose {x,y,z,roll,pitch,yaw} [m],[rad]
	* @param[in] joints_cur current joint angles {base,shoulder,elbow,wrist1,wrist2,wrist3} [rad]
	* @param[out] joints_target optimal joints for target pose
	* @return IK is successful or not
	*/

	rodrigues(pose_target); //pose to matrix
	//current state
	theta1_cur = joints_cur[0];
	theta2_cur = joints_cur[1];
	theta3_cur = joints_cur[2];
	theta4_cur = joints_cur[3];
	theta5_cur = joints_cur[4];
	theta6_cur = joints_cur[5];
	//determine target joints' angles
	calA();
	calB();
	bool success = false;
	while (true) {
		//state1
		//std::cout << "ch1=" << ch1 << ", ch3=" << ch3 << ", ch5=" << ch5 << std::endl;
		if (!bool1) {
			//std::cout << "1" << std::endl;
			if (ch1==1) break;
			bool success1 = state1();
			if (!success1) break;
		}
		//state5&state6
		if (!bool5) {
			//std::cout << "5" << std::endl;
			bool success5 = state5();
			if (!success5) continue;
			//state6
			state6();
		}
		//state3
		if (!bool3)
		{
			//std::cout << "3" << std::endl;
			bool success3 = state3();
			if (!success3) continue;
		}
		//state24
		if (!bool24) {
			//std::cout << "24" << std::endl;
			bool success24 = state24();
			if (!success24) continue;
			success = true;
			break;
		}
	}
	if (success) {
		double th1 = theta1[idx1];
		double th2 = theta2;
		double th3 = theta3[idx3];
		double th4 = theta4;
		double th5 = theta5[idx5];
		double th6 = theta6;
		joints_target = std::vector<double>{ th1,th2,th3,th4,th5,th6 };
		return true;
	}
	else return false;
}

void IK::rodrigues(std::vector<double>& tcp) {
	/**
	* @brief convert pose into matrix-style
	* @param[in] tcp {x,y,z,rx,ry,rz} rx,ry,rz : rotational vector
	*/
	// Convert angles to radians
	double x = tcp[0];
	double y = tcp[1];
	double z = tcp[2];
	double nx = tcp[3];
	double ny = tcp[4];
	double nz = tcp[5];
	double angle = std::pow(nx * nx + ny * ny + nz * nz, 0.5);
	//convert to unit vector
	nx = nx / angle;
	ny = ny / angle;
	nz = nz / angle;
	//rodriguez formula
	Eigen::Matrix3d axisCross;
	axisCross << 0, -nz, ny,
		nz, 0, -nx,
		-ny, nx, 0;

	Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity() + sin(angle) * axisCross + (1 - cos(angle)) * axisCross * axisCross;//rotationMatrix(row,col)

	//https://qiita.com/harmegiddo/items/96004f7c8eafbb8a45d0
	//X->Y->Z rotation -> RxRyRz
	//first row
	r11 = rotationMatrix(0,0);
	r12 = rotationMatrix(0, 1);
	r13 = rotationMatrix(0, 2);
	px = x;
	//second row
	r21 = rotationMatrix(1, 0);
	r22 = rotationMatrix(1, 1);
	r23 = rotationMatrix(1, 2);
	py = y;
	//third row
	r31 = rotationMatrix(2, 0);
	r32 = rotationMatrix(2, 1);
	r33 = rotationMatrix(2, 2);
	pz = z;

}


void IK::pose2matrix(std::vector<double>& tcp) {
	/**
	* @brief convert tool center pose to translational matrix
	* @param[in] tcp tool center pose {x,y,z,roll,pitch,yaw}
	* @param[out] pose_matrix pose in matrix style, i.e. translation matrix
	*/
	// Convert angles to radians
	double x = tcp[0];
	double y = tcp[1];
	double z = tcp[2];
	double roll = tcp[3];
	double pitch = tcp[4];
	double yaw = tcp[5];

	// Calculate trigonometric values
	double croll = cos(roll);
	double sroll = sin(roll);
	double cpitch = cos(pitch);
	double spitch = sin(pitch);
	double cyaw = cos(yaw);
	double syaw = sin(yaw);

	//https://qiita.com/harmegiddo/items/96004f7c8eafbb8a45d0
	//X->Y->Z rotation -> RxRyRz
	//first row
	r11 = cpitch * cyaw;
	r12 = -cpitch * syaw;
	r13 = spitch;
	px = x;
	//second row
	r21 = croll * syaw + cyaw * sroll * spitch;
	r22 = croll * cyaw - sroll * spitch * syaw;
	r23 = -cpitch * sroll;
	py = y;
	//third row
	r31 = sroll * syaw - croll * cyaw * spitch;
	r32 = cyaw * sroll + croll * spitch * syaw;
	r33 = croll * cpitch;
	pz = z;
	//std::cout << "pose_matrix=" << std::endl;
	//std::cout << r11 << "," << r12 << "," << r13 << "," << px << std::endl;
	//std::cout << r21 << "," << r22 << "," << r23 << "," << py << std::endl;
	//std::cout << r31 << "," << r32 << "," << r33 << "," << pz << std::endl;
}

bool IK::state1() {
	/**
	* @brief calculate theta1, C, D, E
	*/

	//first time
	if (ch1 == -1) {
		//calculate theta1
		bool bool_cal1 = cal1();
		if (!bool_cal1) return false;
		checkAngle(theta1);
		//std::cout << "theta1=" << theta1[0] << ", " << theta1[1] << std::endl;
		double dif0 = std::abs(theta1[0] - theta1_cur);
		double dif1 = std::abs(theta1[1] - theta1_cur);
		if (dif0 <= dif1) //dif0 smaller -> 1st
			idx1 = 0; 
		else //dif1 smaller -> 2nd
			idx1 = 1;

		//signal setting
		bool1 = true;
		ch1++;

		//calculate C,D and E
		calC();
		calD();
		calE();
	}
	else { //second time
		if (idx1 == 0)
			idx1 = 1;
		else
			idx1 = 0;

		//signal setting
		bool1 = true;
		ch1++;

		//calculate C,D and E
		calC();
		calD();
		calE();
	}
	return true;
}

bool IK::state5() {
	/**
	* @brief calculate theta5
	* @return whether theta5 is acceptable or not
	*/

	if (ch5 == -1){//first time
		cal5();
		checkAngle(theta5);
		//std::cout << "theta5=" << theta5[0] << ", " << theta5[1] << std::endl;
		double dif0 = std::abs(theta5[0] - theta5_cur);
		double dif1 = std::abs(theta5[1] - theta5_cur);
		if (dif0 <= dif1) //dif0 smaller -> 1st
			idx5 = 0;
		else //dif1 smaller -> 2nd
			idx5 = 1;

		//check theta5 is acceptable
		double criteria = std::abs(sin(theta5[idx5]));
		//std::cout << "state5 criteria: " << criteria << " > 1e-12" << std::endl;
		if (criteria<=1e-12) {//return to state1
			bool1 = false;
			bool5 = false;
			return false;
		}

		//signal setting -> state3
		bool5 = true;
		ch5++;
	}
	else { //second time
		if (idx5 == 0)
			idx5 = 1;
		else
			idx5 = 0;

		//std::cout << "theta5=" << theta5[idx5];
		//check theta5 is acceptable
		double criteria = std::abs(sin(theta5[idx5]));
		//std::cout << "state5 criteria: " << criteria << " > 1e-12" << std::endl;
		if (criteria <= 1e-12) {//-> state1
			bool1 = false; 
			bool5 = false;
			ch5 = -1;
			return false;
		}

		//signal setting -> state3
		bool5 = true;
		ch5++;
	}
	return true;
}

void IK::state6() {
	/**
	* @brief calculate theta6 and F
	*/
	cal6();
	checkAngle_scalar(theta6);
	//std::cout << "theta6=" << theta6  << std::endl;
	calF();
}

bool IK::state3(){
	/**
	* @brief calculate theta3
	*/
	if (ch3 == -1) {
		cal234();
		//std::cout << "theta234=" << theta234 << std::endl;
		bool bool_cal3 = cal3();
		//std::cout << "state3 : c3=" << c3 << std::endl;
		//unrealizable theta
		if (!bool_cal3) {
			if (ch5 == 0) { //return to state5
				bool3 = false;
				bool5 = false;
				return false;
			}
			else if (ch5 == 1 and ch1 == 0) { //return to state1
				bool1 = false;
				bool5 = false;
				bool3 = false;
				ch5 = -1;
				return false;
			}
			else {//break
				bool1 = false;
				bool3 = false;
				return false; //-> break before state1()
			}
		}
		//std::cout << "theta3=" << theta3[0]<<", "<<theta3[1] << std::endl;
		checkAngle(theta3);
		//std::cout << "theta3=" << theta3[0] << ", " << theta3[1] << std::endl;
		double dif0 = std::abs(theta3[0] - theta3_cur);
		double dif1 = std::abs(theta3[1] - theta3_cur);
		if (dif0 <= dif1) //dif0 smaller -> 1st
			idx3 = 0;
		else //dif1 smaller -> 2nd
			idx3 = 1;

		//check theta5 is acceptable
		double criteria = std::abs(sin(theta3[idx3]));
		//std::cout << "state3 criteria: 2" << criteria << " > 1e-12" << std::endl;
		if (criteria<= 1e-12) {
			if (ch5 == 0) { //return to state5
				bool3 = false;
				bool5 = false;
				return false;
			}
			else if (ch5 == 1 and ch1 == 0) { //return to state1
				bool1 = false;
				bool5 = false;
				bool3 = false;
				ch5 = -1;
				ch3 = -1;
				return false;
			}
			else {//break
				bool1 = false;
				bool3 = false;
				return false; //-> break before state1()
			}
		}

		//signal setting
		bool3 = true;
		ch3++;
	}
	else {//second time
		if (idx3 == 0)
			idx3 = 1;
		else
			idx3 = 0;

		//check theta5 is acceptable
		double criteria = std::abs(sin(theta3[idx3]));
		//std::cout << "state3 criteria: 2" << criteria << " > 1e-12" << std::endl;
		if (criteria <= 1e-12) {
			if (ch5 == 0) { //return to state5
				bool5 = false;
				bool3 = false;
				ch3 = -1;
				return false;
			}
			else if (ch5 == 1 and ch1 == 0) { //return to state1
				bool1 = false;
				bool5 = false;
				ch5 = -1;
				bool3 = false;
				ch3 = -1;
				return false;
			}
			else {//break
				bool1 = false;
				return false; //-> break before state1()
			}
		}

		//signal setting
		bool3 = true;
		ch3++;
	}
	return true;	
}

bool IK::state24() {
	/**
	* @brief calculate theta2 and theta4
	*/
	cal2();
	cal4();
	checkAngle_scalar(theta2);
	checkAngle_scalar(theta4);
	//std::cout << "theta2=" << theta2 << std::endl;
	//std::cout << "theta4=" << theta4 << std::endl;
	double det = det_j();
	double criteria = std::abs(d5 * sin(theta2 + theta3[idx3] + theta4) + a2 * cos(theta2) + a3 * cos(theta2 + theta3[idx3]));
	//std::cout << "state24 : criteria=" << criteria << " > 1e-12" << std::endl;
	if (criteria<=1e-12) {//theta is not acceptable
		if (ch3 == 0) {//return to state3
			bool3 = false;
			bool24 = false;
			return false;
		}
		else if (ch3 == 1 and ch5 == 0) {//return state5
			bool3 = false;
			ch3 = -1;
			bool5 = false;
			bool24 = false;
			return false;
		}
		else if (ch3 == 1 and ch5 == 1 and ch1 == 0) {//return to state1
			bool1 = false;
			bool5 = false;
			ch5 = -1;
			bool3 = false;
			ch3 = -1;
			bool24 = false;
			return false;
		}
		else { //failed -> break before state1
			bool1 = false;
			return false;
		}
	}
	return true; //successful
}


//A~F
void IK::calA() {
	/**
	* @brief calculate A
	*/
	A = py - d6 * r23;
}

void IK::calB() {
	/**
	* @brief calculate B
	*/
	B = px - d6 * r13;
}

void IK::calC() {
	/**
	* @brief calculate C
	*/
	C = cos(theta1[idx1]) * r11 + sin(theta1[idx1])* r21;
}

void IK::calD() {
	/**
	* @brief calculate D
	*/
	D = cos(theta1[idx1]) * r22 - sin(theta1[idx1]) * r12;
}

void IK::calE() {
	/**
	* @brief calculate E
	*/
	E = sin(theta1[idx1])*r11-cos(theta1[idx1])*r21;
}

void IK::calF() {
	/**
	* @brief calculate F
	*/
	F = cos(theta5[idx5]) * cos(theta6);
}

bool IK::cal1() {
	/**
	* @brief calculate theta1
	*/
	if (B * B + (-A) * (-A) - d4 * d4 < 0) return false;
	double y_ele1 = std::pow((B * B + (-A) * (-A) - d4 * d4),0.5);
	double ele1 = std::atan2(y_ele1, d4);
	double ele2 = std::atan2(B, -A);
	theta1 = std::vector<double>{ (ele1 + ele2),(-ele1 + ele2) };
	return true;
}

void IK::cal2() {
	/**
	* @brief calculate theta2
	*/
	double ele1 = std::atan2(Ks, Kc);
	double ele2 = atan2((sin(theta3[idx3]) * a3), (cos(theta3[idx3]) * a3 + a2));
	theta2 = ele1 - ele2;
}

bool IK::cal3() {
	/**
	* @brief calculate theta3
	*/
	c3 = (Ks * Ks + Kc * Kc - a2 * a2 - a3 * a3) / (2 * a2 * a3);
	if (c3 > 1) return false;
	s3 = std::pow(1 - c3 * c3, 0.5);
	//std::cout << "c3=" << c3 << ", s3=" << s3 << std::endl;
	double ele = std::atan2(s3, c3);
	theta3 = std::vector<double>{ ele,-ele };
	return true;
}

void IK::cal4() {
	/**
	* @brief calculate theta4
	*/
	theta4 = theta234 - theta2 - theta3[idx3];
}

void IK::cal5() {
	/**
	* @brief calculate theta5
	*/
	double y_ele1 = std::pow((E*E + D*D), 0.5);
	double x_ele1 = sin(theta1[idx1]) * r13 - cos(theta1[idx1]) * r23;
	double ele1 = std::atan2(y_ele1, x_ele1);
	theta5 = std::vector<double>{ ele1,-ele1 };
}

void IK::cal6() {
	/**
	* @brief calculate theta6
	*/
	double y_ele1 = D / sin(theta5[idx5]);
	double x_ele1 = E / sin(theta5[idx5]);
	theta6 = std::atan2(y_ele1, x_ele1);
}

void IK::cal234()
{
	/**
	* @brief calculate theta_234
	*/
	double y_ele = r31 * F - sin(theta6) * C;
	double x_ele = F * C + sin(theta6) * r31;
	theta234 = std::atan2(y_ele, x_ele);
	//std::cout << "cal234: theta1=" << theta1[idx1] << ", theta5=" << theta5[idx5] << std::endl;
	Kc = cos(theta1[idx1]) * px + sin(theta1[idx1]) * py - sin(theta234) * d5 + cos(theta234) * sin(theta5[idx5]) * d6;
	Ks = pz - d1 + cos(theta234) * d5 + sin(theta234) * sin(theta5[idx5]) * d6;
}

double IK::det_j() {
	return sin(theta3[idx3])* sin(theta5[idx5])* a2* a3* (cos(theta2) * a2 + cos(theta2 + theta3[idx3]) * a3 + sin(theta234) * d5);
}

void IK::checkAngle(std::vector<double>& theta) {
	/**
	* @brief transform angle between -pi and pi
	*/
	for (double& angle : theta) {
		while (true) {
			if (angle <= pi and angle >= -pi) break;
			if (angle > pi) angle -= 2 * pi;
			else if (angle < -pi) angle += 2 * pi;
		}
	}
}

void IK::checkAngle_scalar(double& theta) {
	/**
	* @brief transform angle between -pi and pi
	*/
	while (true) {
		if (theta <= pi and theta >= -pi) break;
		if (theta > pi) theta -= 2 * pi;
		else if (theta < -pi) theta += 2 * pi;
	}
}