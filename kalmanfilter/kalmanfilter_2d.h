#pragma once

#ifndef KALMANFILTER_2D_H
#define KALMANFILTER_2D_H

#include "stdafx.h"

class KalmanFilter2D {
private:
    Eigen::Vector<float,6> state_; // State estimate [x, y, vx, vy, ax, ay]
    Eigen::Matrix<float,6,6> P_;     // Estimate error covariance
    Eigen::Matrix<float, 6, 6> Q_;     // Process noise covariance
    Eigen::Matrix<float, 2, 2> R_;     // Measurement noise covariance
    Eigen::Matrix<float, 6, 6> A_;     // State transition matrix
    Eigen::Matrix<float, 2, 6> H_;     // Measurement matrix
    Eigen::Matrix<float, 6, 2> K_;     // Kalman gain

    const float dt_ = 1.0; // Time step (for simplicity, you may adjust this based on your scenario)
public:
    KalmanFilter2D(float initial_x, float initial_y, float initial_vx, float initial_vy,
            float process_noise_pos, float process_noise_vel, float process_noise_acc, float measurement_noise) {
        // Initial state: [x, y, vx, vy, ax, ay]
        state_ << initial_x, initial_y, initial_vx, initial_vy, 0, 9.81; //column vector

        // Initial estimate error covariance
        P_ = Eigen::MatrixXf::Identity(6, 6);

        // Process noise covariance
        Q_ << process_noise_pos, 0, 0, 0, 0, 0,
            0, process_noise_pos, 0, 0, 0, 0,
            0, 0, process_noise_vel, 0, 0, 0,
            0, 0, 0, process_noise_vel, 0, 0,
            0, 0, 0, 0, process_noise_acc, 0,
            0, 0, 0, 0, 0, process_noise_acc; 

        // Measurement noise covariance
        R_ = Eigen::MatrixXf::Identity(2, 2) * measurement_noise;
    }

    // Prediction step
    void predict(Eigen::Vector<float,6>& prediction) {
        // State transition matrix A for constant acceleration model
        A_ << 1, 0, dt_, 0, 0.5 * dt_ * dt_, 0,
              0, 1, 0, dt_, 0, 0.5 * dt_,
              0, 0, 1, 0, dt_, 0,
              0, 0, 0, 1, 0, dt_,
              0, 0, 0, 0, 1, 0,
              0, 0, 0, 0, 0, 1;

        // Predict the next state
        state_ = A_ * state_;
        prediction = state_;
        
        // Update the estimate error covariance
        P_ = A_ * P_ * A_.transpose() + Q_;
    }

    // Update step
    void update(const Eigen::Vector2f& measurement) {
        // Measurement matrix H (we are measuring the position in x and y)
        H_ << 1, 0, 0, 0, 0, 0,
              0, 1, 0, 0, 0, 0; //2*6 vector

        // Kalman gain
        K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse(); //6*2 matrix

        // Update the state estimate
        state_ = state_ + K_ * (measurement - H_ * state_); //6*1 + (6*2)*(2*1) (2*6)*(6*1)

        // Update the estimate error covariance
        P_ = (Eigen::MatrixXf::Identity(6, 6) - K_ * H_) * P_;
    }

    Eigen::Vector<float,6> getState() const {
        return state_;
    }
};

#endif