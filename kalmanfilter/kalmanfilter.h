#pragma once

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "stdafx.h"

class ConstantKalmanFilter {
    /**
    *  kalmanfilter for a constant velocity model
    */
private:
    double x_hat_;       // State estimate
    double P_;           // Estimate error covariance
    double Q_;           // Process noise covariance
    double R_;           // Measurement noise covariance
    double x_hat_minus_; // Predicted state estimate
    double P_minus_;     // Predicted estimate error covariance
    double K_;           // Kalman gain

public:
    ConstantKalmanFilter(double initial_state, double initial_estimate_error, double process_noise, double measurement_noise) {
        x_hat_ = initial_state;
        P_ = initial_estimate_error;
        Q_ = process_noise;
        R_ = measurement_noise;
    }

    // Prediction step
    void predict() {
        x_hat_minus_ = x_hat_;
        P_minus_ = P_ + Q_;
    }

    // Update step
    void update(double measurement) {
        K_ = P_minus_ / (P_minus_ + R_);
        x_hat_ = x_hat_minus_ + K_ * (measurement - x_hat_minus_);
        P_ = (1 - K_) * P_minus_;
    }

    double getState() const {
        return x_hat_;
    }

};

class ParabolicKalmanFilter {
    /**
    * kalmanfilter for a constant acceleration model
    */
private:
    Eigen::Vector3d state_; // State estimate [position, velocity, acceleration]
    Eigen::Matrix3d P_;     // Estimate error covariance
    Eigen::Matrix3d Q_;     // Process noise covariance
    double R_;             // Measurement noise covariance
    Eigen::Matrix3d A_;     // State transition matrix
    Eigen::Vector3d H_;     // Measurement matrix
    Eigen::Vector3d K_;     // Kalman gain

    const double dt_ = 1.0; // Time step (for simplicity, you may adjust this based on your scenario)

public:
    ParabolicKalmanFilter(double initial_position, double initial_velocity, double process_noise, double measurement_noise) {
        // Initial state: [position, velocity, acceleration]
        state_ << initial_position, initial_velocity, 0;

        // Initial estimate error covariance
        P_ = Eigen::Matrix3d::Identity(3, 3);

        // Process noise covariance
        Q_ = Eigen::Matrix3d::Identity(3, 3) * process_noise;

        // Measurement noise covariance
        R_ = measurement_noise;
    }

    // Prediction step
    void predict() {
        // State transition matrix A for constant acceleration model
        A_ << 1, dt_, 0.5 * dt_ * dt_,
            0, 1, dt_,
            0, 0, 1;

        // Predict the next state
        state_ = A_ * state_;

        // Update the estimate error covariance
        P_ = A_ * P_ * A_.transpose() + Q_;
    }

    // Update step
    void update(double measurement) {
        // Measurement matrix H (we are measuring the position)
        H_ << 1, 0, 0; //column vector

        // Kalman gain
        K_ = P_ * H_ / ((H_.transpose() * P_ * H_) + R_);

        // Update the state estimate : measurement is position
        state_ = state_ + K_ * (measurement - (H_.transpose() * state_));  //when getting value by calculating matrix dot -> extract by (A*B)(0,0)

        // Update the estimate error covariance
        P_ = (Eigen::Matrix3d::Identity(3, 3) - K_ * H_.transpose()) * P_;
    }

    Eigen::Vector3d getState() const {
        return state_;
    }
};

#endif
