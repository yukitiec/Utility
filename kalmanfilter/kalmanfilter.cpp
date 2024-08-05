// kalmanfilter.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "stdafx.h"
#include "kalmanfilter.h"
#include "kalmanfilter_2d.h"
#include "ekf.h"

int main()
{


    // Initialize EKF
    EKF ekf(0.1, 9.81, 1.225, 0.47, 0.01, 1.0);

    // Set initial state [x, y, z, v_x, v_y, v_z]
    Eigen::VectorXd x0(6);
    x0 << 0, 0, 0, 10, 10, 10;
    ekf.setInitialState(x0);

    // Simulated measurements (for demonstration purposes)
    std::vector<Eigen::VectorXd> measurements = {
        (Eigen::VectorXd(3) << 1, 1, 1).finished(),
        (Eigen::VectorXd(3) << 2, 2, 2).finished(),
        (Eigen::VectorXd(3) << 3, 3, 3).finished()
    };

    // EKF loop
    for (auto& z : measurements) {
        ekf.predict();
        ekf.update(z);

        std::cout << "Updated state: " << ekf.getState().transpose() << std::endl;
    }

    // Example usage
    ConstantKalmanFilter ckf(0, 10, 0.01, 0.1);
    /**
    * 
    * kf(double initial_state, double initial_estimate_error, double process_noise, double measurement_noise) {
    *    x_hat_ = initial_state;
    *    P_ = initial_estimate_error;
    *    Q_ = process_noise;
    *    R_ = measurement_noise;
    * 
    */
    ParabolicKalmanFilter pkf(0, 0, 0.01, 0.1); //parabolic kalman filter is better for both constant velocity and model with acceleration
    /**
    * (double initial_position, double initial_velocity, double process_noise, double measurement_noise) {
        // Initial state: [position, velocity, acceleration]
        state_ << initial_position, initial_velocity, 0;

        // Initial estimate error covariance
        P_ = Eigen::MatrixXd::Identity(3, 3);

        // Process noise covariance
        Q_ = Eigen::MatrixXd::Identity(3, 3) * process_noise;

        // Measurement noise covariance
        R_ = measurement_noise;
    */
    std::cout << "a constant velocity model" << std::endl;
    for (int i = 0; i < 10; ++i) {
        // Simulate a noisy measurement
        double measurement = i + 0.1 * ((double)rand() / RAND_MAX - 0.5);

        // Prediction step
        ckf.predict();
        pkf.predict();

        // Update step
        ckf.update(measurement);
        pkf.update(measurement);
        std::cout << "True state: " << i << ", Estimated state by a constantKF : " << ckf.getState() << ", by parabollicKF : " << pkf.getState()(0)<< std::endl;
    }
    ConstantKalmanFilter ckf_2(0, 10, 0.01, 0.1);
    ParabolicKalmanFilter pkf_2(0, 5, 0.01, 0.1);
    std::cout << " model with acceleration" << std::endl;
    for (int i = 0; i < 10; ++i) {
        // Simulate a noisy measurement of the position (parabola)
        double true_position = 0.5 * 9.81 * i * i; // Simple parabolic motion formula
        double measurement = true_position + 0.1 * ((double)rand() / RAND_MAX - 0.5);

        // Prediction step
        ckf_2.predict();
        pkf_2.predict();

        // Update step
        ckf_2.update(measurement); 
        pkf_2.update(measurement); //measurement should be only position

        std::cout << "True position: " << true_position << ", Estimated position by a constantKF : "<<ckf_2.getState()<<", by a parabollicKF : " << pkf_2.getState()(0) << std::endl;
    }
    KalmanFilter2D kf2d(0, 0, 1, 1, 0.01, 0.1, 0.1, 0.01); //double initial_x, double initial_y, double initial_vx, double initial_vy,double process_noise_pos, double process_noise_vel, double process_noise_acc, double measurement_noise
    std::vector<KalmanFilter2D> kalmanVector;
    kalmanVector.push_back(kf2d);
    std::cout << " ## 2d kalman filter model ## " << std::endl;
    Eigen::Vector<float, 6> prediction;
    Eigen::Vector<float, 6> estimation;
    for (int i = 0; i < 20; ++i) {
        // Simulate a noisy measurement of the position (parabola)
        auto start_time = std::chrono::high_resolution_clock::now();
        float true_position_x = i;
        float measurement_x = true_position_x + 0.1 * ((float)rand() / RAND_MAX - 0.5);
        float true_position_y = 0.5 * 9.81 * i * i; // Simple parabolic motion formula
        float measurement_y = true_position_y + 0.1 * ((float)rand() / RAND_MAX - 0.5);
        Eigen::Vector2f measurement_2d;
        measurement_2d << measurement_x, measurement_y;
        kalmanVector[0].predict(estimation); //prediction
        kalmanVector[0].update(measurement_2d);
        kalmanVector[0].predict(prediction); //prediction
        // Stop the clock
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate the duration
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float x = kalmanVector[0].getState()[0];
        float y = kalmanVector[0].getState()[1];
        std::cout << "True position: " << true_position_x << "," << true_position_y  <<", prediction ;; x="<<prediction[0]<<","<<prediction[1]<<", Estimated position: " << estimation[0]<<","<<estimation[1] << std::endl;
        // Print the duration
        std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    }

    return 0;

}

