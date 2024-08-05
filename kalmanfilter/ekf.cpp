#include "ekf.h"

void EKF::setInitialState(const Eigen::VectorXd& x0) {
    /*
    * @brief set initial state
    */

    x = x0;
}

void EKF::predict() {
    /**
    * @brief Prediction part
    */
    x = stateTransitionModel(x);

    // Calculate the Jacobian of the state transition model
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
    double v = x.segment<3>(3).norm();
    double drag_acc = 0.5 * rho * Cd * A * v / mass;

    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;

    F(3, 3) -= drag_acc * dt;
    F(4, 4) -= drag_acc * dt;
    F(5, 5) -= (drag_acc * dt + g * dt);

    P = F * P * F.transpose() + Q;
}

void EKF::update(Eigen::VectorXd& z) {
    /**
    * @brief updated data
    * @param[in] z measurement
    */
    Eigen::VectorXd y = z - measurementModel(x);
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (Eigen::MatrixXd::Identity(x.size(), x.size()) - K * H) * P;
}

Eigen::VectorXd EKF::getState() const {
    /**
    * @brief get the latest data.
    */
    return x;
}

Eigen::VectorXd EKF::stateTransitionModel(Eigen::VectorXd& x) {
    /**
    * @brief state transition model. \dot(x)=f(x,u)+epilon
    * @param[in] x current state.
    */
    Eigen::VectorXd x_pred(6);
    double v = x.segment<3>(3).norm();
    double drag_acc = 0.5 * rho * Cd * A * v / mass;

    x_pred(0) = x(0) + x(3) * dt;//x+vx*dt
    x_pred(1) = x(1) + x(4) * dt;//y+vy*dt
    x_pred(2) = x(2) + x(5) * dt - 0.5 * g * dt * dt;//z+vz-1/2*g*dt*dt

    x_pred(3) = x(3) - drag_acc * x(3) * dt;
    x_pred(4) = x(4) - drag_acc * x(4) * dt;
    x_pred(5) = x(5) - g * dt - drag_acc * x(5) * dt;

    return x_pred;
}

Eigen::VectorXd EKF::measurementModel(Eigen::VectorXd& x) {
    /**
    * @brief measurement model 
    * @param[in] x latest state.
    */

    Eigen::VectorXd z(3);
    z(0) = x(0);
    z(1) = x(1);
    z(2) = x(2);
    return z;
}

