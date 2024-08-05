#include "stdafx.h"

class EKF {
public:
    EKF(double dt, double g, double rho, double Cd, double A, double mass)
        : dt(dt), g(g), rho(rho), Cd(Cd), A(A), mass(mass) {
        // Initialize matrices
        P = Eigen::MatrixXd::Identity(6, 6);
        Q = 0.1 * Eigen::MatrixXd::Identity(6, 6);
        H = Eigen::MatrixXd::Zero(3, 6);
        H(0, 0) = H(1, 1) = H(2, 2) = 1;
        R = 0.1 * Eigen::MatrixXd::Identity(3, 3);
    }

    void setInitialState(const Eigen::VectorXd& x0);

    void predict();

    void update(Eigen::VectorXd& z);

    Eigen::VectorXd getState() const;

private:
    double dt, g, rho, Cd, A, mass;
    Eigen::VectorXd x;
    Eigen::MatrixXd P, Q, H, R;


    Eigen::VectorXd stateTransitionModel(Eigen::VectorXd& x);

    Eigen::VectorXd measurementModel(Eigen::VectorXd& x);
};

