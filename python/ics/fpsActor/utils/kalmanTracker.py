import numpy as np


class KalmanAngleTracker3D:
    def __init__(self, initialAngle, initialVelocity=0.0, initialAcceleration=0.0,
                 q_angle=1e-1, q_velocity=1e-1, q_acceleration=5e-2, r_measurement=1e-2):
        """
        Kalman filter for angular position tracking with constant acceleration model.
        State: [angle, angular velocity, angular acceleration]
        """
        # State vector: [angle, velocity, acceleration]
        self.x = np.array([[initialAngle],
                           [initialVelocity],
                           [initialAcceleration]])

        # Control input model (not used here, but could be extended)
        self.B = np.zeros((3, 1))

        # Measurement matrix: we only observe angle
        self.H = np.array([[1, 0, 0]])

        # Covariance matrix
        self.P = np.eye(3)

        # Process noise covariance
        self.Q_base = np.diag([q_angle, q_velocity, q_acceleration])

        # Measurement noise covariance
        self.R = np.array([[r_measurement]])

    def predict(self, steps=1.0):
        dt = steps  # Time step based on number of motor steps (assumed linear mapping)
        F = np.array([[1, dt, 0.5 * dt ** 2],
                      [0, 1, dt],
                      [0, 0, 1]])
        Q = np.diag([self.Q_base[0, 0], self.Q_base[1, 1], self.Q_base[2, 2]])

        # Predict the next state
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x.flatten()

    def predict_external(self, steps=1.0):
        """
        Computes the predicted state after a given number of steps without
        modifying the internal state of the filter.
        """
        dt = steps
        F = np.array([[1, dt, 0.5 * dt ** 2],
                      [0, 1, dt],
                      [0, 0, 1]])
        predicted_state = F @ self.x
        return predicted_state.flatten()

    def update(self, measuredAngle):
        z = np.array([[measuredAngle]])  # Measurement
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

    def get_state(self):
        return self.x.flatten()
