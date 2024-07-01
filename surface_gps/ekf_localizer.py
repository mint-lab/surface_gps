import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import plot_covariance
from collections import deque

"""
TODO:
* test the EKFLocalizer class
* model measurement considering the lever-arm effect
* consider how to model the covariance matrices to account for angles
"""


# TODO: handle exceptions for the case when x or y velocity results in NaN
class EKFLocalizer(ExtendedKalmanFilter):
    """An Extended Kalman Filter for localization"""

    def __init__(
        self,
        v_noise_var=0.0025,
        w_noise_var=5.895184e-06,
        q_noise_var=4.592449e-06,
        dt=0.1,
        T0=20,
        buffer_size=100,
    ):
        """
        @brief A Constructor
        @param v_noise_var: The variance of the velocity noise.
                            The standard deviation is 0.05m/s accuracy
                            with a 68% confidence interval at 30 m/s for dynamic operation
                            (from u-blox ZED-F9R datasheet)
        @param w_noise_var: The variance of the angular velocity noise
                            (from myAHRS+ ROS 2 topic message)
        @param q_noise_var: The variance of the quaternion noise
                            (from myAHRS+ ROS 2 topic message)
        @param dt: The time interval
        """
        super().__init__(dim_x=10, dim_z=6)
        self.motion_noise = np.array([[v_noise_var, 0], [0, w_noise_var]])
        self.h_pos_vel = lambda x: x[0:6]  # x, y, z, vx, vy, vz
        self.h_quat = lambda x: x[6:10]  # qx, qy, qz, qw
        self.h_alt = lambda x: x[2]  # z
        # TODO: model jacobian matrices considering the lever-arm effect
        self.H_pos_vel = lambda x: np.eye(6, 10)
        self.H_quat = lambda x: np.eye(4, 10)
        self.H_alt = lambda p: np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        self.R_pos_vel = np.eye(6) * 0.01  # TODO: modify when wrapping with ROS
        self.R_quat = q_noise_var * np.eye(4)
        # TODO: modify to altitude with error propagation R(p_noise_var, t_noise_var)
        self.R_alt = np.array([[1]])
        self.dt = dt

        self.T0 = T0 + 273.15  # K #! dummy
        self.Tgrad = -0.0065  # K/m
        self.g = 9.80665  # m/s^2
        self.Gc = 287.058  # J/(kg*K)
        self.P0 = None  # Pa

        self.h_buffer = deque([], maxlen=buffer_size)

    def predict(self):
        """Predict the state"""
        x, y, z, vx, vy, vz, qx, qy, qz, qw = self.x.flatten()
        vxt, vyt, vzt = vx * self.dt, vy * self.dt, vz * self.dt

        # Predict the position
        self.x[0] = x + vxt
        self.x[1] = y + vyt
        self.x[2] = z + vzt

        # Predict the covariance
        self.F = np.array(
            [
                [1.0, 0, 0, self.dt, 0, 0, 0, 0, 0, 0],
                [0, 1.0, 0, 0, self.dt, 0, 0, 0, 0, 0],
                [0, 0, 1.0, 0, 0, self.dt, 0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
            ]
        )
        # df/dv, df/dw
        W = np.array(
            [
                [self.dt, 0],
                [self.dt, 0],
                [self.dt, 0],
                [1.0, 0],
                [1.0, 0],
                [1.0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ]
        )
        self.Q = W @ self.motion_noise @ W.T
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, measurement_type="pos_vel"):
        """Update the state"""
        if measurement_type == "pos_vel":
            self.R = self.R_pos_vel
            super().update(z, HJacobian=self.H_pos_vel, Hx=self.h_pos_vel, R=self.R)
        elif measurement_type == "quat":
            self.R = self.R_quat
            super().update(z, HJacobian=self.H_quat, Hx=self.h_quat, R=self.R)

    def calculate_altitude(self, P: float) -> bool:
        """Calculate the altitude"""
        if self.P0 is None:
            self.P0 = P
        h = (self.T0 / self.Tgrad) * (
            1 - (P / self.P0) ** (self.Tgrad * self.Gc / self.g)
        )
        self.h_buffer.append(h)
        return True

    def optimize_altitude(self, h_buffer: deque) -> float:
        """Optimize the altitude"""
        mean = np.mean(h_buffer)
        return mean
