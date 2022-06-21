import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.stats import plot_covariance

class EKFLocalizer(ExtendedKalmanFilter):
    def __init__(self, v_noise_std=1, w_noise_std=1, gps_noise_std=1, dt=1):
        super().__init__(dim_x=5, dim_z=2)
        vv = v_noise_std * v_noise_std
        vw = v_noise_std * w_noise_std
        ww = w_noise_std * w_noise_std
        self.motion_noise = np.array([[vv, vw], [vw, ww]])
        self.h = lambda x: x[0:2]
        self.H = lambda x: np.eye(2, 5)
        self.R = gps_noise_std * gps_noise_std * np.eye(2)
        self.dt = dt

    def predict(self):
        x, y, theta, v, w = self.x.flatten()
        vt, wt = v * self.dt, w * self.dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)

        # Predict the state
        self.x[0] = x + vt * c
        self.x[1] = y + vt * s
        self.x[2] = theta + wt
        #if self.x[2] >= np.pi:
        #    self.x[2] -= 2 * np.pi
        #elif self.x[2] < -np.pi:
        #    self.x[2] += 2 * np.pi
        #self.x[3] = v # Not necessary
        #self.x[4] = w # Not necessary

        # Predict the covariance
        self.F = np.array([
            [1, 0, -vt * s, self.dt * c, -vt * self.dt * s / 2],
            [0, 1,  vt * c, self.dt * s,  vt * self.dt * c / 2],
            [0, 0,       1,           0,               self.dt],
            [0, 0,       0,           1,                     0],
            [0, 0,       0,           0,                     1]])
        W = np.array([
            [self.dt * c, -vt * self.dt * s / 2],
            [self.dt * s,  vt * self.dt * c / 2],
            [0, self.dt],
            [1, 0],
            [0, 1]])
        self.Q = W @ self.motion_noise @ W.T

        self.P = self.F @ self.P @ self.F.T + self.Q

        # Save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z):
        super().update(z, HJacobian=self.H, Hx=self.h, R=self.R)



if __name__ == '__main__':
    # Define experimental configuration
    dt = 0.1
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    gps_noise_std = 1

    # Instantiate EKF for pose (and velocity) tracking
    ekf = EKFLocalizer(v_noise_std=1, w_noise_std=0.1, gps_noise_std=gps_noise_std, dt=dt)

    record = []
    for t in np.arange(0, 6, dt):
        # Generate position observation with additive Gaussian noise
        truth = get_true_position(t)
        obs = truth + np.random.normal(size=truth.shape, scale=gps_noise_std)

        # Predict and update the EKF
        ekf.predict()
        ekf.update(obs)

        record.append([t] + truth.flatten().tolist() + obs.flatten().tolist() + ekf.x.flatten().tolist() + ekf.P.flatten().tolist())
    record = np.array(record)

    # Visualize the results
    plt.figure()
    plt.plot(record[:,1], record[:,2], color='r', label='Truth')
    plt.plot(record[:,3], record[:,4], color='b', label='Observation')
    plt.plot(record[:,5], record[:,6], color='g', label='EKF')
    for i, line in enumerate(record):
        if i % 5 == 0:
            plot_covariance(line[5:7], line[10:].reshape(5, 5)[0:2,0:2], interval=0.98, edgecolor='g', alpha=0.5)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,1], color='r', label='Truth')
    plt.plot(record[:,0], record[:,3], color='b', label='Observation')
    plt.plot(record[:,0], record[:,5], color='g', label='EKF')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,2], color='r', label='Truth')
    plt.plot(record[:,0], record[:,4], color='b', label='Observation')
    plt.plot(record[:,0], record[:,6], color='g', label='EKF')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], (360.-90.)/(6.-0.) * record[:,0] + 90., color='r', label='Truth')
    plt.plot(record[:,0], record[:,7] * 180 / np.pi, color='g', label='EKF')
    plt.xlabel('Time')
    plt.ylabel(r'Orientaiton $\theta$')
    plt.grid()
    plt.legend()