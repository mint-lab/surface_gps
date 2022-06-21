import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.stats import plot_covariance

def fx(state, dt):
    x, y, theta, v, w = state.flatten()
    x = x + np.cos(theta + w*dt/2)
    y = y + np.sin(theta + w*dt/2)
    theta = theta + w*dt
    return [x, y, theta, v, w]

def hx(state):
    x, y, theta, v, w = state
    return [x, y]

def x_mean_fn(sigmas, Wm):
    mean_x, mean_y, mean_theta, mean_v, mean_w = 0, 0, 0, 0, 0
    mean_sin, mean_cos = 0, 0
    for i, (x, y, theta, v, w) in enumerate(sigmas):
        mean_x += x * Wm[i]
        mean_y += y * Wm[i]
        mean_sin += np.sin(theta) * Wm[i]
        mean_cos += np.cos(theta) * Wm[i]
        mean_v += v * Wm[i]
        mean_w += w * Wm[i]
    mean_theta = np.arctan2(mean_sin, mean_cos)
    return np.array([mean_x, mean_y, mean_theta, mean_v, mean_w])

def z_residual_fn(a, b):
    if a.ndim > b.ndim:
        return a.reshape(b.shape) - b
    elif a.ndim < b.ndim:
        return a - b.reshape(a.shape)
    return a - b


if __name__ == '__main__':
    # Define experimental configuration
    dt = 0.1
    r, w = 10., np.pi / 4
    get_true_position = lambda t: r * np.array([[np.cos(w * t)], [np.sin(w * t)]]) # Circular motion
    gps_noise_std = 1

    # Instantiate UKF for pose (and velocity) tracking
    sigma = MerweScaledSigmaPoints(n=5, alpha=.1, beta=2., kappa=1.)
    ukf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, fx=fx, hx=hx, points=sigma, x_mean_fn=x_mean_fn, residual_z=z_residual_fn)
    #ukf = UnscentedKalmanFilter(dim_x=5, dim_z=2, dt=dt, fx=fx, hx=hx, points=sigma, residual_z=z_residual_fn)
    ukf.Q = np.eye(5)
    ukf.R = gps_noise_std * gps_noise_std * np.eye(2)

    record = []
    for t in np.arange(0, 6, dt):
        # Generate position observation with additive Gaussian noise
        truth = get_true_position(t)
        obs = truth + np.random.normal(size=truth.shape, scale=gps_noise_std)

        # Predict and update the Kalman filter
        ukf.predict()
        ukf.update(obs)

        record.append([t] + truth.flatten().tolist() + obs.flatten().tolist() + ukf.x.flatten().tolist() + ukf.P.flatten().tolist())
    record = np.array(record)

    # Visualize the results
    plt.figure()
    plt.plot(record[:,1], record[:,2], color='r', label='Truth')
    plt.plot(record[:,3], record[:,4], color='b', label='Observation')
    plt.plot(record[:,5], record[:,6], color='g', label='UKF')
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
    plt.plot(record[:,0], record[:,5], color='g', label='UKF')
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], record[:,2], color='r', label='Truth')
    plt.plot(record[:,0], record[:,4], color='b', label='Observation')
    plt.plot(record[:,0], record[:,6], color='g', label='UKF')
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(record[:,0], (360.-90.)/(6.-0.) * record[:,0] + 90., color='r', label='Truth')
    plt.plot(record[:,0], record[:,7] * 180 / np.pi, color='g', label='UKF')
    plt.xlabel('Time')
    plt.ylabel(r'Orientaiton $\theta$')
    plt.grid()
    plt.legend()