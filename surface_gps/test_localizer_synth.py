import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from dataset_player import make_cv_trajectory, add_gaussian_noise, DatasetPlayer
from simple_localizer import SimpleLocalizer


if __name__ == '__main__':
    # Generate synthetic data
    # truth = make_cv_trajectory(length=10, timestep=0.5, v=0.1, euler_init=(0, np.deg2rad(-10), 0)) # A straight line with a 10-degree pitch
    truth = make_cv_trajectory(length=10, timestep=0.5, v=0.1, w=np.deg2rad(2), euler_init=(0, np.deg2rad(-10), 0)) # A circular trajectory
    dataset = {
        'position':     add_gaussian_noise(truth['position'], std_dev=[0.03, 0.03, 0.3]),
        'orientation':  add_gaussian_noise(truth['orientation'], std_dev=0.01)}

    # Instantiate the localizer
    localizer = SimpleLocalizer()
    localizer.apply_config({'p_weight': 0.2, 'q_weight': 0.2})

    # Perform localization and record its results
    player = DatasetPlayer(dataset)
    results = {'time': [], 'position': [], 'orientation': []}
    while True:
        type, time, data = player.get_next()
        if type is None:
            break
        success = localizer.apply_data(type, data, time)
        if not success:
            print(f'Failed to localize at timestamp={time} (data type: {type})')
        results['time'].append(time)
        p, q = localizer.get_pose()
        results['position'].append(p)
        results['orientation'].append(q)

    # Prepare the data for plotting
    line_width = 2
    orientation_step = 20
    orientation_length = 0.1
    orientation_width = 0.01
    orientation_alpha = 0.5
    true_ts = np.array([time for time, _ in truth['position']])
    true_ps = np.array([data for _, data in truth['position']])
    data_ts = np.array([time for time, _ in dataset['position']])
    data_ps = np.array([data for _, data in dataset['position']])
    algo_ts = np.array(results['time'])
    algo_ps = np.array(results['position'])

    # Plot the results on the X-Y plnae
    fig = plt.figure()
    plt.plot(true_ps[:, 0], true_ps[:, 1], 'g-', linewidth=line_width, label='Truth')
    plt.plot(data_ps[:, 0], data_ps[:, 1], 'b.', linewidth=line_width, label='Data')
    plt.plot(algo_ps[:, 0], algo_ps[:, 1], 'r-', linewidth=line_width, label='Localizer')
    for i in range(0, len(algo_ps), orientation_step):
        p = results['position'][i]
        q = results['orientation'][i]
        R = Rotation.from_quat(q).as_matrix()
        dx = R @ np.array([orientation_length, 0, 0])
        dy = R @ np.array([0, orientation_length, 0])
        plt.arrow(p[0], p[1], dx[0], dx[1], color='r', width=orientation_width, alpha=orientation_alpha, edgecolor=None)
        plt.arrow(p[0], p[1], dy[0], dy[1], color='g', width=orientation_width, alpha=orientation_alpha, edgecolor=None)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()

    # Plot the results on the time-Z plnae
    fig = plt.figure()
    plt.plot(true_ts, true_ps[:, -1], 'g-', linewidth=line_width, label='Truth')
    plt.plot(data_ts, data_ps[:, -1], 'b.', linewidth=line_width, label='Data')
    plt.plot(algo_ts, algo_ps[:, -1], 'r-', linewidth=line_width, label='Localizer')
    plt.xlabel('Time [sec]')
    plt.ylabel('Z [m]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()