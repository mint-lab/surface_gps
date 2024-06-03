import numpy as np
import matplotlib.pyplot as plt
from synthetic_dataset import make_cv_trajectory, add_gaussian_noise, DatasetPlayer
from simple_localizer import SimpleLocalizer


if __name__ == '__main__':
    # Generate synthetic data
    truth = make_cv_trajectory(length=10, timestep=0.5, v=0.1, euler_init=(0, np.deg2rad(-10), 0))
    dataset = {
        'position':     add_gaussian_noise(truth['position'], std_dev=[0.03, 0.03, 0.3]),
        'orientation':  add_gaussian_noise(truth['orientation'], std_dev=0.01)}

    # Instantiate the localizer
    localizer = SimpleLocalizer()

    # Perform localization
    player = DatasetPlayer(dataset)
    results = {'timestamp': [], 'position': [], 'orientation': []}
    while True:
        type, timestamp, data = player.get_next()
        if type is None:
            break
        elif type == 'position':
            success = localizer.apply_position(data, timestamp)
        elif type == 'orientation':
            success = localizer.apply_orientation(data, timestamp)
        if not success:
            print(f'Failed to localize at timestamp={timestamp} (data type: {type})')
        results['timestamp'].append(timestamp)
        p, q = localizer.get_pose()
        results['position'].append(p)
        results['orientation'].append(q)

    # Plot the results
    true_ts = np.array([time for time, _ in truth['position']])
    true_ps = np.array([data for _, data in truth['position']])
    data_ts = np.array([time for time, _ in dataset['position']])
    data_ps = np.array([data for _, data in dataset['position']])
    estm_ts = np.array(results['timestamp'])
    estm_ps = np.array(results['position'])

    plt.figure()
    plt.plot(true_ps[:, 0], true_ps[:, 1], 'g-', linewidth=2, label='Truth')
    plt.plot(data_ps[:, 0], data_ps[:, 1], 'b.', linewidth=2, label='Data')
    plt.plot(estm_ps[:, 0], estm_ps[:, 1], 'r-', linewidth=2, label='Estimated')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    plt.figure()
    plt.plot(true_ts, true_ps[:, -1], 'g-', linewidth=2, label='Truth')
    plt.plot(data_ts, data_ps[:, -1], 'b.', linewidth=2, label='Data')
    plt.plot(estm_ts, estm_ps[:, -1], 'r-', linewidth=2, label='Estimated')
    plt.xlabel('Time [sec]')
    plt.ylabel('Z [m]')
    plt.grid(True)
    plt.tight_layout()

    plt.show()