import numpy as np
import matplotlib.pyplot as plt
from simple_localizer import SimpleLocalizer


def gen_straight_line(length=10, speed=0.1, time_step=0.5):
    t = 0.
    p_txyz = [(t, np.zeros(3))]                 # [(time, position), ...]
    q_txyzw = [(t, np.array([0., 0., 0., 1.]))] # [(time, orientation), ...]

    displacement = np.array([speed * time_step, 0, 0])
    while p_txyz[-1][-1][0] < length:
        t += time_step
        p_txyz.append((t, p_txyz[-1][-1] + displacement))
        q_txyzw.append((t, q_txyzw[-1][-1].copy()))

    return p_txyz, q_txyzw


def add_gaussian_noise(true_data, mean=0., std_dev=0.1):
    noisy_data = []
    for (t, data) in true_data:
        noisy_data.append((t, data + np.random.normal(mean, std_dev, data.shape)))
    return noisy_data


class DatasetPlayer:
    def __init__(self, dataset):
        self._type = list(dataset.keys())
        self._dataset = list(dataset.values())
        self._cursor = np.zeros(len(self._dataset), dtype=int)

    def reset(self):
        self._cursor = np.zeros(len(self._dataset), dtype=int)

    def get_next(self):
        # Select the next data to play
        next_times = [data[cursor][0] if cursor < len(data) else np.inf for (data, cursor) in zip(self._dataset, self._cursor)]
        select = np.argmin(next_times)
        if next_times[select] == np.inf:
            return None, None, None

        # Advance the cursor and return the selected data
        type = self._type[select]
        timestamp, data = self._dataset[select][self._cursor[select]]
        self._cursor[select] += 1
        return type, timestamp, data


if __name__ == '__main__':
    # Generate synthetic data
    true_pos, true_ori = gen_straight_line()
    dataset = {
        'position':     add_gaussian_noise(true_pos),
        'orientation':  add_gaussian_noise(true_ori)}

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
    true_ts = np.array([time for time, _ in true_pos])
    true_ps = np.array([data for _, data in true_pos])
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
    plt.plot(data_ts, data_ps[:, -1], 'b-', linewidth=2, label='Data')
    plt.plot(estm_ts, estm_ps[:, -1], 'r-', linewidth=2, label='Estimated')
    plt.xlabel('Time [sec]')
    plt.ylabel('Z [m]')
    plt.grid(True)
    plt.tight_layout()

    plt.show()