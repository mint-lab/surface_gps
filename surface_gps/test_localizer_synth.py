import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from simple_localizer import SimpleLocalizer


def conjugate(q_xyzw:np.array) -> np.array:
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]])


def hamilton_product(q1:np.array, q2:np.array) -> np.array:
    return np.array([
        q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1],
        q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0],
        q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3],
        q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]])


def hamilton_rotate(q_xyzw:np.array, v_xyz:np.array) -> np.array:
    return hamilton_product(hamilton_product(q_xyzw, np.array([v_xyz[0], v_xyz[1], v_xyz[2], 0])), conjugate(q_xyzw))[:3]


def make_cv_trajectory(length=100, timestep=1, v=1, w=0, p_init=[0, 0, 0], euler_init=[0, 0, 0]):
    # Calculate the displacement from the constant linear and angular velocity
    if type(v) is float or type(v) is int:
        v = [v, 0, 0] # Assume the 1D velocity is along the x-axis
    if type(w) is float or type(w) is int:
        w = [0, 0, w] # Assume the 1D angular velocity is around the z-axis
    vdt = np.array(v) * timestep
    wdt = np.array(w) * timestep
    wdt = R.from_euler('xyz', wdt).as_quat()

    # Derive the trajectory with the constant displacement
    t = 0.
    p_txyz  = [(t, np.array(p_init))]                           # [(time, position), ...]
    q_txyzw = [(t, R.from_euler('xyz', euler_init).as_quat())]  # [(time, orientation), ...]
    distance = 0
    while distance < length:
        t += timestep
        p_txyz.append((t, p_txyz[-1][-1] + hamilton_rotate(q_txyzw[-1][-1], vdt)))
        q_txyzw.append((t, hamilton_product(q_txyzw[-1][-1], wdt)))
        distance += np.linalg.norm(vdt)
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
    true_pos, true_ori = make_cv_trajectory(length=10, timestep=0.5, v=0.1, euler_init=(0, np.deg2rad(-10), 0))
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