import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pyproj import Transformer
from simple_localizer import SimpleLocalizer
from dataset_player import load_rosbag_file, DatasetPlayer


if __name__ == '__main__':
    # Read the real data
    bag_file = '../../ros2_ws/data/HYWC_linear'
    topics = {'gps'     : '/ublox/fix',
              'ahrs'    : '/imu/data',
              'pressure': '/zed2/zed_node/atm_press'}
    dataset = load_rosbag_file(bag_file, list(topics.values()), list(topics.keys()))

    # Instantiate the localizer
    localizer = SimpleLocalizer()
    localizer.set_config({'ahrs_robot2sensor_quat': Rotation.from_euler('xyz', [0, 0, -np.pi/2]).as_quat()})

    # Perform localization and record its results
    player = DatasetPlayer(dataset)
    results = {'time': [], 'position': [], 'orientation': []}
    while True:
        type, time, data = player.get_next()
        if type is None:
            break
        success = localizer.apply_data(type, data, time)
        if not success:
            print(f'Failed to apply the data at timestamp={time} (data type: {type})')
        results['time'].append(time)
        p, q = localizer.get_pose()
        results['position'].append(p)
        results['orientation'].append(q)

    # Prepare the data for plotting
    line_width = 2
    orientation_step = 100
    orientation_length = 0.2
    orientation_width = 0.02
    orientation_alpha = 0.5
    data_ts, data_ps = [], []
    epsg_convertor = Transformer.from_crs('EPSG:4326', 'EPSG:5186')
    y, x = epsg_convertor.transform(*localizer.get_gps_origin())
    epsg_offset = np.array([x, y, 0])
    for time, (gps_geo, gps_cov) in dataset['gps']:
        data_ts.append(time)
        y, x = epsg_convertor.transform(gps_geo[0], gps_geo[1])
        data_ps.append([x, y, gps_geo[2]] - epsg_offset)
    data_ts, data_ps = np.array(data_ts), np.array(data_ps)
    algo_ts = np.array(results['time'])
    algo_ps = np.array(results['position'])

    # Plot the data on the X-Y plnae
    fig = plt.figure()
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

    # Plot the data on the time-Z plnae
    fig = plt.figure()
    plt.plot(data_ts, data_ps[:, -1], 'b.', linewidth=line_width, label='Data')
    plt.plot(algo_ts, algo_ps[:, -1], 'r-', linewidth=line_width, label='Localizer')
    plt.xlabel('Time [sec]')
    plt.ylabel('Z [m]')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()