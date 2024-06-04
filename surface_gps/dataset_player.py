import numpy as np
from scipy.spatial.transform import Rotation


def conjugate(q_xyzw:np.array) -> np.array:
    '''Return the conjugate of a quaternion q_xyzw = [x, y, z, w] as [-x, -y, -z, w]'''
    return np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]])


def hamilton_product(q1:np.array, q2:np.array) -> np.array:
    '''Return the Hamilton product of two quaternions q1 and q2 as q1*q2 = [x, y, z, w]'''
    return np.array([
        q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1],
        q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0],
        q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3],
        q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]])


def hamilton_rotate(q_xyzw:np.array, v_xyz:np.array) -> np.array:
    '''Return the rotated vector v_xyz by the quaternion q_xyzw as a 3D vector'''
    return hamilton_product(hamilton_product(q_xyzw, np.array([v_xyz[0], v_xyz[1], v_xyz[2], 0])), conjugate(q_xyzw))[:3]


def make_cv_trajectory(length=100, timestep=1, v=1, w=0, p_init=[0, 0, 0], euler_init=[0, 0, 0]):
    '''Generate time-stampled positions and orientations with a constant linear and angular velocity'''
    # Calculate the displacement from the constant linear and angular velocity
    if type(v) is not np.array or type(v) is not list or type(v) is not tuple:
        v = [v, 0, 0] # Assume the 1D velocity is along the x-axis
    if type(w) is not np.array or type(w) is not list or type(w) is not tuple:
        w = [0, 0, w] # Assume the 1D angular velocity is around the z-axis
    vdt = np.array(v) * timestep
    wdt = np.array(w) * timestep
    wdt = Rotation.from_euler('xyz', wdt).as_quat()

    # Derive the trajectory with the constant displacement
    t = 0.
    p_txyz  = [(t, np.array(p_init))]                           # [(time, position), ...]
    q_txyzw = [(t, Rotation.from_euler('xyz', euler_init).as_quat())]  # [(time, orientation), ...]
    distance = 0
    while distance < length:
        t += timestep
        p_txyz.append((t, p_txyz[-1][-1] + hamilton_rotate(q_txyzw[-1][-1], vdt)))
        q_txyzw.append((t, hamilton_product(wdt, q_txyzw[-1][-1])))
        distance += np.linalg.norm(vdt)
    return {'position': p_txyz, 'orientation': q_txyzw}


def add_gaussian_noise(true_data, mean=0., std_dev=0.1):
    '''Return the noisy data with Gaussian noise added to the true data'''
    noisy_data = []
    for (t, data) in true_data:
        noisy_data.append((t, data + np.random.normal(mean, std_dev, data.shape)))
    return noisy_data


def load_rosbag_file(bag_file:str, topic_names:list, target_names:list=[]) -> dict:
    '''Load a ROS bag file and return the desired topics'''
    from pathlib import Path
    from rosbags.highlevel import AnyReader

    bag_file_ = Path(bag_file)
    topic_names_ = [topic if topic.startswith('/') else '/' + topic for topic in topic_names]
    if len(target_names) != len(topic_names_):
        target_names = topic_names_

    dataset = {}
    with AnyReader([bag_file_]) as reader:
        for topic_name, target_name in zip(topic_names_, target_names):
            dataset[target_name] = []
            connections = [x for x in reader.connections if x.topic == topic_name]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                time_sec = timestamp / 10**9 # Convert nanoseconds to seconds
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.msgtype == 'sensor_msgs/msg/NavSatFix':
                    dataset[target_name].append((time_sec, (np.array([msg.latitude, msg.longitude, msg.altitude]), msg.position_covariance)))
                elif connection.msgtype == 'sensor_msgs/msg/Imu':
                    dataset[target_name].append((time_sec, (np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]), msg.orientation_covariance)))
                elif connection.msgtype == 'sensor_msgs/msg/FluidPressure':
                    dataset[target_name].append((time_sec, (msg.fluid_pressure, msg.variance)))
    return dataset


class DatasetPlayer:
    '''A dataset reader to play a dataset of time-stamped data with different types of data'''
    def __init__(self, dataset):
        '''A constructor'''
        self._type = list(dataset.keys())
        self._dataset = list(dataset.values())
        self._cursor = np.zeros(len(self._dataset), dtype=int)

    def reset(self):
        '''Reset the cursor to the beginning of the dataset'''
        self._cursor = np.zeros(len(self._dataset), dtype=int)

    def get_next(self):
        '''Return the next time-stamped data from the dataset'''
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