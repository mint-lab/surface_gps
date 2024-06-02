import numpy as np
import yaml

class SimpleLocalizer:
    '''A simple localizer with moving average'''

    def __init__(self):
        self.p_xyz = np.zeros(3)
        self.q_xyzw = np.array([0., 0., 0., 1.])
        self.p_weight = 0.5
        self.q_weight = 0.5
        self.initialize()

    def initialize(self) -> bool:
        return True

    def load_config_file(self, config_file:str) -> bool:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            if 'SimpleLocalizer' in config_dict:
                return self.apply_config(config_dict['SimpleLocalizer'])
            return self.apply_config(config_dict)
        return False

    def apply_config(self, config_dict:dict) -> bool:
        if 'p_xyz' in config_dict:
            self.p_xyz = np.array(config_dict['p_xyz'])
        if 'q_xyzw' in config_dict:
            self.q_xyzw = np.array(config_dict['q_xyzw'])
        if 'p_weight' in config_dict:
            self.p_weight = config_dict['p_weight']
        if 'q_weight' in config_dict:
            self.q_weight = config_dict['q_weight']
        return True

    def save_config_file(self, config_file:str) -> bool:
        with open(config_file, 'w') as f:
            config_dict = {
                'SimpleLocalizer': {
                    'p_weight': self.p_weight,
                    'q_weight': self.q_weight
                }
            }
            yaml.dump(config_dict, f)
        return False

    def set_pose(self, pose:tuple) -> bool:
        return (self.p_xyz, self.q_xyzw)

    def get_pose(self) -> tuple:
        return (self.p_xyz, self.q_xyzw)

    def apply_position(self, p_xyz:np.array, timestamp:float) -> bool:
        # Apply the linear interpolation (Lerp) between current and new orientation
        self.p_xyz = self.p_weight * p_xyz + (1 - self.p_weight) * self.p_xyz
        return True

    def apply_orientation(self, q_xyzw:np.array, timestamp:float) -> bool:
        # Apply the sphericial linear interpolation (Slerp) between current and new orientation
        theta = np.arccos(np.dot(self.q_xyzw, q_xyzw))
        w0 = np.sin(self.q_weight*theta) / np.sin(theta)
        w1 = np.sin((1 - self.q_weight)*theta) / np.sin(theta)
        self.q_xyzw = w0 * q_xyzw + w1 * self.q_xyzw
        return True

    def apply_gps_data(self, latlonalt:np.array, timestamp:float) -> bool:
        return False

    def apply_ahrs_data(self, latlonalt:np.array, timestamp:float) -> bool:
        return False

    def apply_pressure(self, pressure:float, timestamp:float) -> bool:
        return False

    def apply_image(self, image:np.array, timestamp:float) -> bool:
        return False