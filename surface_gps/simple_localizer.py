import numpy as np
from pyproj import Transformer
import yaml
from dataset_player import conjugate, hamilton_product, hamilton_rotate


class SimpleLocalizer:
    '''A simple localizer with moving average'''

    def __init__(self, p_weight:float=0.5, q_weight:float=0.5,
                 gps_origin_latlon=[], gps_espg_from:str='EPSG:4326', gps_espg_to:str='EPSG:5186', gps_robot2sensor_offset=[0, 0, 0],
                 ahrs_robot2sensor_quat=[0, 0, 0, 1]):
        '''A constructor'''
        self._p_weight = p_weight
        self._q_weight = q_weight
        self._gps_origin_latlon = gps_origin_latlon
        self._gps_espg_from = gps_espg_from
        self._gps_espg_to = gps_espg_to
        self._gps_robot2sensor_offset = np.array(gps_robot2sensor_offset)
        self._ahrs_senor2robobt_quat = conjugate(ahrs_robot2sensor_quat)
        self.initialize()

    def initialize(self) -> bool:
        '''Initialize the localizer'''
        self.p_xyz = np.zeros(3)
        self.q_xyzw = np.array([0., 0., 0., 1.])
        self._is_p_first = True
        self._is_q_first = True
        self._gps_origin_xyz = []
        self._gps_espg_convertor = Transformer.from_crs(self._gps_espg_from, self._gps_espg_to)
        return True

    def set_pose(self, pose:tuple) -> bool:
        '''Set the pose of the localizer'''
        return (self.p_xyz, self.q_xyzw)

    def get_pose(self) -> tuple:
        '''Get the current pose of the localizer'''
        return (self.p_xyz, self.q_xyzw)

    def apply_data(self, data_type:str, data:dict, timestamp:float) -> bool:
        '''Apply the data to the localizer accodring to the data type'''
        if data_type == 'position':
            return self.apply_position(data, timestamp)
        elif data_type == 'orientation':
            return self.apply_orientation(data, timestamp)
        elif data_type == 'gps':
            return self.apply_gps_data(data, timestamp)
        elif data_type == 'ahrs':
            return self.apply_ahrs_data(data, timestamp)
        elif data_type == 'pressure':
            return self.apply_pressure(data, timestamp)
        elif data_type == 'image':
            return self.apply_image(data, timestamp)
        return False

    def apply_position(self, p_xyz:np.array, timestamp:float) -> bool:
        '''Apply the position data to the localizer'''
        if self._is_p_first:
            self._is_p_first = False
            self.p_xyz = p_xyz.copy()
        else:
            # Apply the linear interpolation (Lerp) between current and new orientation
            self.p_xyz = self._p_weight * p_xyz + (1 - self._p_weight) * self.p_xyz
        return True

    def apply_orientation(self, q_xyzw:np.array, timestamp:float) -> bool:
        '''Apply the orientation data to the localizer'''
        if self._is_q_first:
            self._is_q_first = False
            self.q_xyzw = q_xyzw.copy()
        else:
            # Apply the sphericial linear interpolation (Slerp) between current and new orientation
            theta = np.arccos(np.dot(self.q_xyzw, q_xyzw))
            if np.fabs(theta) > 1e-6:
                w0 = np.sin(self._q_weight*theta) / np.sin(theta)
                w1 = np.sin((1 - self._q_weight)*theta) / np.sin(theta)
                self.q_xyzw = w0 * q_xyzw + w1 * self.q_xyzw
            else:
                self.q_xyzw = self._q_weight * q_xyzw + (1 - self._q_weight) * self.q_xyzw
        return True

    def apply_gps_data(self, data:tuple, timestamp:float) -> bool:
        '''Apply the GPS data to the localizer'''
        # Transform the GPS data to the metric position
        p_geo = data[0]
        if len(self._gps_origin_xyz) < 2:
            if len(self._gps_origin_latlon) < 2:
                y, x = self._gps_espg_convertor.transform(p_geo[0], p_geo[1])
            else:
                y, x = self._gps_espg_convertor.transform(self._gps_origin_latlon[0], self._gps_origin_latlon[1])
            self._gps_origin_xyz = np.array([x, y, 0])
        y, x = self._gps_espg_convertor.transform(p_geo[0], p_geo[1])
        p_xyz = [x, y, p_geo[2]] - self._gps_origin_xyz

        # Compensate the GPS sensor offset
        p_xyz = p_xyz - hamilton_rotate(self.q_xyzw, self._gps_robot2sensor_offset)
        return self.apply_position(p_xyz, timestamp)

    def apply_ahrs_data(self, data:tuple, timestamp:float) -> bool:
        '''Apply the AHRS data to the localizer'''
        q_xyzw = hamilton_product(self._ahrs_senor2robobt_quat, data[0]) # Compensate the sensor orientation
        return self.apply_orientation(q_xyzw, timestamp)

    def apply_pressure(self, pressure:float, timestamp:float) -> bool:
        '''Apply the pressure data to the localizer'''
        return False

    def apply_image(self, image:np.array, timestamp:float) -> bool:
        '''Apply the image data to the localizer'''
        return False

    def set_config(self, config_dict:dict) -> bool:
        '''Apply the configuration dictionary to the localizer'''
        if 'p_weight' in config_dict:
            self._p_weight = config_dict['p_weight']
        if 'q_weight' in config_dict:
            self._q_weight = config_dict['q_weight']
        if 'gps_origin_latlon' in config_dict:
            self._gps_origin_latlon = config_dict['gps_origin_latlon']
        if 'gps_espg_from' in config_dict:
            self._gps_espg_from = config_dict['gps_espg_from']
        if 'gps_espg_to' in config_dict:
            self._gps_espg_to = config_dict['gps_espg_to']
        if 'gps_robot2sensor_offset' in config_dict:
            self._gps_robot2sensor_offset = np.array(config_dict['gps_robot2sensor_offset'])
        if 'ahrs_robot2sensor_quat' in config_dict:
            self._ahrs_senor2robobt_quat = conjugate(config_dict['ahrs_robot2sensor_quat'])
        return self.initialize()

    def get_config(self) -> dict:
        '''Get the configuration dictionary of the localizer'''
        return {
            'p_weight'                  : self._p_weight,
            'q_weight'                  : self._q_weight,
            'gps_origin_latlon'         : list(self._gps_origin_latlon),
            'gps_espg_from'             : self._gps_espg_from,
            'gps_espg_to'               : self._gps_espg_to,
            'gps_robot2sensor_offset'   : list(self._gps_robot2sensor_offset),
            'ahrs_robot2sensor_quat'    : list(conjugate(self._ahrs_senor2robobt_quat))
        }

    def load_config_file(self, config_file:str) -> bool:
        '''Load the configuration file for the localizer'''
        with open(config_file, 'r') as f:
            class_name = self.__class__.__name__
            config_dict = yaml.safe_load(f)
            if class_name in config_dict:
                return self.set_config(config_dict[class_name])
            return self.set_config(config_dict)
        return False

    def save_config_file(self, config_file:str) -> bool:
        '''Save the configuration file for the localizer'''
        with open(config_file, 'w') as f:
            class_name = self.__class__.__name__
            config_dict = { class_name: self.get_config() }
            yaml.dump(config_dict, f)
        return False