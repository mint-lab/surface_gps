import numpy as np
import cv2 as cv
import opencx as cx
from sensorpy.zed import ZED, sl
import open3d as o3d
import time
from scipy.linalg import null_space
import random

class RANSACMultiPlane:
    def __init__(self, length_threshold = 0.1, plane_candidate_n = 1000, plane_n = 3):
        self.length_threshold = length_threshold
        self.plane_candidate_n = plane_candidate_n
        self.plane_n = plane_n
        
    def find_plane(self, data):
        null_v_dic = {}
        points_dic = {}
        pcds = []
        
        null_coeff = np.empty((0, 4))

        ran_1 = np.zeros(self.plane_candidate_n)
        ran_2 = np.zeros(self.plane_candidate_n)
        ran_3 = np.zeros(self.plane_candidate_n)

        r_list = [10, 3, 1]
        while np.any(ran_1 == ran_2) or np.any(ran_1 == ran_3) or np.any(ran_3 == ran_2):
            ran_1 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
            ran_2 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
            ran_3 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
        
        ran_pix = np.vstack((ran_1, ran_2, ran_3)).T
        
        for points in ran_pix:
            p1, p2, p3 = points[0], points[1], points[2]
            matrix = np.array([data[p1],
                               data[p2],
                               data[p3]])
            matrix = np.hstack((matrix, np.array([[1],[1],[1]])))
            A, B, C, D = np.squeeze(null_space(matrix))
            null_coeff = np.vstack((null_coeff, [A, B, C, D]))
        
        for n_plane in range(self.plane_n):
            coeff_arr = null_coeff.copy()      
            
            for it in range(3):
                if it<2:
                    s_data = data[random.sample(range(0, np.shape(data)[0]), np.shape(data)[0]//(10**(3-it))),:]
                else:
                    s_data = data
                coeff_arr = self.count_points(s_data, coeff_arr, r_list[it])
        
            A, B, C, D = coeff_arr[0]
            x = data[:,0]
            y = data[:,1]
            z = data[:,2]
            h = np.abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
            inner_index = h<self.length_threshold
        
            points_dic[n_plane] = data[inner_index]
            null_v_dic[n_plane] = coeff_arr[0]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_dic[n_plane].astype(np.float64)))
            pcd.paint_uniform_color(np.abs(coeff_arr[0][:-1]))
            pcds.append(pcd)
            data = data[~inner_index]
        
        return null_v_dic, pcds
    
    def count_points(self, s_datas, coeff_arr, r_num):
        result = np.empty((0,5))
        x = s_datas[:,0]
        y = s_datas[:,1]
        z = s_datas[:,2]
        
        for enum in coeff_arr:
            A, B, C, D = enum
            h = abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
            inner_index = h<self.length_threshold
            in_n = inner_index[np.where(inner_index == True)].size
            result=np.vstack((result, [in_n, A, B, C, D]))
            
        result = np.squeeze(result[[result[:, 0].argsort()[::-1]]])
        
        return result[:r_num, 1:]
