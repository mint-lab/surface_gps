import pickle
import numpy as np
from scipy.linalg import null_space
from time import time
import json
import open3d as o3d
import random

def find_plane(data ,thres, candi_n, plane_n):
    null_v_dic = {}
    points_dic = {}
    
    null_coeff = np.empty((0, 4))

    ran_1 = np.zeros(candi_n)
    ran_2 = np.zeros(candi_n)
    ran_3 = np.zeros(candi_n)

    r_list = [10, 3, 1]
    while np.any(ran_1 == ran_2) or np.any(ran_1 == ran_3) or np.any(ran_3 == ran_2):
        ran_1 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=candi_n, dtype=int)
        ran_2 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=candi_n, dtype=int)
        ran_3 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=candi_n, dtype=int)
    
    ran_pix = np.vstack((ran_1, ran_2, ran_3)).T
    
    for points in ran_pix:
        p1, p2, p3 = points[0], points[1], points[2]
        matrix = np.array([data[p1],
                           data[p2],
                           data[p3]])
        matrix = np.hstack((matrix, np.array([[1],[1],[1]])))
        A, B, C, D = np.squeeze(null_space(matrix))
        null_coeff = np.vstack((null_coeff, [A, B, C, D]))
    
    for n_plane in range(plane_n):
        coeff_arr = null_coeff.copy()      
        
        for it in range(3):
            if it<2:
                s_data = data[random.sample(range(0, np.shape(data)[0]), np.shape(data)[0]//(10**(2-it))),:]
            else:
                s_data = data
            coeff_arr = count_points(s_data, thres, coeff_arr, r_list[it])
    
        A, B, C, D = coeff_arr[0]
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]
        h = np.abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
        inner_index = h<thres
    
        points_dic[n_plane] = data[inner_index]
        null_v_dic[n_plane] = coeff_arr
        data = data[~inner_index]
    
    return points_dic, null_v_dic

def get_3d(data):
    pix_x = np.linspace(0, np.shape(data)[1]-1, np.shape(data)[1])
    pix_y = np.linspace(0, np.shape(data)[0]-1, np.shape(data)[0])
    X, Y = np.meshgrid(pix_x, pix_y)

    pose_x = ((X-cx)*data)/fx
    pose_y = ((cy-Y)*data)/fy

    loc_data = np.dstack([pose_x, pose_y])
    loc_data = np.dstack([loc_data, data])
    
    nan_deleted_data = loc_data[~np.isnan(loc_data)]
    inf_deleted_data = nan_deleted_data[~np.isinf(nan_deleted_data)]
    result_data = np.reshape(inf_deleted_data,(int(inf_deleted_data.size/3),3))
    
    return result_data

def count_points(s_datas, thres, coeff_arr, r_num):
    result = np.empty((0,5))
    x = s_datas[:,0]
    y = s_datas[:,1]
    z = s_datas[:,2]
    
    for enum in coeff_arr:
        A, B, C, D = enum
        h = abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
        inner_index = h<thres
        in_n = inner_index[np.where(inner_index == True)].size
        result=np.vstack((result, [in_n, A, B, C, D]))
        
    result = np.squeeze(result[[result[:, 0].argsort()[::-1]]])
    
    return result[:r_num, 1:]

def plot_planes(planes):
    plane_geo = []
    for it in range(len(planes)):
        geo = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(planes[it]))
        color_v = np.zeros_like(planes[it])
        if it<3:
            color_v[:,it] = 1
        geo.colors = o3d.utility.Vector3dVector(color_v)
        plane_geo.append(geo)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    o3d.visualization.draw_geometries(plane_geo+[sphere])
    
if __name__ == '__main__':
    start_t = time()
    
    with open("data_files/test_0815.pickle", "rb") as f: #load depth_value
        depth_datas = pickle.load(f)
        
    with open("data_files/zed_param.json", "r") as json_file:
        param_data = json.load(json_file)
        
    fx, fy, cx, cy = param_data['fx'], param_data['fy'], param_data['cx'], param_data['cy'] #get camera parameter
    
    point_3d = get_3d(depth_datas)   #get 3d potints origin from cx, cy
    
    points, nullvs = find_plane(point_3d, 0.05, 100, 3)   #find planes
    
    end_t = time()
    print(end_t-start_t)
    
    plane_geos = plot_planes(points)