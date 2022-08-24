import pickle
import numpy as np
from scipy.linalg import null_space
from time import time
import json
import matplotlib.pyplot as plt

def find_plane(data ,thres, repeat, plane_n):
    null_v_dic = {}
    points_dic = {}
    
    for it in range(plane_n):
        ran_1 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=repeat, dtype=int)
        ran_2 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=repeat, dtype=int)
        ran_3 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=repeat, dtype=int)
    
        ran_pix = np.vstack((ran_1, ran_2))
        ran_pix = np.vstack((ran_pix, ran_3)).T
    
        max_v = []
        max_points = []
        max_index = np.full((np.shape(data)[1]), False)
    
        for points in ran_pix:
            p1, p2, p3 = points[0], points[1], points[2]
            matrix = np.array([data[p1],
                               data[p2],
                               data[p3]])
        
            matrix = np.hstack((matrix, np.array([[1],[1],[1]])))
        
            A, B, C, D = np.squeeze(null_space(matrix))
        
            x = data[:,0]
            y = data[:,1]
            z = data[:,2]
            h = abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
            inner_index = h<thres
            
            if len(h[inner_index]) > len(max_index[max_index==True]):
                max_points = data[inner_index]
                max_v = np.array([A, B, C, D])
                max_index = inner_index
        
        points_dic[it] = max_points
        null_v_dic[it] = max_v
        data = data[~max_index]
    
    points_dic[plane_n] = data
    
    return {'null_v' : null_v_dic, 'points' : points_dic}
        
        

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
    

def plot_planes(planes, n_point):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    
    for i in range(len(planes)):
        points = planes[i]
        np.random.shuffle(points)
        ax.scatter(points[:n_point, 0], points[:n_point, 1], points[:n_point, 2])
        
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
if __name__ == '__main__':
    start_t = time()
    show_plot = True
    
    with open("data_files/test_0815.pickle", "rb") as f: #load depth_value
        depth_datas = pickle.load(f)
        
    with open("data_files/zed_param.json", "r") as json_file:
        param_data = json.load(json_file)
        
        
    fx, fy, cx, cy = param_data['fx'], param_data['fy'], param_data['cx'], param_data['cy'] #get camera parameter
    
    point_3d = get_3d(depth_datas)   #get 3d potints origin from cx, cy
    
    planes = find_plane(point_3d, 0.03, 100, 3)   #find planes
    
    if show_plot:
        plot_planes(planes['points'], 2000)

    end_t = time()
    print(end_t-start_t)