import pickle
import numpy as np
from scipy.linalg import null_space
from time import time
from multiprocessing import Pool
import multiprocessing as mp
import json
import matplotlib.pyplot as plt


def set_param(path):
    with open(path, "r") as json_file:
        json_data = json.load(json_file)
        
    fx = json_data["fx"]
    fy = json_data["fy"]
    cx = json_data["cx"]
    cy = json_data["cy"]
    
    return fx, fy, cx, cy

def set_random_data(data, num):
    ran_1 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_2 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_3 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    
    ran_pix = np.vstack((ran_1, ran_2))
    ran_pix = np.vstack((ran_pix, ran_3))
    return ran_pix.T

def get_max_null_v(data ,thres, repeat, n_worker):
    ran_point = set_random_data(data, repeat)
    p = Pool(n_worker)
    
    c_null_v = p.starmap(cal_r, [(data, points, thres) for points in ran_point])
    
    p.close()
    p.join()
    
    c_null_v = np.array(c_null_v)
    max_result = c_null_v[np.argmax(c_null_v[:,0], axis=0)]
    
    return max_result

def cal_r(data, points, thres):   #Compute the number of points that less than threshold
    p1, p2, p3 = points[0], points[1], points[2]
    matrix = np.array([data[:,p1],
                        data[:,p2],
                        data[:,p3]])
    
    matrix = np.hstack((matrix, np.array([[1],[1],[1]])))
    
    A, B, C, D = np.squeeze(null_space(matrix))
        
    x = data[0]
    y = data[1]
    z = data[2]
    
    h = abs(A*x+B*y+C*z+D)/(np.sqrt(A**2+B**2+C**2))
    h = np.size(h[h<0.005])
    
    return h, A, B, C, D
        

def get_3d(data):
    pix_x = np.linspace(0, np.shape(data)[1]-1, np.shape(data)[1])
    pix_y = np.linspace(0, np.shape(data)[0]-1, np.shape(data)[0])
    X, Y = np.meshgrid(pix_x, pix_y)

    pose_x = ((X-cx)*datas)/fx
    pose_y = ((cy-Y)*datas)/fy

    loc_data = np.dstack([pose_x, pose_y])
    loc_data = np.dstack([loc_data, datas])
    
    result_data = delete_nun_inf(loc_data)
    
    return result_data
    
def delete_nun_inf(data):
    nan_deleted_data = data[~np.isnan(data)]
    inf_deleted_data = nan_deleted_data[~np.isinf(nan_deleted_data)]
    result_data = np.reshape(inf_deleted_data,(int(inf_deleted_data.size/3),3)).T
    
    return result_data

def find_in_out(null_v, data, thres):
    data = data
    A, B, C, D = null_v
    
    x = data[0]
    y = data[1]
    z = data[2]
    
    h = abs(A*x+B*y+C*z+D)/(np.sqrt(A**2+B**2+C**2))
    in_data = data.T[h<thres].T
    out_data = data.T[h>=thres].T
    
    return in_data, out_data
    
def find_plane(data ,thres, repeat, n_planes, n_worker=mp.cpu_count()):
    null_v = {}
    points = {}
    for i in range(n_planes):
        max_null_v = get_max_null_v(data, thres, repeat, n_worker)
        in_data, out_data = find_in_out(max_null_v[1:], data, thres)
        points[i] = in_data
        null_v[i] = max_null_v
        data = out_data
        print("Find Plane", i)
    points[n_planes] = data
        
    return {'null_v' : null_v, 'points' : points}

def plot_planes(planes, n_point):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    
    for i in range(len(planes)):
        points = planes[i].T
        np.random.shuffle(points)
        ax.scatter(points[:n_point, 0], points[:n_point, 1], points[:n_point, 2])
        
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
if __name__ == '__main__':
    start_t = time()
    show_plot = True
    
    with open("data_files/test_0815.pickle", "rb") as f: #load depth_value
        datas = pickle.load(f)
        
        
    fx, fy, cx, cy = set_param("data_files/zed_param.json")   #get camera parameter
    height, width = np.shape(datas)
    
    point_3d = get_3d(datas)   #get 3d potints origin from cx, cy
    
    planes = find_plane(point_3d, 0.05, 100, 3)   #find planes
    
    if show_plot:
        plot_planes(planes['points'], 1000)

    end_t = time()
    print(end_t-start_t)