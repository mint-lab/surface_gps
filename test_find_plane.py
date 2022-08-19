import pickle
import numpy as np
from scipy.linalg import null_space
from time import time
from multiprocessing import Pool
import multiprocessing as mp
import json


def set_param(path):
    with open(path, "r") as json_file:
        json_data = json.load(json_file)
        
    fx = json_data["fx"]
    fy = json_data["fy"]
    cx = json_data["cx"]
    cy = json_data["cy"]
    
    return fx, fy, cx, cy

def set_random_data(data, num=100):
    ran_1 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_2 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_3 = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    
    ran_pix = np.vstack((ran_1, ran_2))
    ran_pix = np.vstack((ran_pix, ran_3))
    return ran_pix.T

def find_plane(loc ,thres, repeat, thread):
    ran_point = set_random_data(loc, repeat)
    p = Pool(thread)
    
    c_null_v = p.starmap(cal_r, [(loc, points, thres) for points in ran_point])
    
    p.close()
    p.join()
    
    return np.array(c_null_v)

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

def get_main_plane(data, repeat, threshold=0.001, n_worker=mp.cpu_count()):
    result = find_plane(data, threshold, repeat, n_worker) 
    max_result = result[np.argmax(result[:,0], axis=0)]
    
    return max_result
    
def delete_nun_inf(data):
    nan_deleted_data = data[~np.isnan(data)]
    inf_deleted_data = nan_deleted_data[~np.isinf(nan_deleted_data)]
    result_data = np.reshape(inf_deleted_data,(int(inf_deleted_data.size/3),3)).T
    
    return result_data

if __name__ == '__main__':
    start_t = time()
    
    with open("data_files/test_0815.pickle", "rb") as f: #load depth_value
        datas = pickle.load(f)
        
        
    fx, fy, cx, cy = set_param("data_files/zed_param.json")   #get camera parameter
    height, width = np.shape(datas)
    
    point_3d = get_3d(datas)   #get 3d potints origin from cx, cy
    
    main_plane = get_main_plane(point_3d, 100)   #find main plane
    

    end_t = time()
    print(end_t-start_t)