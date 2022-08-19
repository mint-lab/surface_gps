import pickle
import numpy as np
import matplotlib.pyplot as plt
import json

def set_data(path):
    with open(path, "r") as json_file:
        json_data = json.load(json_file)
        
    fx = json_data["fx"]
    fy = json_data["fy"]
    cx = json_data["cx"]
    cy = json_data["cy"]
    
    return fx, fy, cx, cy

def set_random_data(data, num=100):
    ran_x = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_y = np.random.randint(low=0, high=np.shape(data)[0]-1, size=num, dtype=int)
    ran_pix = np.vstack((ran_x, ran_y)).T
    ran_loc = np.empty((0, 3))
    
    for point in ran_pix:
        pos = data[point[1],point[0],:]
        ran_loc = np.vstack((ran_loc, pos))
    
    return ran_loc

def get_3d(data):
    pix_x = np.linspace(0, 1279, 1280)
    pix_y = np.linspace(0, 719, 720)
    X, Y = np.meshgrid(pix_x, pix_y)

    pose_x = ((X-cx)*datas)/fx
    pose_y = ((cy-Y)*datas)/fy

    loc_data = np.dstack([pose_x, pose_y])
    loc_data = np.dstack([loc_data, datas])
    
    return loc_data
    

if __name__ == '__main__':
    with open("data_files/test_0815.pickle", "rb") as f: #load depth_value
        datas = pickle.load(f)
        
    show_plot = True
    
    fx, fy, cx, cy = set_data("data_files/zed_param.json")
    
    loc_data = get_3d(datas)
            
    if show_plot:
        ran_loc = set_random_data(loc_data, 5000)
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(ran_loc[:,0], ran_loc[:,1], ran_loc[:,2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")