import pickle
import zed
import numpy as np
import matplotlib.pyplot as plt

def set_data(info):
    param = info.calibration_parameters.left_cam
    fx = param.fx
    fy = param.fy
    cx = param.cx
    cy = param.cy
    
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
    

if __name__ == '__main__':
    with open("depth_file.pickle", "rb") as f: #load depth_value
        datas = pickle.load(f)
    
    cam = zed.ZED()
    cam.open()
    cam_info = cam.camera.get_camera_information()
    fx, fy, cx, cy = set_data(cam_info)
    
    width = cam_info.camera_resolution.width
    height = cam_info.camera_resolution.height
    
    col=0
    row=0
    
    loc_data = np.empty((height, width, 3))
    show_plot = True

    for data in np.nditer(datas):
        n_x = col-cx
        n_y = cy-row
        r_x = (n_x*data)/fx
        r_y = (n_y*data)/fy
   
        loc_data[row][col] = (r_x, r_y, data)
   
        col += 1
        if col==width:
            col=0
            row +=1
            
    if show_plot:
        ran_loc = set_random_data(loc_data, 5000)
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(ran_loc[:,0], ran_loc[:,1], ran_loc[:,2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    cam.close()