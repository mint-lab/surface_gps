import pickle
import zed
import numpy as np

def set_data(info):
    param = info.calibration_parameters.left_cam
    fx = param.fx
    fy = param.fy
    cx = param.cx
    cy = param.cy
    
    return fx, fy, cx, cy
    

if __name__ == '__main__':
    with open("depth_data.pickle", "rb") as f: #load depth_value
        datas = pickle.load(f)
    
    cam = zed.ZED()
    cam.open()
    cam_info = cam.camera.get_camera_information()
    fx, fy, cx, cy = set_data(cam_info)
    
    width = cam_info.camera_resolution.width
    height = cam_info.camera_resolution.height
    
    i=0
    j=0
    
    loc_data = np.empty((height, width, 3))

    for data in np.nditer(datas):
        n_x = i-cx
        n_y = cy-j
        r_x = (n_x*data)/fx
        r_y = (n_y*data)/fy
   
        loc_data[j][i] = (r_x, r_y, data)
   
        i += 1
        if i==width:
            i=0
            j +=1

    cam.close()