import pickle
import zed
import numpy as np
from scipy.linalg import null_space
from time import time

def set_data(info):
    param = info.calibration_parameters.left_cam
    fx = param.fx
    fy = param.fy
    cx = param.cx
    cy = param.cy
    
    return fx, fy, cx, cy

def set_random_data(data, num=100):
    ran_loc = np.empty((0, 3))
    ran_x = np.random.randint(low=0, high=np.shape(data)[1]-1, size=num, dtype=int)
    ran_y = np.random.randint(low=0, high=np.shape(data)[0]-1, size=num, dtype=int)
    ran_pix = np.vstack((ran_x, ran_y)).T
    
    for point in ran_pix:
        pos = data[point[1],point[0],:]
        ran_loc = np.vstack((ran_loc, pos))
        
    ran_loc = delete_nan_inf(ran_loc)
    
    return ran_loc

def find_plane(loc ,thres):
    data = np.nan_to_num(loc, copy=True)
    d_v = np.array(([1],[1],[1]))
    i = 0;
    m_count = 0;
    m_null_v = []
    while i<loc.shape[0]:
        print(i)
        j = i+1
        while j<loc.shape[0]:
            k = j+1
            while k<loc.shape[0]:
                data_v = np.empty((0,3))
                count = 0
                if (np.isnan(loc[i]).any() or np.isnan(loc[j]).any() or np.isnan(loc[k]).any()) == False:
                    data_v = np.vstack((data_v, data[i]))
                    data_v = np.vstack((data_v, data[j]))
                    data_v = np.vstack((data_v, data[k]))
                    data_v = np.hstack((data_v, d_v))
                    
                    null_v = null_space(data_v)
                    a, b, c, d = null_v[0], null_v[1], null_v[2], null_v[3]
                    for point in data:
                        h = (np.abs(a*point[0]+b*point[1]+c*point[2]+d))/(np.sqrt(a**2+b**2+c**2))
                            
                        if h<thres:
                            count += 1
                            
                        
                    if count > m_count:
                        m_null_v = null_v
                        m_count = count
                        
                
                k+=1
                
            j+=1
            
        i+=1
        
    return m_null_v, m_count

def delete_nan_inf(data):
    n = 0
    count = 0
    while n<np.shape(data)[0]:
        if np.isnan(data[n]).any() or np.isinf(data[n]).any():
            data = np.delete(data, n, 0)
            n-=1
            count+=1
        n+=1
        
    print("Deleted point :",count)
        
    return data

if __name__ == '__main__':
    start_t = time()
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
            
    ran_loc = set_random_data(loc_data, 70)
    null_v, m_count = find_plane(ran_loc, 0.005)

    cam.close()
    end_t = time()
    print(end_t-start_t)