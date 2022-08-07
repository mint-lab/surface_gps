import zed
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def draw_circle(image, depth):
    img = np.hstack((image, depth))
    img = cv.resize(img, (1280, 720))
    d_img = cv.circle(img.copy(), (30,30), 15, (0, 0, 255), -1)
    return img, d_img

if __name__ == "__main__": 
    cam = zed.ZED()
    output_path = 'C:/Mintlab/zed cam/test_ani_0805'
    cam.open(fps=60)
    cam.start_recording(output_path)
    cam.tracking_pose()
    show_plot = True
    
    sensor_dic = []
    
    if show_plot:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.grid(True)
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel('frame')
        ax2.set_ylabel('baromter')
        ax2.grid(True)
        line1, = ax1.plot([], [], [], 'k')
        line2, = ax2.plot([], [], 'r')
    
    n_frame = 0
    s_frame = 600
    poses = np.empty((0,3))
    frames, baro = [], []
    
    while True:
        if cam.grab():
            if cam.get_sensors_data():
                l_image, r_image, d_image = cam.get_images()
                time, baro_v, t_pose, imu_v = cam.get_sensor()
                sensor_dic.append({'time':time, 'baro':baro_v, 't_pose':t_pose, 'imu':imu_v})
                
                poses = np.vstack((poses,t_pose[:3]))
                baro = np.append(baro, baro_v)
                frames = np.append(frames, n_frame)
                
                img, d_img = draw_circle(l_image, d_image)
                cv.imshow('ZED: RGB, DEPTH', d_img)
                
                if show_plot:
                    ax1.set_xlim([np.min(poses[:,0]-0.1), np.max(poses[:,0]+0.1)])
                    ax1.set_ylim([np.min(poses[:,1]-0.1), np.max(poses[:,1]+0.1)])
                    ax1.set_zlim([np.min(poses[:,2]-0.1), np.max(poses[:,2]+0.1)])
                    ax2.set_ylim([np.min(baro)-0.1, np.max(baro)+0.1])
                    if n_frame%20==0:
                        if n_frame<s_frame:
                            ax2.set_xlim([0, frames[-1]+1])
                            line2.set_data(frames[:], baro[:])
                        
                        else:
                            ax2.set_xlim([n_frame-s_frame, frames[-1]+1])
                            line2.set_data(frames[n_frame-s_frame:], baro[n_frame-s_frame:])
                            
                        line1.set_data_3d(poses[:,0], poses[:,1], poses[:,2])
                        plt.draw()
        
                n_frame += 1
            
        key = cv.waitKey(1)
        if key == ord(' '): # Space
            cv.imshow("ZED: RGB, DEPTH", img)
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break
    
    zed.save_pickle(sensor_dic, output_path)
    cv.destroyAllWindows()
    cam.close()
    plt.ioff()
    plt.close()
    