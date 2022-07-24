import pyzed.sl as sl
from sys import exit
import pickle
from time import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def save_data(pic_dic):
    baro_value = sensors_data.get_barometer_data().pressure
    
    tx = round(cam_pose.get_translation(py_translation).get()[0], 3)
    ty = round(cam_pose.get_translation(py_translation).get()[1], 3)
    tz = round(cam_pose.get_translation(py_translation).get()[2], 3)
    
    # Display the orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(cam_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(cam_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(cam_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(cam_pose.get_orientation(py_orientation).get()[3], 3)
    t_poses = [tx, ty, tz, ox, oy, oz, ow]
    
    #Display the IMU acceleratoin
    acceleration = [0,0,0]
    cam_imu.get_linear_acceleration(acceleration)
    ax = round(acceleration[0], 3)
    ay = round(acceleration[1], 3)
    az = round(acceleration[2], 3)
    
    #Display the IMU angular velocity
    a_velocity = [0,0,0]
    cam_imu.get_angular_velocity(a_velocity)
    vx = round(a_velocity[0], 3)
    vy = round(a_velocity[1], 3)
    vz = round(a_velocity[2], 3)

    # Display the IMU orientation quaternion
    zed_imu_pose = sl.Transform()
    ox = round(cam_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
    oy = round(cam_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
    oz = round(cam_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
    ow = round(cam_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
    imu_poses = [ax, ay, az, vx, vy, vz, ox, oy, oz, ow]
    
    pic_dic.append({'time' : times, 'pose' : t_poses, 'baro' : baro_value, 'imu' : imu_poses})
    return tx, ty, tz, baro_value
    
def get_image():
    cam.retrieve_image(rgb_image, sl.VIEW.LEFT)
    cam.retrieve_image(depth_image, sl.VIEW.DEPTH)
    cam.retrieve_measure(depth_float, sl.MEASURE.DEPTH)
    img = np.hstack((rgb_image.get_data(), depth_image.get_data()))
    img = cv.resize(img, (1280, 720))
    d_img = cv.circle(img.copy(), (30,30), 15, (0, 0, 255), -1)
    return img, d_img
    
if __name__ == "__main__": 
    path_output = 'C:/Mintlab/zed cam/test_ani.svo'
    cam = sl.Camera()
    init = sl.InitParameters()
    
    #recording option
    init.camera_resolution = sl.RESOLUTION.HD720
    rgb_image = sl.Mat()
    depth_image = sl.Mat()
    depth_float = sl.Mat()
    recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H264)
    
    #pose&sensors option
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.coordinate_units = sl.UNIT.METER
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    sensors_data = sl.SensorsData()
    cam_pose = sl.Pose()
    py_translation = sl.Translation()
    
    
    #error test
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
        
    
    err = cam.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
        
    
    
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
    
    #pickle data
    pic_dic = []
    
    #plot data&option
    poses = np.empty((0,3))
    frames = []
    baro = []
    show_animation=True
    n_frame = 0
    
    runtime = sl.RuntimeParameters()
    
    if show_animation:
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
        plt.show()
    
    timestamp = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT)
    ex_times = 0
    s_frames = 300
    
    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            if cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS: 
                timestamp = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT)
                times = timestamp.get_milliseconds()
            
                cam.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
                cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                cam_imu = sensors_data.get_imu_data()
                
                x, y, z, baro_v = save_data(pic_dic)
                poses = np.vstack((poses, [x, y, z]))
                baro = np.append(baro, baro_v)

                print(n_frame+1)
                frames = np.append(frames, n_frame+1)
                
                p_time = time()
                
                img, d_img = get_image()
                cv.imshow("ZED: RGB, DEPTH", d_img)
                
                c_time = time()
                print("cv Time : ", c_time-p_time)
                p_time = time()
                
                if show_animation:
                    if n_frame%20==0:
                        if n_frame<s_frames:
                            ax2.cla()
                            ax2.plot(frames[:], baro[:], 'r')
                        
                        else:
                            ax2.cla()
                            ax2.plot(frames[n_frame-s_frames:], baro[n_frame-s_frames:], 'r')
                        ax1.cla()
                        ax1.plot(poses[:,0], poses[:,1], poses[:,2], 'k')
                        plt.draw()
                    
                c_time = time()
                print("draw Time : ", c_time-p_time)
                n_frame +=1
                
                diff_times = times-ex_times
                print('Frames : ', int(1000/diff_times))
                ex_times = times
                
                key = cv.waitKey(1)
                
        if key == 32:   # Space
            cv.imshow("ZED: RGB, DEPTH", img)
            key = cv.waitKey(0)
            
        if key == 27:   # ESC
            break
                        
    with open('C:/Mintlab/zed cam/test_ani.pickle', "wb") as f:
        pickle.dump(pic_dic, f)
    
    plt.ioff()
    cv.destroyAllWindows()
    cam.close()
    plt.close()
