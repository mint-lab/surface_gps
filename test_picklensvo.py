import pyzed.sl as sl
from sys import exit
import pickle
import keyboard
from time import localtime

cam = sl.Camera()

if __name__ == "__main__": 
    path_output = 'path/file_name.svo'
    cam = sl.Camera()

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.coordinate_units = sl.UNIT.METER

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
        
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = cam.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
        
    recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H264)
    
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
    
    
    runtime = sl.RuntimeParameters()
    sensors_data = sl.SensorsData()
    baro = []
    i = 0
    pic_dic = []
    cam_pose = sl.Pose()
    py_translation = sl.Translation()
    
    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            if cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS: 
                timestamp = cam.get_timestamp(sl.TIME_REFERENCE.CURRENT)
                times = timestamp.get_milliseconds()
                baro_value = sensors_data.get_barometer_data().pressure
                baro.append(baro_value)
            
                cam.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
                cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                cam_imu = sensors_data.get_imu_data()
                
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
                print(i)
                i +=1
                    

        try:                                    #Press 'q' to stop recording
            if keyboard.is_pressed('q'):
                print('q')
                break
        except:
            break
        
    with open('path/file_name.pickle', "wb") as f:
        pickle.dump(pic_dic, f)
                
    cam.close()