import pyzed.sl as sl
from sys import exit
import pickle
import keyboard

cam = sl.Camera()

if __name__ == "__main__":
    path_output = 'your_path/your_name.svo'
    

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.ULTRA

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.H264)
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    sensors_data = sl.SensorsData()
    baro = []
    frames_recorded = 0

    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            if cam.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS: 
                baro.append(sensors_data.get_barometer_data().pressure)
                
            try:                                    #Press 'q' to stop recording
                if keyboard.is_pressed('q'):
                    print('q')
                    break
            except:
                break
                
    with open('your_path/your_name.pickle', "wb") as f:
        pickle.dump(baro, f)

    cam.close()