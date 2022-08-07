import zed
import cv2 as cv
import numpy as np

if __name__ == "__main__": 
    output_path = 'file_path/file_name'
    cam = zed.ZED()
    cam.open(fps=60)
    cam.start_recording(output_path)
    cam.tracking_pose()
    
    sensor_dic = []
    zoom = 0.5
    
    while True:
        if cam.grab():
            if cam.get_sensors_data():
                l_image, r_image, d_image = cam.get_images()
                time, baro_v, t_pose, imu_v = cam.get_sensor()
                sensor_dic.append({'time':time, 'baro':baro_v, 't_pose':t_pose, 'imu':imu_v})
                
                color, _, depth = cam.get_images()

                # Show the images
                depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
                merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
                cv.imshow('ZED Live: Color and Depth', merge)
                
        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break
                
    zed.save_pickle(sensor_dic, output_path)
    cv.destroyAllWindows()
    cam.close()