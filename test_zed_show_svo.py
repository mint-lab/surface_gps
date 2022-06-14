import pyzed.sl as sl
import cv2

filepath = 'your svo file path'
    
input_type = sl.InputType()
input_type.set_from_svo_file(filepath)
init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False, coordinate_units=sl.UNIT.MILLIMETER)
cam = sl.Camera()
status = cam.open(init)

runtime = sl.RuntimeParameters()
runtime.sensing_mode = sl.SENSING_MODE.FILL
mat = sl.Mat()
depth_data = sl.Mat()
depth_image = sl.Mat()

key = ''

while key != 113:    # press 'q' to quit
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat)
        cam.retrieve_measure(depth_data, sl.MEASURE.DEPTH)
        cam.retrieve_image(depth_image, sl.VIEW.DEPTH)
        cv2.imshow("ZED", mat.get_data())
        cv2.imshow("DEPTH", depth_image.get_data())
        images_np = mat.get_data()
        depth_np = depth_data.get_data()
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(1)

cv2.destroyAllWindows()