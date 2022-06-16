import cv2 as cv
import pyzed.sl as sl
import numpy as np


fld = cv.ximgproc.createFastLineDetector(length_threshold=50, canny_th1=10, canny_th2=200)
def find_line(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lines = fld.detect(gray_img)
    lines = np.squeeze(lines)
    drawn_images = fld.drawSegments(img, lines)
    for line in lines:
        st_point = (int(line[0]), int(line[1]))
        end_point = (int(line[2]), int(line[3]))
        drawn_images = cv.line(img, st_point, end_point,(0,0,255), 1)
    return drawn_images

if __name__ == '__main__':
    svo_file = 'C:/Users/Hyunkil/OneDrive/Documents/ZED/HD1080_SN2860_14-28-02.svo'
    svo_realtime = True
    
    input_type = sl.InputType()
    input_type.set_from_svo_file(svo_file)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=svo_realtime)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Prepare to grab images
    runtime = sl.RuntimeParameters()
    rgb_img = sl.Mat()
    
    # Grab images and show them
    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(rgb_img, sl.VIEW.LEFT)
            images_np = rgb_img.get_data()
            result = np.delete(images_np, 3, axis=2)
            drawn_img = find_line(result)
            cv.imshow('Line', drawn_img)
            key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    cv.destroyAllWindows()
    cam.close()