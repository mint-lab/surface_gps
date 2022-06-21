import cv2 as cv
import pyzed.sl as sl
import numpy as np

if __name__ == '__main__':
    svo_file = 'data/zed_MR327_220614.svo'
    svo_realtime = False

    # Instantiate the camera object
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
    rgb_image = sl.Mat()

    # Grab images and show them
    ls_detector = cv.ximgproc.createFastLineDetector(length_threshold=50, canny_th1=10, canny_th2=200)
    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Grab an image
            cam.retrieve_image(rgb_image, sl.VIEW.LEFT)
            image = np.delete(rgb_image.get_data(), 3, axis=2) # Remove the 4th channel

            # Detect line segments
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            lines = ls_detector.detect(gray)

            # Visualize line segments
            result = ls_detector.drawSegments(image, lines, linecolor=(0, 0, 255))
            cv.imshow('ZED: Line Segment Detection', result)

        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    cv.destroyAllWindows()
    cam.close()