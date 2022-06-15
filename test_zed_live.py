import cv2 as cv
import pyzed.sl as sl

if __name__ == '__main__':
    # Instantiate the camera object
    init = sl.InitParameters()
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # Prepare to grab images
    runtime = sl.RuntimeParameters()
    rgb_img = sl.Mat()
    depth_img = sl.Mat()

    # Grab images and show them
    while True:
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(rgb_img, sl.VIEW.LEFT)
            cam.retrieve_image(depth_img, sl.VIEW.DEPTH)
            cv.imshow("ZED: RGB", rgb_img.get_data())
            cv.imshow("ZED: Depth", depth_img.get_data())
        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    cv.destroyAllWindows()
    cam.close()