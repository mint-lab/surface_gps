import cv2 as cv
import zed
import numpy as np

if __name__ == '__main__':
    svo_file = 'file_path/file_name.svo'
    svo_realtime = False
    
    cam = zed.ZED()
    cam.load_svo(svo_file, svo_realtime)
    zoom=0.5

    while True:
        if cam.grab():
            color, _, depth = cam.get_images()
            depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('ZED Live: Color and Depth', merge)
        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    cv.destroyAllWindows()
    cam.close()