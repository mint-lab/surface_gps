import cv2 as cv
from sensorpy.zed import ZED
import opencx as cx
from LID_test import LID


if __name__ == '__main__':
    svo_file = 'data/220902_Gym/short.svo'
    svo_realtime = True
    
    cam = ZED()
    cam.open(svo_file=svo_file, svo_realtime=svo_realtime)
    
    lid = LID()

    # Grab images and show them  
    # output = cx.VideoWriter('220902_Gym_FLD.avi')
    
    while True:
        if cam.grab():
            # Grab an image
            color, _, _ = cam.get_images()
            
            # Detect line segments
            # Visualize line segments
            result = lid.fsd(color)
            # output.write(result)
            cv.imshow('ZED: Line Segment Detection', result)

        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    # output.release()
    cv.destroyAllWindows()
    cam.close()