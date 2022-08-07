import cv2 as cv
import zed

if __name__ == '__main__':
    svo_file = 'file_path/file_name.svo'
    svo_realtime = False
    
    cam = zed.ZED()
    cam.load_svo(svo_file, svo_realtime)

    # Grab images and show them
    ls_detector = cv.ximgproc.createFastLineDetector(length_threshold=50, canny_th1=10, canny_th2=200)
    
    while True:
        if cam.grab():
            # Grab an image
            color, _, _ = cam.get_images()

            # Detect line segments
            gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
            lines = ls_detector.detect(gray)

            # Visualize line segments
            result = ls_detector.drawSegments(color.copy(), lines, linecolor=(0, 0, 255))
            cv.imshow('ZED: Line Segment Detection', result)

        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    cv.destroyAllWindows()
    cam.close()