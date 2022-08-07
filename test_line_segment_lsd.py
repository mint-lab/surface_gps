import cv2 as cv
import zed

if __name__ == '__main__':
    svo_file = 'file_path/file_name.svo'
    svo_realtime = False
    
    cam = zed.ZED()
    cam.load_svo(svo_file, svo_realtime)

    # Grab images and show them
    ls_detector = cv.line_descriptor.LSDDetector.createLSDDetector() # Note) createLSDDetectorWithParams()
    
    while True:
        if cam.grab():
            # Grab an image
            color, _, _ = cam.get_images()

            # Detect line segments
            gray = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
            lines = ls_detector.detect(gray, 2, 1) # Arguments) image, scale, numOctaves

            # Visualize line segments
            result = cv.line_descriptor.drawKeylines(color, lines, color=(0, 0, 255))
            cv.imshow('ZED: Line Segment Detection', result)

        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    # Deallocate resources
    cv.destroyAllWindows()
    cam.close()