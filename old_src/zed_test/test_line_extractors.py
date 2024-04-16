import cv2 as cv
import opencx as cx
from line_extractors import get_line_extractor, draw_line_segments
from sensorpy.zed import ZED
import time

def test_line_extractor(line_name='FLD', line_options={}, svo_file='', svo_realtime=False, output_file=''):
    # Open input and output videos
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime, depth_mode='performance')
    
    if output_file:
        output = cx.VideoWriter(output_file)

    # Run the line segment extractor
    extractor = get_line_extractor(line_name, **line_options)
    while True:
        # Grab an image
        if not zed.grab():
            break
        color, _, _ = zed.get_images()

        # Run the extractor
        elapse = time.time()
        lines = extractor.extract(color)
        elapse = time.time() - elapse

        # Visualize its result
        result = cv.cvtColor(cv.cvtColor(color, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) # Make the image gray to highlight lines
        draw_line_segments(result, lines)
        cx.putText(result, f'{1/elapse:.1f} Hz', (10, 10))
        cv.imshow('test_line_extractors', result)
        if output_file:
            output.write(result)

        key = cv.waitKey(1)
        if key == 32:   # Space
            key = cv.waitKey(0)
        if key == 27:   # ESC
            break

    cv.destroyAllWindows()
    if output_file:
        output.release()
    zed.close()



if __name__ == '__main__':
    test_line_extractor(svo_file = '../data/220720_M327/short.svo')
