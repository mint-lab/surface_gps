import cv2 as cv
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sensorpy.zed import ZED

def capture_svo(svo_file, svo_realtime=False, cap_file='%04d.png', frame_step=1, frame_offset=[10, 30]):
    '''Play the given SVO file and capture an image if specified'''

    zed = ZED()
    zed.open(svo_file=svo_file, depth_mode='performance')

    cv.namedWindow('ZED SVO Capture')
    frame_total = zed.camera.get_svo_number_of_frames()
    is_paused = True
    while zed.is_open():
        frame = zed.camera.get_svo_position()
        if zed.grab():
            # Get and show the left image
            left, *_ = zed.get_images()
            show = left.copy()
            cv.putText(show, f'{frame} / {frame_total}', frame_offset, cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0))
            cv.imshow('ZED SVO Capture', show)
        else:
            # Play again
            zed.camera.set_svo_position(0)

        key = cv.waitKey(1)
        if is_paused:
            key = cv.waitKey(0)
            if key == ord('c') or key == ord('C'):      # 'c' or 'C'
                # Save the left image
                cv.imwrite(cap_file % frame, left)
            elif key == ord('<') or key == ord(','):    # '<' or ','
                # Move backward
                zed.camera.set_svo_position(max(frame-frame_step, 0))
            elif key == ord('>') or key == ord('.'):    # '>' or '.'
                # Move forward
                zed.camera.set_svo_position(min(frame+frame_step, frame_total-1))
        if key == ord(' '):                             # Space
            # Toggle pause and resume
            is_paused = not is_paused
        if key == 27:                                   # ESC
            # Terminate the program
            break

    cv.destroyAllWindows()
    zed.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZED SVO Capture')
    parser.add_argument('svo_file',             nargs=1, type=str,                         help='the name of the SVO file (e.g. test.svo)')
    parser.add_argument('--svo_realtime', '-r', nargs=1, type=bool,  default=[False],      help='the flag to enable realtime play (default: False)')
    parser.add_argument('--cap_file',     '-c', nargs=1, type=str,   default=['%04d.png'], help='the name of the capture files (default: "%04d.png")')
    args = parser.parse_args()

    capture_svo(args.svo_file[0], svo_realtime=args.svo_realtime[0], cap_file=args.cap_file[0])
