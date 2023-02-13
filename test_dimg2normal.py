import numpy as np
import cv2 as cv
from sensorpy.zed import ZED
from time import time

newVal = (1,1,1)

loDiff, upDiff = (0.01,0.01,0.01), (0.01,0.01,0.01)

zed = ZED()
# zed.open(svo_file='data/230116_M327/auto_v.svo', svo_realtime=True, min_depth=0.2, max_depth = 20)
# zed.open(svo_file='data/220902_Gym/short.svo', svo_realtime=True, min_depth=0.2, max_depth = 20)
zed.open(min_depth=0.2, max_depth = 20)

zed_info = zed.camera.get_camera_information()
width = zed_info.camera_resolution.width
height = zed_info.camera_resolution.height
seed_x_n=8
seed_y_n=4

grad_size = 10

xs = np.linspace(0, width,  seed_x_n+2, dtype=np.int32)[1:-1]
ys = np.linspace(0, height, seed_y_n+2, dtype=np.int32)[1:-1]
xx, yy = np.meshgrid(xs, ys)
seed_pts = np.dstack((xx, yy)).reshape(-1, 2)

while zed.is_open():
    if zed.grab():
        s_time = time()
        d_v = zed.get_depth()*500
        np.nan_to_num(d_v, copy=False, nan=0, posinf=0, neginf=0)
        color,_,_ = zed.get_images()
        norm_dv = d_v-d_v.min()
        norm_dv = norm_dv/d_v.max()
        
        #Increase step -> Decrease noise, Make contour
        grad_x = (d_v[:,grad_size:]-d_v[:,:-grad_size])/(2*grad_size)
        grad_y = (d_v[grad_size:]-d_v[:-grad_size])/(2*grad_size)
        
        direction = np.dstack((-grad_x[grad_size//2:-grad_size//2], -grad_y[:,grad_size//2:-grad_size//2], norm_dv[grad_size//2:-grad_size//2,grad_size//2:-grad_size//2]))
        # direction = np.dstack((-grad_x[5:-5], -grad_y[:,5:-5], np.ones_like(grad_x[5:-5])))
        
        magnitude = np.sqrt((direction**2).sum(axis=2))
        magnitude = np.dstack((magnitude, magnitude, magnitude))
        
        normal = (direction/magnitude+1)/2          #Make -1~1 scale to 0~1 scale,   num<0.5 : minus
        # normal = direction/magnitude
        
        rows, cols = normal.shape[:2]
        
        
        label_arr = np.zeros_like(normal[:,:,0])
        label_n = 1
        for pts in seed_pts:
            if label_arr[pts[1], pts[0]] == 0:
                mask = np.zeros((rows+2, cols+2), np.uint8)
                retval = cv.floodFill(normal.copy(), mask, pts, newVal, loDiff, upDiff, flags=8)
                if retval[0] > 3000:
                    label_arr[(mask==1)[1:-1,1:-1]] = label_n
                    label_n += 1
        
        print(time()-s_time)
        
        cv.imshow('normal', normal)
        cv.imshow('color', color)
        cv.imshow('labeling', label_arr/label_arr.max())
        
        
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey(0)
        if key == 27:
            break;
    
cv.destroyAllWindows()
zed.close()