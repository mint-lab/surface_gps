import numpy as np
import cv2 as cv
from sensorpy.zed import ZED
from time import time

ret_list = []

def onMouse(event, x, y, flags, param) :
    global mask, normal
    if event == cv.EVENT_LBUTTONDOWN:
        seed = (x,y)
        retval = cv.floodFill(normal, mask, seed, newVal, loDiff, upDiff)
        ret_list.append(retval)
        cv.imshow('normal', normal)


newVal = (1,1,1)

loDiff, upDiff = (0.01,0.01,0.01), (0.01,0.01,0.01)

zed = ZED()
zed.open(svo_file='data/230116_M327/auto_v.svo', min_depth=0.2, max_depth = 20)
# zed.open(min_depth=0.2, max_depth = 20)

while zed.is_open():
    if zed.grab():
        s_time = time()
        d_v = zed.get_depth()*1000
        color,_,_ = zed.get_images()
        norm_dv = d_v-d_v.min()
        norm_dv = norm_dv/d_v.max()
        
        #Increase step -> Decrease noise, Make contour
        grad_x = (d_v[:,20:]-d_v[:,:-20])/20
        grad_y = (d_v[20:]-d_v[:-20])/20
        
        direction = np.dstack((-grad_x[10:-10], -grad_y[:,10:-10], norm_dv[10:-10,10:-10]))
        
        magnitude = np.sqrt((direction**2).sum(axis=2))
        magnitude = np.dstack((magnitude, magnitude, magnitude))
        
        normal = (direction/magnitude+1)/2          #Make -1~1 scale to 0~1 scale,   num<0.5 : minus
        print(time()-s_time)
        
        rows, cols = normal.shape[:2]
        mask = np.zeros((rows+2, cols+2), np.uint8)
        
        cv.imshow('normal', normal)
        cv.imshow('color', color)
        cv.setMouseCallback('normal', onMouse)
        
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey(0)
        if key == 27:
            break;
    
cv.destroyAllWindows()
zed.close()