import numpy as np
import cv2 as cv

d_v = np.load('Gym_depth.npy')*255

grad_x = (d_v[:,2:]-d_v[:,:-2])/2
grad_y = (d_v[2:]-d_v[:-2])/2
direction = np.dstack((-grad_x[1:-1], -grad_y[:,1:-1], np.ones_like(grad_x[1:-1])))
magnitude = np.sqrt((direction**2).sum(axis=2))
magnitude = np.dstack((magnitude, magnitude, magnitude))
normal = direction/magnitude

cv.imshow('depth', d_v)
cv.imshow('normal', np.abs(normal))

cv.waitKey(0)
cv.destroyAllWindows()