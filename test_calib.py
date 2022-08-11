import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
import os

left_images=[]
right_images=[]

left_results=[]
right_results=[]

img_path = 'img_folder/'

def conv_Rt2pose(rvec, tvec):
    '''Convert the given rotation vector (rvec) and translation vector (tvec) to the camera rotation matrix and position.'''
    Rt = Rotation.from_rotvec(-rvec.flatten()).as_matrix() # Transpose
    pos = -Rt @ tvec.flatten()
    return Rt, pos

def calib(images, position):
    for image in images:
        ret, corners = cv.findChessboardCorners(image, (9,6), None)
        success, rvec, tvec = cv.solvePnP(obj_pts, corners, matrix_camera, distortion_coeffs, flags=0)
        ori, pos = conv_Rt2pose(rvec, tvec)
        
        if position == 'right':
            right_results.append([ori, pos])
            
        elif position == 'left':
            left_results.append([ori, pos])
            

for i in os.listdir(img_path):
    path = img_path+i
    if '.jpg' in path:
        images = cv.imread(path)
        gray = cv.cvtColor(images, cv.COLOR_BGR2GRAY)
        
        if 'left' in path:
            left_images.append(gray)
        
        elif 'right' in path:
            right_images.append(gray)
        

        

        
ret, corners = cv.findChessboardCorners(gray, (9,6), None)

obj_pts = np.zeros([54, 3])
print(obj_pts.shape[0])
corners = np.squeeze(corners)
obj_pts[:,2] = 0.5
j = 0
for i in range(corners.shape[0]):
    obj_pts[i, 0] += 0.03*j
    j +=1
    if j==9:
        j = 0
        
    obj_pts[i, 1] += 0.03*(int(i/9))
    
matrix_camera = np.array(
                            [[535.09, 0, 634.2],
                              [0, 535.09, 357.5],
                              [0, 0, 1]], dtype="double"
    )


distortion_coeffs = np.zeros((4,1))
calib(left_images, 'left')
calib(right_images, 'right')