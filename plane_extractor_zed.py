import numpy as np
import cv2 as cv
import opencx as cx
from sensorpy.zed import ZED, sl
import open3d as o3d
import time
from scipy.linalg import null_space
import random

class ZEDPlaneExtractor:
    def __init__(self, zed, length_threshold=0.1, area_threshold=0.1, x_n = 8, y_n = 4):
        self.zed = zed
        self.zed_planes = []
        self.length_threshold = length_threshold
        self.area_threshold = area_threshold

        # Generate the sampling points based on the image resolution
        zed_info = zed.camera.get_camera_information()
        self.zed_width = zed_info.camera_resolution.width
        self.zed_height = zed_info.camera_resolution.height
        x_p = np.linspace(0, self.zed_width, x_n+2).astype(np.int32)[1:-1]
        y_p = np.linspace(0, self.zed_height, y_n+2).astype(np.int32)[1:-1]
        xx, yy = np.meshgrid(x_p, y_p)
        yx = np.dstack((yy, xx))
        self.sample_pts = yx.reshape(-1, 2).astype(np.float32)

        # Generate the camera matrix
        cam = zed_info.calibration_parameters.left_cam
        self.zed_K = np.array([[cam.fx,      0, cam.cx],
                               [     0, cam.fy, cam.cy],
                               [     0,      0,      1]])

    def extract(self, depth=None, get_meshs=False):
        self.zed_planes = []
        self.zed_meshs = []

        plane_eqs = []
        center_arrs = np.empty((0,3))
        for pt in self.sample_pts:
                plane = sl.Plane()
                mesh = sl.Mesh()
                find_plane_status = self.zed.camera.find_plane_at_hit(pt, plane)
                if str(find_plane_status) == 'SUCCESS' and np.max(plane.get_extents())> self.length_threshold and np.prod(plane.get_extents())>self.area_threshold:
                    if (center_arrs==np.round(plane.get_center(),2)).any(axis=1).any() == False:
                        center_arrs = np.vstack((center_arrs, np.round(plane.get_center(),2)))
                        bounds_m = plane.get_bounds()
                        bounds_px = bounds_m @ self.zed_K.T
                        bounds_px = bounds_px / bounds_px[:,-1].reshape(-1, 1)
                        self.zed_planes.append((plane, bounds_px[:,0:2].astype(np.float32)))
                        plane_eqs.append(plane.get_plane_equation())
                        if get_meshs:
                            mesh = plane.extract_mesh()
                            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)), triangles=o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)))
                            mesh.paint_uniform_color(np.abs(plane.get_plane_equation()[:3]))
                            self.zed_meshs.append(mesh)
        if get_meshs:
            return np.vstack(plane_eqs), self.zed_meshs
        return np.vstack(plane_eqs)

    def get_normal_image(self, depth=None):
        image = np.zeros((self.zed_height, self.zed_width, 3), dtype=np.uint8)
        for plane, polygon2d in self.zed_planes:
            rgb = (255 * np.abs(plane.get_normal())).astype(np.uint8)
            cv.fillPoly(image, [polygon2d.astype(np.int32)], rgb.tolist())
        return image
    
class RansacExtractor:
    def __init__(self, length_threshold = 0.1, plane_candidate_n = 1000, plane_n = 3):
        self.length_threshold = length_threshold
        self.plane_candidate_n = plane_candidate_n
        self.plane_n = plane_n
        
    def find_plane(self, data):
        null_v_dic = {}
        points_dic = {}
        pcds = []
        
        null_coeff = np.empty((0, 4))

        ran_1 = np.zeros(self.plane_candidate_n)
        ran_2 = np.zeros(self.plane_candidate_n)
        ran_3 = np.zeros(self.plane_candidate_n)

        r_list = [10, 3, 1]
        while np.any(ran_1 == ran_2) or np.any(ran_1 == ran_3) or np.any(ran_3 == ran_2):
            ran_1 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
            ran_2 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
            ran_3 = np.random.randint(low=0, high=np.shape(data)[0]-1, size=self.plane_candidate_n, dtype=int)
        
        ran_pix = np.vstack((ran_1, ran_2, ran_3)).T
        
        for points in ran_pix:
            p1, p2, p3 = points[0], points[1], points[2]
            matrix = np.array([data[p1],
                               data[p2],
                               data[p3]])
            matrix = np.hstack((matrix, np.array([[1],[1],[1]])))
            A, B, C, D = np.squeeze(null_space(matrix))
            null_coeff = np.vstack((null_coeff, [A, B, C, D]))
        
        for n_plane in range(self.plane_n):
            coeff_arr = null_coeff.copy()      
            
            for it in range(3):
                if it<2:
                    s_data = data[random.sample(range(0, np.shape(data)[0]), np.shape(data)[0]//(10**(3-it))),:]
                else:
                    s_data = data
                coeff_arr = self.count_points(s_data, coeff_arr, r_list[it])
        
            A, B, C, D = coeff_arr[0]
            x = data[:,0]
            y = data[:,1]
            z = data[:,2]
            h = np.abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
            inner_index = h<self.length_threshold
        
            points_dic[n_plane] = data[inner_index]
            null_v_dic[n_plane] = coeff_arr[0]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_dic[n_plane].astype(np.float64)))
            pcd.paint_uniform_color(np.abs(coeff_arr[0][:-1]))
            pcds.append(pcd)
            data = data[~inner_index]
        
        return null_v_dic, pcds
    
    def count_points(self, s_datas, coeff_arr, r_num):
        result = np.empty((0,5))
        x = s_datas[:,0]
        y = s_datas[:,1]
        z = s_datas[:,2]
        
        for enum in coeff_arr:
            A, B, C, D = enum
            h = abs(A*x+B*y+C*z+D)/np.sqrt(A**2+B**2+C**2)
            inner_index = h<self.length_threshold
            in_n = inner_index[np.where(inner_index == True)].size
            result=np.vstack((result, [in_n, A, B, C, D]))
            
        result = np.squeeze(result[[result[:, 0].argsort()[::-1]]])
        
        return result[:r_num, 1:]
        
    

def test_plane_extractor(svo_file='', svo_realtime=False, output_file=''):
    # Open input and output videos
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime)
    if output_file:
        output = cx.VideoWriter(output_file)

    # Run the line segment extractor
    extractor = ZEDPlaneExtractor(zed)
    while True:
        # Grab an image
        if not zed.grab():
            break
        color, _, depth = zed.get_images()

        # Run the extractor
        elapse = time.time()
        planes = extractor.extract(depth)
        elapse = time.time() - elapse

        # Visualize its result
        gray = cv.cvtColor(cv.cvtColor(color, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) # Make the image gray to highlight result
        normal = extractor.get_normal_image(depth)
        result = cv.addWeighted(gray, 0.5, normal, 0.5, 0)
        cx.putText(result, f'{1/max(elapse, 0.001):.1f} Hz, Planes: {len(planes)}', (10, 10))
        cv.imshow('test_plane_extractors', result)
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
    test_plane_extractor('data/230116_M327/auto_v.svo')
