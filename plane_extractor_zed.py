import numpy as np
import cv2 as cv
import open3d as o3d
import opencx as cx
from sensorpy.zed import ZED, sl
import time

class ZEDSinglePlane:
    def __init__(self, zed, seed_pt=None):
        self.zed = zed
        self.zed_plane = None

        # Get the image resolution and the camera matrix
        zed_info = zed.camera.get_camera_information()
        self.zed_width = zed_info.camera_resolution.width
        self.zed_height = zed_info.camera_resolution.height
        cam = zed_info.calibration_parameters.left_cam
        self.zed_K = np.array([[cam.fx,      0, cam.cx],
                               [     0, cam.fy, cam.cy],
                               [     0,      0,      1]])

        # Initialize the default seed point
        if seed_pt is not None:
            self.seed_pt = seed_pt
        else:
            self.seed_pt = (self.zed_width / 2, self.zed_height / 2)

    def extract(self, depth=None, seed_pt=None):
        self.zed_plane = None
        if seed_pt is None:
            seed_pt = self.seed_pt
        plane = sl.Plane()
        status = self.zed.camera.find_plane_at_hit(seed_pt, plane)
        if status == sl.ERROR_CODE.SUCCESS:
            self.zed_plane = plane
            return plane.get_plane_equation().reshape(1, -1)
        return np.array([])

    def get_plane_image(self, depth=None, seed_pt=None, seed_pt_radius=10, seed_pt_thickness=2, seed_pt_color=(0, 255, 255)):
        image = np.zeros((self.zed_height, self.zed_width, 3), dtype=np.uint8)

        if self.zed_plane is not None:
            # Draw the plane
            bounds_meter = self.zed_plane.get_bounds()
            bounds_pixel = bounds_meter @ self.zed_K.T
            bounds_pixel = bounds_pixel[:,0:2] / bounds_pixel[:,-1].reshape(-1, 1)
            rgb = (255 * np.abs(self.zed_plane.get_normal())).astype(np.uint8)
            cv.fillPoly(image, [bounds_pixel.astype(np.int32)], rgb[::-1].tolist())

        if seed_pt_radius > 0:
            # Draw 'seed_pt'
            if seed_pt is None:
                seed_pt = self.seed_pt
            seed_pt = np.array(seed_pt).astype(np.int32)
            cv.line(image, seed_pt-(seed_pt_radius,0), seed_pt+(seed_pt_radius,0), thickness=seed_pt_thickness, color=seed_pt_color)
            cv.line(image, seed_pt-(0,seed_pt_radius), seed_pt+(0,seed_pt_radius), thickness=seed_pt_thickness, color=seed_pt_color)

        return image

    def get_plane_mesh(self, depth=None, seed_pt=None):
        if self.zed_plane is not None:
            mesh = self.zed_plane.extract_mesh()
            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
                                             triangles=o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)))
            mesh.paint_uniform_color(np.abs(self.zed_plane.get_normal()))
            return [mesh]
        return []

class ZEDMultiPlane:
    def __init__(self, zed, length_threshold=0.1, area_threshold=0.1, dist_threshold=0.1, seed_x_n=8, seed_y_n=4):
        self.zed = zed
        self.zed_planes = []
        self.length_threshold = length_threshold
        self.area_threshold = area_threshold
        self.dist_threshold = dist_threshold

        # Generate the seed points based on the image resolution
        zed_info = zed.camera.get_camera_information()
        self.zed_width = zed_info.camera_resolution.width
        self.zed_height = zed_info.camera_resolution.height
        xs = np.linspace(0, self.zed_width,  seed_x_n+2, dtype=np.int32)[1:-1]
        ys = np.linspace(0, self.zed_height, seed_y_n+2, dtype=np.int32)[1:-1]
        xx, yy = np.meshgrid(xs, ys)
        self.seed_pts = np.dstack((xx, yy)).reshape(-1, 2)

        # Generate the camera matrix
        cam = zed_info.calibration_parameters.left_cam
        self.zed_K = np.array([[cam.fx,      0, cam.cx],
                               [     0, cam.fy, cam.cy],
                               [     0,      0,      1]])

    def extract(self, depth=None):
        self.zed_planes = []
        plane_eqs = []
        center_pts = np.empty((0, 3))
        for seed_pt in self.seed_pts:
            plane = sl.Plane()
            status = self.zed.camera.find_plane_at_hit(seed_pt, plane)
            if status == sl.ERROR_CODE.SUCCESS and np.max(plane.get_extents())> self.length_threshold and np.prod(plane.get_extents())>self.area_threshold:
                center_pt = plane.get_center()
                delta = center_pts - center_pt.reshape(-1, 3)
                dist = np.linalg.norm(delta, axis=1)
                if np.all(dist > self.dist_threshold):
                    plane_eqs.append(plane.get_plane_equation())
                    center_pts = np.vstack((center_pts, center_pt))
                    self.zed_planes.append(plane)
        return np.vstack(plane_eqs)

    def get_plane_image(self, depth=None, seed_pt_radius=10, seed_pt_thickness=2, seed_pt_color=(0, 255, 255)):
        image = np.zeros((self.zed_height, self.zed_width, 3), dtype=np.uint8)

        for plane in self.zed_planes:
            # Draw each plane
            bounds_meter = plane.get_bounds()
            bounds_pixel = bounds_meter @ self.zed_K.T
            bounds_pixel = bounds_pixel[:,0:2] / bounds_pixel[:,-1].reshape(-1, 1)
            rgb = (255 * np.abs(plane.get_normal())).astype(np.uint8)
            cv.fillPoly(image, [bounds_pixel.astype(np.int32)], rgb[::-1].tolist())

        if seed_pt_radius > 0:
            # Draw each seed point
            for seed_pt in self.seed_pts:
                cv.line(image, seed_pt-(seed_pt_radius,0), seed_pt+(seed_pt_radius,0), thickness=seed_pt_thickness, color=seed_pt_color)
                cv.line(image, seed_pt-(0,seed_pt_radius), seed_pt+(0,seed_pt_radius), thickness=seed_pt_thickness, color=seed_pt_color)

        return image

    def get_plane_mesh(self, depth=None):
        meshes = []
        for plane in self.zed_planes:
            mesh = plane.extract_mesh()
            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64)),
                                             triangles=o3d.utility.Vector3iVector(mesh.triangles.astype(np.int32)))
            mesh.paint_uniform_color(np.abs(plane.get_normal()))
            meshes.append(mesh)
        return meshes

def test_plane_extractor(svo_file='', svo_realtime=False, output_file=''):
    # Open input and output videos
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime)
    if output_file:
        output = cx.VideoWriter(output_file)

    # Run the line segment extractor
    extractor = ZEDMultiPlane(zed)
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
        normal = extractor.get_plane_image(depth)
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
    test_plane_extractor('data/220720_M327/short.svo')
    # test_plane_extractor('data/220902_Gym/short.svo')
    # test_plane_extractor('data/230116_M327/auto_v.svo')
