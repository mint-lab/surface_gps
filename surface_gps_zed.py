import numpy as np
from scipy.spatial.transform import Rotation
import json, copy
import cv2 as cv
import open3d as o3d
from sensorpy.zed import ZED, print_zed_info
import opencx as cx
from concurrent.futures import ThreadPoolExecutor
from line_extractors import get_line_extractor, draw_line_segments
from plane_extractor_zed import ZEDMultiPlane, ZEDSinglePlane
from plane_extractor_ransac import RANSACMultiPlane

def get_transformation(R, tvec):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,-1] = tvec
    return T

def visualize_pcd(vis, zed, config, zed_color, localizer, robot_vis, zed_vis):
    if config['vis_show_pcd']:
        zed_xyz = zed.get_xyz().reshape(-1,3).astype(np.float64)
        zed_rgb = (zed_color / 255).reshape(-1,3).astype(np.float64)
        valid = zed_xyz[:,-1] > 0
        zed_xyz = zed_xyz[valid,:]
        zed_rgb = zed_rgb[valid,:]
        zed_pcd = o3d.geometry.PointCloud()
        zed_pcd.points = o3d.utility.Vector3dVector(zed_xyz)
        zed_pcd.colors = o3d.utility.Vector3dVector(zed_rgb)
        zed_pcd.transform(localizer.get_T_zed())
        vis.remove_geometry('PointCloud')
        vis.add_geometry('PointCloud', zed_pcd)
        
    if robot_vis is not None:
        robot_copy = copy.deepcopy(robot_vis)
        robot_copy.transform(localizer.get_T_robot())
        vis.remove_geometry('Robot')
        vis.add_geometry('Robot', robot_copy)
    zed_copy = copy.deepcopy(zed_vis)
    zed_copy.transform(localizer.get_T_zed())
    vis.remove_geometry('ZED-Left')
    vis.add_geometry('ZED-Left', zed_copy)
        
def process_line(vis, zed, localizer, zed_color, line_name='FLD', line_options={}):
    xyz = zed.get_xyz()
    extractor = get_line_extractor(line_name, **line_options)
    lines = extractor.extract(zed_color).astype(np.int32)
    l_pts = np.empty((0,3))
    
    # Visualize line segments
    for li in lines:
        l_pts = np.vstack((l_pts, xyz[li[1], li[0], :], xyz[li[3], li[2], :]))
        
    lines_coup = [ [i, i+1] for i in range(0, l_pts.shape[0]//2, 2)]
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(l_pts.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(lines_coup)
    l_colors = [[1, 0, 0] for i in range(len(lines_coup))]
    ls.colors = o3d.utility.Vector3dVector(l_colors)
    ls.transform(localizer.get_T_zed())
    vis.remove_geometry('line')
    vis.add_geometry('line', ls)
    
def process_plane(vis, zed, localizer, thres, length_threshold=0.1, area_threshold=0.1, x_n = 8, y_n = 4):
    extractor = ZEDMultiPlane(zed)
    planes = extractor.extract()
    meshs = extractor.get_plane_mesh()
    for i in range(x_n*y_n):
        vis.remove_geometry('mesh'+str(i))
    for idx,mesh in enumerate(meshs):
        mesh.transform(localizer.get_T_zed())
        vis.add_geometry('mesh'+str(idx), mesh)
    return planes

# def process_plane(vis, zed, localizer, length_threshold = 0.1, plane_candidate_n = 1000, plane_n = 3):
#     extractor = RANSACMultiPlane(length_threshold = length_threshold, plane_candidate_n=plane_candidate_n, plane_n=plane_n)
#     xyz = zed.get_xyz()
#     planes, meshs = extractor.find_plane(xyz.reshape(-1,3))
#     for i in range(plane_n):
#         vis.remove_geometry('mesh'+str(i))
#     for idx,mesh in enumerate(meshs):
#         mesh.transform(localizer.get_T_zed())
#         vis.add_geometry('mesh'+str(idx), mesh)
#     return planes

class SurfaceGPSZED:
    def __init__(self, init_rvec=np.zeros(3), init_tvec=np.zeros(3), zed_K=np.eye(3), zed_distort=np.zeros(5), zed_rvec=np.zeros(3), zed_tvec=np.zeros(3), project_threshold=1.):
        T_world2robot = get_transformation(Rotation.from_rotvec(init_rvec).as_matrix(), init_tvec)
        self.T_zed2robot = get_transformation(Rotation.from_rotvec(zed_rvec).as_matrix(), zed_tvec)
        self.T_robot2zed = np.linalg.inv(self.T_zed2robot)
        self.T_world2zed = T_world2robot @ self.T_robot2zed
        self.T_odom_prev = None
        self.surface_mesh = None
        self.surface_impl = None

        self.project_threshold = project_threshold

    def load_surface(self, surface):
        self.surface_mesh = copy.deepcopy(surface)
        self.surface_mesh.compute_triangle_normals()
        self.surface_impl = o3d.t.geometry.RaycastingScene()
        self.surface_impl.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.surface_mesh))
        return True

    def apply_odometry(self, zed_qxyzw, zed_tvec):
        zed_r = Rotation.from_quat(zed_qxyzw)
        T_odom_curr = get_transformation(zed_r.as_matrix(), zed_tvec)
        if self.T_odom_prev is not None:
            T_delta = T_odom_curr @ np.linalg.inv(self.T_odom_prev)
            self.T_world2zed = T_delta @ self.T_world2zed
        self.T_odom_prev = T_odom_curr
        return True

    # def apply_orientation(self, zed_qxyzw):
    #     zed_r = Rotation.from_quat(zed_qxyzw)
    #     self.T_world2zed[0:3,0:3] = zed_r.as_matrix() # TODO
    #     return True

    def project_on_surface(self):
        if self.surface_impl is not None:
            T_world2robot = self.get_T_robot()
            robot_pt = T_world2robot[0:3,-1]
            query_pt = o3d.core.Tensor(robot_pt.reshape(1,-1), dtype=o3d.core.Dtype.Float32)
            result = self.surface_impl.compute_closest_points(query_pt)
            if len(result['points']) > 0:
                closest_pt = result['points'][0].numpy()
                dist = np.linalg.norm(closest_pt - robot_pt)
                if dist < self.project_threshold:
                    T_world2robot[0:3,-1] = closest_pt
                    self.T_world2zed = T_world2robot @ self.T_robot2zed
                    return True
        return False

    def get_T_robot(self):
        T_world2robot = self.T_world2zed @ self.T_zed2robot
        return T_world2robot

    def get_T_zed(self):
        return self.T_world2zed

def load_config(config_file):
    '''Load configuration from the given file'''

    # The default configuration
    config = {
        'surface_file'      : '',
        'surface_color'     : [0.6, 0.6, 0.6],
        'robot_size'        : [],
        'robot_color'       : [0.9, 0.6, 0.3],
        'zed_depth_mode'    : 'neural',
        'zed_print_info'    : True,
        'localizer_name'    : 'SurfaceGPSZED',
        'localizer_option'  : {
            'init_rvec'     : [0, 0, 0],
            'init_tvec'     : [0, 0, 0],
            'zed_K'         : [[500, 0, 640], [0, 500, 360], [0, 0, 1]],
            'zed_distort'   : [0, 0, 0, 0, 0],
            'zed_rvec'      : [0, 0, 0],
            'zed_tvec'      : [0, 0, 0]
        },
        'vis_axes_length'   : 1,
        'vis_cam_fov'       : 90,
        'vis_cam_lookat'    : [0, 0, 0],
        'vis_cam_eye'       : [0, 0, -1],
        'vis_cam_up'        : [0, -1, 0],
        'vis_image_zoom'    : 0.5,
        'vis_ground_plane'  : 'XZ',
        'vis_show_depth'    : True,
        'vis_show_pcd'      : False,
        'vis_show_skybox'   : False,
        'vis_show_axes'     : True,
        'vis_show_ground'   : True,
        'vis_show_settings' : True,
        'vis_show_plane'    : True,
        'vis_show_line'     : True
    }

    # Update the configuration from the given file
    if config_file:
        with open(config_file, 'r') as f:
            config_from_file = json.load(f)
            config.update(config_from_file)

    # Preprocess the configuration
    for key in ['init_rvec', 'init_tvec', 'zed_K', 'zed_distort', 'zed_rvec', 'zed_tvec']:
        config['localizer_option'][key] = np.array(config['localizer_option'][key], dtype=np.float64)
    for key in ['vis_cam_lookat', 'vis_cam_eye', 'vis_cam_up']:
        config[key] = np.array(config[key], dtype=np.float64)
    return config

def test_localizer(config_file='', svo_file='', svo_realtime=True):
    '''Test a localizer'''

    # Load the configuration file
    config = load_config(config_file)
    if not svo_file and ('svo_file' in config):
        svo_file = config['svo_file']

    # Open the ZED camera
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime, depth_mode=config['zed_depth_mode'], min_depth=0.2, max_depth=20)
    zed.start_tracking()
    if config['zed_print_info']:
        print_zed_info(zed)

    # Prepare the image viewer
    win_title = 'SurfaceGPS Tester'
    cv.namedWindow(win_title + ': Image')

    # Prepare the 3D viewer
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer(win_title + ': 3D Data')
    vis.setup_camera(config['vis_cam_fov'], config['vis_cam_lookat'], config['vis_cam_eye'], config['vis_cam_up'])
    GROUND_PLANE = {'XY': o3d.visualization.rendering.Scene.XY, 'XZ': o3d.visualization.rendering.Scene.XZ, 'YZ': o3d.visualization.rendering.Scene.YZ}
    vis.ground_plane = GROUND_PLANE[config['vis_ground_plane']]
    vis.show_skybox(config['vis_show_skybox'])
    vis.show_axes = config['vis_show_axes']
    vis.show_ground = config['vis_show_ground']
    vis.show_settings = config['vis_show_settings']
    vis.line_width = 5
    app.add_window(vis)

    # Prepare the surface
    surface = None
    if config['surface_file']:
        surface = o3d.io.read_triangle_mesh(config['surface_file'])
        surface.paint_uniform_color(config['surface_color'])
        vis.add_geometry('Surface', surface)

    # Prepare the robot and ZED camera for visualization
    robot_vis = None
    if len(config['robot_size']) >= 3:
        width, height, depth, *_ = config['robot_size']
        robot_vis = o3d.geometry.TriangleMesh().create_box(width, height, depth)
        robot_vis.vertices = o3d.utility.Vector3dVector(np.array(robot_vis.vertices) - [width/2, height, depth/2])
        robot_vis.compute_triangle_normals()
        robot_vis.paint_uniform_color(config['robot_color'])
    zed_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(config['vis_axes_length'])

    # Prepare the localizer
    localizer = SurfaceGPSZED(**config['localizer_option'])
    if surface is not None:
        localizer.load_surface(surface)

    # Run the localizer with sensor data
    frame = 0
    while zed.is_open():
        # Get images and 3D pose
        if not zed.grab():
            break
        zed_color, _, zed_depth = zed.get_images()
        zed_state, zed_qxyzw, zed_tvec = zed.get_tracking_pose()
        

        # Perform the localizer
        if zed_state:
            localizer.apply_odometry(zed_qxyzw, zed_tvec)
            localizer.project_on_surface()

        # Show the 3D information
        if not vis.is_visible:
            break
        # Prepare for MultiThreading
        with ThreadPoolExecutor() as executor:
            executor.submit(visualize_pcd, vis, zed, config, zed_color, localizer, robot_vis, zed_vis)
            # if config['vis_show_plane']:
            #     executor.submit(process_plane, vis, zed, localizer, 0.1)
            if config['vis_show_plane']:
                executor.submit(process_plane, vis, zed, localizer, 0.02, 1000, 3)
            if config['vis_show_line']:
                executor.submit(process_line, vis, zed, localizer, zed_color, line_name='lsd')
        vis.post_redraw()
        
        print(zed.camera.get_current_fps())

        # Show the images
        merge = zed_color
        if config['vis_show_depth']:
            zed_depth_color = cv.applyColorMap(zed_depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((zed_color, zed_depth_color)), (0, 0), fx=config['vis_image_zoom'], fy=config['vis_image_zoom'])
        robot_T = localizer.get_T_robot()
        robot_euler = Rotation.from_matrix(robot_T[0:3,0:3]).as_euler('xyz', degrees=True)
        info_text = f'P: ({robot_T[0,-1]:.3f}, {robot_T[1,-1]:.3f}, {robot_T[2,-1]:.3f}) [m]\n'
        info_text += f'O: ({robot_euler[0]:.1f}, {robot_euler[1]:.1f}, {robot_euler[2]:.1f}) [deg]'
        cx.putText(merge, info_text, (10, 10))
        cv.imshow(win_title + ': Image', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break
        frame += 1

    cv.destroyAllWindows()
    vis.close()
    zed.close()



if __name__ == '__main__':
    # test_localizer('config_220720_M327.json', svo_file='data/220720_M327/short.svo')
    # test_localizer('config_220720_M327.json', svo_file='data/220720_M327/long.svo')

    test_localizer('config_230116_M327.json', svo_file='data/230116_M327/auto_v.svo')
    # test_localizer('config_230116_M327.json', svo_file='data/230116_M327/manual_o.svo')
    # test_localizer('config_230116_M327.json', svo_file='data/230116_M327/manual_o_speed.svo')
    # test_localizer('config_230116_M327.json', svo_file='data/230116_M327/manual_s.svo')
    # test_localizer('config_230116_M327.json', svo_file='data/230116_M327/manual_s_head.svo')
    # test_localizer('config_230116_M327.json', svo_file='data/230116_M327/manual_z.svo')
    # test_localizer('config_230116_M327_angle.json', svo_file='data/230116_M327/auto_rotate.svo')

    # test_localizer(svo_file='data/220902_Gym/short.svo')
