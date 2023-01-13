import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2 as cv
import open3d as o3d
from sensorpy.zed import ZED, print_zed_info

def get_transformation(R, tvec):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,-1] = tvec
    return T

class SurfaceGPSZED:
    def __init__(self, init_rvec=np.zeros(3), init_tvec=np.zeros(3)):
        # self.state_robj = Rotation.from_rotvec(init_rvec)
        # self.state_tvec = init_tvec
        self.state_T = get_transformation(Rotation.from_rotvec(init_rvec).as_matrix(), init_tvec)
        self.odom_T_prev = None

    def apply_odometry(self, odom_qxyzw, odom_tvec):
        robj = Rotation.from_quat(odom_qxyzw)
        odom_T_curr = get_transformation(robj.as_matrix(), odom_tvec)
        if self.odom_T_prev is not None:
            T_delta = odom_T_curr @ np.linalg.inv(self.odom_T_prev)
            self.state_T = T_delta @ self.state_T
        self.odom_T_prev = odom_T_curr

    # def get_pose(self):
    #     return self.state_robj.as_rotvec(), self.state_tvec

    def get_poseT(self):
        return self.state_T

def load_config(config_file):
    '''Load configuration from the given file'''

    # The default configuration
    config = {
        'surface_file'      : '',
        'zed_depth_mode'    : 'neural',
        'zed_print_info'    : True,
        'local_name'        : 'SurfaceGPSZED',
        'local_init_rvec'   : [0, 0, 0],
        'local_init_tvec'   : [0, 0, 0],
        'vis_axes_length'   : 1,
        'vis_cam_fov'       : 90,
        'vis_cam_center'    : [0, 0, 0],
        'vis_cam_eye'       : [0, 0, -1],
        'vis_cam_up'        : [0, -1, 0],
        'vis_image_zoom'    : 0.5,
        'vis_ground_plane'  : 'XZ',
        'vis_show_depth'    : True,
        'vis_show_skybox'   : False,
        'vis_show_axes'     : True,
        'vis_show_ground'   : True,
        'vis_show_settings' : True,
    }

    # Update the configuration from the given file
    if config_file:
        with open(config_file, 'r') as f:
            config_from_file = json.load(f)
            config.update(config_from_file)

    # Preprocess the configuration
    for key in ['local_init_rvec', 'local_init_tvec', 'vis_cam_eye', 'vis_cam_center', 'vis_cam_up']:
        config[key] = np.array(config[key], dtype=np.float64)
    return config

def test_localizer(config_file='', svo_file='', svo_realtime=False):
    '''Test a localizer'''

    # Load the configuration file
    config = load_config(config_file)
    if not svo_file and ('svo_file' in config):
        svo_file = config['svo_file']

    # Open the ZED camera
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime, depth_mode=config['zed_depth_mode'])
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
    vis.setup_camera(config['vis_cam_fov'], config['vis_cam_center'], config['vis_cam_eye'], config['vis_cam_up'])
    GROUND_PLANE = {'XY': o3d.visualization.rendering.Scene.XY, 'XZ': o3d.visualization.rendering.Scene.XZ, 'YZ': o3d.visualization.rendering.Scene.YZ}
    vis.ground_plane = GROUND_PLANE[config['vis_ground_plane']]
    vis.show_skybox(config['vis_show_skybox'])
    vis.show_axes = config['vis_show_axes']
    vis.show_ground = config['vis_show_ground']
    vis.show_settings = config['vis_show_settings']
    app.add_window(vis)

    # Prepare the surface
    surface = None
    if config['surface_file']:
        surface = o3d.io.read_triangle_mesh(config['surface_file'])
        vis.add_geometry('Surface', surface)

    # Prepare the localizer
    localizer = SurfaceGPSZED(init_rvec=config['local_init_rvec'], init_tvec=config['local_init_tvec'])

    # Run the localizer with sensor data
    frame = 0
    while zed.is_open():
        # Get images and 3D pose
        if not zed.grab():
            break
        zed_color, _, zed_depth = zed.get_images()
        zed_state, zed_qxyzw, zed_tvec = zed.get_tracking_pose()

        # Perform the localizer
        localizer.apply_odometry(zed_qxyzw, zed_tvec)

        # Show the 3D pose
        robot_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(config['vis_axes_length'])
        robot_vis.transform(localizer.get_poseT())
        if not vis.is_visible:
            break
        vis.remove_geometry('Robot')
        vis.add_geometry('Robot', robot_vis)
        vis.post_redraw()

        # Show the images
        merge = zed_color
        if config['vis_show_depth']:
            zed_depth_color = cv.applyColorMap(zed_depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((zed_color, zed_depth_color)), (0, 0), fx=config['vis_image_zoom'], fy=config['vis_image_zoom'])
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
    test_localizer('config_M327.json', svo_file='data/220720_M327/short.svo')
    # test_localizer(svo_file='data/220902_Gym/short.svo')
