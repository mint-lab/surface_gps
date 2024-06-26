import numpy as np
from scipy.spatial.transform import Rotation
import cv2 as cv
import open3d as o3d
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from sensorpy.zed import ZED, print_zed_info

# 3 is not  valid INPUT_TYPE enum.py

def get_transformation(R, tvec):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,-1] = tvec
    return T

def play_3d_data(svo_file=None, svo_realtime=False, depth_mode='neural', zoom=0.5, auto_track=False):
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=True, depth_mode=depth_mode, min_depth=0.2)
    zed.start_tracking()
    # print_zed_info(zed)

    # Prepare the 3D viewer
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer('ZED 3D Viewer: 3D Data')
    vis.ground_plane = o3d.visualization.rendering.Scene.XY
    vis.show_skybox(False)
    vis.show_axes = True
    vis.show_ground = True
    vis.show_settings = True
    app.add_window(vis)
    T_c2w = np.array([[ 1,  0, 0, 0],
                      [0,  0, 1, 0],
                      [ 0, -1, 0, 0],
                      [ 0,  0, 0, 1]])

    # Prepare the image viewer
    cv.namedWindow('ZED 3D Viewer: Color and Depth')

    is_first = True
    while zed.is_open():
        if zed.grab():
            # Get all images
            color, _, depth = zed.get_images()
            status, qxyzw, tvec = zed.get_tracking_pose()
            robj = Rotation.from_quat(qxyzw)

            # Show the 3D pose
            zed_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
            zed_vis.transform(T_c2w @ get_transformation(robj.as_matrix(), tvec))
            pcd = o3d.t.geometry.PointCloud(zed.get_xyz().reshape(-1,3))
            pcd.point.colors=(color/255).reshape(-1,3).astype(np.float32)
            pcd.transform(T_c2w @ get_transformation(robj.as_matrix(), tvec))
            
            if not vis.is_visible:
                break
            vis.remove_geometry('Camera')
            vis.add_geometry('Camera', zed_vis)
            vis.remove_geometry('pcd')
            vis.add_geometry('pcd', pcd)
            vis.post_redraw()
            if is_first or auto_track:
                is_first = False
                vis.reset_camera_to_default()

            # Show the images
            depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('ZED 3D Viewer: Color and Depth', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    vis.close()
    zed.close()



if __name__ == '__main__':
    play_3d_data('../data/220720_M327/short.svo')
    # play_3d_data('data/220902_Gym/short.svo')
    # play_3d_data()
