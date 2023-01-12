import numpy as np
from scipy.spatial.transform import Rotation
import cv2 as cv
import open3d as o3d
from sensorpy.zed import ZED, print_zed_info

def get_transformation(R, tvec):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,-1] = tvec
    return T

def extract_3d_pose(svo_file=None, svo_realtime=False, depth_mode='neural', zoom=0.5, output_file='zed_3d_pose_extract.npy'):
    zed = ZED()
    zed.open(svo_file=svo_file, svo_realtime=svo_realtime, depth_mode=depth_mode)
    zed.start_tracking()
    print_zed_info(zed)

    # Prepare the 3D viewer
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer('SurfaceGPS 3D Viewer: 3D Pose')
    vis.ground_plane = o3d.visualization.rendering.Scene.XZ
    vis.show_skybox(False)
    vis.show_axes = True
    vis.show_ground = True
    vis.show_settings = True
    app.add_window(vis)

    # Prepare the image viewer
    cv.namedWindow('SurfaceGPS 3D Viewer: Color and Depth')

    poses = []
    is_first = True
    while zed.is_open():
        # Get images and 3D pose
        if not zed.grab():
            break
        color, _, depth = zed.get_images()
        status, qxyzw, tvec = zed.get_tracking_pose()
        robj = Rotation.from_quat(qxyzw)

        # Record the 3D pose
        poses.append(np.hstack([tvec, robj.as_rotvec(), status]))

        # Show the 3D pose
        zed_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
        zed_vis.transform(get_transformation(robj.as_matrix(), tvec))
        if not vis.is_visible:
            break
        vis.remove_geometry('Camera')
        vis.add_geometry('Camera', zed_vis)
        vis.post_redraw()
        if is_first:
            is_first = False
            vis.reset_camera_to_default()

        # Show the images
        depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
        merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
        cv.imshow('SurfaceGPS 3D Viewer: Color and Depth', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    vis.close()
    zed.close()

    # Write the 3D poses
    if len(output_file) > 3 and (len(poses) > 0):
        poses = np.vstack(poses)
        if output_file[-4:].lower() == '.npy':
            np.save(output_file, poses)
        else:
            np.savetxt(output_file, poses, delimiter=',')



if __name__ == '__main__':
    extract_3d_pose('data/220720_M327/short.svo')
    # extract_3d_pose('data/220902_Gym/short.svo')
    # extract_3d_pose()
