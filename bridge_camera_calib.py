import numpy as np
import cv2 as cv
import open3d as o3d
import sys, glob, json
from scipy.spatial.transform import Rotation
from sensorpy.camera_calib import load_config, get_2d_points

def get_grid_lineset(start=(0., 0., 0.), cellsize=0.1, extend=10, color=[0.5, 0.5, 0.5]):
    half = cellsize * extend
    xs = np.arange(start[0]-half, start[0]+half+cellsize, cellsize)
    y = start[1]
    zs = np.arange(start[2]-half, start[2]+half+cellsize, cellsize)

    pts_xs = [[x, y, zs[0]] for x in xs]
    pts_xe = [[x, y, zs[-1]] for x in xs]
    pts_zs = [[xs[0], y, z] for z in zs]
    pts_ze = [[xs[-1], y, z] for z in zs]

    lines_x = [[i, len(pts_xs)+i] for i in range(len(pts_xs))]
    offset = len(pts_xs) + len(pts_xe)
    lines_y = [[offset+i, offset+len(pts_zs)+i] for i in range(len(pts_zs))]

    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_xs + pts_xe + pts_zs + pts_ze),
        lines=o3d.utility.Vector2iVector(lines_x + lines_y),
    )
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lineset.lines))
    return lineset

def get_chessboard_points(chess_pattern_rows, chess_pattern_cols, chess_cellsize, offset_front=0., offset_height=0., offset_left=0.):
    x, y = np.meshgrid(range(chess_pattern_cols), range(chess_pattern_rows))
    z = np.zeros_like(y)
    pts = chess_cellsize * np.dstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    pts[:,:,0] = pts[:,:,0] - offset_left
    pts[:,:,1] = pts[:,:,1] - offset_height
    pts[:,:,2] = pts[:,:,2] + offset_front
    return pts

def get_pose_transformation(rvec, tvec):
    T = np.eye(4)
    T[:3,:3] = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    T[:3, 3] = tvec.flatten()
    return np.linalg.pinv(T)



if __name__ == '__main__':
    config_file = 'camera_calib.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Load configuration
    config = load_config(config_file)

    # Extract 2D points on the given images
    if not config['cam_file']:
        raise('An image sequence for the camera is not given')
    all_files = sorted(glob.glob(config['cam_file']))
    cam_pts, cam_img, cam_files = get_2d_points(all_files, config['chess_pattern_rows'], config['chess_pattern_cols'], config['subpx_winsize'], config['subpx_criteria'])

    # Prepare 3D points
    chessboard = get_chessboard_points(config['chess_pattern_rows'], config['chess_pattern_cols'], config['chess_cellsize'], config['offset_front'], config['offset_height'], config['offset_left'])
    obj_pts = [chessboard.astype(np.float32)] * len(cam_img)

    # Calibrate the camera
    if 'CALIB_USE_INTRINSIC_GUESS' not in config['calib_option']:
        # Enforce 'CALIB_USE_INTRINSIC_GUESS' for non-uniform Z values
        config['calib_flags'] += cv.CALIB_USE_INTRINSIC_GUESS
        h, w, *_ = cam_img[0].shape
        if config['cam_K'] is None:
            config['cam_K'] = np.array([[1000, 0, w/2], [0, 1000, h/2], [0, 0, 1]])
    cam_rms, cam_K, cam_distort, cam_rvecs, cam_tvecs = cv.calibrateCamera(obj_pts, cam_pts, cam_img[0].shape[::-1], config['cam_K'], config['cam_distort'], flags=config['calib_flags'])

    # Write the calibration result
    if config['calib_output']:
        with open(config['calib_output'], 'wt') as f:
            calib_result = {'cam_K': cam_K.tolist(), 'cam_distort': cam_distort.flatten().tolist()}
            if config['save_cam_pose']:
                calib_result['cam_pose'] = []
                for file, rvec, tvec in zip(cam_files, cam_rvecs, cam_tvecs):
                    calib_result['cam_pose'].append({'file': file, 'rvec': rvec.flatten().tolist(), 'tvec': tvec.flatten().tolist()})
            json.dump(calib_result, f, indent=4)

    # Print the calibration result briefly
    print('### Brief Calibration Report')
    print(f'* Calibration flags: {bin(config["calib_flags"])}')
    print(f'* Camera files: {config["cam_file"]}')
    print(f'* The number of used images: {len(cam_files)} / {len(all_files)}')
    print(f'* RMS error: {cam_rms:.6f} [pixel]')
    for file, rvec, tvec in zip(cam_files, cam_rvecs, cam_tvecs):
        T = get_pose_transformation(rvec, tvec)
        print(f'* Camera file: {file}')
        print(f'  * Camera height: {-T[1,-1]:.6f} [m]')
        print(f'  * Camera lateral offset: {T[0, -1]:.6f} [m]')
        print(f'  * Camera frontal offset: {T[2, -1]:.6f} [m]')
        print(f'  * Camera rotation: {np.rad2deg(np.linalg.norm(rvec)):.3f} [deg]')

    # Visualize the result
    grid_viz = get_grid_lineset()
    chessboard_viz = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(chessboard.reshape(-1, 3)))
    cam_viz = [o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)] * len(cam_rvecs)
    for (viz, rvec, tvec) in zip(cam_viz, cam_rvecs, cam_tvecs):
        viz.transform(get_pose_transformation(rvec, tvec))
    o3d.visualization.draw([grid_viz, chessboard_viz] + cam_viz, show_ui=True, show_skybox=False)
