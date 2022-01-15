import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from ahrs import myAHRSPlus, AHRSCube

rad2deg = lambda rad: rad * 180 / np.pi
deg2rad = lambda deg: deg * np.pi / 180

class SphereSurface:
    def __init__(self, plotter, radius=1., color='gray', opacity=0.5):
        self.radius = radius
        self.mesh = pv.Sphere(radius=radius)
        self.actor = plotter.add_mesh(self.mesh, color=color, opacity=opacity)

    def normal2position(self, q_xyzw):
        R = Rotation.from_quat(q_xyzw)
        return self.radius * R.as_matrix()[:,-1]

class SurfaceGPS:
    def __init__(self, surface):
        self.p = [0, 0, 0]
        self.q = [0, 0, 0, 1]
        self.surface = surface

    def get_pose(self):
        return (self.p, self.q)

    def apply_ahrs(self, q_xyzw):
        self.p = self.surface.normal2position(q_xyzw)
        self.q = q_xyzw
        return True



if __name__ == '__main__':
    # Open the VHRS device
    ahrs_dev = myAHRSPlus()
    if ahrs_dev.open('COM4'):

        # Instantiate objects
        plotter = pv.Plotter()
        surface = SphereSurface(plotter)
        ahrs_viz = AHRSCube(plotter, scale=10)
        localizer = SurfaceGPS(surface)

        # Configure visualization
        plotter.add_axes_at_origin('r', 'g', 'b')
        plotter.show(title='SeoulTech SurfaceGPS', interactive_update=True)

        # Localize the AHRS and show it
        try:
            while True:
                q_xyzw = ahrs_dev.get_xyzw()
                localizer.apply_ahrs(q_xyzw)
                p, q = localizer.get_pose()
                ahrs_viz.set_position(p)
                ahrs_viz.set_orientation(q)
                plotter.update()
        except KeyboardInterrupt:
            pass

        plotter.close()