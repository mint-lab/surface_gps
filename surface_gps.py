import numpy as np
import pyvista as pv
import time

rad2deg = lambda rad: rad * 180 / np.pi

if __name__ == '__main__':
    # Prepare
    sphere_radius = 1
    plotter = pv.Plotter()
    axes = pv.Axes(show_actor=True, actor_scale=1.5, line_width=4)
    plotter.add_actor(axes.actor)
    sphere = pv.Sphere(radius=sphere_radius)
    sphere_actor = plotter.add_mesh(sphere)
    ahrs = pv.Cube(x_length=0.2, y_length=0.1, z_length=0.001)
    ahrs_actor = plotter.add_mesh(ahrs, color='r')

    # Move 'ahrs' on 'sphere'
    plotter.show(interactive_update=True)
    for theta in np.linspace(0, 2 * np.pi, 100):
        x = sphere_radius * np.cos(theta)
        y = sphere_radius * np.sin(theta)
        ahrs_actor.SetPosition(x, y, 0)
        ahrs_actor.SetOrientation(0, 90, rad2deg(theta))
        plotter.update()
        time.sleep(0.1)
    plotter.show()