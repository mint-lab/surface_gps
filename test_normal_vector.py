import pyvista as pv

mesh = pv.Sphere(radius=10)
#mesh = pv.Box((-10, 10, -10, 10, -10, 10), level=10)
#mesh = pv.ParametricRandomHills(1, 20, 20, 40)
#mesh = pv.ParametricRandomHills(randomseed=-1)
#mesh = pv.examples.load_nut()

mesh = mesh.compute_normals()
mesh = mesh.elevation()
arrow = mesh.glyph(scale='Normals', orient='Normals', tolerance=0.02)

plotter = pv.Plotter()
plotter.add_axes()
plotter.add_mesh(mesh, scalars='Elevation', cmap='terrain')
plotter.add_mesh(arrow, color='black')

plotter.show()