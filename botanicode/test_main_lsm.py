import numpy as np
import os

from lsm import LevelSetMethod
from vec import VectorField, Gravitropism, Phototropism
from lightEngine import Sky
from plotter import Plotter



def initial_phi(x, y):
    a = 0.3  # Semi-major axis (along x-axis)
    b = .5  # Semi-minor axis (along y-axis)
    cx, cy = 0.0, 0.6  # Center of the ellipse

    # superellipse
    n = 4
    phi = (np.abs((x - cx) / a)**n + np.abs((y - cy) / b)**n)**(1/n) - 1
    return phi

# Initialize solver and vector field
nx, ny = 200, 200  # Grid size
dt = 0.001  # Time step

# space grid
x = np.linspace(-1, 1, nx, endpoint=True)
y = np.linspace(0, 2, ny, endpoint=True)

vector_field = VectorField(x,y, nx, ny)
# Add tropisms to the vector field
#vector_field.add_tropism(Gravitropism(), weight=0.1)
sky = Sky(position=(2, 1.5))
vector_field.add_tropism(Phototropism(sky), weight=0.9)

solver = LevelSetMethod(
    x,y,nx=nx, ny=ny,dt=dt,
    vector_field=vector_field,
    phi_initial_func=initial_phi,
    spatial_scheme="upwind",
)

plotter = Plotter([solver, vector_field], dpi=500)


# Number of time steps
time_steps = 300

# Main loop for simulation
for t in range(time_steps):
    solver.time_step()  # Update the level set
   
    if t % 100 == 0:
        plotter.plot()


#plotter.animate(img_folder="result_lsm", fps=1)