import numpy as np
import os

from lightEngine import Sky
from plant import Plant, GrowthRegulation
from plotter import Plotter

reg = GrowthRegulation(
    leaf_arrangement="alternate",
    length_to_shoot=2
)


my_plant = Plant(reg)
sky = Sky(position=np.array([0, 0, 20]))


object_to_plot = [my_plant.structure, my_plant, my_plant.structure, my_plant.structure]
plot_methods = [None, "plot", "plot_lighting", "plot_auxin"]
plot_3ds = [False, True, False, False]
plotter = Plotter(objects_to_plot=object_to_plot, plot_methods=plot_methods, plot_3ds=plot_3ds, ncols=2, dpi=500, figsize=(15,8))

folder = "results_alternate"

my_plant.update(sky=sky)
my_plant.print()
plotter.plot()


# Simulation parameters
time_steps = 9 # Number of growth steps
delta_t = 1     # Time increment for each growth step



print("Starting plant growth simulation...")


# Simulation loop
for step in range(time_steps):
    print(f"\n--- Growth Step {step + 1} ---")

   
    my_plant.grow(delta_t)
   
    my_plant.update(sky=sky)
    my_plant.print() 
    plotter.plot()
   
    
    



print("Growth simulation completed.")


#plotter.animatemg_folder=folder,fps=1)
