import numpy as np
import os

import logging
logging.basicConfig(
    filename="plant_sim.log",
    encoding="utf-8",
    filemode="w",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,

)

logger = logging.getLogger("plant_sim")


from lightEngine import Sky
from soil import Soil
from env import Environment
from plant import Plant, GrowthRegulation
from plotter import Plotter

reg = GrowthRegulation(
    leaf_arrangement="opposite",
    length_to_shoot=3
)

my_plant = Plant(reg)
sky = Sky(position=np.array([0, 0, 20]))
soil = Soil()
env = Environment(sky=sky, soil=soil)

plotter_graph_objects = [my_plant.structure]*4
plotter_graph_methods = [lambda ax: my_plant.structure.plot_value(ax=ax,resource="elongation_rate"),
                lambda ax: my_plant.structure.plot_value(ax=ax,resource="sugar"),
                lambda ax: my_plant.structure.plot_value(ax=ax,resource="auxin"),
                lambda ax: my_plant.structure.plot_value(ax=ax,resource="water")]
plotter_graph_3d = [False]*4
plotter_grafo = Plotter(objects_to_plot=plotter_graph_objects, plot_methods=plotter_graph_methods, plot_3ds=plotter_graph_3d, ncols=2, dpi=500, figsize=(15,8))

plotter_struttura = Plotter(objects_to_plot=[my_plant], plot_methods=[None], plot_3ds=[True], ncols=1, dpi=500, figsize=(15,8))


my_plant.update(env)
logger.info("Plant created.")
my_plant.log(logger=logger)
plotter_grafo.plot()
plotter_struttura.plot()

# Simulation parameters
time_steps = 10 # Number of growth steps
delta_t = 1     # Time increment for each growth step

folder = "results_plant_branch"

logger.info("Starting plant growth simulation...")


# Simulation loop
for step in range(time_steps):
    logger.info(f"\n--- Growth Step {step + 1} ---")

   
    my_plant.grow(delta_t)

    my_plant.update(env)
    my_plant.log(logger=logger)
    #plotter_grafo.plot()
    #plotter_struttura.plot()
    

logger.info("Growth simulation completed.")


plotter_history_objects = [my_plant.structure.history]*4
plotter_history_methods = [lambda ax: my_plant.structure.history.plot_value(ax=ax,resource="elongation_rate"),
                lambda ax: my_plant.structure.history.plot_value(ax=ax,resource="sugar"),
                lambda ax: my_plant.structure.history.plot_value(ax=ax,resource="auxin"),
                lambda ax: my_plant.structure.history.plot_value(ax=ax,resource="water")]

plotter_history_3d = [False]*4

plotter_history = Plotter(objects_to_plot=plotter_history_objects, plot_methods=plotter_history_methods, plot_3ds=plotter_history_3d, ncols=2, dpi=500, figsize=(15,8))

plotter_history.plot()

#plotter.animatemg_folder=folder,fps=1)
