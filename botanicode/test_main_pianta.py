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


from light import Sky
from soil import Soil
from air import Air
from env import Environment
from plant import Plant
from plotter import Plotter
from tuner import GrowthRegulation
from simclock import SimClock


#####################################

timer = SimClock(start_time=0)

reg = GrowthRegulation('botanicode/tomato_data/tomato.json')
my_plant = Plant(reg, age=0, timer=timer)

sky = Sky(position=np.array([0, 0, 100]))
soil = Soil(moisture=0.5)
air = Air(temperature=20, water_concentration=0.5)

env = Environment(sky=sky, soil=soil, air=air)

plotter_graph_objects = [my_plant.structure]*4
plotter_graph_methods = [lambda ax: my_plant.structure.plot(ax=ax, pos= True),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="lenght"),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="leaf_size"),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="radius"),]
plotter_graph_3d = [False]*4
plotter_grafo = Plotter(objects_to_plot=plotter_graph_objects, plot_methods=plotter_graph_methods, plot_3ds=plotter_graph_3d, ncols=2, dpi=500, figsize=(15,8))

plotter_struttura = Plotter(objects_to_plot=[my_plant], plot_methods=[None], plot_3ds=[True], ncols=1, dpi=500, figsize=(15,8))



folder = "results_ODEs"

my_plant.update(env = env)
logger.info("Plant created.")
my_plant.log(logger=logger)
#plotter_grafo.plot(save_folder=folder, name=f"grafo_{0}")
#plotter_struttura.plot(save_folder=folder, name=f"struttura_{0}")
#plotter_grafo.plot()
#plotter_struttura.plot()

# Simulation parameters
time_steps = 30 # one week
delta_t = 1 # hours


logger.info("Starting plant growth simulation...")


# Simulation loop
for step in range(time_steps):
    logger.info(f"\n--- Growth Step {step + 1}/{time_steps} ---")   

    my_plant.grow(delta_t)
    
    my_plant.update(env)
    my_plant.structure.snapshot(timer.elapsed_time)
    my_plant.log(logger=logger)

    #plotter_grafo.plot(save_folder=folder, name=f"grafo_{step}")
    #plotter_struttura.plot(save_folder=folder, name=f"struttura_{step}")

    #plotter_grafo.plot()
    #plotter_struttura.plot()
    timer.tick(delta_t)



    

logger.info("Growth simulation completed.")

plotter_struttura.plot()


my_plant.structure.history.plot(values=["lenght","radius"], node_type="Stem")
my_plant.structure.history.plot(values=["leaf_size"], node_type="Leaf")