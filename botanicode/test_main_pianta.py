import numpy as np
import os

from lightEngine import Sky
from plant import Plant, GrowthRegulation
from plotter import Plotter

reg = GrowthRegulation('botanicode/tomato_data/tomato.json')
my_plant = Plant(reg, age=0)

sky = Sky(position=np.array([0, 0, 20]))
soil = Soil()

env = Environment(sky=sky, soil=soil)

plotter_graph_objects = [my_plant.structure]*4
plotter_graph_methods = [lambda ax: my_plant.structure.plot(ax=ax, pos= True),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="lenght"),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="leaf_size"),
                         lambda ax: my_plant.structure.plot_value(ax=ax, var="radius"),]
plotter_graph_3d = [False]*4
plotter_grafo = Plotter(objects_to_plot=plotter_graph_objects, plot_methods=plotter_graph_methods, plot_3ds=plotter_graph_3d, ncols=2, dpi=500, figsize=(15,8))

plotter_struttura = Plotter(objects_to_plot=[my_plant], plot_methods=[None], plot_3ds=[True], ncols=1, dpi=500, figsize=(15,8))



folder = "results_ODEs"

my_plant.update(env)
logger.info("Plant created.")
my_plant.log(logger=logger)
#plotter_grafo.plot(save_folder=folder, name=f"grafo_{0}")
#plotter_struttura.plot(save_folder=folder, name=f"struttura_{0}")
#plotter_grafo.plot()
#plotter_struttura.plot()

# Simulation parameters
time_steps = 30 # one week
delta_t = 1 # hours



print("Starting plant growth simulation...")


# Simulation loop
for step in range(time_steps):
    logger.info(f"\n--- Growth Step {step + 1}/{time_steps} ---")   

    my_plant.grow(delta_t)
    
    my_plant.update(env)
    my_plant.log(logger=logger)

    #plotter_grafo.plot(save_folder=folder, name=f"grafo_{step}")
    #plotter_struttura.plot(save_folder=folder, name=f"struttura_{step}")

    #plotter_grafo.plot()
    #plotter_struttura.plot()

    

logger.info("Growth simulation completed.")

data = my_plant.structure.history.get_data()

stem_data = data['Stem']
SAM_data = data['SAM']
leaves_data = data['Leaf']


# get number of stems
number_of_stems = len(stem_data)
number_of_sams = len(SAM_data)
number_of_leaves = len(leaves_data)


number_of_plots = 2

# make subplots for each stem
import matplotlib.pyplot as plt

fig, axs = plt.subplots(number_of_plots, 1, figsize=(10, 10))

for i in range(number_of_plots):
    for key, value in stem_data.items():
        # key is the stem number
        # value is the history of the stem 

        # get the stem lenght vs time, time is the key of the dictionary value
        time_steps = list(value.keys())
        if i == 0:
            val = [value[time_step]['lenght'] for time_step in time_steps]
        else:
            val = [value[time_step]['radius'] for time_step in time_steps]
        

        axs[i].plot(time_steps, val, label=f"{key}")
        if i == 0:
            axs[i].set_ylabel("Lenght [cm]")
        else:
            axs[i].set_ylabel("Radius [cm]")
        
    axs[i].grid()
    axs[i].legend()
        

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(number_of_plots, 1, figsize=(10, 10))

for i in range(number_of_plots):
    for key, value in SAM_data.items():
        # key is the stem number
        # value is the history of the stem 

        # get the stem lenght vs time, time is the key of the dictionary value
        time_steps = list(value.keys())
        if i == 0:
            val = [value[time_step]['time_to_next_shoot'] for time_step in time_steps]
        else:
            val = [value[time_step]['time_to_next_shoot'] for time_step in time_steps]
        

        axs[i].plot(time_steps, val, label=f"{key}")
        if i == 0:
            axs[i].set_ylabel("Lenght [cm]")
        else:
            axs[i].set_ylabel("Radius [cm]")
        
    axs[i].grid()
    axs[i].legend()
        

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(number_of_plots, 1, figsize=(10, 10))

for i in range(number_of_plots):
    for key, value in leaves_data.items():
        # key is the stem number
        # value is the history of the stem 

        # get the stem lenght vs time, time is the key of the dictionary value
        time_steps = list(value.keys())
        if i == 0:
            val = [value[time_step]['leaf_size'] for time_step in time_steps]
        else:
            val = [value[time_step]['z_angle'] for time_step in time_steps]
        

        axs[i].plot(time_steps, val, label=f"{key}")
        if i == 0:
            axs[i].set_ylabel("Lenght [cm]")
        else:
            axs[i].set_ylabel("Radius [cm]")
        
    axs[i].grid()
    axs[i].legend()
        

plt.tight_layout()
plt.show()




#plotter.animatemg_folder=folder,fps=1)
