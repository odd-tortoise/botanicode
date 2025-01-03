import numpy as np

from simulator import SimClock, Simulation, plotter

from env import Environment
from light import Sky
from soil import Soil
from air import Air

from plant import Plant
from tuner import GrowthRegulation

from botanical_nodes import Stem, Leaf


# set a clock for all the simulations
Simulation.set_clock(start_time=0, photo_period=(8,18))
# this clock is also the one used by the plant and environment objects, single source of truth

folder = "results"
# create a folder to save the results


# create an environment object
sky = Sky(position=np.array([0, 0, 100]))
soil = Soil(moisture=0.5)
air = Air(temperature=20, water_concentration=0.5)

env = Environment(sky=sky, soil=soil, air=air) 

# create a plant object
reg = GrowthRegulation('botanicode/tomato_data/tomato.json')
plant = Plant(reg, age=0)


extra_tasks = {
    "before": {
        "plot_iniziale_struttura":{
            "method": plotter,
            "kwargs": {
                "plot_methods": [lambda ax: plant.plot(ax=ax)],
                "plot_3ds": [True],
                "ncols": 1,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        },
        "plot_iniziale_grafo":{
            "method": plotter,
            "kwargs": {
                "plot_methods": [lambda ax: plant.structure.plot(ax=ax, pos= True),
                                lambda ax: plant.structure.plot_value(ax=ax, var="shape.lenght", node_types=[Stem]),
                                lambda ax: plant.structure.plot_value(ax=ax, var="shape.size", node_types=Leaf),
                                lambda ax: plant.structure.plot_value(ax=ax, var="shape.radius", node_types=Stem),
                                lambda ax: plant.structure.plot_value(ax=ax, var="env_data.temperature", node_types=Leaf),
                                lambda ax: plant.structure.plot_value(ax=ax, var="device_data.temperature.val",node_types=Stem)],
                "plot_3ds": [False, False, False, False, False, False],
                "ncols": 2,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        }
    },
    "during": {
        
    },
    "after": {
        "plot_finale_struttura":{
            "method": plotter,
            "kwargs": {
                "plot_methods": [lambda ax: plant.plot(ax=ax)],
                "plot_3ds": [True],
                "ncols": 1,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        },
    }
}


# create a simulation object
sim = Simulation(config_file="botanicode/sim_settings.json", env=env, plant=plant, tasks=extra_tasks, folder=folder)

sim.run(10,1)