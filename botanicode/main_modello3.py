import numpy as np
from typing import List
from dataclasses import dataclass
import networkx as nx

from simulator import SimClock, Simulation

from env import Environment
from light import Sky
from soil import Soil
from air import Air

from plant import Plant
from plant_reg import PlantRegulation

from botanical_nodes import Stem, Leaf, SAM, RAM, Root

from model import Model, NodeStateBlueprint, NodeRuleBlueprint
from shapes import CylinderShape, LeafShape, PointShape

from utils import NumericalIntegrator as NI
from utils import plotter

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
logger.info("Initializing simulation...")


# set a clock for all the simulations
Simulation.set_clock(photo_period=(8,18),step="hour")
Simulation.logger = logger
# this clock is also the one used by the plant and environment objects, single source of truth

# create a folder to save the results
folder = "results"


################### READ A PLANT FILE ###################
plantfile_path = "botanicode/tomato_data/tomato_3.json"
# this will be used to initialize the blue prints and the plant object paramters
plant_reg = PlantRegulation(plantfile_path)



################### DEFINE A MODEL ###################
# define the state variables for each node type

@dataclass
class StemState(NodeStateBlueprint):
    lenght: float
    radius: float
    water: float

@dataclass
class LeafState(NodeStateBlueprint):
    size: float
    petioles_size: float
    rachid_size: float
    water: float

@dataclass
class SAMState(NodeStateBlueprint):
    pass

@dataclass
class RAMState(NodeStateBlueprint):
    pass

@dataclass
class RootState(NodeStateBlueprint):
    lenght: float
    radius: float

# QUI SI FA IL TUNING !!
# define the rules for each node type
def stem_length_ode(t, y, nodes: list[Stem]):
    return 1

def stem_derived_rules(node):
    node.state.radius = 0.5*node.state.lenght

def leaf_derived_rules(node):
    node.state.rachid_size = 0.5*node.state.size
    node.state.petiole_size = 0.2*node.state.size

def leaf_size_ode(t, y, nodes: List[Leaf]): 
    return 1

stem_rules = NodeRuleBlueprint(dynamics={"lenght": stem_length_ode},
                               derived=stem_derived_rules,
                               env_reading = ["temp"])
leaf_rules = NodeRuleBlueprint(dynamics={"size": leaf_size_ode},
                               derived= leaf_derived_rules,
                               env_reading = ["temp"])
sam_rules = NodeRuleBlueprint()
ram_rules = NodeRuleBlueprint()
root_rules = NodeRuleBlueprint()

def shooting_rule(node):
    return isinstance(node, SAM) #boolean expression to decide if to shoot or not from node 

def water_diffusion(t, y, args):
    # this kind of ode is for the whole plant so it is not a method of a single node
    # we have a system of ODEs, one for each variable/node
    nodes, plant = args

    graph = plant.structure.G

    # extract the subgraph of the plant made by the nodes of interest
    subgraph = graph.subgraph(nodes)


    L = nx.laplacian_matrix(subgraph).todense() 

    

    rhs = - (L @ y)

    return rhs






##!!!!!!!!!

# create a model object
model = Model(model_name="empirical_time_tomato")
model.add_blueprint(Stem, StemState, stem_rules, CylinderShape)
model.add_blueprint(Leaf, LeafState, leaf_rules, LeafShape)
model.add_blueprint(SAM, SAMState, sam_rules, PointShape)
model.add_blueprint(RAM, RAMState, ram_rules, PointShape)
model.add_blueprint(Root, RootState, root_rules, CylinderShape)

model.add_shooting_rule(shooting_rule)

model.add_whole_plant_dynamic(var = "water", node_types=[Stem, Leaf], ode = water_diffusion)


################### CREATE AN ENVIRONMENT #############
sky = Sky(position=np.array([0, 0, 100]))
soil = Soil(moisture=0.5)
air = Air(temperature=20, water_concentration=0.5)
env = Environment(sky=sky, soil=soil, air=air) 


################### CREATE A PLANT ###################
# create a plant object
plant = Plant(plant_reg,model)
plant.probe(env)


################### DEFINE A SOLVER ###################
solver = NI("forward_euler", dt=1)


################### DEFINE EXTRA TASKS for the simulation ###################
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
                "plot_methods": [lambda ax: plant.structure.plot(ax=ax),
                                lambda ax: plant.structure.plot_value(ax=ax, var="lenght", node_types=[Stem]),
                                lambda ax: plant.structure.plot_value(ax=ax, var="size", node_types=Leaf),
                                lambda ax: plant.structure.plot_value(ax=ax, var="radius", node_types=Stem),
                                lambda ax: plant.structure.plot_value(ax=ax, var="temp", node_types=Leaf),
                                lambda ax: plant.structure.plot_value(ax=ax, var="temp",node_types=Stem)],
                "plot_3ds": [False, False, False, False, False, False],
                "ncols": 2,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        },
        "log":{
            "method": plant.log,
            "args": [Simulation.logger]
        },
    },
    "during": {
        "log":{
            "method": plant.log,
            "kwargs": {
                "logger": Simulation.logger
            }
        },
        "plot_iniziale_grafo":{
            "method": plotter,
            "kwargs": {
                "plot_methods": [lambda ax: plant.structure.plot(ax=ax),
                                lambda ax: plant.structure.plot_value(ax=ax, var="lenght", node_types=[Stem]),
                                lambda ax: plant.structure.plot_value(ax=ax, var="size", node_types=Leaf),
                                lambda ax: plant.structure.plot_value(ax=ax, var="radius", node_types=Stem),
                                lambda ax: plant.structure.plot_value(ax=ax, var="water", node_types=[Stem,Leaf]),
                                lambda ax: plant.structure.plot_value(ax=ax, var="temp",node_types=Stem)],
                "plot_3ds":  [False, False, False, False, False, False],
                "ncols": 2,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        },
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
        "plot_finale_history":{
            "method": plotter,
            "kwargs": {
                "plot_methods": [lambda ax: plant.history.plot(variable="water",ax=ax, node_types=["Leaf", "Stem"]),
                                 lambda ax: plant.history.plot(variable="lenght",ax=ax, node_types=["Stem"]),],
                "plot_3ds": [False,False],
                "ncols": 1,
                "dpi": 500,
                "figsize": (15,8),
                "save_folder": folder,
            }
        },
    }
}

extra_tasks["before"] = {}


################### CREATE THE SIMULATION ###################
# create a simulation object, it is the orchestrator of the simulation
sim = Simulation(config_file="botanicode/tomato_data/sim_settings.json",
                env=env, plant=plant,
                solver = solver,
                model = model,
                tasks=extra_tasks,
                folder=folder)


################### RUN THE SIMULATION ###################
sim.run()