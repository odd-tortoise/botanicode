import numpy as np
from typing import List
from dataclasses import dataclass
import networkx as nx


from env import Environment
from botanical_nodes import NodeFactory, NodeState
from botanical_nodes import Stem, Leaf, Root, SAM, RAM
from shapes import CylinderShape, PointShape, LeafShape
from simulator import SimClock, Simulation

from plant import Plant, PlantState
from plant_reg import PlantRegulation

from model import Model, Rule



result_folder = "results"

env_setting_file = "botanicode/settings_file/env_setting.json"



# create a clock
clock = SimClock(photo_period=(8,18),step="hour")

#create environment
env = Environment().set_env(env_setting_file)

# create a factory for the nodes
node_factory = NodeFactory()

@dataclass
class StemState(NodeState):
    length: float = 1.0
    age: int = 0

@dataclass
class LeafState(NodeState):
    size: float = 1.0
    petioles_size: float = 0.1
    rachid_size: float = 0.1
    age: int = 0

@dataclass
class RootState(NodeState):
    length: float = 1.0
    age: int = 0

@dataclass
class SAMState(NodeState):
    age: int = 0

@dataclass
class RAMState(NodeState):
    age: int = 0


node_factory.add_blueprint(Stem, StemState, CylinderShape)
node_factory.add_blueprint(Leaf, LeafState, LeafShape)
node_factory.add_blueprint(Root, RootState, CylinderShape)
node_factory.add_blueprint(SAM, SAMState, PointShape)
node_factory.add_blueprint(RAM, RAMState, PointShape)

node_factory.read_blueprint_file("botanicode/settings_file/node_blueprints.json")

# create a plant regulation file
plant_regulation_file = "botanicode/settings_file/tomato_1.json"
plant_reg = PlantRegulation(plant_regulation_file)

# create a plant state
@dataclass
class PlantState(PlantState):
    internodes_no : int = 0

    def reset(self):
        self.internodes_no = 0

state = PlantState()


# create a plant
plant = Plant(reg = plant_reg, node_factory = node_factory, plant_state = state)


# create a rule for the stems
stem_rule = Rule( [Stem], trainable = True, is_dynamic = False, no_params = 2)
def stem_length_rule(nodes : List[Stem], params : np.array):
    for node in nodes:
        #logistic growth
        node.state.length = (5 + (node.id -1  - 4)**2 ) / (1 + np.exp(-params[0] * (node.state.age - params[1]))) 
stem_rule.set_action(stem_length_rule)


def shoots_if_rule(plant : Plant):
    list_to_shoot = []
    for node in plant.structure.G.nodes():
        if isinstance(node, SAM) and plant.plant_state.internodes_no < 8:
            list_to_shoot.append(node)
            
    return list_to_shoot

# create a model to store the rules
model = Model("tomato")
model.add_node_rule(stem_rule)
model.add_shooting_rule(shoots_if_rule)

env_reads = { Stem: ["temp"],
              Leaf: ["temp","light"],
              SAM: ["temp"],
            }
model.env_reads = env_reads


# create a simulation
sim = Simulation(plant, model, clock, env)


# create a dataset
from utils import Dataset
data = Dataset()


model.set_trainable_params([0.5, 30])

# tune the paramters
sim.tune(data, 60,1)


