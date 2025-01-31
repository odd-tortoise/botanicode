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
from utils import Dataset,NumericalIntegrator



result_folder = "results"

env_setting_file = "botanicode/settings_file/env_setting.json"



# create a clock
clock = SimClock(photo_period=(8,18),step="hour")

#create environment
env = Environment().set_env(env_setting_file)

# create a solver 
ni = NumericalIntegrator("forward_euler")

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
    #it also has the plant height
    internodes_no : int = 0
    
state = PlantState()


# create a plant
plant = Plant(reg = plant_reg, node_factory = node_factory, plant_state = state)


# create a rule for the stems
stem_rule = Rule( trainable = True, is_dynamic = False, no_params = 2)
def stem_length_rule(plant : Plant, params : np.array):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Stem)]
    for node in nodes:
        #logistic growth
        node.state.length = (5 + (node.id -1  - 4)**2 ) / (1 + np.exp(-params[0] * (node.state.age - params[1]))) 
stem_rule.set_action(stem_length_rule)

# create a dynamic rule for the leaves
leaf_rule = Rule(trainable = False, is_dynamic = True, no_params = 0)
def leaf_size_rule(t,y, plant: Plant, params : np.array):
    # deve essere una funzione che restituisce il rhs dell'equazione differenziale 
    # perché viene data in pasto a ODEINT che la risolve -> t,y, args
    rhs = []
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        node.state.s_max = 4*(node.parent.id+1)
        rhs.append(0.5/node.state.s_max * node.state.size * ( node.state.s_max -  node.state.size))

    return np.array(rhs)
leaf_rule.set_action(leaf_size_rule, "size",[Leaf])

# create a rule for the plant
plant_rule = Rule( trainable = False, is_dynamic = False, no_params = 0)
def plant_rule_action(plant : Plant, params : np.array):
    plant.plant_state.internodes_no = len([node for node in plant.structure.G.nodes() if isinstance(node, Stem)])
plant_rule.set_action(plant_rule_action)


# create a rule for shooting
def shoots_if_rule(plant : Plant):
    list_to_shoot = []
    for node in plant.structure.G.nodes():
        if isinstance(node, SAM) and plant.plant_state.internodes_no < 3:
            list_to_shoot.append(node)
            
    return list_to_shoot

# create a rule for the branch
def branch_if_rule(plant : Plant):
    list_to_branch = []
    for node in plant.structure.G.nodes():
        if isinstance(node, Leaf) and node.state.age > 10:
            list_to_branch.append(node)
            
    return list_to_branch


# create a model to store the rules
model = Model("tomato")
model.add_rule(stem_rule)
model.add_dynamic_rule(leaf_rule)
model.add_shooting_rule(shoots_if_rule)
model.add_branching_rule(branch_if_rule)


env_reads = { 
                Stem: ["temp"],
                Leaf: ["temp","light"],
                SAM: ["temp"]
            }
model.env_reads = env_reads


# create a simulation
sim = Simulation(plant, model, clock, env, ni)

ni.set_dt(1)

sim.run(5,1)
sim.reset_simulation()
sim.run(6,1)


# create a dataset
from utils import Dataset
data = Dataset()


# tune the paramters
#sim.tune(data, 60,1)


