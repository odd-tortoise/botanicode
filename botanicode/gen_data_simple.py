import numpy as np
from dataclasses import dataclass, field

from env import Environment, Clock
from env_components import *
from botanical_nodes import NodeFactory, NodeState
from botanical_nodes import Stem, Leaf, Root, SAM, RAM
from shapes import CylinderShape, PointShape, LeafShape
from simulator import Simulation

from plant import Plant, PlantState
from plant_reg import PlantRegulation

from development_engine import StaticRule, DynamicRule, DevelopmentEngine
from utils import NumericalIntegrator, Plotter

# create a clock
clock = Clock(photo_period=(8,18),step="hour")

# create a numerical integrator for the ODEs 
ni = NumericalIntegrator("forward_euler")

# create a plant regulation file
plant_regulation_file = "botanicode/single_run_files/tomato.json"
plant_reg = PlantRegulation(plant_regulation_file)

# create a factory for the nodes
node_factory = NodeFactory()

@dataclass
class StemState(NodeState):
    length: float = 1.0
    radius: float = 0.1
    direction: np.ndarray =  field(default_factory=lambda: np.array([0, 0, 1]))

@dataclass
class LeafState(NodeState):
    size: float = 1.0
    petioles_size: float = 0.1
    rachid_size: float = 1
    leaflets_number: int = plant_reg.phylotaxis["leaflets_number"]
    leaf_bending_rate: float = plant_reg.phylotaxis["leaf_bending_rate"]
    outline_function: callable = plant_reg.phylotaxis["outline_function"]
    y_angle: float = plant_reg.phylotaxis["y_angle"]
    z_angle: float = 0

@dataclass
class RootState(NodeState):
    length: float = 1.0
    radius: float = 0.1 
    direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))

@dataclass
class SAMState(NodeState):
    pass

@dataclass
class RAMState(NodeState):
    pass


node_factory.add_blueprint(Stem, StemState, CylinderShape)
node_factory.add_blueprint(Leaf, LeafState, LeafShape)
node_factory.add_blueprint(Root, RootState, CylinderShape)
node_factory.add_blueprint(SAM, SAMState, PointShape)
node_factory.add_blueprint(RAM, RAMState, PointShape)


# create a plant state
@dataclass
class PlantState(PlantState):
    #it also has the plant height
    internodes_no : int = 0
    expected_internodes_no : int = 0
    
state = PlantState()


# create a plant
plant = Plant(reg = plant_reg, node_factory = node_factory, plant_state = state)

# create a rule for the stems
def stem_length_rule(plant : Plant, params : np.array):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Stem)]
    for node in nodes:
        node.state.length = (10+0.8*node.state.temp) /(1+5*np.exp(-0.062*node.state.age))

stem_rule = StaticRule(action=stem_length_rule)


def leaf_rachid_rule(plant : Plant, params : np.array ):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        #logistic growth
        node.state.rachid_size = 0.5*node.state.size
rachid_rule = StaticRule(action=leaf_rachid_rule)

# create a dynamic rule for the leaves

def leaf_size_rule(t,y, plant: Plant, params : np.array):
    # deve essere una funzione che restituisce il rhs dell'equazione differenziale 
    # perché viene data in pasto a ODEINT che la risolve -> t,y, args
    rhs = []
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        node.state.s_max = 4*(node.state.rank+1)
        rhs.append((0.05*node.state.temp)/node.state.s_max * node.state.size * ( node.state.s_max -  node.state.size))

    return np.array(rhs)
leaf_rule = DynamicRule(action=leaf_size_rule, var = "size", types = [Leaf])

# create a rule for the plant

def plant_rule_action(plant : Plant, params : np.array):
    plant.state.internodes_no = len([node for node in plant.structure.G.nodes() if isinstance(node, Stem)])
    plant.state.expected_internodes_no = 2+plant.state.age if plant.state.age < 4 else 5
plant_rule = StaticRule(action=plant_rule_action)


# create a rule for shooting
def shoots_if_rule(plant : Plant):
    list_to_shoot = [] #this is a list of tuples (nodes from which to shoot, the amout of shoots to do from that node)


    #list of sams 
    sams = [node for node in plant.structure.G.nodes() if isinstance(node, SAM)]

    #distribute the nodes to shoot on the sams equally
    nodes_to_shoot = int(np.ceil(plant.state.expected_internodes_no - plant.state.internodes_no))

    if sams:
        shoots_per_sam = nodes_to_shoot // len(sams)
        remaining_shoots = nodes_to_shoot % len(sams)

        for sam in sams:
            shoots = shoots_per_sam + (1 if remaining_shoots > 0 else 0)
            if remaining_shoots > 0:
                remaining_shoots -= 1
            list_to_shoot.append((sam, shoots))

    return list_to_shoot


# create a model to store the rules
model = DevelopmentEngine("tomato")
model.add_rule(stem_rule)
model.add_rule(leaf_rule)
model.add_rule(plant_rule)
model.add_rule(rachid_rule)
model.add_shooting_rule(shoots_if_rule)


env_reads = { 
                Stem: ["temp"],
                Leaf: ["temp","light"],
                SAM: ["temp"]
            }
model.env_reads = env_reads


# create a simulation
dt = 1
max_time = 20
sim = Simulation(solver=ni)

env = Environment(
    sky=Sky(light_intensity=10),
    air=Air(temperature=20, humidity=0.5),
    soil=Soil(moisture=0.5)
)
temperatures = [15, 20, 25]

import matplotlib.pyplot as plt
import pickle

# Create subplots with appropriate projections
fig = plt.figure(figsize=(10, 10))
axes = []

for i,temp in enumerate(temperatures):
    plant.reset()
    clock.elapsed_time = 0
    env.air.temperature = temp
    sim.history.reset()

    sim.run(max_t=max_time,delta_t=dt, dev_eng=model, plant=plant, clock=clock, env = env)
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    plant.plot(ax=ax)

    # save the plant and the env
    with open(f"botanicode/training_files/simple/data_{temp}.pkl", "wb") as f:
        pickle.dump((env,sim.history), f)
    
plt.show()