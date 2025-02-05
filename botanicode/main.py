import numpy as np
from typing import List
from dataclasses import dataclass, field
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



result_folder = "results_main"

env_setting_file = "botanicode/single_run_files/env_setting.json"

# create a clock
clock = SimClock(photo_period=(8,18),step="hour")

#create environment
env = Environment().set_env(env_setting_file)

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
    age: int = 0
    water: float = 0.0
    direction: np.ndarray =  field(default_factory=lambda: np.array([0, 0, 1]))
    tt: float = 0.0

@dataclass
class LeafState(NodeState):
    size: float = 1.0
    petioles_size: float = 0.1
    rachid_size: float = 1
    age: int = 0
    water : float = 0.0
    leaflets_number: int = plant_reg.phylotaxis["leaflets_number"]
    leaf_bending_rate: float = plant_reg.phylotaxis["leaf_bending_rate"]
    outline_function: callable = plant_reg.phylotaxis["outline_function"]
    y_angle: float = plant_reg.phylotaxis["y_angle"]
    z_angle: float = 0
    tt: float = 0.0




@dataclass
class RootState(NodeState):
    length: float = 1.0
    radius: float = 0.1 
    age: int = 0
    direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))

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


# create a plant state
@dataclass
class PlantState(PlantState):
    #it also has the plant height
    internodes_no : int = 0
    expected_internodes_no : int = 0
    tt : float = 0.0
    
state = PlantState()


# create a plant
plant = Plant(reg = plant_reg, node_factory = node_factory, plant_state = state)

# create a rule for the stems
stem_rule = Rule( trainable = False, is_dynamic = False, no_params = 0)
def stem_length_rule(plant : Plant, params : np.array):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Stem)]
    for node in nodes:
        node.state.tt += max(0, node.state.temp - 10)
        #logistic growth
        if node.id + 1 >= 14:
            node.state.max_lenght = 10.33+ (4.5-10.33)/(1 + np.exp( (14 - 10.9)/1.7))
        else:
            node.state.max_lenght = 10.33+ (4.5-10.33)/(1 + np.exp( (node.id +1 - 10.9)/1.7))

        node.state.length = node.state.max_lenght /(1+5.022*np.exp(-0.062*node.state.tt))
        if node.id == 0:
            node.state.water = 10 
stem_rule.set_action(stem_length_rule)


rachid_rule = Rule( trainable = True, is_dynamic = False, no_params = 1)
def leaf_rachid_rule(plant : Plant, params : np.array):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        #logistic growth
        node.state.rachid_size = params[0]*node.state.size
rachid_rule.set_action(leaf_rachid_rule)

# create a dynamic rule for the leaves
leaf_rule = Rule(trainable = True, is_dynamic = True, no_params = 1)
def leaf_size_rule(t,y, plant: Plant, params : np.array):
    # deve essere una funzione che restituisce il rhs dell'equazione differenziale 
    # perché viene data in pasto a ODEINT che la risolve -> t,y, args
    rhs = []
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        node.state.s_max = 4*(node.parent.id+1)
        rhs.append(params[0]/node.state.s_max * node.state.size * ( node.state.s_max -  node.state.size))

    return np.array(rhs)
leaf_rule.set_action(leaf_size_rule, "size", [Leaf])

# create a rule for the plant
plant_rule = Rule( trainable = False, is_dynamic = False, no_params = 0)

def plant_rule_action(plant : Plant, params : np.array):
    plant.state.internodes_no = len([node for node in plant.structure.G.nodes() if isinstance(node, Stem)])
    apical_temp = np.mean([node.state.temp for node in plant.structure.G.nodes() if isinstance(node, SAM)])
    plant.state.tt = plant.state.tt + max(0, apical_temp - 10)
    plant.state.expected_internodes_no = 36.61/(1+ 43.05* np.exp(-0.008*plant.state.tt))
plant_rule.set_action(plant_rule_action)

water_dynamic = Rule(trainable = False, is_dynamic = True, no_params = 0)
def water_diffusion(t, y, plant: Plant, params : np.array):
    graph = plant.structure.G

    # extract the subgraph of the plant made by the nodes of interest
    subgraph = graph.subgraph([node for node in graph.nodes() if isinstance(node, Stem) or isinstance(node, Leaf)])


    L = nx.laplacian_matrix(subgraph).todense() 

    

    rhs = - (L @ y)
    return np.array(rhs)
water_dynamic.set_action(water_diffusion, "water",[Stem,Leaf])


# create a rule for shooting
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
model = Model("tomato")
model.add_rule(stem_rule)
model.add_dynamic_rule(leaf_rule)
model.add_rule(plant_rule)
model.add_rule(rachid_rule)
model.add_dynamic_rule(water_dynamic)
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
sim = Simulation(plant, model, clock, env, ni)

ni.set_dt(dt)
sim.run(max_t=max_time,delta_t=dt)

plant.plot()
import matplotlib.pyplot as plt
plt.show()


fig, ax = plt.subplots(2,1)
plant.structure.plot_value("water",[Stem,Leaf],ax[0])
plant.structure.plot(ax[1])
plt.show()

