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
    direction: np.ndarray =  field(default_factory=lambda: np.array([0, 0, 1]))
    tt : float = 0.0

@dataclass
class LeafState(NodeState):
    size: float = 1.0
    petioles_size: float = 0.1
    rachid_size: float = 1
    age: int = 0
    leaflets_number: int = plant_reg.phylotaxis["leaflets_number"]
    leaf_bending_rate: float = plant_reg.phylotaxis["leaf_bending_rate"]
    outline_function: callable = plant_reg.phylotaxis["outline_function"]
    y_angle: float = plant_reg.phylotaxis["y_angle"]
    z_angle: float = 0
    tt : float = 0.0


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

# create a rule for the evolution of the stem, based on the termal time 
stem_rule = Rule(trainable = False, is_dynamic = False, no_params = 0)
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
stem_rule.set_action(stem_length_rule)


leaf_rule = Rule( trainable = False, is_dynamic = False, no_params = 0)
def leaf_size_rule(plant : Plant, params : np.array):
    nodes = [node for node in plant.structure.G.nodes() if isinstance(node, Leaf)]
    for node in nodes:
        #logistic growth
        node.state.tt = node.state.tt + max(0, node.state.temp - 10) # thermal time is the sum of the daily thermal time
   
        if node.id + 1 >= 8:
            node.state.size = 21.69 + (-4.09- 21.69)/(1 + np.exp( (node.id+1 - 20.69)/48.84))
            node.state.petioles_size =21.30 /(1+7.387*np.exp(-0.021*node.state.tt))
        else:
            node.state.size = 19.91 + (2.8 - 19.91)/(1 + np.exp( (node.id +1 - 58.95)/33.87))
            node.state.petioles_size =21.30 /(1+7.387*np.exp(-0.021*node.state.tt))
        
        node.state.rachid_size = node.state.petioles_size
leaf_rule.set_action(leaf_size_rule)



# create a rule for the plant
plant_rule = Rule( trainable = False, is_dynamic = False, no_params = 0)

def plant_rule_action(plant : Plant, params : np.array):
    plant.state.internodes_no = len([node for node in plant.structure.G.nodes() if isinstance(node, Stem)])
    apical_temp = np.mean([node.state.temp for node in plant.structure.G.nodes() if isinstance(node, SAM)])
    plant.state.tt = plant.state.tt + max(0, apical_temp - 10)
    plant.state.expected_internodes_no = 36.61/(1+ 43.05* np.exp(-0.008*plant.state.tt))
plant_rule.set_action(plant_rule_action)


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
model.add_rule(leaf_rule)
model.add_rule(plant_rule)
model.add_shooting_rule(shoots_if_rule)

env_reads = { 
                Stem: ["temp"],
                Leaf: ["temp","light"],
                SAM: ["temp"]
            }
model.env_reads = env_reads


# create a simulation

dt = 1
max_time = 40
sim = Simulation(plant, model, clock, env, ni)

ni.set_dt(dt)
sim.run(max_t=max_time,delta_t=dt)

plant.plot()
import matplotlib.pyplot as plt
plt.show()


def aftermath(plant, file, node_type):

        node_data = plant.history.data[node_type]
        plant_data = plant.history.data["Plant"]
        # è un dict con chiavi il nome del nodo 
        # dentro ogni dict c'è una lista con timestamp + dati {}

        #voglio fare i plot vs thermal time
        # in questo caso è facile perché i thermal time sono tutti uguali solo scalati
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,1)
        thermal_time = np.array([ node_data["S0"][i][1]["tt"] for i in range(len(node_data["S0"]))])
        pn = np.array([ plant_data["Plant"][i][1]["internodes_no"] for i in range(len(plant_data["Plant"]))])
        exp_pn = np.array([ plant_data["Plant"][i][1]["expected_internodes_no"] for i in range(len(plant_data["Plant"]))])
        ax[0].plot(thermal_time,pn, "k", label="pn")
        ax[0].plot(thermal_time,exp_pn, "r", label="expected_pn")
        ax[0].legend()
        ax[0].grid()
        
        all_length = []
        for node_name, data in node_data.items():
            # data è una lista di dict
            # ogni dict ha timestamp e dati
            # devo fare un plot per ogni nodo
            data_len = []
            for d in data:
                data_len.append(d[1]["length"])

            all_length.append([data_len, node_name])

    
        # nodes to plot
        nodes_to_plot = ["S3","S4","S5","S6","S8","S10","S12","S14"]


        for data_len,names in all_length:
            
            # pick the len(el) elemts of thermal time
            if names in nodes_to_plot:
            
                ax[1].plot(thermal_time[:len(data_len)],data_len,label=names)
        ax[1].grid()
        
        ax[1].legend()
        
        plt.show()

aftermath(plant,"botanicode/single_run_files/history.txt",Stem.__name__)