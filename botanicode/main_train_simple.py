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

from development_engine import DevelopmentEngine, StaticRule, DynamicRule
from utils import NumericalIntegrator, EvolutionaryOptimizer

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
        node.state.length = (10+params[0]*node.state.temp) /(1+params[1]*np.exp(-0.062*node.state.age))

stem_rule = StaticRule(action=stem_length_rule, trainable= True, no_params=2)
stem_rule.set_bounds([(0, 10), (0, 10)])


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
        rhs.append((params[0]*node.state.temp)/node.state.s_max * node.state.size * ( node.state.s_max -  node.state.size))

    return np.array(rhs)
leaf_rule = DynamicRule(action=leaf_size_rule, var = "size", types = [Leaf], trainable = True, no_params = 1)
leaf_rule.set_bounds([(0.02, 0.06)])
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



def loss(dict_obs: dict, dict_exp: dict):
    # Retrieve the history dictionaries.
    # Get the history dictionaries for Stem nodes.
    stem_obs = dict_obs["Nodes"].get(Stem, {})
    stem_exp = dict_exp["Nodes"].get(Stem, {})

    total_loss = 0.0
    # Compute the union of all node names (keys) for Stem nodes.
    node_names = set(stem_obs.keys()).union(set(stem_exp.keys()))

    # For each node in the union, compare their snapshot histories.
    for node_name in node_names:
        snapshots_obs = stem_obs.get(node_name, [])
        snapshots_exp = stem_exp.get(node_name, [])
        
        # Determine the number of snapshots to compare.
        n_snapshots = max(len(snapshots_obs), len(snapshots_exp))

        node_loss = 0.0
        
        # For each snapshot index, extract the length. If a snapshot is missing, default to 0.
        for i in range(n_snapshots):
            if i < len(snapshots_obs):
                # Each snapshot is assumed to be [timestamp, node_data]
                obs_length = snapshots_obs[i][1].get("length", 0)
            else:
                obs_length = 0

            if i < len(snapshots_exp):
                exp_length = snapshots_exp[i][1].get("length", 0)
            else:
                exp_length = 0

            node_loss += (obs_length - exp_length) ** 2

        total_loss += node_loss/n_snapshots

    
    return total_loss


# create a model to store the rules
model = DevelopmentEngine("tomato")
model.add_rule(stem_rule)
model.add_rule(leaf_rule)
model.add_rule(plant_rule)
model.add_rule(rachid_rule)
model.add_shooting_rule(shoots_if_rule)
model.add_loss_function(loss)

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

# load the dataset, read it from the folder
dataset = []
import pickle
import os 

data_folder = "botanicode/training_files/simple/"

for file in os.listdir(data_folder):
    if file.endswith(".pkl") and "data" in file:
        data = pickle.load(open(data_folder+file, "rb"))
        dataset.append([data[0], data[1].data]) #data[0] is the env, data[1] is the history, data[1].data is the dictionary of the history


optimizer = EvolutionaryOptimizer(max_epochs=1, pop_size=1, mutation_scale=0.1, loss_threshold=1e-3)

best_params, opt_info = sim.tune(plant, clock, dataset, max_t=20, delta_t=1, batch_size=3, dev_eng=model,optimizer=optimizer, folder="botanicode/training_files/simple/results/")


print("Best parameters found:", best_params)


# run a simulation with the best parameters

model.set_trainable_params(best_params)


sim.inspect_tuning(
    plant=plant,
    clock=clock,
    dataset_folder=data_folder,
    max_t=20,
    delta_t=1,
    dev_eng=model,
    node_type=Stem,
    node_names=["S0","S1","S2"],
    var = "length",
    folder="botanicode/training_files/simple/results/",
    name="stem_length"
)

sim.inspect_tuning(
    plant=plant,
    clock=clock,
    dataset_folder=data_folder,
    max_t=20,
    delta_t=1,
    dev_eng=model,
    node_type=Leaf,
    node_names=["L00","L10","L20"],
    var = "size",
    folder="botanicode/training_files/simple/results/",
    name="leaf_size"
)