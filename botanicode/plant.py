from botanical_nodes import Stem, Leaf, Root, SAM, RAM
from botanical_nodes import NodeState, NodeFactory, Part
from plant_reg import PlantRegulation
from graph import Structure


import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Dict, List, Any
import copy

class Tracker:
    def __init__(self):
        self.data = {}

    def snapshot(self, timestamp, node):
        """
        Records a snapshot of the state of nodes.

        Args:
            timestamp (str): The timestamp of the snapshot.
            nodes (iterable): An iterable of nodes with `name`, `state.__dict__`, and class type.
        """
        
        node_data = copy.deepcopy(node.state.__dict__)
        node_name = node.name
        node_type = type(node).__name__
        
        if node_type not in self.data:
            self.data[node_type] = {}
        if node_name not in self.data[node_type]:
            self.data[node_type][node_name] = []

        self.data[node_type][node_name].append([timestamp, node_data])
    
    def snap_plant(self, timestamp, plant):
        plant_data = copy.deepcopy(plant.plant_state.__dict__)
        if "Plant" not in self.data:
            self.data["Plant"] = {
                "Plant": []
            }
        self.data["Plant"]["Plant"].append([timestamp, plant_data])

    def get_variable_over_time(self, node_type, node_name, variable):
        """
        Extracts timestamps and variable values for a specific node type and node name.

        Args:
            node_type (str): The type of node.
            node_name (str): The name of the node.
            variable (str): The variable to extract.

        Returns:
            tuple: A tuple of (timestamps, values).
        """
        if node_type not in self.data or node_name not in self.data[node_type]:
            print(f"Warning: Node type '{node_type}' or node name '{node_name}' not found in data.")
            return [], []

        timestamps = []
        values = []
        for snapshot in self.data[node_type][node_name]:
            timestamps.append(snapshot[0])
            values.append(snapshot[1].get(variable, None))
        return timestamps, values

    def plot(self, variable, ax=None, node_types = []):
        """
        Plots the specified variable for the given node types.

        Args:
            node_types (list of str): List of node types to include in the plot.
            variable (str): The variable to plot.

        """

        if not isinstance(node_types, list):
            node_types = [node_types]

        if ax is None:
            fig, ax = plt.figure(figsize=(10, 5))
        
        
        
        for node_type in node_types:
            if node_type in self.data:
               
                
                for node_name, history in self.data[node_type].items():

                    timestamps, values = self.get_variable_over_time(node_type, node_name, variable)
                    if timestamps and values:
                        ax.plot(timestamps, values, label=f"{node_type} - {node_name}")
                        
                        

        ax.grid()
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{variable}")
                    
        if ax is None:
            plt.tight_layout()
            plt.show()

    def save_to_file(self, path):
        """
        Saves the tracked data to a txt file.

        Args:
            path (str): The file path to save the data.
        """
        try:
            with open(path, 'w') as file:
                for node_type, values in self.data.items():
                    for node_name, history in values.items():
                        file.write(f"{node_type} - {node_name}\n")
                        for timestamp, data in history:
                            file.write(f"{timestamp}: {data}\n")
                        
            print(f"Data successfully saved to {path}")
        except Exception as e:
            print(f"Error saving data to file: {e}")

@dataclass
class PlantState:
    plant_height : float = 0
    def reset(self):
        self.__init__()

class Plant:
    def __init__(self, reg : PlantRegulation, node_factory : NodeFactory, plant_state : PlantState):

        self.growth_regulation = reg
        self.plant_state = plant_state
        self.node_factory = node_factory

        self.history = Tracker()

        # utils
        self.leaf_z_angle_offset = 0
        self.phylotaxy_data = self.growth_regulation.get_phylotaxis()
        
        self.initialize_plant()
     
    def reset(self):
        for node_type in self.node_factory.node_blueprints.keys():
            node_type.counter = 0
        self.__init__(self.growth_regulation, self.node_factory, self.plant_state)
        self.plant_state.reset()    
     
    def probe(self, env, reads,t):
        def probe_recursive(node):
            # get the rules to probe the env
            if type(node) not in reads:
                return
            
            vars_to_read = reads[type(node)]
            # this is a list of things to read from the env

            for env_var in vars_to_read:
                value = env.measure(node.shape.position,env_var,t)
                setattr(node.state, env_var, value)

        self.structure.traverse(action=probe_recursive)

    def snapshot(self, timestamp):
        for node in self.structure.G.nodes():
            self.history.snapshot(timestamp, node)
 
           
        self.history.snap_plant(timestamp, self)

    def make_leaves(self):
        if self.phylotaxy_data["leaf_arrangement"] == "alternate":
            self.leaf_z_angle_offset+=self.phylotaxy_data["angle"]
            self.leaf_z_angle_offset = self.leaf_z_angle_offset % (2 * np.pi)
        elif self.phylotaxy_data["leaf_arrangement"] == "decussate":
            if self.leaf_z_angle_offset == 0:
                self.leaf_z_angle_offset = np.pi/2
            else:
                self.leaf_z_angle_offset = 0
        elif self.phylotaxy_data["leaf_arrangement"] == "opposite":
            self.leaf_z_angle_offset = 0
        else:
            raise ValueError("Invalid leaf arrangement.")
        
        leaves = []
        for i in range(self.phylotaxy_data["leaves_number"]):
            z_angle = 2 * np.pi * i / self.phylotaxy_data["leaves_number"] + self.leaf_z_angle_offset
            shape_variations = {"z_angle": z_angle,
                                "id": i,
                                "leaflets_number": self.phylotaxy_data["leaflets_number"],
                                "y_angle": self.phylotaxy_data["y_angle"],
                                "outline_function": self.phylotaxy_data["outline_function"],
                                "leaf_bending_rate": self.phylotaxy_data["leaf_bending_rate"],}

            leaves.append(self.node_factory.create(Leaf, shape_variations=shape_variations))

        return leaves

    def initialize_plant(self):
        # Create the initial plant structure
        # assume initial age of the stem to be 1 

        # qui mettere le regole per inizializzare la pianta in maniera diversa come "variations"
        stem = self.node_factory.create(Stem)
        sam = self.node_factory.create(SAM)
        ram = self.node_factory.create(RAM)
        root = self.node_factory.create(Root)
        leaves = self.make_leaves()

        self.structure = Structure(seed=stem)

        self.structure.add_node(stem, sam, "tip")
        sam.name =  sam.name[:2] + str(stem.id)
        self.structure.add_node(self.structure.seed, root, "base")
        self.structure.add_node(root, ram, "tip")

        for leaf in leaves:
            self.structure.add_node(stem, leaf, "tip")
            leaf.name = leaf.name[0] + str(stem.id) + leaf.name[1:]



        self.update()

    def apply_rule(self,rule):
        rule.action(self, rule.params)

    def apply_dynamic_rule(self,rule,t,solver):
        y = []
        for node_type in rule.types:
            nodes = [node for node in self.structure.G.nodes() if isinstance(node, node_type)]
            y.append([ getattr(node.state, rule.var) for node in nodes])

        # concatenate the lists
        y = [item for sublist in y for item in sublist]
        y = np.array(y)
    
        new_y = solver.integrate(
            rhs_function = rule.action,
            plant = self,
            params = rule.params,
            t = t,
            y = y
            )
        
        
        current = 0
        for i, node_type in enumerate(rule.types):
            nodes = [node for node in self.structure.G.nodes() if isinstance(node, node_type)]
            for node in nodes:
                setattr(node.state, rule.var, new_y[current])
                current += 1

    def grow(self,dt,env,model,t):
        
        # apply derived rules changes

        self.update() #update the shapes and the positions
        self.age_nodes(dt)
        list_to_shoot = model.shooting_rule(self)
        
        for node in list_to_shoot:
            self.shoot(node)

        # probe the environment for the new nodes

        self.probe(env,model.env_reads,t)

        #self.model.plant_rules(self)
        
    def age_nodes(self, dt):
        def age_node(node):
            node.state.age += dt

        self.structure.traverse(action=age_node)
        
    def update(self):
        self.update_shapes()
        self.update_positions_and_realpoints()
        self.plant_state.plant_height = self.compute_plant_height()

    def update_shapes(self):
        def update_shape(node):
            node.shape.generate_points(node.state)

        self.structure.traverse(action=update_shape)

    def update_positions_and_realpoints(self):
        def update_position(node):
            node.update_position()
            
            node.shape.generate_real_points()
            
            
        self.structure.traverse(action=update_position)

    def compute_plant_height(self):
        max = 0
        for node in self.structure.G.nodes():
            if node.shape.position[2] > max:
                max = node.shape.position[2]
        return max
  
    def log(self, logger):        
        message = str(self.plant_state)
        logger.warning(message) 
  
    def shoot(self, node): # this is basically a FACTORY!! 
        # idea for the future: use a factory pattern in the C++ implementation
        
        stem = self.node_factory.create(Stem)
        sam = self.node_factory.create(SAM)
        sam.name = sam.name[2] + str(stem.id) + sam.name[2:]

        leaves = self.make_leaves()
    
        if isinstance(node, SAM):
            self.structure.add_node(node.parent, stem, "tip")
            self.structure.remove_node(node)
        else:
            self.structure.add_node(node, stem, "tip")
        
        self.structure.add_node(stem, sam, "tip")
        for leaf in leaves:
            self.structure.add_node(stem, leaf, "tip")
            leaf.name = leaf.name[0] + str(stem.id) + leaf.name[1:]

        self.update()
    
    def plot(self, ax=None):
         # Plotting in 3D
    
        show = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            show = True
        
        def plot_node(node):
            if isinstance(node, Leaf):
                
                leaf_skeletons = node.shape.real_leaves_points
                rachid_skeleton = node.shape.real_rachid_points
                # plot the rachid
                rachid_skeleton = np.array(rachid_skeleton)
                if rachid_skeleton.size > 0:
                    ax.plot(rachid_skeleton[:, 0], rachid_skeleton[:, 1], rachid_skeleton[:, 2],
                            color=node.rachid_color, label='Rachid Skeleton', linewidth=2)
                    
                # plot the leaves 
                for leaf in leaf_skeletons:
                    leaf = np.array(leaf)
                    if leaf.size > 0:
                        ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2],
                                color=node.color, label='Leaf Skeleton', linewidth=2, marker='o',markersize=2)
                        ax.plot([leaf[0, 0], leaf[-1, 0]], [leaf[0, 1], leaf[-1, 1]], [leaf[0, 2], leaf[-1, 2]], color=node.color, linewidth=2)
                
            else:
                skeleton = node.shape.real_points
                skeleton = np.array(skeleton)
                if skeleton.size > 0:  # Check if any stem nodes exist
                    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
                            color=node.color, linewidth=2)
                    
                    
        # Recursively traverse each child node  
        self.structure.traverse(action=plot_node)    

        # Setting the plot labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
                
        size = int(self.plant_state.plant_height) + 1
        size = size if size%2== 0 else size + 2 - size % 2
        
        ax.set_xlim([-size//2, size//2])
        ax.set_ylim([-size//2, size//2])
        ax.set_zlim([0, size ])

        if show:
            # Show plot
            
            plt.show()