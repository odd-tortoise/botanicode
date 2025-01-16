from botanical_nodes import Stem, Leaf, Root, SAM, RAM, Seed
from graph import Structure
import numpy as np
from light import Sky
import matplotlib.pyplot as plt
from plant_reg import PlantRegulation
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Any
import json
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



    def plotold(self, ax=None, value = "node_data.position", node_types = []):

        if not isinstance(node_types, list):
            node_types = [node_types]

        if ax is None:
            fig, ax = plt.figure(figsize=(10, 5))

        attrs = value.split('.')
        

        
        for node_type in node_types:
            if node_type not in self.data:
                raise ValueError(f"Node type {node_type.__name__} not found.")
            if len(self.data[node_type]) == 0:
                raise ValueError(f"No data found for the node type {node_type.__name__}")
            
            node_data = self.data[node_type] # qui dizioanrio con chiavi i nomi dei node_types

            markers = ['o', "x"]
            standard_marker = ''
            marker_index = 0
            plotted_lines = []
           
            for key, value in node_data.items():
                # key is the name of the node
                # value is the history of the node
                time_steps = list(value.keys())
                extracted_values = []
                for time, data in value.items():    
                    # data is the dictionary with the node data per each timestamp 
                    
                    for attr in attrs:
                        if isinstance(data, dict):
                            data = data.get(attr, None)
                        elif hasattr(data, attr):
                            data = getattr(data, attr, None)
                        else:
                            raise ValueError(f"Attribute {attr} not found in node {key}")
                    extracted_values.append(data)
                
                # chck for using differen markers for close lines 
                use_marker = False
                for line in plotted_lines:
                    if len(line) != len(extracted_values):
                        continue
                    else:
                        distance = np.linalg.norm(np.array(extracted_values) - np.array(line))
                        if distance < 0.05:  # Threshold for considering lines as close
                            use_marker = True
                            marker_index = (marker_index + 1) % len(markers)
                            break

                if use_marker:
                    ax.plot(time_steps, extracted_values,label=f"{key}",marker=markers[marker_index])
                else:
                    ax.plot(time_steps, extracted_values,label=f"{key}", marker=standard_marker)
                
                plotted_lines.append(extracted_values)

            
                
        ax.grid()
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(".".join(attrs))
                    
        if ax is None:
            plt.tight_layout()
            plt.show()

@dataclass
class PlantState:
    plant_height: float
    age: float

class Plant:
    def __init__(self, reg, model):
        self.growth_regulation = reg
        self.clock = None
        self.model = model

        # check that model and growth reg have all the need params
        #... to be done

        #unpack the model
        self.StemStuff = model.nodes_blueprint[Stem]
        self.LeafStuff = model.nodes_blueprint[Leaf]
        self.RootStuff = model.nodes_blueprint[Root]
        self.SAMStuff = model.nodes_blueprint[SAM]
        self.RAMStuff = model.nodes_blueprint[RAM]

        seed = Seed()
        self.structure = Structure(seed=seed)

        self.plant_state = PlantState(plant_height=0, age=0)

        self.history = Tracker()

        # utils
        self.leaf_z_angle_offset = 0
        self.phylotaxy_data = self.growth_regulation.get_phylotaxis()
        
        self.initialize_plant()
     
    def probe(self, env):
        def probe_recursive(node):
            # get the rules to probe the env
            if type(node) not in self.model.nodes_blueprint:
                return
            
            vars_to_read = self.model.nodes_blueprint[type(node)]["rules"].env_reading 
            # this is a list of things to read from the env

            for env_var in vars_to_read:
                value = env.measure(node.shape.position,env_var)
                setattr(node.state, env_var, value)

        self.structure.traverse(action=probe_recursive)

    def set_clock(self, clock):
        self.clock = clock

    def snapshot(self, timestamp):
        for node in self.structure.G.nodes():
            self.history.snapshot(timestamp, node)
 
            #self.history.track(node_data,node_name, node_type, timestamp)

    def make_leaves(self):
        leaves_states = [self.LeafStuff["state"](**self.growth_regulation.get_leaf_data()["initial_state"]) for i in range(self.phylotaxy_data["leaves_number"])]
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
        for i,state in enumerate(leaves_states):
            z_angle = 2 * np.pi * i / self.phylotaxy_data["leaves_number"] + self.leaf_z_angle_offset
            shape = self.LeafStuff["shape"](state, self.phylotaxy_data, z_angle)
            leaves.append(Leaf(state=state, shape=shape, id = i))

        return leaves

    def initialize_plant(self):
        # Create the initial plant structure
        # assume initial age of the stem to be 1 

        stem_state = self.StemStuff["state"](**self.growth_regulation.get_stem_data()["initial_state"])
        stem_shape = self.StemStuff["shape"](stem_state, direction = np.array([0, 0, 1]))
        stem = Stem(state=stem_state, shape=stem_shape)


        sam_state = self.SAMStuff["state"]()
        sam_shape = self.SAMStuff["shape"](sam_state)
        sam = SAM(state=sam_state, shape=sam_shape)

        ram_state = self.RAMStuff["state"]()
        ram_shape = self.RAMStuff["shape"](ram_state)
        ram = RAM(state=ram_state, shape=ram_shape)

        root_state = self.RootStuff["state"](**self.growth_regulation.get_root_data()["initial_state"])
        root_shape = self.RootStuff["shape"](root_state,direction = np.array([0, 0, -1]))
        root = Root(state=root_state, shape=root_shape)

        
        leaves = self.make_leaves()
        

        self.structure.add_node(self.structure.seed, stem, "base")
        self.structure.add_node(stem, sam, "tip")
        sam.name =  sam.name[2] + str(stem.id) + sam.name[2:]
        self.structure.add_node(self.structure.seed, root, "base")
        self.structure.add_node(root, ram, "tip")

        for leaf in leaves:
            self.structure.add_node(stem, leaf, "tip")
            leaf.name = leaf.name[0] + str(stem.id) + leaf.name[1:]

        self.update()

    def get_dynamic_info(self):
        ret = {}
        for node_type, values in self.model.nodes_blueprint.items():
            rules = values["rules"]
            if hasattr(rules, "dynamics"):
                for key, rhs in rules.dynamics.items():
                    ret[key+"_"+node_type.__name__] = {}
                    ret[key+"_"+node_type.__name__]["value"]= self.structure.get_nodes_attribute(key, node_type)[0]
                    ret[key+"_"+node_type.__name__]["node_obj"] = self.structure.get_nodes_attribute(key, node_type)[1]
                    ret[key+"_"+node_type.__name__]["rhs"] = rhs

        
        return ret
                    
    def grow(self, ret, dt):
        # apply dynamic rules changes
        for key, values in ret.items():
            var = key.split("_")[0]
            node_type = key.split("_")[1]
            
            if "new_value" in values:
                new_value = values["new_value"]
                
                for node,val in zip(values["node_obj"], new_value):
                    setattr(node.state, var, val)

        # apply derived rules changes

        for node_type, values in self.model.nodes_blueprint.items():
            rules = values["rules"]
            if hasattr(rules, "derived"):
                rule = rules.derived
                for node in self.structure.G.nodes():
                    if hasattr(node.state, key):
                        rule(node.state)

        self.update()

        self.age_nodes(dt)


        list_to_shoot = [ node for node in self.structure.G.nodes() if self.model.shooting_rule(node)]
        for node in list_to_shoot:
            self.shoot(node)

    def age_nodes(self, dt):
        def age_node(node):
            node.state.age += dt

        self.structure.traverse(action=age_node)
        
    def update(self):
        self.update_shapes()
        self.update_positions_and_realpoints()
        self.compute_plant_height()

    def update_shapes(self):
        def update_shape(node):
            node.shape.generate_points()

        self.structure.traverse(action=update_shape)

    def update_positions_and_realpoints(self):
        def update_position(node):
            node.update_position()
            
            node.shape.generate_real_points()
            
            
        self.structure.traverse(action=update_position)

    def compute_plant_height(self):
        def compute_plant_height_recursive(node):
            if node.shape.position[2] > self.plant_state.plant_height:
                self.plant_state.plant_height = node.shape.position[2]

        self.structure.traverse(action=compute_plant_height_recursive)
        
    def log(self, logger):        
        message = str(self.plant_state)
        logger.warning(message) 
  
    def shoot(self, node): # this is basically a FACTORY!! 
        # idea for the future: use a factory pattern in the C++ implementation
        
        stem_state = self.StemStuff["state"](**self.growth_regulation.get_stem_data()["generation"])
        stem_shape = self.StemStuff["shape"](stem_state, direction = np.array([0, 0, 1]))
        stem = Stem(state=stem_state, shape=stem_shape)


        sam_state = self.SAMStuff["state"]()
        sam_shape = self.SAMStuff["shape"](sam_state)
        sam = SAM(state=sam_state, shape=sam_shape)
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
                            color=node.rachid_color, label='Rachid Skeleton', linewidth=2, marker='o')
                    
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
                            color=node.color, linewidth=2, marker='o')
                    
                    
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