from typing import Dict, Any
import numpy as np
from dataclasses import dataclass
from shapes import NodeShape
import json

class Part:
    def __init__(self, state = None, shape = None):
        self.state = state #this is a dataclass with the state of the part, for example the age, the conductance, etc.
        self.shape = shape #this is a class with the shape of the part
        
        self.parent = None  
        self.children = []
        self.attached_to = None # plug point of the parent to which the part is attached
        
        self.name = None
        self.color = "black"
           
    
    def get_data(self):
        data = self.state.__dict__
        data["position"] = self.shape.position
        data["orientation"] = self.shape.orientation
        data["color"] = self.color
        data["name"] = self.name
        data["parent"] = self.parent
        data["children"] = self.children
        data["attached_to"] = self.attached_to
        return data
    
    def update_position(self):
        if self.parent is None:
            self.shape.position = np.array([0,0,0])
        else:
            self.shape.position = self.parent.shape.compute_plug_points()[self.attached_to]
        

class Stem(Part):
    counter = 0

    def __init__(self, shape = None, state = None):
        super().__init__( state = state, shape = shape)
        

        self.name = f"S{Stem.counter}"
        self.id = Stem.counter
        Stem.counter += 1
        
        self.color = "green"

    
    def is_apical(self):
        # check if there is a SAM object into the relational children
        for child in self.children:
            if isinstance(child, SAM):
                return child
        return False
    
class Root(Part):
    counter = 0
    def __init__(self,shape = None, state = None):
        super().__init__(state = state, shape = shape)
        self.name = f"R{Root.counter}"
        self.id = Root.counter
        Root.counter += 1
        
        self.color = "brown"
    
    def is_apical(self):
        # check if there is a SAM object into the relational children
        for child in self.children:
            if isinstance(child, RAM):
                return child
        return False
    
class SAM(Part): 
    def __init__(self, shape = None, state = None):
        super().__init__( state = state, shape = shape)
   
        if self.parent is not None:
            self.name = f"SAM{self.parent.id}"
        else:
            self.name = f"SAM"
        
        self.color = "lightblue"

class RAM(Part):
    def __init__(self, shape = None, state = None):
        super().__init__(state = state, shape = shape)
   
        if self.parent is not None:
            self.name = f"RAM{self.parent.id}"
        else:
            self.name = f"RAM"
        
        self.color = "red"
    
class Leaf(Part):
    def __init__(self, shape = None, state = None, id = 0):
        super().__init__(state = state, shape = shape)
        self.id = id
        self.parent_rank = 0
        if self.parent is not None:
            self.name = f"L{self.parent.id}{id}"
            self.parent_rank = self.parent.id
        else:
            self.name = f"L{id}"

        self.color = "orange"
        self.rachid_color = "purple"
    

class NodeState:
    """Node state"""
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, value)

class NodeFactory:
    def __init__(self):
        self.node_blueprints : Dict[str, Dict[str, Any]] = {}

    def add_blueprint(self, 
                      nodetype : Part,
                      state : NodeState,
                      shape : NodeShape,
                      initial_state_values : dict = {},
                      initial_shape_info : dict = {}) -> None:
        """
        Add a blueprint for a node type.
        Args:
            state (NodeState): The template state for nodes of this type.
            shape (NodeShape): The template shape for nodes of this type.
            initial_values (dict): The initial values for the state variables.

        Raises:
            ValueError: If a blueprint with the same name already exists.
            if there is a shape variable that is not in the state.
        """
        if nodetype in self.node_blueprints:
            raise ValueError(f"A blueprint for the type '{nodetype.__name__}' already exists.")

        self.node_blueprints[nodetype] = {
            "state": state,
            "shape": shape,
            "initial_state_values": initial_state_values,
            "initial_shape_info": initial_shape_info
        }

        # check that inside the state shape variables there are all the variables needed by the shape
        # TODO check that the shape variables are in the state

    def create(self, nodetype : Part, initial_values_variation : dict = {}, shape_variations : dict = {}) -> Part:
        """Create a node of the given type."""
        blueprint = self.node_blueprints.get(nodetype)
        if not blueprint:
            raise ValueError(f"No blueprint found for node type '{nodetype.__name__}'.")
        
        state_params = blueprint["initial_state_values"].copy()
        for key, value in initial_values_variation.items():
            if key not in state_params:
                raise ValueError(f"Invalid state variable '{key}' for node type '{nodetype.__name__}'. The variations can act only on the initial state values.")
            state_params[key] = value
        
        shape_params = blueprint["initial_shape_info"].copy()
        for key, value in shape_variations.items():
            shape_params[key] = value
        
        state = blueprint["state"](**state_params)
        shape = blueprint["shape"](state, shape_params)
        node = nodetype(state=state, shape=shape)
        return node
    
    def read_blueprint_file(self, filename : str) -> None:
        
        with open(filename, 'r') as file:
            data = json.load(file)

        for nodetype, blueprint in data.items():
            if nodetype not in [k.__name__ for k in self.node_blueprints.keys()]:
                raise ValueError(f"Blueprint for node type '{nodetype}' not found.")
            
            for k in self.node_blueprints.keys():
                if k.__name__ == nodetype:
                    nodeclass = k
                    break

            self.node_blueprints[nodeclass]["initial_state_values"]  = blueprint["initial_state_values"]
            self.node_blueprints[nodeclass]["initial_shape_info"] = blueprint["initial_shape_info"]
           
            for key, value in self.node_blueprints[nodeclass]["initial_state_values"].items():
                if isinstance(value, list):
                    self.node_blueprints[nodeclass]["initial_state_values"][key] = np.array(value)
                else:
                    self.node_blueprints[nodeclass]["initial_state_values"][key] = value

            for key, value in self.node_blueprints[nodeclass]["initial_shape_info"].items():
                if isinstance(value, list):
                    self.node_blueprints[nodeclass]["initial_shape_info"][key] = np.array(value)
                else:
                    self.node_blueprints[nodeclass]["initial_shape_info"][key] = value