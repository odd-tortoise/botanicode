from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from shapes import NodeShape

@dataclass
class NodeState:
    """Base class for all node states."""
    age : float = 0
    rank : float = 0

class Part:
    """Base class for all plant parts."""
    def __init__(self, state : NodeState, shape : NodeShape):
        """
        Initialize a Part instance.

        Args:
            stat : The state of the part, e.g., age, conductance, etc.
            shape : The shape of the part.
        """
        self.state = state
        self.shape = shape 

        self.parent = None  
        self.children = []
        self.attached_to = None # plug point of the parent to which the part is attached
        
        self.name = None
        self.color = "black"
    
    def update_position(self) -> None:
        """Update the position of the part based on its parent's position."""
        if self.parent is None:
            self.shape.position = np.array([0,0,0])
        else:
            self.shape.position = self.parent.shape.compute_plug_points()[self.attached_to]
        
class Stem(Part):
    # TODO: pensare un modo migliore per gestire gli id e i rank, credo questo possa dare problemi con le branch
    counter = 0

    def __init__(self, shape , state):
        super().__init__( state = state, shape = shape)

        self.name = f"S{Stem.counter}"
        self.id = Stem.counter 
        self.state.rank = self.id
        Stem.counter += 1
        
        self.color = "green"

    
class Root(Part):
    counter = 0
    def __init__(self,shape , state ):
        super().__init__(state = state, shape = shape)
        self.name = f"R{Root.counter}"
        self.id = Root.counter
        Root.counter += 1
        
        self.color = "brown"
    
class SAM(Part): 
    def __init__(self, shape, state):
        super().__init__( state = state, shape = shape)
   
        self.name = f"SAM"
        
        self.color = "lightblue"

class RAM(Part):
    def __init__(self, shape , state ):
        super().__init__(state = state, shape = shape)
   
       
        self.name = f"RAM"
        
        self.color = "red"
    
class Leaf(Part):
    def __init__(self, shape , state, id = 0):
        super().__init__(state = state, shape = shape)
        self.id = 0 # in this case is the position at the same internode of which the leaf is attached to

        self.state.rank = 0 
        self.name = "L"
        self.color = "orange"
        self.rachid_color = "purple"
    

class NodeFactory:
    def __init__(self):
        self.node_blueprints : Dict[str, Dict[str, Any]] = {}

    def add_blueprint(self, 
                      nodetype : Part,
                      state : NodeState,
                      shape : NodeShape) -> None:
        """
        Add a blueprint for a node type.

        Args:
            nodetype (Part): The type of the node.
            state (NodeState): The template state for nodes of this type.
            shape (NodeShape): The template shape for nodes of this type.

        Raises:
            ValueError: If a blueprint with the same name already exists or if there is a shape variable that is not in the state.
        """
        if nodetype in self.node_blueprints:
            raise ValueError(f"A blueprint for the type '{nodetype.__name__}' already exists.")
        
        # check that the state HAS all the variables needed by the shape
        for key in shape.required_state_variables:
            if key not in state.__annotations__:
                raise ValueError(f"Invalid state variable '{key}' for node type '{nodetype.__name__}'. The shape requires this variable.")
            
        self.node_blueprints[nodetype] = {
            "state": state,
            "shape": shape
        }

    def create(self, nodetype : Part) -> Part:
        """
        Create a node of the given type.

        Args:
            nodetype (Part): The type of the node to create.

        Returns:
            Part: The created node.

        Raises:
            ValueError: If no blueprint is found for the given node type.
        """

        blueprint = self.node_blueprints.get(nodetype)
        if not blueprint:
            raise ValueError(f"No blueprint found for node type '{nodetype.__name__}'.")
        
        state = blueprint["state"]()
        shape = blueprint["shape"](state)
        node = nodetype(state=state, shape=shape)
        return node
    

    # TODO: sempre una questione da risovlere, i dictionary hanno come keys le classi, forse si può fare in modo che siano le stringhe dei nomi delle classi
    # con il .__name__, in questo modo è più facile da gestire per i file di configurazione

    # TODO: cambiare il meccanismo di creazione, non è detto che un utente voglia creare tutti gli organi disponibili 
    # (es. se voglio creare solo foglie e steli, non mi serve creare anche le radici)

