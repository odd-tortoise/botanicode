import numpy as np
from general_nodes import Part
from dataclasses import dataclass
from shapes import PointShape
from scipy.spatial.transform import Rotation as R

@dataclass
class SeedState:
    age : float = 0

class Seed(Part):

    def __init__(self):
        super().__init__(state=SeedState(), shape=PointShape(None))
        self.color = "darkgray"
        self.name = "Seed"

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
    