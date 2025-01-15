import numpy as np

class Part:
    def __init__(self, state = None, shape = None):
        
        self.name = None
        
        self.state = state #this is a dataclass with the state of the part, for example the age, the conductance, etc.
        self.shape = shape #this is a dataclass with the shape of the part, for example the size of the leaf, the length of the stem, etc.
        
        self.state.age = 0
        
        self.parent = None  
        self.children = []
        self.attached_to = None
        
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
        
          
           