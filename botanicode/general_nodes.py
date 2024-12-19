import numpy as np
from dataclasses import dataclass, asdict
from typing import Union


@dataclass
class DeviceShape:
    size: float

@dataclass
class StructuralShape:
    lenght: float
    radius: float
    direction: np.ndarray


class Part:
    def __init__(self, position = np.array([0,0,0]), age = 0):
        
        # position is the position of the leaves in the stem
        # position is the starting point of the leaf for the leaves
        # is always zero for the root
        # is the position of the parent + 0.1 for the stem
        self.name = None
        
        # this are the things that need to be consistent within the tree structure
        self.position = position 
        self.age = age


        self.shape = None #this is a dataclass with the dimensions of the part, 
        # for example for a leaf we have the length and the width
        # for stems we have the length and the radius and direction
        # basically its everything that is needed to "draw" the part
        
        self.parent = None  
        
        self.points = []
        self.real_points = []
        
        self.color = "black"
        
        
    def grow(self, dt, new_shape: Union[StructuralShape, DeviceShape] = None):
        """
        Grow the part. This method should be implemented in the subclasses.

        It modifies the part's properties, such as the length of the stem or the size of the leaf.

        Acts on the dimesion paramter and age, conductance, etc.

        It's kinda of an update method.
        """
        pass

    def generate_points(self, n_points = 10):
        """
        Generate the points of the part. This method should be implemented in the subclasses.
        The points are at the origin. It ueses the shape parameter to generate the points.
        """
        pass

    def get_real_points(self):

        return self.real_points

    def compute_real_points(self, offset = np.array([0, 0, 0])):
        """
        Get the real points in the 3d space. 

        The real points are the points of the part translated to the correct position in the 3d space.
        """
        self.real_points = [point + offset for point in self.points]
        
    
    def get_data(self):
        node_data = {
            "name": self.name,
            "position": self.position.tolist(),
            "age": self.age,
            "color": self.color,
        }
        data = {
            "node_data": node_data,
        }
        return data
    
    def compute_conductance(self):
        return 1
    
    def update_position(self):
        pass
        

class DevicePart(Part):

    def __init__(self, position = np.array([0,0,0]), age = 0, shape : DeviceShape = None):
        # shape must be a dataclass with the size of the device
        if shape is None or not isinstance(shape, DeviceShape):
            raise ValueError("shape must be a DeviceShape dataclass")
        super().__init__(position, age)

        self.shape = shape
        self.is_generator = False
        self.env_data = {}


    def probe(self): #from env
        pass

    def emit(self): # to env
        pass


    def get_data(self):
        data = super().get_data()
        device_data = {
            "shape": asdict(self.shape),
            "is_generator": self.is_generator,
        }
        data["device_data"] = device_data
        data["env_data"] = self.env_data
        return data
    
    def update_position(self):
        # the position of the device is the same as the parent
        self.position = self.parent.position 

    def compute_conductance(self):
        # the conductance of the device is 1
        return 1



class StructuralPart(Part):
    def __init__(self, position = np.array([0,0,0]), age = 0, shape : StructuralShape = None):
        # shape must be a dataclass with the size of the device
        if shape is None or not isinstance(shape, StructuralShape):
            raise ValueError("shape must be a StructuralShape dataclass")
        super().__init__(position, age)
        self.shape = shape

        
        self.grow(0, shape)
           
        self.structural_children = []
        self.device_children = []
        self.device_data = {}

    def generate_points(self, n_points = 10):
        
        # for structural parts the points are generated from the shape as a curve
        # generate n_points, along the shape direction and length
        # the points are generated from the origin

        points = []
        for i in range(n_points+1):
            points.append(self.shape.direction * i * self.shape.lenght/n_points)

        self.points = points

                
    def grow(self, dt, new_shape: StructuralShape = None):
        self.age += dt
        if new_shape is not None:
            self.shape = new_shape
            self.generate_points()
            self.compute_direction()


    def compute_conductance(self):
        return self.shape.radius/self.shape.lenght + self.age
        
    def compute_direction(self):
        self.shape.direction = self.shape.direction

    def get_data(self):
        data = super().get_data()
        structural_data = {
            "shape": asdict(self.shape)
        }
        data["structural_data"] = structural_data
        data["device_data"] = self.device_data
        return data
    
    def update_position(self):
        self.position = self.parent.position + self.points[-1]


    def grab(self): # read data from devices
        # reset the data
        self.device_data = {}
        for device in self.device_children:
            for key, value in device.env_data.items():
                # key is the name of the data es "temperature"
                # value is the value of the data read from the device "device"
                if key in self.device_data:
                    self.device_data[key]["val"].append(value)
                    self.device_data[key]["device"].append(device)
                else:
                    self.device_data[key] = {"val": [value], "device": [device]}

    def send(self): # send data to devices
        pass
  