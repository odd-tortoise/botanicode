import numpy as np

class Resources:
    def __init__(self, water=0, sugar=0, auxin=0):
        self.water = water
        self.sugar = sugar
        self.auxin = auxin

    def get_dict(self):
        return {
            "water": self.water,
            "sugar": self.sugar,
            "auxin": self.auxin,
        }

    def set_water(self, water):
        self.water = water

    def set_auxin(self, auxin):
        self.auxin = auxin

    def set_sugar(self, sugar):
        self.sugar = sugar  


class Part:
    def __init__(self, position = np.array([0,0,0]), age = 0):      
        
        # position is the position of the leaves in the stem
        # position is the starting point of the leaf for the leaves
        # is always zero for the root
        # is the position of the parent + 0.1 for the stem
        self.name = None
        
        # this are the things that need to be consistent within the tree structure
        self.position = position 
        self.conductance = 1

        self.age = age
        
        self.parent = None  
        
        self.points = []
        self.real_points = []

        self.color = "black"
        
        
    def grow(self, dt):
        pass

    def get_real_points(self):
        return self.real_points
    
    def compute_real_points(self, offset = np.array([0, 0, 0])):
        self.real_points = [point + offset for point in self.points]

    def get_data(self):
        node_data = {
            "name": self.name,
            "position": self.position.tolist(),
            "age": self.age,
            "conductance": self.conductance,
        }
        data = {
            "node_data": node_data,
        }
        return data

class DevicePart(Part):
    def __init__(self, position = np.array([0,0,0]), age = 0):
        super().__init__(position, age)

        self.conductance = 1

        self.is_generator = False
        
        self.env_data = {}


    def probe(self): #from env
        pass

    def emit(self): # to env
        pass


    def get_data(self):
        data = super().get_data()
        device_data = {
            "is_generator": self.is_generator,
        }
        data["device_data"] = device_data
        data["env_data"] = self.env_data
        return data
    
    def update_position(self):
        self.position = self.parent.position 

    def compute_conductance(self):
        self.conductance = 1

class StructuralPart(Part):
    def __init__(self, position = np.array([0,0,0]), age = 0, lenght = None, radius = None, direction = np.array([0, 0, 1])):
        super().__init__(position, age)


        self.radius = radius
        self.lenght = lenght
        self.direction = direction


        
        if lenght is not None:
            self.points = [np.array([0, 0, 0]), direction*lenght]
            self.compute_conductance()

        self.structural_children = []
        self.device_children = []
        self.device_data = {}

                
    def grow(self, dt, new_l, new_r):
        self.age += dt

        # Generate new skeleton points
        self.points.append(self.points[-1] + self.direction*(new_l - self.lenght))

        self.lenght = new_l
        self.radius = new_r

    def compute_conductance(self):
        self.conductance = self.radius/self.lenght + self.age
        
    def compute_direction(self):
        pass

    def get_data(self):
        data = super().get_data()
        structural_data = {
            "lenght": self.lenght,
            "radius": self.radius,
            "direction": self.direction.tolist(),
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
  