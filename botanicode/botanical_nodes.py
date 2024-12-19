import numpy as np
from general_nodes import StructuralPart, DevicePart, DeviceShape, StructuralShape, Part
from dataclasses import dataclass

class Stem(StructuralPart):
    counter = 0

    def __init__(self, position = np.array([0,0,0]), age = 0, shape = None):
        super().__init__(position, age, shape)
        self.name = f"S{Stem.counter}"
        self.id = Stem.counter
        Stem.counter += 1
        
        self.color = "green"
    
    def compute_direction(self):
        self.shape.direction = np.array([0, 0, 1])
        return
    
    def is_apical(self):
        # check if there is a SAM object into the relational children
        for child in self.device_children:
            if isinstance(child, SAM):
                return child
        return False
    
    def __str__(self):
        message = f"""
    Stem Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units

    Shape                  : {str(self.shape)}
    
    Parent                  : {self.parent.name}
    
    Structural Children     : {", ".join([child.name for child in self.structural_children])}
    Device Cihldren         : {", ".join([child.name for child in self.device_children])}
    Number of Devices       : {len(self.device_children)}
    SAM                     : {self.is_apical()}
    """
        return message

# class Rootlings(DevicePart): ....

class Seed(Part):
    def __init__(self):
        super().__init__(position=np.array([0,0,0]), age = 0)
        self.color = "darkgray"
        self.name = "Seed"

        self.points = [np.array([0, 0, 0])]
        self.structural_children = []
        
    def __str__(self):
        message = f"""
    Seed Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    """
        return message
    
    def update_position(self):
        self.position = np.array([0,0,0])  

    def generate_points(self):
        self.points = [np.array([0, 0, 0])]

    def grow(self, dt):
        super().grow(dt, new_shape=None)
        
    def compute_conductance(self):
        return 0
    



class Root(StructuralPart):
    counter = 0

    def __init__(self, position = np.array([0,0,0]), age = 0, shape = StructuralShape(lenght=0, radius=0, direction=np.array([0, 0, -1]))):
        super().__init__(position, age, shape=shape)
        self.name = f"R{Root.counter}"
        self.id = Root.counter
        Root.counter += 1
       
        self.color = "brown"
    
    def compute_direction(self):
        self.shape.direction = np.array([0, 0, -1])
        return
    
    def is_apical(self):
        # check if there is a RAM object into the relational children
        for child in self.device_children:
            if isinstance(child, RAM):
                return child
        return False
    
    def __str__(self):
        message = f"""
    Root Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units

    Shape                  : {str(self.shape)}
    
    Parent                  : {self.parent.name}
    
    Structural Children     : {", ".join([child.name for child in self.structural_children])}
    Device Cihldren         : {", ".join([child.name for child in self.device_children])}
    Number of Devices       : {len(self.device_children)}
    RAM                     : {self.is_apical()}
    """
        return message
  
class SAM(DevicePart):
    
    def __init__(self, position = np.array([0,0,0]), age = 0):
        super().__init__(position = position,age=age, shape= DeviceShape(size=0.1))
        
        if self.parent is not None:
            self.name = f"SAM{self.parent.id}"
        else:
            self.name = f"SAM"
        self.is_generator = True

        
        self.color = "lightblue"

        self.time_to_next_shoot = 0
       
    def __str__(self):
        message = f"""
    SAM Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Parent                  : {self.parent.name}

    Points                  : {np.round(self.points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        return message 

    def get_data(self):
        data = super().get_data()
        part_data = {
            "time_to_next_shoot": self.time_to_next_shoot,
        }
        data["part_data"] = part_data
        return data
    
    def compute_points(self):
        self.points = [np.array([0, 0, 0]), np.array([0, 0, 0.1])]

class RAM(DevicePart):
    
    def __init__(self, position= np.array([0,0,0]), age=0):
        super().__init__(position=position,age=age, shape= DeviceShape(size=0.1))
        
        if self.parent is not None:
            self.name = f"RAM{self.parent.id}"
        else:
            self.name = f"RAM"
        
        self.is_generator = True

        self.color = "red"
       
    def __str__(self):
        message = f"""
    RAM Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Parent                  : {self.parent.name}

    Points                  : {np.round(self.points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        return message
    
    def compute_points(self):
        self.points = [np.array([0, 0, 0]), np.array([0, 0, -0.1])]


@dataclass
class LeafShape(DeviceShape):
    size: float
    leaf_function: callable
    rachid_size: float
    petioles_size: float
    
class Leaf(DevicePart):


    def __init__(self,position = np.array([0,0,0]), age=0, id = 0, y_angle = 0, z_angle = 0, y_bending_rate = 1, leaflets_number = 1, shape : LeafShape = None):
        super().__init__(position, age=age, shape=shape)

        self.id = id
        self.parent_rank = 0
        if self.parent is not None:
            self.name = f"L{self.parent.id}{id}"
            self.parent_rank = self.parent.id
        else:
            self.name = f"L{id}"

        self.color = "orange"
        self.rachid_color = "purple"

        

        if leaflets_number == 1:
            self.shape.rachid_size = 0 # this is the size of the rachid block


        self.leaflets_number = leaflets_number

        self.z_angle = z_angle
        self.y_angle = y_angle

        self.y_bending_rate = y_bending_rate
      
    
        if self.shape.leaf_function is None:
            def leaf_function_standard(angle, t):
                t = t/5
                def gieles(theta, m, n1, n2, n3, a, b):
                    r = (np.abs(np.cos(m*theta/4)/a))**n2 + (np.abs(np.sin(m*theta/4)/b))**n3
                    r = r**(-1/n1)
                    return r

                r = gieles(angle, 2,1,1,1,2*t,t)
                x = r*np.cos(angle)+t
                y = r*np.sin(angle)

                return np.array([x, y, 0])

            self.shape.leaf_function = leaf_function_standard
        

        # paramter for the light of each leaflet
        self.leaflets_positions = [self.position] * self.leaflets_number
        
        self.generate_points()

    def probe(self, env):
        self.env_data = env.measure(self.position)

    def send(self):
        self.parent.device_data[self.name] = self.env_data

    def generate_points(self):
        rachid_points = [np.array([0, 0, 0])]
        leaves_points = []

        y_angle = self.y_angle
        leaves_to_plot = self.leaflets_number

        while leaves_to_plot > 0:
            #add rachid point
            rachid_point = rachid_points[-1] + self.shape.rachid_size * np.array([np.cos(y_angle),0, np.sin(y_angle)])
            rachid_points.append(rachid_point)
            
            if leaves_to_plot >= 2:
               
                leaf_points_up = self.generate_leaf_points(angle_with_z = np.pi/2, angle_wiht_y = y_angle)
                leaf_points_down = self.generate_leaf_points(angle_with_z = -np.pi/2, angle_wiht_y = y_angle)

                petiole_up = np.array([0,self.shape.petioles_size,0])
                petiole_down = np.array([0, -self.shape.petioles_size,0])

                # translate the leaf points to the tip of the rachid
                leaf_points_up = [point + rachid_point + petiole_up for point in leaf_points_up]
                lead_points_down = [point + rachid_point + petiole_down for point in leaf_points_down]

                leaves_points.append(leaf_points_up)
                leaves_points.append(lead_points_down)

                leaves_to_plot -= 2

            if leaves_to_plot == 1:
                # add the leaves on the sides
                 # add the leaf on the tip 
                leaf_point = self.generate_leaf_points(angle_with_z = 0, angle_wiht_y=- y_angle)
                petiole = np.array([self.shape.petioles_size,0,0])
                # translate the leaf points to the tip of the rachid
                leaf_point = [point + rachid_point + petiole for point in leaf_point]

                leaves_points.append(leaf_point)
                leaves_to_plot -= 1

            y_angle -= self.y_bending_rate*y_angle

                
        # rotate the rachid points
        z_rotation_angle = self.z_angle

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        rotated_leaves = []
        
        for leaf in leaves_points:
            leaf = [np.dot(rot_z, point) for point in leaf]
            
            rotated_leaves.append([leaf])

        rotated_rachid = [np.dot(rot_z, point) for point in rachid_points]
        
        self.leaves_points = rotated_leaves
        self.rachid_points = rotated_rachid
  
    def generate_leaf_points(self,angle_with_z = 0,angle_wiht_y = 0,n_points=11):

        temp_points = []
        angles = np.linspace(0, 2*np.pi, n_points)
        for theta in angles:
            point = self.shape.leaf_function(theta, self.shape.size)
            temp_points.append(point)

        y_rotation_angle = angle_wiht_y

        rot_y = np.array([[np.cos(y_rotation_angle), 0, np.sin(y_rotation_angle)],
                        [0, 1, 0],
                        [-np.sin(y_rotation_angle), 0, np.cos(y_rotation_angle)]])

        z_rotation_angle = angle_with_z

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        points = [np.dot(rot_y,point) for point in temp_points]
        points = [np.dot(rot_z,point) for point in points]   

        return points
        
    def grow(self, dt, new_shape = None):
        self.age += dt
        
        if new_shape is not None:
            self.shape = new_shape
            self.generate_points()


    def __str__(self):
        message = f"""
    Leaf Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Parent                  : {self.parent.name}
    Shape                   : {str(self.shape)} 

    Y angle                 : {self.y_angle:.2f} units
    Z angle                 : {self.z_angle:.2f} units
    """
        return message

    def compute_real_points(self, offset = np.array([0, 0, 0])):
        # override the compute_real_points method
        # in the leaves we have 2 lists, one for the rachid and one for the leaves, leaves are a list of list 

        # double comprehension list to get all the points
        self.real_points = [point + offset for leaf in self.leaves_points for point in leaf]
        self.real_rachid_points = [point + offset for point in self.rachid_points]

    def get_real_points(self):
        return self.real_points,self.real_rachid_points
    
        
    def get_data(self):
        data = super().get_data()

        part_data ={
            "y_angle": self.y_angle,
            "z_angle": self.z_angle,
            "leaflets_number": self.leaflets_number,
            "parent_rank": self.parent_rank,
        }
        data["part_data"] = part_data
        return data
    
    def update_position(self):
        # we assume that the leaf is attached to the parent and parent is a structural part
        self.position = self.parent.position + np.array([self.parent.shape.radius * np.cos(self.z_angle), self.parent.shape.radius * np.sin(self.z_angle),0])