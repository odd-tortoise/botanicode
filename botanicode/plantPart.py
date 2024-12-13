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

        self.age = age
        self.resources = Resources()
        
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
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "age": self.age,
            "resources": self.resources.get_dict(),
        }

class DevicePart(Part):
    def __init__(self, position = np.array([0,0,0]), age = 0):
        super().__init__(position, age)

        self.is_generator = False
        self.conductance = 1

    def give(self): # 2 of these with the env, 2  with the structure
        pass

    def emit(self):
        pass

    def take(self):
        pass

    def receive(self):
        pass

    def get_data(self):
        data = super().get_data()
        data["is_generator"] = self.is_generator
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
        self.conductance = 0

        if lenght is not None:
            self.points = [np.array([0, 0, 0]), direction*lenght]
            self.compute_conductance()

        self.structural_children = []
        self.device_children = []

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
        data["lenght"] = self.lenght
        data["radius"] = self.radius
        data["direction"] = self.direction.tolist()
        return data
    
    def update_position(self):
        self.position = self.parent.position + self.points[-1]
    
class Stem(StructuralPart):
    counter = 0

    def __init__(self, position = np.array([0,0,0]), age = 0, lenght = None, radius = None, direction = np.array([0, 0, 1])):
        super().__init__(position, age, lenght, radius, direction)
        self.name = f"S{Stem.counter}"
        self.id = Stem.counter
        Stem.counter += 1
       
        
        self.color = "green"
    
    def compute_direction(self):
        self.direction = np.array([0, 0, 1])
        return
    
    def is_apical(self):
        # check if there is a SAM object into the relational children
        for child in self.device_children:
            if isinstance(child, SAM):
                return child
        return False
    
    def print(self):
        message = f"""
    Stem Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units

    lenght                  : {self.lenght:.2f} units
    Radius                  : {self.radius:.2f} units
    Direction               : {np.round(self.direction, 2).tolist()}
    
    Parent                  : {self.parent.name}
    
    Structural Children     : {", ".join([child.name for child in self.structural_children])}
    Device Cihldren         : {", ".join([child.name for child in self.device_children])}
    Number of Devices       : {len(self.device_children)}
    SAM                     : {self.is_apical()}
    
    Points                  : {np.round(self.points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        return message

# class Rootlings(DevicePart): ....

class Seed(StructuralPart):
    def __init__(self):
        super().__init__(position=np.array([0,0,0]), age = 0)
        self.color = "darkgray"
        self.name = "Seed"

        self.points = [np.array([0, 0, 0])]
        
    def __str__(self):
        message = f"""
    Seed Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    """
        return message
    
    def update_position(self):
        self.position = np.array([0,0,0])  

    def grow(self, dt):
        pass

    def compute_conductance(self):
        pass


class Root(StructuralPart):
    counter = 0

    def __init__(self, position = np.array([0,0,0]), age = 0, lenght = None, radius = None, direction = np.array([0, 0, -1])):
        super().__init__(position, age, lenght, radius, direction)
        self.name = f"R{Root.counter}"
        self.id = Root.counter
        Root.counter += 1
       
        self.color = "brown"
    
    def compute_direction(self):
        self.direction = np.array([0, 0, -1])
        return
    
    def is_apical(self):
        # check if there is a RAM object into the relational children
        for child in self.device_children:
            if isinstance(child, RAM):
                return child
        return False
    
    def __str__(self):
        message = f"""
    Stem Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units

    lenght                  : {self.lenght:.2f} units
    Radius                  : {self.radius:.2f} units
    Direction               : {np.round(self.direction, 2).tolist()}
    
    Parent                  : {self.parent.name}
    
    Structural Children     : {", ".join([child.name for child in self.structural_children])}
    Device Cihldren         : {", ".join([child.name for child in self.device_children])}
    Number of Devices       : {len(self.device_children)}
    RAM                     : {self.is_apical()}
    """
        return message
  
class SAM(DevicePart):
    
    def __init__(self, position = np.array([0,0,0]), age = 0):
        super().__init__(position = position,age=age)
        
        if self.parent is not None:
            self.name = f"SAM{self.parent.id}"
        else:
            self.name = f"SAM"
        self.is_generator = True

        self.points = [np.array([0, 0, 0]), np.array([0, 0, 0.1])]
        self.color = "lightblue"

        self.time_to_next_shoot = 0
       
    def print(self):
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
        data["time_to_next_shoot"] = self.time_to_next_shoot
        return data

class RAM(DevicePart):
    
    def __init__(self, position= np.array([0,0,0]), age=0):
        super().__init__(position=position,age=age)
        
        if self.parent is not None:
            self.name = f"RAM{self.parent.id}"
        else:
            self.name = f"RAM"
        
        self.is_generator = True

        self.points = [np.array([0, 0, 0]), np.array([0, 0, -0.1])]

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
       
class Leaf(DevicePart):
    def __init__(self,position = np.array([0,0,0]), age=0, id = 0, y_angle = 0, z_angle = 0, leaflets_number = 1, leaf_function = None, leaf_size = 0, rachid_size = 0, petioles_size = 0):
        super().__init__(position, age=age)

        self.id = id
        if self.parent is not None:
            self.name = f"L{self.parent.id}{id}"
        else:
            self.name = f"L{id}"

        self.color = "orange"
        self.rachid_color = "purple"

        # as of now the leaf and petioles are 1D, no radius/arch
        self.leaf_size = leaf_size
        self.petioles_size = petioles_size # is the size of a petiole

        if leaflets_number == 1:
            self.rachid_size = 0
        else:
            self.rachid_size = rachid_size # is the size of the rachid block


        self.leaflets_number = leaflets_number

        self.z_angle = z_angle
        self.y_angle = y_angle
      
    
        if leaf_function is None:
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

            self.leaf_function = leaf_function_standard
        else:
            self.leaf_function = leaf_function
        
        # paramter for the light of each leaflet
        self.leaflets_positions = [self.position] * self.leaflets_number
        self.lighting = [0]*self.leaflets_number

        #self.position = np.array(self.position, dtype=float) +  np.array([radius * np.cos(z_angle), radius * np.sin(z_angle),0])
        # position of the base of the leaf is the position of the parent + the radius of the parent in the direction of the leaf
        
        #self.leaflets_number = leaflets_number
        #self.rachid_number = int(max(np.floor(leaflets_number/2),1))
        #self.petioles_length = 0.1
        

        # the leaflets are the leaves that are attached to the rachid
        # the points are stored in a list of list (skelton_points)

        #self.rachid_points = []
        #self.real_rachid_points = []

        #self.skeleton_points = [[np.array([0, 0, 0])]]

        #self.SAM_distance = []

        self.generate_total_points()

    def generate_total_points(self):
        rachid_points = [np.array([0, 0, 0])]
        leaves_points = []

        y_angle = self.y_angle
        leaves_to_plot = self.leaflets_number

        while leaves_to_plot > 0:
            #add rachid point
            rachid_point = rachid_points[-1] + self.rachid_size * np.array([np.cos(y_angle),0, np.sin(y_angle)])
            rachid_points.append(rachid_point)
            
            if leaves_to_plot >= 2:
               
                leaf_points_up = self.generate_leaf_points(angle = np.pi/2)
                leaf_points_down = self.generate_leaf_points(angle = -np.pi/2)

                petiole_up = np.array([0,self.petioles_size,0])
                petiole_down = np.array([0, -self.petioles_size,0])

                # translate the leaf points to the tip of the rachid
                leaf_points_up = [point + rachid_point + petiole_up for point in leaf_points_up]
                lead_points_down = [point + rachid_point + petiole_down for point in leaf_points_down]

                leaves_points.append(leaf_points_up)
                leaves_points.append(lead_points_down)

                leaves_to_plot -= 2

            if leaves_to_plot == 1:
                # add the leaves on the sides
                 # add the leaf on the tip 
                leaf_point = self.generate_leaf_points(angle = 0)
                petiole = np.array([self.petioles_size,0,0])
                # translate the leaf points to the tip of the rachid
                leaf_point = [point + rachid_point + petiole for point in leaf_point]

                leaves_points.append(leaf_point)
                leaves_to_plot -= 1

                
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
  
    def generate_leaf_points(self,angle = 0,n_points=11):

        temp_points = []
        angles = np.linspace(0, 2*np.pi, n_points)
        for theta in angles:
            point = self.leaf_function(theta, self.leaf_size)
            temp_points.append(point)

        z_rotation_angle = angle

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        
        points = [np.dot(rot_z,point) for point in temp_points]   

        return points
        
    def grow(self, dt, new_leaf_size, new_rachid_size,new_petioles_size):
        self.age += dt
        self.leaf_size = new_leaf_size
        self.rachid_size = new_rachid_size if self.leaflets_number > 1 else 0
        self.petioles_size = new_petioles_size

        self.generate_total_points()

    def __str__(self):
        message = f"""
    Leaf Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Parent                  : {self.parent.name}
    Leaf size               : {self.leaf_size:.2f} units
    Rachid size             : {self.rachid_size:.2f} units
    Petioles size           : {self.petioles_size:.2f} units

    Y angle                 : {self.y_angle:.2f} units
    Z angle                 : {self.z_angle:.2f} units
    """
        print(message)
        return

    def compute_real_points(self, offset=np.array([0, 0, 0])):
        # override the compute_real_points method
        # in the leaves we have 2 lists, one for the rachid and one for the leaves, leaves are a list of list 

        # double comprehension list to get all the points
        self.real_points = [point + offset for leaf in self.leaves_points for point in leaf]
        self.real_rachid_points = [point + offset for point in self.rachid_points]

    def get_real_points(self):
        return self.real_points,self.real_rachid_points
    
    def get_data(self):
        data = super().get_data()
        data["leaf_size"] = self.leaf_size
        data["rachid_size"] = self.rachid_size
        data["petioles_size"] = self.petioles_size
        data["y_angle"] = self.y_angle
        data["z_angle"] = self.z_angle
        data["leaflets_number"] = self.leaflets_number
        data["lighting"] = self.lighting
        return data
    
    def update_position(self):
        self.position = self.parent.position + np.array([self.parent.radius * np.cos(self.z_angle), self.parent.radius * np.sin(self.z_angle),0])