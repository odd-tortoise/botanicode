import numpy as np

class PlantPart:
    def __init__(self, position = np.array([0,0,0]), age=0):      
        self.position = position 
        # position is the position of the leaves in the stem
        # position is the starting point of the leaf for the leaves
        # is always zero for the root
        # is the position of the parent + 0.1 for the stem

        self.name = None

        self.age = age
        self.parent = None  
        self.stop_growing = False
        self.auxin = 0

        self.real_points = []
        self.skeleton_points = [np.array([0, 0, 0])]

    def grow(self, dt):
        pass

    def get_real_points(self):
        return self.real_points
    
    def compute_real_points(self, offset = np.array([0, 0, 0])):
        self.real_points = [point + offset for point in self.skeleton_points]
    
    def get_auxin(self):
        return self.auxin
    
class Stem(PlantPart):
    counter = 0

    def __init__(self, position, age=1, direction = np.array([0, 0, 1])):

        super().__init__(position+direction, age)
        Stem.counter += 1
        
        self.skeleton_points.append(direction) # OGNI classe derivata nell init crea dei punti inizial
        self.radius = 0.1
        self.direction = direction

        self.stem_children = []
        self.leaf_children = []
        self.sam = None

        self.name = f"S{Stem.counter}"
        self.id = Stem.counter

    
    def compute_direction(self):
        vert = self.direction
        
        weighted_direction = np.zeros(3)
        total_light = 0
        for leaf in self.leaf_children:
            weighted_direction += leaf.lighting * np.array([np.cos(leaf.z_angle), np.sin(leaf.z_angle), 0])
            total_light += leaf.lighting
        
        if total_light > 0:
            weighted_direction /= total_light

        direction = 0.4*vert + 10*weighted_direction
        direction /= np.linalg.norm(direction)
        
        self.direction = direction
        return
    
    def grow(self, dt):
        if not self.stop_growing:
            self.age += dt
            # Generate new skeleton points
            self.skeleton_points.append(self.skeleton_points[-1] + self.direction)
            self.radius += 0.1

            self.position = self.real_points[-1] + self.direction

            if self.age > 3:
                self.stop_growing = True

        #self.auxin *= 0.7

    def get_length(self):
        return np.linalg.norm(self.skeleton_points[-1] - self.skeleton_points[0])
    
    def print(self):
        message = f"""
    Stem Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Length                  : {self.get_length():.2f} units
    radius                  : {self.radius:.2f} units
    
    Number of Leaves        : {len(self.leaf_children)}

    Parent                  : {self.parent.name}
    Stem Children           : {", ".join([child.name for child in self.stem_children])}
    Leaves Cihldren         : {", ".join([child.name for child in self.leaf_children])}
    SAM                     : {self.sam.name if self.sam is not None else None}

    Direction               : {np.round(self.direction, 2).tolist()}
    Skeleton Points         : {np.round(self.skeleton_points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        print(message)
  
  

class Root(PlantPart):
    counter = 0
    def __init__(self, position = np.array([0,0,0])):
        super().__init__(position, age=1)
        self.skeleton_points.append(np.array([0, 0, -1]))
        self.direction = np.array([0, 0, -1])
        self.stop_growing = True
        self.stem_children = []
        Root.counter += 1
        self.id = Root.counter
        self.name = f"R{self.id}"

    def print(self):
        message = f"""
    Root Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units

    Skeleton Points         : {np.round(self.skeleton_points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        print(message)
        return


class SAM(PlantPart):
    
    def __init__(self, position, direction = np.array([0, 0, .1])):
        super().__init__(position + direction, age=1)

        self.skeleton_points.append(direction)
        self.stop_growing = True
        
        self.name = f"SAM"
       
    def print(self):
        message = f"""
    SAM Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    parent                  : {self.parent.name}

    Skeleton Points         : {np.round(self.skeleton_points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        print(message)
        return
    
    
class Leaf(PlantPart):
    def __init__(self,position, z_angle = 0, y_angle = -np.pi/4, radius = 0.1, id = 0):
        super().__init__(position, age=1)

        self.z_angle = z_angle
        self.y_angle = y_angle
        self.radius = radius
        
        self.lighting = 0
        self.auxin = 0

        self.position = np.array(self.position, dtype=float) +  np.array([radius * np.cos(z_angle), radius * np.sin(z_angle),0])

        self.SAM_distance = []
        
        if self.parent is not None:
            self.name = f"L{self.parent.id}{id}"
        else:
            self.name = f"L{id}"


        def leaf_function(angle, t):
            t = t/5
            def gieles(theta, m, n1, n2, n3, a, b):
                r = (np.abs(np.cos(m*theta/4)/a))**n2 + (np.abs(np.sin(m*theta/4)/b))**n3
                r = r**(-1/n1)
                return r

            r = gieles(angle, 2,1,1,1,2*t,t)
            x = r*np.cos(angle)+t
            y = r*np.sin(angle)

            return np.array([x, y, 0])

        self.leaf_function = leaf_function

        self.generate_leaf_points()
    
    def generate_leaf_points(self,n_points=10):

        temp_points = []
        angles = np.linspace(0, 2*np.pi, n_points)
        for theta in angles:
            point = self.leaf_function(theta, self.age)
            temp_points.append(point)

        z_rotation_angle = self.z_angle

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        rot_y = np.array([[np.cos(self.y_angle), 0, np.sin(self.y_angle)],  
                        [0, 1, 0],
                        [-np.sin(self.y_angle), 0, np.cos(self.y_angle)]])

        points = [np.dot(rot_z, np.dot(rot_y,point)) for point in temp_points]   

        self.skeleton_points = points
        

    def grow(self, dt):
        if not self.stop_growing:
            self.age += dt
            self.y_angle *= 0.7

            self.generate_leaf_points()
            if self.age > 4:
                self.stop_growing = True

    def print(self):
        message = f"""
    Leaf Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    parent                  : {self.parent.name}

    Z Angle                 : {self.z_angle:.2f} units
    Y Angle                 : {self.y_angle:.2f} units
    Radius                  : {self.radius:.2f} units

    Lighting                : {self.lighting:.2f} units
    Auxin                   : {self.auxin:.2f} units
    """
        print(message)
        return


    def produce_auxin(self):
        # produce auxin based on the distace from the SAM
        # if close to the SAM, produce more auxin
        # if far from the SAM, produce less auxin

        self.auxin = 0

        for sam_distace in self.SAM_distance:
            self.auxin += 1/sam_distace