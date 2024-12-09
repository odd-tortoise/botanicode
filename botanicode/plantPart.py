import numpy as np

class Resources:
    def __init__(self, water=0, sugar=0, auxin=0, cytokinins=0):
        self.water = water
        self.sugar = sugar
        self.auxin = auxin

        self.elongation_rate = 0

    def get_dict(self):
        return {
            "water": self.water,
            "sugar": self.sugar,
            "auxin": self.auxin,
            "elongation_rate": self.elongation_rate
        }

    def set_elongation_rate(self, elongation_rate):
        self.elongation_rate = elongation_rate

    def set_water(self, water):
        self.water = water

    def set_auxin(self, auxin):
        self.auxin = auxin

    def set_sugar(self, sugar):
        self.sugar = sugar
    
    def set_cytokinins(self, cytokinins):
        self.cytokinins = cytokinins

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
        
        self.resources = Resources()

        self.real_points = []
        self.skeleton_points = [np.array([0, 0, 0])]

    def grow(self, dt):
        pass

    def get_real_points(self):
        return self.real_points
    
    def compute_real_points(self, offset = np.array([0, 0, 0])):
        self.real_points = [point + offset for point in self.skeleton_points]
    
    
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
            self.skeleton_points.append(self.skeleton_points[-1] + self.direction*self.resources.elongation_rate)
            self.radius += 0.1*self.resources.elongation_rate

            self.position = self.position + self.direction*self.resources.elongation_rate

            if self.age > 8:
                self.stop_growing = True

        #self.auxin *= 0.7

    def get_length(self):
        return np.linalg.norm(self.skeleton_points[-1] - self.skeleton_points[0])
    
    def __str__(self):
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
        return message
  
  

class Root(PlantPart):
    counter = 0
    def __init__(self, position = np.array([0,0,0])):
        super().__init__(position, age=1)
        self.skeleton_points.append(np.array([0, 0, -1]))
        self.direction = np.array([0, 0, -1])
        self.stem_children = []
        Root.counter += 1
        self.id = Root.counter
        self.name = f"R{self.id}"
        self.radius = 0.1


    def __str__(self):
        message = f"""
    Root Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    Radius                  : {self.radius:.2f} units

    Skeleton Points         : {np.round(self.skeleton_points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    Water                   : {self.resources.water:.2f} units
    """
        return message
    
    def grow(self, dt):
        self.age += dt
        # Generate new skeleton points
        self.skeleton_points.append(self.skeleton_points[-1] + self.direction*self.resources.elongation_rate)
        self.radius += 0.1*self.resources.elongation_rate

class SAM(PlantPart):
    
    def __init__(self, position, direction = np.array([0, 0, .1])):
        super().__init__(position + direction, age=1)

        self.skeleton_points.append(direction)
        self.stop_growing = True
        
        self.name = f"SAM"
       
    def __str__(self):
        message = f"""
    SAM Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    parent                  : {self.parent.name}

    Skeleton Points         : {np.round(self.skeleton_points, 2).tolist()}
    Real Points             : {np.round(self.real_points, 2).tolist()}
    """
        return message
    
class Leaf(PlantPart):
    def __init__(self,position, z_angle = 0, y_angle = -np.pi/4, radius = 0.1, id = 0, leaflets_number = 1):
        super().__init__(position, age=1)

        self.z_angle = z_angle
        self.y_angle = y_angle
        self.radius = radius

        self.size = 1
        
        self.lighting = 0

        self.position = np.array(self.position, dtype=float) +  np.array([radius * np.cos(z_angle), radius * np.sin(z_angle),0])
        # position of the base of the leaf is the position of the parent + the radius of the parent in the direction of the leaf
        
        self.leaflets_number = leaflets_number
        self.rachid_number = int(max(np.floor(leaflets_number/2),1))
        self.petioles_length = 0.1

        self.leaf_size = 0.5

        

        # the leaflets are the leaves that are attached to the rachid
        # the points are stored in a list of list (skelton_points)

        self.rachid_points = []
        self.real_rachid_points = []

        self.skeleton_points = [[np.array([0, 0, 0])]]

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

        self.generate_total_points()

    def generate_total_points(self):
        rachid_points = [np.array([0, 0, 0])]
        leaves_points = []

        y_angle = self.y_angle
        leaves_to_plot = self.leaflets_number

        for i in range(self.rachid_number):
            #add rachid points
            rachid_segment = np.array([0.5, 0, 0.5*np.sin(y_angle)])
            rachid_point = rachid_points[-1] + rachid_segment
            rachid_points.append(rachid_point)

            y_angle *= 0.5

            if leaves_to_plot >= 2:
               
                leaf_points_up = self.generate_leaf_points(angle = np.pi/2)
                leaf_points_down = self.generate_leaf_points(angle = -np.pi/2)

                petiole_up = np.array([0,self.petioles_length,0])
                petiole_down = np.array([0, -self.petioles_length,0])

                # translate the leaf points to the tip of the rachid
                leaf_points_up = [point + rachid_point + petiole_up for point in leaf_points_up]
                lead_points_down = [point + rachid_point + petiole_down for point in leaf_points_down]

                leaves_points.append(leaf_points_up)
                leaves_points.append(lead_points_down )

                leaves_to_plot -= 2

            if leaves_to_plot == 1:
                # add the leaves on the sides
                 # add the leaf on the tip 
                leaf_point = self.generate_leaf_points(angle = 0)
                petiole = np.array([self.petioles_length,0,0])
                # translate the leaf points to the tip of the rachid
                leaf_point = [point + rachid_point + petiole for point in leaf_point]

                leaves_points.append(leaf_point)

                
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
        
        self.skeleton_points = rotated_leaves
        self.rachid_points = rotated_rachid




    
    def generate_leaf_points(self,angle = 0,n_points=10):

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
        

    def grow(self, dt):
        if not self.stop_growing:
            self.age += dt
            self.generate_total_points()
            if self.age > 4:
                self.stop_growing = True



    def __str__(self):
        message = f"""
    Leaf Information:
    -----------------
    Position                : {np.round(self.position, 2).tolist()}
    Age                     : {self.age:.2f} units
    parent                  : {self.parent.name}
    size                    : {self.size:.2f} units

    Z Angle                 : {self.z_angle:.2f} units
    Y Angle                 : {self.y_angle:.2f} units
    Radius                  : {self.radius:.2f} units

    Lighting                : {self.lighting:.2f} units
    Auxin                   : {self.resources.auxin:.2f} units
    """
        return message


    def produce_auxin(self):
        # produce auxin based on the distace from the SAM
        # if close to the SAM, produce more auxin
        # if far from the SAM, produce less auxin

        self.auxin = 0

        for sam_distace in self.SAM_distance:
            self.auxin += 1/sam_distace


    def compute_real_points(self, offset=np.array([0, 0, 0])):
        # override the compute_real_points method
        # in the leaves we have 2 lists, one for the rachid and one for the leaves, leaves are a list of list 

        # double comprehension list to get all the points
        self.real_points = [point + offset for leaf in self.skeleton_points for point in leaf]
        self.real_rachid_points = [point + offset for point in self.rachid_points]

    def get_real_points(self):
        return self.real_points,self.real_rachid_points