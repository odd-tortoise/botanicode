from botanical_nodes import Stem, Leaf, Root, SAM, RAM, Seed, LeafShape
from general_nodes import StructuralShape, DeviceShape
from graph import Structure
import numpy as np
from light import Sky
import matplotlib.pyplot as plt
from tuner import GrowthRegulation


class Plant:
    def __init__(self, reg, age=0, timer=None):
        self.growth_regulation = reg

        seed = Seed()
        self.structure = Structure(seed=seed)

        self.plant_height = 0
        self.age = age
        self.timer = timer  

        self.leaf_z_angle_offset = 0
        
        self.initialize_plant()
        

    def initialize_plant(self):
        # Create the initial plant structure
        # assume initial age of the stem to be 1 

        stem_shape = StructuralShape(lenght = self.growth_regulation.initial_stem_lenght,
                                     radius = self.growth_regulation.initial_stem_radius,
                                     direction = np.array([0, 0, 1]))
        stem = Stem(position=np.array([0, 0, 0]), age = 1, shape=stem_shape)

        leaf_shape = LeafShape(size = self.growth_regulation.new_leaf_size,
                               leaf_function= None, # use the default leaf function
                               rachid_size= 0,
                               petioles_size= self.growth_regulation.new_petioles_size)
        leaves = []
        for i in range(self.growth_regulation.initial_leaf_number):
            z_angle = 2 * np.pi * i / self.growth_regulation.initial_leaf_number
            y_angle = self.growth_regulation.leaf_y_angle

            leaf = Leaf(position=stem.position, age = 1, z_angle=z_angle, y_angle=y_angle, y_bending_rate=self.growth_regulation.leaf_bending_rate,
                        leaflets_number=self.growth_regulation.initial_leaflets_number,
                        shape=leaf_shape,
                        id = i)
            leaves.append(leaf)

        sam = SAM(stem.position)

        self.structure.shoot(self.structure.seed, {"structure": stem, "generator": sam, "devices": leaves})

        
        # add the root 
        root_shape = StructuralShape(lenght = self.growth_regulation.initial_root_lenght,
                                        radius = self.growth_regulation.initial_root_radius,
                                        direction = np.array([0, 0, -1]))
        root = Root(position=np.array([0, 0, 0]), age = 1, shape=root_shape)
        
        ram = RAM(root.position)

        self.structure.shoot(self.structure.seed, {"structure": root, "generator": ram})
        self.structure.ensure_consistency()
        self.compute_plant_height()

    def compute_plant_height(self):
        def compute_plant_height_recursive(node):
            if node.position[2] > self.plant_height:
                self.plant_height = node.position[2]

        self.structure.traverse(action=compute_plant_height_recursive)
       
    def update(self, env=None):
        self.structure.ensure_consistency()
        self.compute_plant_height()
        self.compute_real_points_for_nodes()
        self.probe_environment(env)
       

    def log(self, logger=None):        
        def log(node):
            message = str(node)
            logger.warning(message)

        self.structure.traverse(action=log)

    def grow(self, dt):

        # get the current length of the structural elements
        stem_len, stem_nodes = self.structure.get_nodes_attribute(var = "shape.lenght", node_types=[Stem])
        stem_rank, _ = self.structure.get_nodes_attribute("id", node_types=[Stem])
        stem_temp, _ = self.structure.get_nodes_attribute("device_data.temperature.val", node_types=[Stem])

        for id, el in enumerate(stem_temp):
            #substitute with the average
            stem_temp[id] = np.mean(el)

        stem_len = np.array(stem_len)
        stem_rank = np.array(stem_rank)
        stem_temp = np.array(stem_temp)


        # leaf data
        leaf_size, leaf_node = self.structure.get_nodes_attribute(var = "shape.size", node_types=[Leaf])
        leaf_rank, _ = self.structure.get_nodes_attribute("parent_rank", node_types=[Leaf])

        leaf_size = np.array(leaf_size)
        leaf_rank = np.array(leaf_rank)

    
        l = self.growth_regulation.l_max*self.growth_regulation.stem_lenght_scaler(stem_rank, stem_temp)
        s = self.growth_regulation.s_max*self.growth_regulation.leaf_size_scaler(leaf_rank,None)

        # ODE for the lenght of the structural elements
        k = self.growth_regulation.K * np.ones_like(stem_len)

        der = []

        for i in range(len(stem_len)):
            coef = k[i] * (l[i] - stem_len[i])
            der.append(coef)

        
        new_length = stem_len + np.array(der) * dt
        new_radius = self.growth_regulation.new_stem_radius * np.ones_like(new_length) 

        der = []
        k= self.growth_regulation.K * np.ones_like(leaf_size)
        for i in range(len(leaf_size)):
            coef = max(0,k[i] * (s[i] - leaf_size[i]))
            der.append(coef)

        
        new_size = leaf_size + np.array(der) * dt
    


        # update the lenght of the structural elements
        for node,new_len,new_r in zip(stem_nodes, new_length, new_radius):
            new_shape = StructuralShape(lenght = new_len, radius = new_r, direction = node.shape.direction)
            node.grow(dt, new_shape)
           
        for node,new_s in zip(leaf_node, new_size):
            new_shape = LeafShape(size = new_s, leaf_function= node.shape.leaf_function, rachid_size= new_s/2, petioles_size= self.growth_regulation.new_petioles_size )
            node.grow(dt, new_shape)
            
            
        self.age+=dt

        SAM_nodes = [node for node in self.structure.G.nodes if isinstance(node, SAM)]

        LAR_normal = self.growth_regulation.LAR


        h_max = 60

        LAR = LAR_normal if self.plant_height < h_max else np.inf

        for node in SAM_nodes:
            node.time_to_next_shoot += dt
            if node.time_to_next_shoot >= LAR:
                self.shoot(node)

        
    def shoot(self, node):
        # check it is a SAM
        if not isinstance(node, SAM):
            raise ValueError("Cannot shoot from a non-SAM node.")
        
        stem_shape = StructuralShape(lenght = self.growth_regulation.new_stem_lenght,
                                        radius = self.growth_regulation.new_stem_radius,
                                        direction = np.array([0, 0, 1]))
        
        stem = Stem(position=np.array([0, 0, 0]), age = 1, shape=stem_shape)
        

        if self.growth_regulation.leaf_arrangement == "alternate":
            self.leaf_z_angle_offset+=self.growth_regulation.leaf_z_angle_alternate_offset
            self.leaf_z_angle_offset = self.leaf_z_angle_offset % (2 * np.pi)
        elif self.growth_regulation.leaf_arrangement == "decussate":
            if self.leaf_z_angle_offset == 0:
                self.leaf_z_angle_offset = np.pi/2
            else:
                self.leaf_z_angle_offset = 0
        elif self.growth_regulation.leaf_arrangement == "opposite":
            self.leaf_z_angle_offset = 0
        else:
            raise ValueError("Invalid leaf arrangement.")
        
        leaves = []
        leaf_shape = LeafShape(size = self.growth_regulation.new_leaf_size,
                                 leaf_function= self.growth_regulation.leaf_function,
                                 rachid_size= self.growth_regulation.new_rachid_size,
                                 petioles_size= self.growth_regulation.new_petioles_size)
        for i in range(self.growth_regulation.leaves_number):
            z_angle = 2 * np.pi * i / self.growth_regulation.leaves_number + self.leaf_z_angle_offset
            y_angle = self.growth_regulation.leaf_y_angle

            leaf = Leaf(position=stem.position, age = 1, z_angle=z_angle, y_angle=y_angle, y_bending_rate=self.growth_regulation.leaf_bending_rate,
                        leaflets_number=self.growth_regulation.leaflets_number,
                        shape=leaf_shape,
                        id = i)
    
            leaves.append(leaf)

        sam = SAM(stem.position)

        self.structure.shoot(node, {"structure": stem, "generator": sam, "devices": leaves})

    def plot(self, ax=None):
         # Plotting in 3D
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        def plot_node(node):
            if isinstance(node, Leaf):
                
                leaf_skeletons, rachid_skeleton = node.get_real_points()
                # plot the rachid
                rachid_skeleton = np.array(rachid_skeleton)
                if rachid_skeleton.size > 0:
                    ax.plot(rachid_skeleton[:, 0], rachid_skeleton[:, 1], rachid_skeleton[:, 2],
                            color=node.rachid_color, label='Rachid Skeleton', linewidth=2, marker='o')
                    
                # plot the leaves 
                for leaf in leaf_skeletons:
                    leaf = np.array(leaf)
                    if leaf.size > 0:
                        ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2],
                                color=node.color, label='Leaf Skeleton', linewidth=2, marker='o',markersize=2)
                        ax.plot([leaf[0, 0], leaf[-1, 0]], [leaf[0, 1], leaf[-1, 1]], [leaf[0, 2], leaf[-1, 2]], color=node.color, linewidth=2)
                
                
            else:
                skeleton = node.get_real_points()
                
                skeleton = np.array(skeleton)
                if skeleton.size > 0:  # Check if any stem nodes exist
                    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
                            color=node.color, label='Stem Skeleton', linewidth=2, marker='o')
                    
                    
        # Recursively traverse each child node  
        self.structure.traverse(action=plot_node)    

        # Setting the plot labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
                
        size = int(self.plant_height) + 1
        size = size if size%2== 0 else size + 2 - size % 2
        
        ax.set_xlim([-size//2, size//2])
        ax.set_ylim([-size//2, size//2])
        ax.set_zlim([0, size ])

        if ax is None:
            # Show plot
            plt.show()

    def compute_real_points_for_nodes(self):
        def compute_real_points_recursive(node):
            if node.parent is not None:
                if isinstance(node, Leaf):
                    node.compute_real_points(offset=node.position)
                else:
                    node.compute_real_points(offset=node.parent.position)
            else:
                node.compute_real_points()

        self.structure.traverse(action=compute_real_points_recursive)

    def probe_environment(self, env):
        def probe_recursive(node):
            if isinstance(node, Leaf):
                node.probe(env)

        def transfer_to_structure(node):
            if isinstance(node, Stem):
                node.grab()
                

        self.structure.traverse(action=probe_recursive)
        self.structure.traverse(action=transfer_to_structure)


