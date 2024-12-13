from plantPart import Stem, Leaf, Root, SAM, RAM, Seed
from structure import Structure
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
        stem = Stem(position=np.array([0, 0, 0]), direction=np.array([0, 0, 1]), age = 1, 
                    lenght = self.growth_regulation.initial_stem_lenght,
                    radius = self.growth_regulation.initial_stem_radius)
        
        leaves = []
        for i in range(self.growth_regulation.initial_leaf_number):
            z_angle = 2 * np.pi * i / self.growth_regulation.initial_leaf_number
            y_angle = self.growth_regulation.leaf_y_angle

            leaf = Leaf(position=stem.position, age = 1, z_angle=z_angle, y_angle=y_angle, 
                        leaf_size=self.growth_regulation.new_leaf_size, 
                        leaflets_number=self.growth_regulation.initial_leaflets_number,
                        rachid_size=0,
                        petioles_size=self.growth_regulation.new_petioles_size,
                        id = i)
            leaves.append(leaf)

        sam = SAM(stem.position)

        self.structure.shoot(self.structure.seed, {"structure": stem, "generator": sam, "devices": leaves})

        
        # add the root 
        root = Root(position=np.array([0, 0, 0]), direction=np.array([0, 0, -1]), age = 1, 
                    lenght = self.growth_regulation.initial_root_lenght,
                    radius = self.growth_regulation.initial_root_radius)
        
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

        # environment interactions
        #self.compute_lighting(env.sky)
        #self.compute_water(env.soil)

        # plant processes
        #self.compute_directions()
        #self.compute_auxin()
        #self.compute_sugar()

    def log(self, logger=None):
        
        def log(node):
            message = str(node)
            logger.warning(message)


        self.structure.traverse(action=log)

    def grow(self, dt):

        """ 
        k_e = 0
        k_s = 0
        k_w = 0
        K = 0

        def elongation_rate(S, W):
            return np.ones_like(S)  # Simplified model
            return k_e * S * W / (K + S + W)


        L = self.structure.get_laplacian()
        A_rtl, L_rtl, A_ltr, L_ltr= self.structure.get_laplacian_directed()

        # contrario?! :O

        def water_dynamic(t, W):
            T = - L @ W  - k_w*elongation_rate(S, W)
            return T
        
        def auxin_dynamic(t, A):
            T = -L @ A
            return T
        
        def sugar_dynamic(t, S):
            return - L @ S - k_s * elongation_rate(S, W)

        


        t_span = [0, dt]

        W = self.structure.get_nutrients("water")
        A = self.structure.get_nutrients("auxin")
        S = self.structure.get_nutrients("sugar")

        from scipy.integrate import solve_ivp
        sol_w = solve_ivp(water_dynamic, t_span, W, t_eval=[dt])
        sol_a = solve_ivp(auxin_dynamic, t_span, A, t_eval=[dt])
        sol_s = solve_ivp(sugar_dynamic, t_span, S, t_eval=[dt])

        self.structure.set_nutrients(nutrients=sol_w.y[:, -1], nutrient_name="water")
        self.structure.set_nutrients(nutrients=sol_a.y[:, -1], nutrient_name="auxin")
        self.structure.set_nutrients(nutrients=sol_s.y[:, -1], nutrient_name="sugar")

        elongation_rate = elongation_rate(sol_s.y[:, -1], sol_w.y[:, -1])
        elongation = elongation_rate * dt  # Simplified integration

        self.structure.set_nutrients(nutrients=elongation, nutrient_name="elongation_rate")
                                     

        # recursive function to grow the plant
        def grow_recursive(node):
            node.grow(dt)

        def eleongate_recursive(node):
            if isinstance(node, SAM):
                if node.parent.get_length() > self.growth_regulation.length_to_shoot:
                    # Add a new stem with leaves based on the growth regulation leaf arrangement
                    if self.growth_regulation.leaf_arrangement == "opposite":
                        stem, leaves, sam = self.gen_prolongation(node, 2, 0)
                    elif self.growth_regulation.leaf_arrangement == "decussate":
                        if node.parent.leaf_children[0].z_angle == 0:
                            z_angle_offset = np.pi/2
                        else:
                            z_angle_offset = 0
                        stem, leaves, sam = self.gen_prolongation(node, 2, z_angle_offset)
                    elif self.growth_regulation.leaf_arrangement == "alternate":

                        previous_angle = node.parent.leaf_children[0].z_angle
                        
                        if self.growth_regulation.leaf_z_angle_alternate_offset is None:
                            raise ValueError("Leaf z angle alternate offset must be set for alternate leaf arrangement.")
                       
                        z_angle_offset = previous_angle + self.growth_regulation.leaf_z_angle_alternate_offset[0]
                        # make sure the angle is between 0 and 2pi
                        z_angle_offset = z_angle_offset % (2 * np.pi)

                        stem, leaves, sam = self.gen_prolongation(node, 1, z_angle_offset)
                        
                    else:
                        raise ValueError("Invalid leaf arrangement.")

                    self.prolongate(node, stem, leaves, sam)
                    #self.structure.G.remove_edge(node.parent, node)
                    self.structure.G.remove_node(node)
                    node.parent.sam = None

        def branch_recursive(node):
            if isinstance(node, Leaf):
                if node.auxin < 0.12 and node.auxin > 0:
                    print(f"Branching on leaf: {node.name}")

                    stem, leaves, sam = self.gen_prolongation(node, 2, 0)
                    self.prolongate(node, stem, leaves, sam)

                    # delete the leaf
                    self.structure.G.remove_edge(node.parent, node)
                    self.structure.G.remove_node(node)
                    node.parent.leaf_children.remove(node)
       
        self.structure.traverse(action=grow_recursive)
        self.structure.traverse(action=eleongate_recursive)
        #self.structure.traverse(action=branch_recursive)

        """

        # get the current length of the structural elements

        current_length = self.structure.get_node_attributes("lenght")
        current_radius = self.structure.get_node_attributes("radius")
        current_size = self.structure.get_node_attributes("leaf_size")

        # remove the None values
        current_length = np.array([l for l in current_length if l is not None])
        current_radius = np.array([r for r in current_radius if r is not None])
        current_size = np.array([s for s in current_size if s is not None])

        # ODE for the lenght of the structural elements
        k = self.growth_regulation.growth_data["k_internodes"]
        l_max = self.growth_regulation.growth_data["internode_lenght_max"]
        r_max = self.growth_regulation.growth_data["internode_radius_max"]

        k_leaf = self.growth_regulation.growth_data["k_leaves"]
        s_max = self.growth_regulation.growth_data["leaves_size_max"]

        # solve the ODE
        new_length = current_length + k * (l_max - current_length) * dt
        new_radius = current_radius + k * (r_max - current_radius) * dt
        new_size = current_size + k_leaf * (s_max - current_size) * dt
        new_petioles_size = 0.1


        # update the lenght of the structural elements
        counter_stems = 0
        counter_leaves = 0
        for node in self.structure.G.nodes:
            if isinstance(node, Stem) or isinstance(node, Root):
                node.grow(dt, new_length[counter_stems], new_radius[counter_stems])
                counter_stems += 1
            elif isinstance(node, Leaf):
                node.grow(dt, new_size[counter_leaves], new_rachid_size= new_size[counter_leaves]/2, new_petioles_size= new_petioles_size)
                counter_leaves += 1

        self.age+=dt

        SAM_nodes = [node for node in self.structure.G.nodes if isinstance(node, SAM)]

        for node in SAM_nodes:
            node.time_to_next_shoot += dt
            if node.time_to_next_shoot >= self.growth_regulation.growth_data["internode_appereace_rate"]:
                self.shoot(node)

        
    def shoot(self, node):
        # check it is a SAM
        if not isinstance(node, SAM):
            raise ValueError("Cannot shoot from a non-SAM node.")
        
        stem = Stem(position=np.array([0, 0, 0]), direction=np.array([0, 0, 1]), age = 1, 
                    lenght = self.growth_regulation.new_stem_lenght,
                    radius = self.growth_regulation.new_stem_radius)
    
        

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
        for i in range(self.growth_regulation.leaves_number):
            z_angle = 2 * np.pi * i / self.growth_regulation.leaves_number + self.leaf_z_angle_offset
            y_angle = self.growth_regulation.leaf_y_angle

            leaf = Leaf(position=stem.position, age = 1, z_angle=z_angle, y_angle=y_angle, 
                        leaf_size=self.growth_regulation.new_leaf_size, 
                        leaflets_number=self.growth_regulation.leaflets_number,
                        rachid_size=self.growth_regulation.new_rachid_size,
                        petioles_size=self.growth_regulation.new_petioles_size,
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
                                color=node.color, label='Leaf Skeleton', linewidth=2, marker='o')
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
                

        self.structure.traverse(action=probe_recursive)


    def compute_auxin(self):
        def compute_auxin_recursive(node):
            node.produce_auxin()
        
        self.structure.traverse_leaves(action=compute_auxin_recursive)

    def compute_water(self, soil):
        def compute_water_recursive(node):
            if isinstance(node, Root):
                node.resources.water = self.plant_height

        self.structure.traverse(action=compute_water_recursive)

    def compute_lighting(self, sky):
        def compute_lighting_recursive(node):  
            distance = sky.compute_distance(node.position)
            node.lighting = 1 / (distance)
            
        self.structure.traverse_leaves(action=compute_lighting_recursive)

    def compute_sugar(self):
        def compute_sugar_recursive(node):
            node.produce_sugar()

        self.structure.traverse_leaves(action=compute_sugar_recursive)

    def compute_directions(self):
        def compute_directions_recursive(node):
           if not isinstance(node, Root) and not isinstance(node, SAM):
             node.compute_direction()

        self.structure.traverse_stems(action=compute_directions_recursive)
        
