from plantPart import Stem, Leaf, Root, SAM
from structure import Structure
import numpy as np
from lightEngine import Sky
import matplotlib.pyplot as plt

class GrowthRegulation:
    def __init__(self, 
                 leaf_arrangement = "opposite",
                 length_to_shoot = 3):
        self.leaf_arrangement = leaf_arrangement
        self.length_to_shoot = length_to_shoot
        self.leaf_y_angle = -np.pi/4
        self.leaf_z_angle_alternate_offset= np.pi/4,
    

class Plant:
    def __init__(self, reg=GrowthRegulation()):
        self.growth_regulation = reg
        self.structure = Structure()

        # Create the root node
        root = Root(position=np.array([0, 0, 0]))
        self.plant_height = 0
        
        initial_stem, leaves, initial_sam = self.gen_prolongation(root, 2, 0)
        self.structure.new_plant(root, initial_stem, leaves, initial_sam)

    def compute_plant_height(self):

        def compute_plant_height_recursive(node):
            if isinstance(node, Leaf):
                if node.position[2] > self.plant_height:
                    self.plant_height = node.position[2]
            elif isinstance(node, Stem):
                if node.position[2] > self.plant_height:
                    self.plant_height = node.position[2]
            elif isinstance(node, Root):
                if node.position[2] > self.plant_height:
                    self.plant_height = node.position[2]
            elif isinstance(node, SAM):
                if node.position[2] > self.plant_height:
                    self.plant_height = node.position[2]

        self.structure.traverse(action=compute_plant_height_recursive)
       
    def update(self, sky=None):
        self.structure.ensure_consistency()
        self.compute_lighting(sky)
        #self.compute_directions()
        self.compute_auxin()
        self.structure.diffuse_auxin()



        # plant height is the maximum height of the plant 
        self.compute_plant_height()
   
    def print(self):
        def print_node(node):
            node.print()
        self.structure.traverse(action=print_node)

    def gen_prolongation(self, parent, n_leaves, z_angle_offset):

        # parent can be 
        # - a root: initial growth
        # - a leaf: branching occourse from the leaf!
        # - a SAM: elongation with internode 

        if isinstance(parent, Stem):
            raise ValueError("Cannot prolongate from a Stem node.")
        
        if isinstance(parent, Root):
            direction = np.array([0, 0, 1])
            pos = parent.position
        elif isinstance(parent, Leaf):
            angle = parent.z_angle
            direction = np.array([np.cos(angle), np.sin(angle), 1])
        elif isinstance(parent, SAM):
            direction = parent.parent.direction
            pos = parent.parent.position
        
        
        new_stem = Stem(
            position = pos,
            direction = direction)
        
        leaves = []
        for i in range(n_leaves):
            z_angle = 2 * np.pi * i / n_leaves + z_angle_offset
            y_angle = self.growth_regulation.leaf_y_angle
            radius = new_stem.radius
            leaf = Leaf(new_stem.position, z_angle, y_angle, radius, i)
            

            leaves.append(leaf)

        sam = SAM(new_stem.position)

        return new_stem, leaves, sam

    def prolongate(self, parent, stem, leaves, sam):
        self.structure.add_stuff(parent, stem, leaves, sam)

    def grow(self, dt):
        
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
                        

                    self.prolongate(node, stem, leaves, sam)
                    #self.structure.G.remove_edge(node.parent, node)
                    self.structure.G.remove_node(node)
                    node.parent.sam = None


                        
        self.structure.traverse(action=grow_recursive)
        self.structure.traverse(action=eleongate_recursive)
    
    def branch(self):
        def branch_recursive(node):
            if node.auxin > 6:
                # choose the leaf with the highest lighting
                leaf = max(node.leaf_children, key=lambda x: x.lighting)
                # drop the leaf
                self.structure.drop_leaf(node, leaf)
                
                # get the direction of the leaf
                angle = leaf.z_angle + leaf.angle_offset
                direction = np.array([np.cos(angle), np.sin(angle), 1])
                self.prolongate(parent=node, n_leaves=2, initial_leaf_angle= node.initial_leaf_angle + np.pi/2, direction = direction)
                node.auxin = 0

        self.structure.traverse_stems(action=branch_recursive)
        self.structure.ensure_consistency()

    def compute_directions(self):
        def compute_directions_recursive(node):
           if not isinstance(node, Root) and not isinstance(node, SAM):
             node.compute_direction()

        self.structure.traverse_stems(action=compute_directions_recursive)
        
    def compute_lighting(self, sky):
        def compute_lighting_recursive(node):  
            distance = sky.compute_distance(node.position)
            node.lighting = 1 / (distance)

        self.structure.traverse_leaves(action=compute_lighting_recursive)

    def plot(self, ax=None):
         # Plotting in 3D
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        plant_high = 0

        # Recursive function to traverse the plant structure
        def traverse(node):
            if isinstance(node, Leaf):
                leaf_skeletons = node.get_real_points()
                leaf_skeletons = np.array(leaf_skeletons)
                if leaf_skeletons.size > 0:
                    ax.plot(leaf_skeletons[:, 0], leaf_skeletons[:, 1], leaf_skeletons[:, 2],
                            color='orange', label='Leaf Skeleton', linewidth=2)
                    #close the leaf
                    ax.plot([leaf_skeletons[0, 0], leaf_skeletons[-1, 0]], [leaf_skeletons[0, 1], leaf_skeletons[-1, 1]], [leaf_skeletons[0, 2], leaf_skeletons[-1, 2]], color='orange', linewidth=2)
            elif isinstance(node, Root):
                root_skeletons = node.get_real_points()
                root_skeletons = np.array(root_skeletons)
                if root_skeletons.size > 0:  # Check if any root nodes exist
                    ax.plot(root_skeletons[:,0], root_skeletons[:,1], root_skeletons[:,2],
                            color='red', label='Root Skeleton', linewidth=3, marker='x', markersize=10)
            elif isinstance(node, SAM):
                sam_skeletons = node.get_real_points()
                sam_skeletons = np.array(sam_skeletons)

                if sam_skeletons.size > 0:  # Check if any root nodes exist
                    ax.plot(sam_skeletons[:,0], sam_skeletons[:,1], sam_skeletons[:,2],
                            color='blue', label='SAM Skeleton', linewidth=3, marker='x', markersize=10)
                    
            elif isinstance(node, Stem):
                stem_skeletons = node.get_real_points()
                stem_skeletons = np.array(stem_skeletons)
                if stem_skeletons.size > 0:  # Check if any stem nodes exist
                    ax.plot(stem_skeletons[:, 0], stem_skeletons[:, 1], stem_skeletons[:, 2],
                            color='green', label='Stem Skeleton', linewidth=2, marker='o')
                    
                    
        # Recursively traverse each child node  
        self.structure.traverse(action=traverse)    

        # Setting the plot labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')


                
        size = int(self.plant_height)
        size = size if size%2== 0 else size + 2 - size % 2
        
        ax.set_xlim([-size//2, size//2])
        ax.set_ylim([-size//2, size//2])
        ax.set_zlim([0, size ])

        if ax is None:
            # Show plot
            plt.show()

    def compute_auxin(self):
        def compute_auxin_recursive(node):
            if isinstance(node, Leaf):
                node.produce_auxin()
        
        self.structure.traverse(action=compute_auxin_recursive)