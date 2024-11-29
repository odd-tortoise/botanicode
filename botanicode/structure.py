import networkx as nx
import matplotlib.pyplot as plt
from plantPart import Stem, Leaf, Root, SAM
import numpy as np

class Tracker:
    def __init__(self):
        self.data = {}
    
    def track(self, node, node_data):
        if node not in self.data:
            self.data[node] = {}

        for key, value in node_data.items():
            if key not in self.data[node]:
                self.data[node][key] = []
            self.data[node][key].append(value)

    def get_data(self):
        return self.data

    def clear(self):
        self.data = []

    def plot_value(self, ax, resource="auxin", legend=False):

        line_styles = ['-', '--', '-.', ':']
        color = ['blue', 'green', 'red', 'black']

        for node, node_data in self.data.items():
        
            if isinstance(node, Leaf):
                line_style = line_styles[0]
            elif isinstance(node, Stem):
                line_style = line_styles[1]

            elif isinstance(node, SAM):
                line_style = line_styles[2]
                continue
            elif isinstance(node, Root):
                line_style = line_styles[3]

            # check 2nd character of the node name
            if node.name[1] == "1":
                color_line = color[0]
            elif node.name[1] == "2":
                color_line = color[1]
            elif node.name[1] == "3":
                color_line = color[2]
            
           
            ax.plot(node_data[resource], label=node.name, linestyle=line_style, color=color_line)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{resource} over time")
        if legend:
            ax.legend()
        
        



class Structure:
    def __init__(self):
        self.G = nx.Graph()
        self.root = None
        self.history = Tracker()

    def record(self):
        for node in self.G.nodes():
            node_data = node.resources.get_dict()
            self.history.track(node, node_data)

    def new_plant(self, root, stem, leaves, sam): 
        #root.resources.water = 1
        #stem.resources.water = 1
        #for leaf in leaves:
        #    leaf.resources.water = 1
        #sam.resources.water = 1

        self.add_root(root)
        self.add_stem(root, stem)
        for leaf in leaves:
            self.add_leaf(stem, leaf)
        self.add_sam(stem, sam)    

    def add_stuff(self, parent, stem, leaves, sam):
        self.add_stem(parent.parent, stem)
        for leaf in leaves:
            self.add_leaf(stem, leaf)

        self.add_sam(stem, sam)    

    def add_root(self, root):
        if not isinstance(root, Root):
            raise ValueError("You're not adding a Root node.")
        self.root = root
        self.G.add_node(root, type="root")

    def add_sam(self, parent, sam):
        if not isinstance(sam, SAM):
            raise ValueError("You're not adding a SAM node.")
        if not isinstance(parent, Stem):
            raise ValueError(f"Cannot add a SAM to a {type(parent).__name__} node only STEM nodes are allowed.")
        
        self.G.add_node(sam, type = "sam")
        self.G.add_edge(parent, sam, weight=1, length=1)
        parent.sam = sam
        sam.parent = parent
        sam.name = f"SAM{parent.id}"
 
    def add_stem(self, parent, child):
        #check if child is a stem
        if not isinstance(child, Stem):
            raise ValueError("You're not adding a Stem node.")

        # prevent adding a stem to a leaf
        if isinstance(parent, Leaf):
            raise ValueError("Cannot add a stem to a leaf node.")

        parent.stem_children.append(child)
        child.parent = parent

        # Add the nodes to the graph
        self.G.add_node(parent, type="stem")

        # weight is inversely proportional to the length of the stem
        lenght = child.get_length() + 1e-6
        weight = 1/lenght + child.radius
        
        # weight = conductivity of the stem
        # lenght = physical length of the stem
        
        self.G.add_edge(parent, child, weight=weight, length=lenght)

    def add_leaf(self, parent, child):

        #check if child is a leaf
        if not isinstance(child, Leaf):
            raise ValueError("You're not adding a Leaf node.")

        # prevent adding a leaf to a leaf
        if not isinstance(parent, Stem):
            raise ValueError("Can only add a leaf to a stem node.")
            
        parent.leaf_children.append(child)
        child.parent = parent

        # Add the nodes to the graph
        self.G.add_node(child, type="leaf")
        # Add the edge to the graph
        self.G.add_edge(parent, child, weight=1, length=1)
        child.name = child.name[0] + str(parent.id) + child.name[1:]
        
    def traverse(self, node=None, action=lambda node: None):
        if node is None:
            node = self.root

        # Perform the action on the current node
        action(node)

        # Traverse the childrens if presents

        # build the target list
        target = []
        if isinstance(node, Root):
            target = node.stem_children
        elif isinstance(node, Stem):
            target = node.stem_children + node.leaf_children
            if node.sam is not None:
                target.append(node.sam)
        elif isinstance(node, SAM):
            target = []
        elif isinstance(node, Leaf):
            target = []
        
        for child in target:
            self.traverse(child, action)
        
    def traverse_stems(self, node=None, action=lambda node: None):
        if node is None:
            node = self.root
    
        action(node)

        # Traverse stem children
        for child in node.stem_children:
            self.traverse_stems(child, action)

        # Traverse SAM if present
        if isinstance(node, Stem) and node.sam is not None:
            self.traverse(node.sam, action)

    def traverse_leaves(self, node=None, action=lambda node: None):
        def new_action(node):
            if isinstance(node, Leaf):
                action(node)

        self.traverse(node, new_action)

    def ensure_consistency(self):

        def update_position(node):
            if isinstance(node, Root):
                node.compute_real_points()
            elif isinstance(node, Stem):
                offset = node.parent.position
                node.compute_real_points(offset)
                node.position = node.real_points[-1]
                
            elif isinstance(node, SAM):
               offset = node.parent.position
               node.compute_real_points(offset)
               node.position = node.real_points[-1]
               
            elif isinstance(node, Leaf):
                radius = node.parent.radius
                angle = node.z_angle
                parent_position = node.parent.position
                node.position = np.array(parent_position, dtype=float) +  np.array([radius * np.cos(angle), radius * np.sin(angle),0])
                node.compute_real_points(node.position)

        def update_weight_edge(node):
            if isinstance(node, Root) or isinstance(node, SAM):
                pass
            else:
                lenght = node.get_length() + 1e-6
                weight = 1/lenght + node.radius
                self.G[node.parent][node]["weight"] = weight
                self.G[node.parent][node]["length"] = lenght

        def update_distances_to_SAM(node):
            if isinstance(node, Leaf):
                distances = []
                for sam in self.G.nodes():
                    if isinstance(sam, SAM):
                        distances.append(nx.shortest_path_length(self.G, source=node, target=sam, weight="length"))
                node.SAM_distance = distances
            
        self.traverse(action=update_position)
        self.traverse_stems(action=update_weight_edge)
        self.traverse(action=update_distances_to_SAM)

    def plot(self, ax = None):
        G = self.G
        # Use a layout algorithm to compute positions automatically
        pos = nx.bfs_layout(G, self.root, align='horizontal')

        # Define node colors based on node types
        node_colors = []
        for node in G.nodes():
            if isinstance(node, Leaf):
                node_colors.append('orange')
            elif isinstance(node, Root):
                node_colors.append('red')
            elif isinstance(node, SAM):
                node_colors.append('lightblue')
            elif isinstance(node, Stem):
                node_colors.append('green')
            else:
                print("Unknown node type.")

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos ,ax=ax, node_color=node_colors, node_size=500)

        # Draw the edges
        nx.draw_networkx_edges(G, pos,ax=ax, edge_color='gray', arrows=True)

        # node labels are the postiton and the end position of the stem nodes
        labels = {}
        for node in G.nodes():
            labels[node] = node.name
           
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # edge labels are the weights of the edges
        edge_labels = {}
        for edge in G.edges():
            edge_labels[edge] = f"w:{G[edge[0]][edge[1]]['weight']:.2f}\nl:{G[edge[0]][edge[1]]['length']:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    def plot_lighting(self, ax = None):
        
        G = self.G
        positions = nx.bfs_layout(G, self.root, align='horizontal')
        # Collect lighting values for leaf nodes
        
        leaf_lighting = [node.lighting for node in self.G.nodes() if isinstance(node, Leaf)]
        
        # Normalize lighting values between 0 and 1
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=min(leaf_lighting), vmax=max(leaf_lighting))
        cmap = plt.cm.Reds  # Choose a colormap
        
        # Assign colors to nodes
        node_colors = []
        for node in self.G.nodes():
            if isinstance(node, Leaf):
                lighting_value = node.lighting
                color = cmap(norm(lighting_value))
                node_colors.append(color)
            elif isinstance(node, Root):
                node_colors.append('red')
            elif isinstance(node, SAM):
                node_colors.append('lightblue')
            elif isinstance(node, Stem):
                node_colors.append('green')
            else:
                node_colors.append('gray')  # Default color
        
            # Draw the graph
        nx.draw_networkx_nodes(
            self.G, 
            pos=positions, 
            node_color=node_colors, 
            ax=ax
        )
        # Draw the edges
        nx.draw_networkx_edges(G, positions,ax=ax, edge_color='gray', arrows=True)

        # node labels on the leaves
        labels = {}
        for node in G.nodes():
            if isinstance(node, Leaf):
                labels[node] = f"{node.lighting:.2f}\n{np.round(node.position,2)}"
            else:
                labels[node] = ""
        nx.draw_networkx_labels(G, positions, labels, font_size=8, ax=ax)

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Lighting")

    def plot_value(self, ax = None, resource="auxin"):
        
        G = self.G
        positions = nx.bfs_layout(G, self.root, align='horizontal')
        # Collect lighting values for leaf nodes
        # Define a mapping for the value keyword to the corresponding node attribute
        value_mapping = {
            "auxin": lambda node: node.resources.auxin,
            "water": lambda node: node.resources.water,
            "sugar": lambda node: node.resources.sugar,
            "elongation_rate": lambda node: node.resources.elongation_rate
        }

        if resource not in value_mapping:
            raise ValueError(f"Unknown resource: {resource}")
        
        value_amount = [value_mapping[resource](node) for node in self.G.nodes()]
        
        # Normalize lighting values between 0 and 1
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=min(value_amount), vmax=max(value_amount))
        cmap = plt.cm.Blues  # Choose a colormap
        
        node_colors = [cmap(norm(value_mapping[resource](node))) for node in G.nodes()]

            
            # Draw the graph
        nx.draw_networkx_nodes(
            self.G, 
            pos=positions, 
            node_color=node_colors, 
            ax=ax
        )
        # Draw the edges
        nx.draw_networkx_edges(G, positions,ax=ax, edge_color='gray', arrows=True)

        # node labels on the leaves
        labels = {node: f"{value_mapping[resource](node):.2f}" for node in G.nodes()}

        nx.draw_networkx_labels(G, positions, labels, font_size=8, ax=ax)

        # edge labels are the weights of the edges
        edge_labels = {}
        for edge in G.edges():
            edge_labels[edge] = f"w:{G[edge[0]][edge[1]]['weight']:.2f}\nl:{G[edge[0]][edge[1]]['length']:.2f}"
        
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels, ax=ax)

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=str(resource))

    def get_laplacian(self):
        return nx.laplacian_matrix(self.G).toarray()
    
    def get_laplacian_directed(self):
        # make a copy of the graph
        G = self.G.to_directed()

        A = nx.adjacency_matrix(G).toarray()

       
        A_ltr = np.tril(A)
        A_rtl = np.triu(A)

        nodes = list(G.nodes())

        G_ltr = nx.DiGraph()
        
        G_rtl = nx.DiGraph()
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if A_ltr[i,j] != 0:
                    G_ltr.add_node(nodes[i])
                    G_ltr.add_node(nodes[j])
                    G_ltr.add_edge(nodes[i], nodes[j], weight=A_ltr[i,j])
                if A_rtl[i,j] != 0:
                    G_rtl.add_node(nodes[i])
                    G_rtl.add_node(nodes[j])
                    G_rtl.add_edge(nodes[i], nodes[j], weight=A_rtl[i,j])

        

        # get the laplacian matrix
        L_rtl = nx.laplacian_matrix(G_rtl).toarray()
        L_ltr = nx.laplacian_matrix(G_ltr).toarray()
        return A_rtl,L_rtl, A_ltr, L_ltr

    def get_nutrients(self, nutrient_name = "water"):
        nutrients = []
        
        mapping = {
            "water": lambda node: node.resources.water,
            "auxin": lambda node: node.resources.auxin,
            "sugar": lambda node: node.resources.sugar,
            "elongation_rate": lambda node: node.resources.elongation_rate
        }

        if nutrient_name not in mapping:
            raise ValueError(f"Unknown nutrient: {nutrient_name}")
        
        for node in self.G.nodes():
            nutrients.append(mapping[nutrient_name](node))

        return np.array(nutrients)
    
    def set_nutrients(self, nutrients, nutrient_name = "water"):
        mapping = {
            "water": lambda node, value: node.resources.set_water(value),
            "auxin": lambda node, value: node.resources.set_auxin(value),
            "sugar": lambda node, value: node.resources.set_sugar(value),
            "elongation_rate": lambda node, value: node.resources.set_elongation_rate(value),
        }

        if nutrient_name not in mapping:
            raise ValueError(f"Unknown nutrient: {nutrient_name}")
        
        for node, value in zip(self.G.nodes(), nutrients):
            mapping[nutrient_name](node, value)

        
# Path: botanicode/plantPart.py