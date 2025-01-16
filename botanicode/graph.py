import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

class Structure:
    def __init__(self, seed=None):
        self.G = nx.Graph()
        self.seed = seed
        self.G.add_node(seed)

    """ 
    QUESTE COSE NON DOVREBBERSO STARE QUI MA NELLA PIANTA !
    def shoot(self, parent, stuff):
        # shoot is a function that adds from the generators nodes
        # to the structure
        #if not isinstance(parent, DevicePart) or not parent.is_generator:
        #    raise ValueError("You're not shooting from Generator node.")
        # implement later

        # add the structure to the parent parent, the generator node will be removed
        if parent.parent is None:
            # if the parent is the seed
            self.join_stuff(parent, stuff)
        else: 
            self.join_stuff(parent.parent, stuff)
            # remove the old generator node
            self.G.remove_node(parent)
            parent.parent.device_children.remove(parent)

    def branch(self, parent, stuff):
        # idelmente si dovrebbe aggiungere solo a generatori, branch
        # permette di aggiungere alle parti strutturali
         
        #if isinstance(parent, GeneratorPart):
        #    raise ValueError("Oopsie you're using a branch function from a generator node.")
        # implement later

        self.join_stuff(parent.parent, stuff)

        # remove the old generator node
        self.G.remove_node(parent)    
    """     
    """
        # add the edge between the generator and the structure
       
        stuff["generator"].name = stuff["generator"].name + str(stuff["structure"].id)
        

        # if there are devices add them
        if "devices" in stuff:
            for device in stuff["devices"]:
                if not isinstance(device, DevicePart):
                    raise ValueError("You're not adding a device part.")
                self.G.add_node(device)
                self.G.add_edge(stuff["structure"], device, weight=device.compute_conductance())
                device.parent = stuff["structure"]
                stuff["structure"].device_children.append(device)
                device.name = device.name[0] + str(stuff["structure"].id) + device.name[1:]
                device.parent_rank = stuff["structure"].id
                

        """


    def add_node(self, parent, node, plug_point=None):
        # can we check if stuff is a dict with the right keys?
        # check if stuff is a dict with at least a structure and a generator
        
        self.G.add_node(node)
        self.G.add_edge(parent, node)
        node.parent = parent
        parent.children.append(node)
        if plug_point is not None:
            node.attached_to = plug_point

        # this part of the name changes move to plant

    def remove_node(self, node):
        self.G.remove_node(node)
        node.parent.children.remove(node)
        node.parent = None
    
    def traverse(self, node=None, action=lambda node: None):
        if node is None:
            node = self.seed
            

        # Perform the action on the current node
        action(node)

        # build the target list
        target = node.children
        
        for child in target:
            self.traverse(child, action)

    def plot(self, ax = None):
        G = self.G
        # Use a layout algorithm to compute positions automatically
        pos = nx.bfs_layout(G, self.seed, align='horizontal')

        # Define node colors based on node types
        node_colors = [mcolors.to_rgb(node.color) for node in G.nodes()]
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos ,ax=ax, node_color=node_colors, node_size=500)
        # Draw the edges
        nx.draw_networkx_edges(G, pos,ax=ax, edge_color='gray', arrows=True)

        # node labels are the postiton and the end position of the stem nodes
        labels = {}
        for node in G.nodes():
            if pos:
                labels[node] = node.name + f"\n{node.shape.position[0]:.2f},{node.shape.position[1]:.2f},{node.shape.position[2]:.2f}"
            else:
                labels[node] = node.name
           
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    def plot_value(self, var, node_types, ax = None):
        
        G = self.G
        positions = nx.bfs_layout(G, self.seed, align='horizontal')
        
        value_amount, nodes= self.get_nodes_attribute(var=var, node_types=node_types)

        # check if is list of list 
        if isinstance(value_amount[0], list):
            # simplly draw the graph
            nx.draw_networkx_nodes(
                self.G, 
                pos=positions, 
                node_color="gray", 
                ax=ax
            )
            # Draw the edges
            nx.draw_networkx_edges(G, positions,ax=ax, edge_color='gray', arrows=True)

            labels = {}
            for node,val in zip(nodes, value_amount):
                labels[node] = f"{val}"
            nx.draw_networkx_labels(G, positions, labels, font_size=8, ax=ax
            )   

            plt.ylabel(var)
        else:

            # Get the minimum and maximum values of the variable, ignore the None values
            vmin = min(value_amount)
            vmax = max(value_amount)
            
            # Normalize lighting values between 0 and 1
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.Blues  # Choose a colormap

            node_colors = []
            for node in G.nodes:
                if node in nodes:
                    value = value_amount[nodes.index(node)]
                    node_colors.append(cmap(norm(value)))
                else:
                    node_colors.append("gray")

            
            # Draw the graph
            nx.draw_networkx_nodes(
                self.G, 
                pos=positions, 
                node_color=node_colors, 
                ax=ax
            )
            # Draw the edges
            nx.draw_networkx_edges(G, positions,ax=ax, edge_color='gray', arrows=True)

            labels = {}
            for node,val in zip(nodes, value_amount):
                labels[node] = f"{val:.2f}"

            nx.draw_networkx_labels(G, positions, labels, font_size=8, ax=ax)

            # colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=var)

    def get_nodes_attribute(self, var, node_types):
        values = []
        nodes = []

        if not isinstance(node_types, list):
            node_types = [node_types]

        for node in self.G.nodes():
            # check if the node is of the right type
            if len(node_types) > 0 and type(node) in node_types:
                # Split the attribute path by dots    
                value = getattr(node.state, var)
                values.append(value)
                nodes.append(node)

        return values, nodes