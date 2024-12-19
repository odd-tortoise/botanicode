import networkx as nx
import matplotlib.pyplot as plt
from general_nodes import StructuralPart, DevicePart
import numpy as np
import matplotlib.colors as mcolors

class Tracker:
    def __init__(self):
        self.data = {}
    
    def track(self, node_data, node_type, timestamp):
        if node_type not in self.data:
            self.data[node_type] = {}

        if node_data["node_data"]["name"] not in self.data[node_type]:
            self.data[node_type][node_data["node_data"]["name"]] = {}

        self.data[node_type][node_data["node_data"]["name"]][timestamp] = node_data

    def get_data(self):
        return self.data

    def clear(self):
        self.data = []


    def plot(self, ax=None, value = "node_data.position", node_types = []):

        if not isinstance(node_types, list):
            node_types = [node_types]

        if ax is None:
            fig, ax = plt.figure(figsize=(10, 5))

        attrs = value.split('.')
        

        
        for node_type in node_types:
            if node_type not in self.data:
                raise ValueError(f"Node type {node_type.__name__} not found.")
            if len(self.data[node_type]) == 0:
                raise ValueError(f"No data found for the node type {node_type.__name__}")
            
            node_data = self.data[node_type] # qui dizioanrio con chiavi i nomi dei node_types

            markers = ['o', "x"]
            standard_marker = ''
            marker_index = 0
            plotted_lines = []
           
            for key, value in node_data.items():
                # key is the name of the node
                # value is the history of the node
                time_steps = list(value.keys())
                extracted_values = []
                for time, data in value.items():    
                    # data is the dictionary with the node data per each timestamp 
                    
                    for attr in attrs:
                        if isinstance(data, dict):
                            data = data.get(attr, None)
                        elif hasattr(data, attr):
                            data = getattr(data, attr, None)
                        else:
                            raise ValueError(f"Attribute {attr} not found in node {key}")
                    extracted_values.append(data)
                
                # chck for using differen markers for close lines 
                use_marker = False
                for line in plotted_lines:
                    if len(line) != len(extracted_values):
                        continue
                    else:
                        distance = np.linalg.norm(np.array(extracted_values) - np.array(line))
                        if distance < 0.05:  # Threshold for considering lines as close
                            use_marker = True
                            marker_index = (marker_index + 1) % len(markers)
                            break

                if use_marker:
                    ax.plot(time_steps, extracted_values,label=f"{key}",marker=markers[marker_index])
                else:
                    ax.plot(time_steps, extracted_values,label=f"{key}", marker=standard_marker)
                
                plotted_lines.append(extracted_values)

            
                
        ax.grid()
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel(".".join(attrs))
                    
        if ax is None:
            plt.tight_layout()
            plt.show()

class Structure:
    def __init__(self, seed=None):
        self.G = nx.Graph()

        #if not isinstance(seed, seed):
        #    raise ValueError("You're not adding a seed node.")
        # move this into the plant logic

        self.seed = seed
        self.G.add_node(seed)

        self.history = Tracker()

    def snapshot(self, timestamp):
        for node in self.G.nodes():
            node_data = node.get_data()
            node_type = type(node)
            self.history.track(node_data, node_type, timestamp)


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
        
    def join_stuff(self, parent, stuff):
        # can we check if stuff is a dict with the right keys?
        # check if stuff is a dict with at least a structure and a generator
        if not isinstance(stuff, dict):
            raise ValueError("You're not adding a dict.")
        if "structure" not in stuff or  not isinstance(stuff["structure"], StructuralPart):
            raise ValueError("You're not adding a structural part.")
        if "generator" not in stuff or not isinstance(stuff["generator"], DevicePart) or not stuff["generator"].is_generator:
            raise ValueError("You're not adding a generator part.")
        
        
        # add the structure to the graph
        self.G.add_node(stuff["structure"])

        # add the generator to the graph
        self.G.add_node(stuff["generator"])

        # add the edge between the generator and the structure
        self.G.add_edge(stuff["generator"], stuff["structure"], weight=stuff["generator"].compute_conductance())
        stuff["generator"].parent = stuff["structure"]
        stuff["structure"].device_children.append(stuff["generator"])
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

        # add the edge between the parent and the structure
        self.G.add_edge(parent, stuff["structure"], weight=stuff["structure"].compute_conductance())
        parent.structural_children.append(stuff["structure"])
        stuff["structure"].parent = parent
          
    def traverse(self, node=None, action=lambda node: None):
        if node is None:
            node = self.seed

        # Perform the action on the current node
        action(node)

        # build the target list
        target = []
        if isinstance(node, DevicePart):
            target = []
        elif isinstance(node, StructuralPart):
            target = node.device_children + node.structural_children
        else:
            # this is the seed
            target = node.structural_children
        
        for child in target:
            self.traverse(child, action)

    def ensure_consistency(self):
        def update_position(node):
            node.update_position()

        def update_conductance(node):
            # update the conductance of the edge between the parent and the node
            if node.parent is not None:
    
                self.G[node.parent][node]['weight'] = node.compute_conductance()    



        self.traverse(action=update_position)
        self.traverse(action=update_conductance)
    

    def plot(self, ax = None, pos = False):
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
                labels[node] = node.name + f"\n{node.position[0]:.2f},{node.position[1]:.2f},{node.position[2]:.2f}"
            else:
                labels[node] = node.name
           
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # edge labels are the weights of the edges
        edge_labels = {}
        for edge in G.edges():
            edge_labels[edge] = f"w:{G[edge[0]][edge[1]]['weight']:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    def plot_value(self, ax = None, var="", node_types = []):
        
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
                attrs = var.split('.')
                value = node
                for attr in attrs:
                    if isinstance(value, dict):
                        value = value.get(attr, None)
                    else:
                        value = getattr(value, attr, None)
                    if value is None:
                        raise ValueError(f"Attribute {var} not found in node {node.name}")
                values.append(value)
                nodes.append(node)

        return values, nodes



if __name__ == "__main__":

    from plantPart import Seed, Stem, Leaf, SAM, Root, RAM

    # create a seed
    seed = Seed()
    fig, ax = plt.subplots(1,4, figsize=(10,5))

    # create the shoot objects
    stem = Stem(lenght=1, radius=0.1)
    leaf = Leaf()
    sam = SAM()

    # create the root objects
    root = Root(lenght=1.4, radius=0.1)
    ram = RAM()


    # assemble the structure
    structure = Structure(seed)
    structure.shoot(seed, {"structure": stem, "generator": sam, "devices": [leaf]})
    structure.shoot(seed, {"structure": root, "generator": ram})

    # plot the structure
    
    structure.plot(ax=ax[0])
    #plt.show()


    # add another part on the shoot

    leaf2 = Leaf()
    stem2 = Stem(lenght=2, radius=0.1)
    sam2 = SAM()

    structure.shoot(sam, {"structure": stem2, "generator": sam2, "devices": [leaf2]})

    # plot the structure
    
    structure.plot(ax=ax[1])
    #plt.show()


    # plot a value
    
    structure.plot_value(ax=ax[2], var="lenght")
    
    structure.plot_value(ax=ax[3], var="leaf_size")
    plt.show()

