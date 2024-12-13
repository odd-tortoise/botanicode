import networkx as nx
import matplotlib.pyplot as plt
from plantPart import StructuralPart, DevicePart
import numpy as np
import matplotlib.colors as mcolors

class Tracker:
    def __init__(self):
        self.data = {}
    
    def track(self, node_data, node_type, timestamp):
        if node_type not in self.data:
            self.data[node_type] = {}

        if node_data["name"] not in self.data[node_type]:
            self.data[node_type][node_data["name"]] = {}

        self.data[node_type][node_data["name"]][timestamp] = node_data

    def get_data(self):
        return self.data

    def clear(self):
        self.data = []


    def plot(self, ax=None, values = ["position"], node_type = "Stem"):

        number_of_plots = len(values)
        if ax is not None and number_of_plots != 1:
            raise ValueError("Only one plot is allowed when passing an ax object")
        if node_type not in self.data:
            raise ValueError("Node type not found.")
        if len(self.data[node_type]) == 0:
            raise ValueError("No data found.")
        
        node_data = self.data[node_type]

        # check if the values are in the node data
        for value in values:
            if value not in node_data[list(node_data.keys())[0]][0]:
                raise ValueError(f"Value {value} not found in the node data.")

        if ax is None:
            fig, axs = plt.subplots(number_of_plots, 1, figsize=(10, 10))
            axs = np.atleast_1d(axs)  # Ensure axs is always an array
        else:
            axs = [ax]

        markers = ['o', "x"]
        standard_marker = ''
        marker_index = 0


        for i in range(number_of_plots):
            plotted_lines = []
            for key, value in node_data.items():
                # key is the stem number
                # value is the history of the stem 

                # get the stem lenght vs time, time is the key of the dictionary value
                time_steps = list(value.keys())
                val = [value[time_step][values[i]] for time_step in time_steps]
                
                use_marker = False
                for line in plotted_lines:
                    if len(line) != len(val):
                        continue
                    else:
                        distance = np.linalg.norm(np.array(val) - np.array(line))
                        if distance < 0.05:  # Threshold for considering lines as close
                            use_marker = True
                            marker_index = (marker_index + 1) % len(markers)
                            break

                if use_marker:
                    axs[i].plot(time_steps, val, label=f"{key}", marker=markers[marker_index])
                else:
                    axs[i].plot(time_steps, val, label=f"{key}", marker=standard_marker)
                
                plotted_lines.append(val)

               
                axs[i].set_ylabel(values[i])
                
            axs[i].grid()
            axs[i].legend()
                
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
            node_type = type(node).__name__
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
        self.G.add_edge(stuff["generator"], stuff["structure"], weight=stuff["generator"].conductance)
        stuff["generator"].parent = stuff["structure"]
        stuff["structure"].device_children.append(stuff["generator"])
        stuff["generator"].name = stuff["generator"].name + str(stuff["structure"].id)
        

        # if there are devices add them
        if "devices" in stuff:
            for device in stuff["devices"]:
                if not isinstance(device, DevicePart):
                    raise ValueError("You're not adding a device part.")
                self.G.add_node(device)
                self.G.add_edge(stuff["structure"], device, weight=device.conductance)
                device.parent = stuff["structure"]
                stuff["structure"].device_children.append(device)
                device.name = device.name[0] + str(stuff["structure"].id) + device.name[1:]

        # add the edge between the parent and the structure
        self.G.add_edge(parent, stuff["structure"], weight=stuff["structure"].conductance)
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
        
        for child in target:
            self.traverse(child, action)

    def ensure_consistency(self):
        def update_position(node):
            node.update_position()

        def update_conductance(node):
            # update the conductance of the edge between the parent and the node
            if node.parent is not None:
                node.compute_conductance()
                
                self.G[node.parent][node]['weight'] = node.conductance



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

    def plot_value(self, ax = None, var=""):
        
        G = self.G
        positions = nx.bfs_layout(G, self.seed, align='horizontal')
        
        value_amount = self.get_node_attributes(var)
    
        # Get the minimum and maximum values of the variable, ignore the None values
        vmin = min([value for value in value_amount if value is not None])
        vmax = max([value for value in value_amount if value is not None])
        
        # Normalize lighting values between 0 and 1
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Blues  # Choose a colormap

        node_colors = []
        for value in value_amount:
            if value is None:
                color = "gray"
            else: 
                color = cmap(norm(value))
            node_colors.append(color)
            
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
        for node,value in zip(G.nodes(), value_amount):
            if value is None:
                labels[node] = "N"
            else:
                labels[node] = f"{value:.2f}"

        nx.draw_networkx_labels(G, positions, labels, font_size=8, ax=ax)

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=var)

    def get_node_attributes(self, var):
        value_amount = []

        for node in self.G.nodes():
            # if the node has the attribute var
            if hasattr(node, var):
                value_amount.append(getattr(node, var))
            else:
                value_amount.append(None)

        return value_amount
    
    def set_node_attributes(self, var, values):
        for node,value in zip(self.G.nodes(), values):
            if hasattr(node, var):
                setattr(node, var, value)
            else:
                pass


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

