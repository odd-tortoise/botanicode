from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class GraphStructure:
    def __init__(self, seed):
        """
        Initialize the GraphStructure.

        Args:
            seed: The seed node to initialize the graph with. Usually it is the first stem.
        """
        self.G = nx.Graph()
        self.seed = seed
        self.G.add_node(seed)

    def add_node(self, parent: Any, node: Any, plug_point: Optional[str] = None) -> None:
        """
        Add a node to the graph.

        Args:
            parent (Any): The parent node.
            node (Any): The node to add.
            plug_point (Optional[str]): The plug point to attach the node to.
        """
        if parent not in self.G:
            raise ValueError("Parent node does not exist in the graph.")
        
        self.G.add_node(node)
        self.G.add_edge(parent, node)
        
        if not hasattr(node, 'parent'):
            node.parent = None
        if not hasattr(node, 'children'):
            node.children = []

        node.parent = parent
        parent.children.append(node)
        
        if plug_point is not None:
            node.attached_to = plug_point

    def remove_node(self, node: Any) -> None:
        """
        Remove a node from the graph.

        Args:
            node (Any): The node to remove.
        """
        if node not in self.G:
            raise ValueError("Node does not exist in the graph.")
        
        self.G.remove_node(node)
        if node.parent:
            node.parent.children.remove(node)
        node.parent = None

    
    def traverse(self, node=None, action=lambda node: None) -> None:
        """
        Traverse the graph and perform an action on each node.

        Args:
            node (Optional[Any]): The starting node for traversal.
            action (Callable[[Any], None]): The action to perform on each node.
        """
        if node is None:
            node = self.seed
            

        # Perform the action on the current node
        action(node)

        # build the target list
        target = node.children
        
        for child in target:
            self.traverse(child, action)

    def plot(self, plot_positions = False, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the graph.

        Args:
            ax (Optional[plt.Axes]): The matplotlib axes to plot on.
        """
        pos = nx.bfs_layout(self.G, self.seed, align='horizontal')
        node_colors = [mcolors.to_rgb(getattr(node, 'color', 'gray')) for node in self.G.nodes()]
        
        nx.draw_networkx_nodes(self.G, pos, ax=ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color='gray', arrows=True)
        
        labels = {}
        for node in self.G.nodes():
            if plot_positions:
                labels[node] = f"{node.name}\n{node.shape.position[0]:.2f},{node.shape.position[1]:.2f},{node.shape.position[2]:.2f}"
            else:
                labels[node] = node.name
        
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, ax=ax)

    def plot_value(self, var: str, node_types: Union[type, List[type]], ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the values of a variable for specific node types.

        Args:
            var (str): The variable to plot.
            node_types (Union[type, List[type]]): The types of nodes to plot.
            ax (Optional[plt.Axes]): The matplotlib axes to plot on.
        """
        if not isinstance(node_types, list):
            node_types = [node_types]
        
        pos = nx.bfs_layout(self.G, self.seed, align='horizontal')
        values, nodes = self.get_nodes_attribute(var, node_types)
        
        if isinstance(values[0], list):
            self._plot_list_values(pos, nodes, values, var, ax)
        else:
            self._plot_scalar_values(pos, nodes, values, var, ax)

    def _plot_list_values(self, pos: Dict[Any, Tuple[float, float]], nodes: List[Any], values: List[List[Any]], var: str, ax: Optional[plt.Axes]) -> None:
        """
        Plot list values for nodes.

        Args:
            pos (Dict[Any, Tuple[float, float]]): Positions of nodes.
            nodes (List[Any]): List of nodes.
            values (List[List[Any]]): List of list values.
            var (str): The variable to plot.
            ax (Optional[plt.Axes]): The matplotlib axes to plot on.
        """
        nx.draw_networkx_nodes(self.G, pos, node_color="gray", ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color='gray', arrows=True)
        
        labels = {}
        for node,val in zip(nodes,values):
            str_val = str([f"{v:.2f}" for v in val])
            labels[node] = str_val
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, ax=ax)
        plt.ylabel(var)

    def _plot_scalar_values(self, pos: Dict[Any, Tuple[float, float]], nodes: List[Any], values: List[float], var: str, ax: Optional[plt.Axes]) -> None:
        """
        Plot scalar values for nodes.

        Args:
            pos (Dict[Any, Tuple[float, float]]): Positions of nodes.
            nodes (List[Any]): List of nodes.
            values (List[float]): List of scalar values.
            var (str): The variable to plot.
            ax (Optional[plt.Axes]): The matplotlib axes to plot on.
        """
        vmin = min(values)
        vmax = max(values)
        
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Blues
        
        node_colors = [cmap(norm(values[nodes.index(node)])) if node in nodes else "gray" for node in self.G.nodes()]
        
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, ax=ax)
        nx.draw_networkx_edges(self.G, pos, ax=ax, edge_color='gray', arrows=True)
        
        labels = {node: f"{val:.2f}" for node, val in zip(nodes, values)}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8, ax=ax)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=var)


    def get_nodes_attribute(self, var: str, node_types: List[type]) -> Tuple[List[Any], List[Any]]:
        """
        Get the values of a variable for specific node types.

        Args:
            var (str): The variable to get values for.
            node_types (List[type]): The types of nodes to get values for.

        Returns:
            Tuple[List[Any], List[Any]]: A tuple containing the values and the corresponding nodes.
        """
        values = []
        nodes = []

        if not isinstance(node_types, list):
            node_types = [node_types]
        
        for node in self.G.nodes():
            if any(isinstance(node, node_type) for node_type in node_types):
                value = getattr(node.state, var, None)
                values.append(value)
                nodes.append(node)
                
            
        return values, nodes
    
    def set_nodes_attribute(self, var : str, nodes : List, values : List[Any]) -> None:
        """
        Set the values of a variable for specific node types.

        Args:
            var (str): The variable to set values for.
            nodes (List[type]): The nodes to set values for.
            values (List[Any]): The values to set.
        """

        for nod, value in zip(nodes, values):
            if not hasattr(nod.state, var):
                raise ValueError(f"Node {nod} does not have attribute {var}.")
            setattr(nod.state, var, value)

        
        
        