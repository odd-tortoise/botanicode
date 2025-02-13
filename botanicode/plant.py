from botanical_nodes import Stem, Leaf, Root, SAM, RAM
from botanical_nodes import NodeState, NodeFactory, Part
from plant_reg import PlantRegulation
from graph_structure import GraphStructure

from env import Environment
from development_engine import DevelopmentEngine

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy


@dataclass
class PlantState:
    # default fields for the plant state
    plant_height : float = 0
    age : float = 0.0
    def reset(self):
        """Reset the plant state."""
        self.__init__()

class Plant:
    def __init__(self, reg : PlantRegulation, node_factory : NodeFactory, plant_state : PlantState):

        """
        Initialize a Plant instance.

        Args:
            reg (PlantRegulation): The plant regulation instance.
            node_factory (NodeFactory): The node factory instance.
            plant_state (PlantState): The plant state instance.
        """

        self.growth_regulation = reg
        self.state = plant_state
        self.node_factory = node_factory

        # utils
        self.leaf_z_angle_offset = 0
        self.phylotaxy_data = self.growth_regulation.phylotaxis
        
        self._initialize_plant()
     
    def reset(self) -> None:
        """Reset the plant to its initial state."""
        for node_type in self.node_factory.node_blueprints.keys():
            node_type.counter = 0
        self.__init__(self.growth_regulation, self.node_factory, self.state)
        self.state.reset()    
     
     
    def probe(self, env: Any, reads: Dict[type, List[str]], t: float) -> None:
        """
        Probe the environment for the new nodes.

        Args:
            env (Any): The environment instance.
            reads (Dict[type, List[str]]): The variables to read from the environment.
            t (float): The current time.
        """
        def probe_recursive(node: Part) -> None:
            if type(node) not in reads:
                return
            
            vars_to_read = reads[type(node)]
            for env_var in vars_to_read:
                value = env.measure(node.shape.position, env_var, t)
                setattr(node.state, env_var, value)

        self.structure.traverse(action=probe_recursive)
            

    
    def _make_leaves(self) -> List[Leaf]:
        """
        Create leaves based on the phylotaxy data.

        Returns:
            List[Leaf]: A list of created leaves.
        """
        if self.phylotaxy_data["leaf_arrangement"] == "alternate":
            self.leaf_z_angle_offset += self.phylotaxy_data["angle"]
            self.leaf_z_angle_offset = self.leaf_z_angle_offset % (2 * np.pi)
        elif self.phylotaxy_data["leaf_arrangement"] == "decussate":
            if self.leaf_z_angle_offset == 0:
                self.leaf_z_angle_offset = np.pi / 2
            else:
                self.leaf_z_angle_offset = 0
        elif self.phylotaxy_data["leaf_arrangement"] == "opposite":
            self.leaf_z_angle_offset = self.node_factory.node_blueprints[Leaf]["state"].z_angle
        else:
            raise ValueError("Invalid leaf arrangement.")
        
        leaves = []
        for i in range(self.phylotaxy_data["leaves_number"]):
            z_angle = 2 * np.pi * i / self.phylotaxy_data["leaves_number"] + self.leaf_z_angle_offset

            leaf = self.node_factory.create(Leaf)
            leaf.id = i
            leaf.state.z_angle = z_angle
            leaves.append(leaf)

        return leaves
    
    def _attach_leaves(self, stem: Stem, leaves: List[Leaf]) -> None:
        """
        Attach leaves to a stem.

        Args:
            stem (Stem): The stem to attach leaves to.
            leaves (List[Leaf]): The leaves to attach.
        """
        for leaf in leaves:
            self.structure.add_node(stem, leaf, "tip")
            leaf.state.rank = stem.state.rank
            leaf.name = leaf.name[0] + str(stem.id) + str(leaf.id)

    
    def _initialize_plant(self) -> None:
        """Initialize the plant structure."""

        stem = self.node_factory.create(Stem)
        sam = self.node_factory.create(SAM)
        ram = self.node_factory.create(RAM)
        root = self.node_factory.create(Root)
        leaves = self._make_leaves()

        self.structure = GraphStructure(seed=stem)

        self.structure.add_node(stem, sam, "tip")
        sam.name =  sam.name[:2] + str(stem.id)
        self.structure.add_node(self.structure.seed, root, "base")
        self.structure.add_node(root, ram, "tip")

        self._attach_leaves(stem, leaves)

        self._update()


    def grow(self, dt: float, env: Environment, dev_eng : DevelopmentEngine, t: float) -> None:
        """
        Grow the plant over a time step.

        Args:
            dt (float): The time step.
            env (Environment): The environment instance.
            dev_eng (DevelopmentEngine)): The engine instance. Needed for shooting and branching.
            t (float): The current time.
        """

        self._update() #update the shapes and the positions
        self._age_nodes(dt)

        # age plant
        self.state.age += dt


        # Shoot from the SAMs
        list_to_shoot = dev_eng.shooting_rule(self)
        for node, shoots in list_to_shoot:
            current_node = node
            for _ in range(shoots):
                current_node = self._shoot(current_node)

        # TODO: Branching
        # TODO: Shoot from the RAMs

        # probe the environment for the new nodes
        self.probe(env, dev_eng.env_reads,t)

    def _age_nodes(self, dt: float) -> None:
        """
        Age the nodes of the plant.

        Args:
            dt (float): The time step.
        """
        def age_node(node: Part) -> None:
            node.state.age += dt

        self.structure.traverse(action=age_node)
        
    def _update(self) -> None:
        """Update the plant's shapes, positions, and real points."""
        self._update_shapes()
        self._update_positions_and_realpoints()
        self.state.plant_height = self._compute_plant_height()

    def _update_shapes(self) -> None:
        """Update the shapes of the plant's nodes."""
        def update_shape(node: Part) -> None:
            node.shape.generate_points(node.state)

        self.structure.traverse(action=update_shape)

    def _update_positions_and_realpoints(self) -> None:
        """Update the positions and real points of the plant's nodes."""
        def update_position(node: Part) -> None:
            node.update_position()
            node.shape.generate_real_points()
            
        self.structure.traverse(action=update_position)

    def _compute_plant_height(self) -> float:
        """
        Compute the height of the plant.

        Returns:
            float: The height of the plant.
        """
        max_height = 0
        for node in self.structure.G.nodes():
            if node.shape.position[2] > max_height:
                max_height = node.shape.position[2]
        return max_height
    
  
    def _shoot(self, node: Part) -> SAM:
        """
        Create a new shoot from a node.

        Args:
            node (Part): The node to shoot from.

        Returns:
            SAM: The new SAM node.
        """
        stem = self.node_factory.create(Stem)
        sam = self.node_factory.create(SAM)
        sam.name = sam.name[2] + str(stem.id) + sam.name[2:]

        leaves = self._make_leaves()
    
        if isinstance(node, SAM):
            self.structure.add_node(node.parent, stem, "tip")
            self.structure.remove_node(node)
        else:
            self.structure.add_node(node, stem, "tip")
        
        self.structure.add_node(stem, sam, "tip")
        self._attach_leaves(stem, leaves)
        self._update()

        # we need to return the SAM in case of multiple shooting from the orginal node
        return sam
    
    def plot(self, ax: Optional[Any] = None) -> None:
        """
        Plot the plant in 3D.

        Args:
            ax (Optional[Any]): The matplotlib axis to plot on.
        """
        show = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            show = True
        
        def plot_node(node: Part) -> None:
            if isinstance(node, Leaf):
                leaf_skeletons = node.shape.real_leaves_points
                rachid_skeleton = node.shape.real_rachid_points
                rachid_skeleton = np.array(rachid_skeleton)
                if rachid_skeleton.size > 0:
                    ax.plot(rachid_skeleton[:, 0], rachid_skeleton[:, 1], rachid_skeleton[:, 2],
                            color=node.rachid_color, label='Rachid Skeleton', linewidth=2)
                for leaf in leaf_skeletons:
                    leaf = np.array(leaf)
                    if leaf.size > 0:
                        ax.plot(leaf[:, 0], leaf[:, 1], leaf[:, 2],
                                color=node.color, label='Leaf Skeleton', linewidth=2, marker='o', markersize=2)
                        ax.plot([leaf[0, 0], leaf[-1, 0]], [leaf[0, 1], leaf[-1, 1]], [leaf[0, 2], leaf[-1, 2]], color=node.color, linewidth=2)
            else:
                skeleton = node.shape.real_points
                skeleton = np.array(skeleton)
                if skeleton.size > 0:
                    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
                            color=node.color, linewidth=2)
                    
        self.structure.traverse(action=plot_node)    

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
                
        size = int(self.state.plant_height) + 1
        size = size if size % 2 == 0 else size + 2 - size % 2
        
        ax.set_xlim([-size // 2, size // 2])
        ax.set_ylim([-size // 2, size // 2])
        ax.set_zlim([0, size])

        if show:
            plt.show()