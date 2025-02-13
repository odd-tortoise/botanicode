from typing import Any, Dict, List
import numpy as np
from dataclasses import dataclass


# Future:
# add more shapes
# add rotations objects to deal btter with orientations
# make the leaf pattern generation more flexible by integrating it into the plant regulation file (similar to the leaf positioning)


class NodeShape:
    """Base class for all node shapes."""
    required_state_variables: List[str] = []

    def __init__(self) -> None:
        self.points: List[np.ndarray] = []  # Local points in the shape's coordinate system
        self.real_points: List[np.ndarray] = []  # Global points in 3D space
        self.position: np.ndarray = np.array([0, 0, 0])  # Global position of the shape

    def generate_real_points(self) -> None:
        """
        Transform local points into global 3D points using the shape's position and orientation.
        """
        self.real_points = [point + self.position for point in self.points]

    def generate_points(self, state: Any, n_points: int = 10) -> None:
        """
        Generate the shape's local points. To be implemented by subclasses.

        Args:
            state (Any): The state of the node containing required variables.
            n_points (int): The number of points to generate.
        """
        pass


class CylinderShape(NodeShape):
    required_state_variables: List[str] = ["length", "direction", "radius"]

    def __init__(self, state: Any) -> None:
        super().__init__()
        self.generate_points(state)

    def generate_points(self, state: Any, n_points: int = 10) -> None:
        """
        Generate local points for a cylinder along the given direction.

        Args:
            state (Any): The state of the node containing required variables.
            n_points (int): The number of points to generate along the cylinder.
        """
        # Points along the cylinder axis
        axis_points = [state.direction * i * state.length / n_points for i in range(n_points + 1)]
        self.points = axis_points

    def compute_plug_points(self) -> Dict[str, np.ndarray]:
        """
        Return attachment points in absolute coordinates:
        - 'base': The bottom center of the cylinder.
        - 'tip': The top center of the cylinder.

        Returns:
            Dict[str, np.ndarray]: A dictionary with 'base' and 'tip' points.
        """
        return {"base": self.real_points[0], "tip": self.real_points[-1]}


class LeafShape(NodeShape):
    required_state_variables: List[str] = [
        "size", "petioles_size", "rachid_size", "leaflets_number", 
        "leaf_bending_rate", "outline_function", "y_angle", "z_angle"
    ]

    def __init__(self, state: Any) -> None:
        super().__init__()
        self.generate_points(state)

        
    def generate_points(self, state: Any, n_points: int = 10) -> None:
        """
        Generate local points for a leaf shape.

        Args:
            state (Any): The state of the node containing required variables.
            n_points (int): The number of points to generate along the leaf.
        """
        rachid_points = [np.array([0, 0, 0])]
        leaves_points = []

        y_angle = state.y_angle
        leaves_to_plot = state.leaflets_number

        while leaves_to_plot > 0:
            #add rachid point
            rachid_point = rachid_points[-1] + state.rachid_size * np.array([np.cos(y_angle),0, np.sin(y_angle)])
            rachid_points.append(rachid_point)
            
            if leaves_to_plot >= 2:
               
                leaf_points_up = self._generate_leaf_points(state=state,angle_with_z = np.pi/2, angle_with_y = y_angle)
                leaf_points_down = self._generate_leaf_points(state=state,angle_with_z = -np.pi/2, angle_with_y = y_angle)

                petiole_up = np.array([0,state.petioles_size,0])
                petiole_down = np.array([0, -state.petioles_size,0])

                # translate the leaf points to the tip of the rachid
                leaf_points_up = [point + rachid_point + petiole_up for point in leaf_points_up]
                lead_points_down = [point + rachid_point + petiole_down for point in leaf_points_down]

                leaves_points.append(leaf_points_up)
                leaves_points.append(lead_points_down)

                leaves_to_plot -= 2

            if leaves_to_plot == 1:
                # add the leaves on the sides
                 # add the leaf on the tip 
                leaf_point = self._generate_leaf_points(state=state,angle_with_z = 0, angle_with_y=- y_angle)
                petiole = np.array([state.petioles_size,0,0])
                # translate the leaf points to the tip of the rachid
                leaf_point = [point + rachid_point + petiole for point in leaf_point]

                leaves_points.append(leaf_point)
                leaves_to_plot -= 1

            y_angle -= state.leaf_bending_rate*state.y_angle

                
         # Rotate the rachid points
        z_rotation_angle = state.z_angle
        rot_z = np.array([
            [np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],
            [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
            [0, 0, 1]
        ])
        
        rotated_leaves = []
        
        for leaf in leaves_points:
            leaf = [np.dot(rot_z, point) for point in leaf]
            
            rotated_leaves.append([leaf])
        rotated_rachid = [np.dot(rot_z, point) for point in rachid_points]
        
        self.leaves_points = rotated_leaves
        self.rachid_points = rotated_rachid
        
  
    def _generate_leaf_points(self, state: Any, angle_with_z: float = 0, angle_with_y: float = 0, n_points: int = 11) -> List[np.ndarray]:
        """
        Generate points for a single leaf.

        Args:
            state (Any): The state of the node containing required variables.
            angle_with_z (float): The angle to rotate around the z-axis.
            angle_with_y (float): The angle to rotate around the y-axis.
            n_points (int): The number of points to generate along the leaf.

        Returns:
            List[np.ndarray]: A list of points representing the leaf.
        """
        temp_points = [state.outline_function(theta, state.size) for theta in np.linspace(0, 2 * np.pi, n_points)]

        rot_y = np.array([
            [np.cos(angle_with_y), 0, np.sin(angle_with_y)],
            [0, 1, 0],
            [-np.sin(angle_with_y), 0, np.cos(angle_with_y)]
        ])

        rot_z = np.array([
            [np.cos(angle_with_z), -np.sin(angle_with_z), 0],
            [np.sin(angle_with_z), np.cos(angle_with_z), 0],
            [0, 0, 1]
        ])

        points = [np.dot(rot_y, point) for point in temp_points]
        points = [np.dot(rot_z, point) for point in points]

        return points
    
    def generate_real_points(self) -> None:
        """
        Transform local points into global 3D points using the shape's position and orientation.
        Leaves are different, they have 2 sets of points, one for the rachid and one for the leaves.
        """
        self.real_rachid_points = [point + self.position for point in self.rachid_points]
        self.real_leaves_points = [point + self.position for leaf in self.leaves_points for point in leaf]


            
    def compute_plug_points(self) -> Dict[str, np.ndarray]:
        """
        Return the attachment point for the leaf:
        - 'base': The base of the leaf where it connects to a stem.

        Returns:
            Dict[str, np.ndarray]: A dictionary with the 'base' point.
        """
        return {"base": self.real_rachid_points[0]}


class PointShape(NodeShape):
    def __init__(self, state: Any) -> None:
        super().__init__()
        self.generate_points(state)

    def generate_points(self, state: Any, n_points: int = 0) -> None:
        """
        Generate a single point for the shape.

        Args:
            state (Any): The state of the node.
            n_points (int): The number of points to generate (default is 0).
        """
        self.points = [np.array([0, 0, 0])]

    def compute_plug_points(self) -> Dict[str, np.ndarray]:
        """
        Return the single point for a PointShape.

        Returns:
            Dict[str, np.ndarray]: A dictionary with the 'base' point.
        """
        return {"base": self.real_points[0]}
    