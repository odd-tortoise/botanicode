from typing import Any, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

class NodeShapeBlueprint:
    def __init__(self, state: Any):
        self.points: List[np.ndarray] = []  # Local points in the shape's coordinate system
        self.real_points: List[np.ndarray] = []  # Global points in 3D space
        self.state = state
        self.position = np.array([0, 0, 0])  # Global position of the shape
        self.orientation = R.identity()  # Orientation as a rotation object

        self.plug_points = {}
        
    def generate_real_points(self):
        """
        Transform local points into global 3D points using the shape's position and orientation.
        """
        self.real_points = [
            point + self.position for point in self.points
        ]

    def generate_points(self, **kwargs):
        """Generate the shape's local points. To be implemented by subclasses."""
        pass


class CylinderShape(NodeShapeBlueprint):
    def __init__(self, state: Any, direction=np.array([0, 0, 1])):
        super().__init__(state)
        self.direction = direction
        self.generate_points()
        

    def generate_points(self, n_points: int = 10):
        """
        Generate local points for a cylinder along the given direction.
        """
        # Points along the cylinder axis
        axis_points = [
           self.direction * i * self.state.lenght / n_points for i in range(n_points + 1)
        ]
        
        self.points = axis_points

    def compute_plug_points(self) -> dict:
        """
        Return attachment points in local coordinates:
        - 'base': The bottom center of the cylinder.
        - 'tip': The top center of the cylinder.
        """
        return {"base": self.real_points[0], "tip": self.real_points[-1]}
    


class LeafShape(NodeShapeBlueprint):
    @dataclass 
    class PetiolesState:
        lenght: float
        radius: float
        stiffness: float

    
    def __init__(self, state: Any, additional_info: dict, z_angle = 0):
        super().__init__(state)
        self.additional_info = additional_info
        self.z_angle = z_angle
        self.generate_points()
        


    @property
    def size(self):
        return self.state.size
    
    @property
    def petioles_size(self):
        return self.state.petioles_size
    
    @property
    def rachid_size(self):
        return self.state.rachid_size

    
    def generate_points(self):
        rachid_points = [np.array([0, 0, 0])]
        leaves_points = []

        y_angle = self.additional_info["y_angle"]
        leaves_to_plot = self.additional_info["leaflets_number"]

        while leaves_to_plot > 0:
            #add rachid point
            rachid_point = rachid_points[-1] + self.rachid_size * np.array([np.cos(y_angle),0, np.sin(y_angle)])
            rachid_points.append(rachid_point)
            
            if leaves_to_plot >= 2:
               
                leaf_points_up = self.generate_leaf_points(angle_with_z = np.pi/2, angle_wiht_y = y_angle)
                leaf_points_down = self.generate_leaf_points(angle_with_z = -np.pi/2, angle_wiht_y = y_angle)

                petiole_up = np.array([0,self.petioles_size,0])
                petiole_down = np.array([0, -self.petioles_size,0])

                # translate the leaf points to the tip of the rachid
                leaf_points_up = [point + rachid_point + petiole_up for point in leaf_points_up]
                lead_points_down = [point + rachid_point + petiole_down for point in leaf_points_down]

                leaves_points.append(leaf_points_up)
                leaves_points.append(lead_points_down)

                leaves_to_plot -= 2

            if leaves_to_plot == 1:
                # add the leaves on the sides
                 # add the leaf on the tip 
                leaf_point = self.generate_leaf_points(angle_with_z = 0, angle_wiht_y=- y_angle)
                petiole = np.array([self.petioles_size,0,0])
                # translate the leaf points to the tip of the rachid
                leaf_point = [point + rachid_point + petiole for point in leaf_point]

                leaves_points.append(leaf_point)
                leaves_to_plot -= 1

            y_angle -= self.additional_info["leaf_bending_rate"]*y_angle

                
        # rotate the rachid points
        z_rotation_angle = self.z_angle

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        rotated_leaves = []
        
        for leaf in leaves_points:
            leaf = [np.dot(rot_z, point) for point in leaf]
            
            rotated_leaves.append([leaf])

        rotated_rachid = [np.dot(rot_z, point) for point in rachid_points]
        
        self.leaves_points = rotated_leaves
        self.rachid_points = rotated_rachid
        
  
    def generate_leaf_points(self,angle_with_z = 0,angle_wiht_y = 0,n_points=11):

        temp_points = []
        angles = np.linspace(0, 2*np.pi, n_points)
        for theta in angles:
            point = self.additional_info["outline_function"](theta, self.state.size)
            temp_points.append(point)

        y_rotation_angle = angle_wiht_y

        rot_y = np.array([[np.cos(y_rotation_angle), 0, np.sin(y_rotation_angle)],
                        [0, 1, 0],
                        [-np.sin(y_rotation_angle), 0, np.cos(y_rotation_angle)]])

        z_rotation_angle = angle_with_z

        rot_z = np.array([[np.cos(z_rotation_angle), -np.sin(z_rotation_angle), 0],  
                        [np.sin(z_rotation_angle), np.cos(z_rotation_angle), 0],
                        [0, 0, 1]])
        
        points = [np.dot(rot_y,point) for point in temp_points]
        points = [np.dot(rot_z,point) for point in points]   

        return points
    
    def generate_real_points(self):
        """
        Transform local points into global 3D points using the shape's position and orientation.
        """
        self.real_rachid_points = [
            point + self.position for point in self.rachid_points
        ]
        self.real_leaves_points = [
            point + self.position for leaf in self.leaves_points for point in leaf
        ]


            
    def compute_plug_points(self) -> dict:
        """
        Return the attachment point for the leaf:
        - 'base': The base of the leaf where it connects to a stem.
        """
        return {"base": self.real_rachid_points[0]}


class PointShape(NodeShapeBlueprint):
    def __init__(self, state: Any):
        super().__init__(state)
        self.generate_points()
        

    def generate_points(self, n_points=0):
        self.points = [np.array([0, 0, 0])]
       
    def compute_plug_points(self) -> dict:
        """
        Return the single point for a PointShape.
        """
        return {"base": self.points[0]}
    