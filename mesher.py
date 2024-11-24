import gmsh
from plantPart import Stem, Leaf


class MeshGenerator:
    def __init__(self):
        pass
    
    def generate_mesh(self, part):
        if isinstance(part, Stem):
            # Generate mesh for stem
            part.skeleton_points = [self.gmsh.model.occ.addPoint(p[0], p[1], p[2], 0.1) for p in part.skeleton_points]
        elif isinstance(part, Leaf):
            # Generate mesh for leaf
            part.leaf_points = [self.gmsh.model.occ.addPoint(p[0], p[1], p[2], 0.1) for p in part.leaf_points]
