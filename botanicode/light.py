import numpy as np

class Sky:
    def __init__(self, position):
        self.position = position

    def measure_light(self, point):
        return np.linalg.norm(self.position - point)
    