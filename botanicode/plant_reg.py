import numpy as np
import json
import pandas as pd
from typing import Any, Dict


class PlantRegulation:
    def __init__(self, json_path: str):
        """
        Initialize by loading the JSON file.

        Args:
            json_path (str): The path to the JSON configuration file.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.general: Dict[str, Any] = self.data.get("general", {})
        self.phylotaxis: Dict[str, Any] = self.data.get("phylotaxis", {})

        # Check that if the leaf arrangement is alternate, there is an angle
        if self.phylotaxis.get("leaf_arrangement") == "alternate":
            assert "angle" in self.phylotaxis, "Angle is required for alternate leaf arrangement"
            # Convert degrees to radians
            self.phylotaxis["angle"] = np.deg2rad(self.phylotaxis["angle"])

        # Leaf shape function
        if self.phylotaxis.get("leaf_style") == "simple":
            self.phylotaxis["outline_function"] = self._leaf_function_standard
        elif self.phylotaxis.get("leaf_style") == "lingulate":
            self.phylotaxis["outline_function"] = self._leaf_function_lingulate
        else:
            raise ValueError("Invalid leaf style")

    def _leaf_function_standard(self, angle: float, t: float) -> np.ndarray:
        """
        Standard leaf shape function.

        Args:
            angle (float): The angle for the leaf shape.
            t (float): The time parameter for the leaf shape.

        Returns:
            np.ndarray: The coordinates of the leaf shape.
        """
        t = t / 5

        def gieles(theta: float, m: int, n1: float, n2: float, n3: float, a: float, b: float) -> float:
            r = (np.abs(np.cos(m * theta / 4) / a)) ** n2 + (np.abs(np.sin(m * theta / 4) / b)) ** n3
            r = r ** (-1 / n1)
            return r

        r = gieles(angle, 2, 1, 1, 1, 2 * t, t)
        x = r * np.cos(angle) + t
        y = r * np.sin(angle)

        return np.array([x, y, 0])

    def _leaf_function_lingulate(self, angle: float, t: float) -> np.ndarray:
        """
        Lingulate leaf shape function.

        Args:
            angle (float): The angle for the leaf shape.
            t (float): The time parameter for the leaf shape.

        Returns:
            np.ndarray: The coordinates of the leaf shape.
        """
        t = t / 5

        def gieles(theta: float, m: int, n1: float, n2: float, n3: float, a: float, b: float) -> float:
            r = (np.abs(np.cos(m * theta / 4) / a)) ** n2 + (np.abs(np.sin(m * theta / 4) / b)) ** n3
            r = r ** (-1 / n1)
            return r

        r = gieles(angle, 2, 1, 1, 1, 10 * t, 0.5 * t)
        x = r * np.cos(angle) + t
        y = r * np.sin(angle)

        return np.array([x, y, 0])

    