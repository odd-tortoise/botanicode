import numpy as np
import json
import pandas as pd
from typing import Any, Dict


class PlantRegulation:
    def __init__(self, json_path: str):
        """Initialize by loading the JSON file."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.general = self.data.get("general", {})
        self.phylotaxis = self.data.get("phylotaxis", {})
        self.growth_params = self.data.get("growth_params", {})
        # check that if the leaf arrangement is alternate, there is an angle
        if self.phylotaxis.get("leaf_arrangement") == "alternate":
            assert "angle" in self.phylotaxis, "Angle is required for alternate leaf arrangement"
            # degrees to radians
            self.phylotaxis["angle"] = np.deg2rad(self.phylotaxis["angle"])

        # leaf shape function
        if self.phylotaxis.get("leaf_style") == "simple":
            def leaf_function_standard(angle, t):
                t = t/5
                def gieles(theta, m, n1, n2, n3, a, b):
                    r = (np.abs(np.cos(m*theta/4)/a))**n2 + (np.abs(np.sin(m*theta/4)/b))**n3
                    r = r**(-1/n1)
                    return r

                r = gieles(angle, 2,1,1,1,2*t,t)
                x = r*np.cos(angle)+t
                y = r*np.sin(angle)

                return np.array([x, y, 0])
            self.phylotaxis["outline_function"] = leaf_function_standard
        elif self.leaf_style == "lingulate":
            def leaf_function_lingulate(angle, t):
                t = t/5
                def gieles(theta, m, n1, n2, n3, a, b):
                    r = (np.abs(np.cos(m*theta/4)/a))**n2 + (np.abs(np.sin(m*theta/4)/b))**n3
                    r = r**(-1/n1)
                    return r

                r = gieles(angle, 2,1,1,1,10*t,0.5*t)
                x = r*np.cos(angle)+t
                y = r*np.sin(angle)

                return np.array([x, y, 0])
            self.phylotaxis["outline_function"] = leaf_function_lingulate
        else:
            raise ValueError("Invalid leaf style")


        
        self.stem_data = self.data.get("stem_data", {})
        self.leaf_data = self.data.get("leaf_data", {})
        self.root_data = self.data.get("root_data", {})


    def get_general_info(self) -> Dict[str, str]:
        """Get general information about the plant."""
        return self.general

    def get_phylotaxis(self) -> Dict[str, Any]:
        """Get phylotaxis configuration."""
        phyllotaxis = self.phylotaxis
        return self.phylotaxis

    def get_stem_data(self) -> Dict[str, Dict[str, float]]:
        """Get stem data."""
        return {
            "initial_state": self.stem_data.get("initial_state", {}),
            "generation": self.stem_data.get("generation", {})
        }

    def get_leaf_data(self) -> Dict[str, Dict[str, float]]:
        """Get leaf data."""
        return {
            "initial_state": self.leaf_data.get("initial_state", {}),
            "generation": self.leaf_data.get("generation", {})
        }

    def get_root_data(self) -> Dict[str, Dict[str, float]]:
        """Get root data."""
        return {
            "initial_state": self.root_data.get("initial_state", {}),
            "generation": self.root_data.get("generation", {})
        }
