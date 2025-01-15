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



if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Generate artificial data for internode lengths at different temperature levels
    def generate_internode_data(num_points, temp_level, amplitude, peak_positions, decay_rate):
        """
        Generate artificial internode length data.

        Parameters:
            num_points (int): Number of internode positions.
            temp_level (str): Temperature level label.
            amplitude (float): Maximum height of the curve.
            peak_positions (list of int): Internode positions where peaks occur.
            decay_rate (float): Controls the width and decay of the curve.

        Returns:
            DataFrame: Internode position and length data.
        """
        internode_positions = np.arange(1, num_points + 1)
        # Create a curve with multiple Gaussian-like peaks
        lengths = np.zeros_like(internode_positions, dtype=float)
        for peak_position in peak_positions:
            lengths += amplitude * np.exp(-decay_rate * (internode_positions - peak_position) ** 2)
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.5, size=num_points)
        lengths = lengths + noise
        lengths = np.clip(lengths, 0, None)  # Ensure no negative lengths

        return pd.DataFrame({
            "rank": internode_positions,
            "len": lengths,
            "Temp": temp_level
        })

    # Parameters for the temperature levels
    temperature_levels = [
        {"temp_level": 1, "amplitude": 14, "peak_positions": [12,15], "decay_rate": 0.01, "num_points": 25},
        {"temp_level": 2, "amplitude": 12, "peak_positions": [12, 28], "decay_rate": 0.015, "num_points": 28},
        {"temp_level": 3, "amplitude": 10, "peak_positions": [11, 25], "decay_rate": 0.02, "num_points": 35},
        {"temp_level": 4, "amplitude": 8, "peak_positions": [10, 32], "decay_rate": 0.025, "num_points": 45}
    ]

    # Generate datasets for each temperature level
    datasets = [
        generate_internode_data(num_points=params["num_points"], **{k: v for k, v in params.items() if k != "num_points"}) for params in temperature_levels
    ]

    # Combine all datasets into one DataFrame
    combined_data = pd.concat(datasets, ignore_index=True)

    # Plot the data
    plt.figure(figsize=(10, 6))
    for temp_level in combined_data["Temp"].unique():
        subset = combined_data[combined_data["Temp"] == temp_level]
        plt.plot(
            subset["rank"],
            subset["len"],
            marker="o",
            label=temp_level
        )

    plt.title("Artificial Internode Length Data at Different Temperature Levels")
    plt.xlabel("Internode Position")
    plt.ylabel("Mean Internode Length (cm)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the data to a CSV file
    combined_data.to_csv("botanicode/tomato_data/tuning_tomato.csv", index=False)
