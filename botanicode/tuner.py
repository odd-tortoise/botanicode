import numpy as np
import json
from plotter import Plotter 
import pandas as pd


class GrowthRegulation:
    def __init__(self, 
                 file_name = None):
        
        self.file_name = file_name
        
        self.read_data()



    def read_data(self):
        with open(self.file_name, 'r') as f:
            data = json.load(f)


        # phylotaxis data
        self.leaflets_number = data['phylotaxis']['leaflets_number']
        self.leaf_arrangement = data['phylotaxis']['leaf_arrangement']
        if self.leaf_arrangement == "alternate":
            # there must be a angle for the alternate arrangement
            self.leaf_z_angle_alternate_offset = data['phylotaxis']['angle']
            # degrees to radians
            self.leaf_z_angle_alternate_offset = np.deg2rad(self.leaf_z_angle_alternate_offset)
        

        # leaf data
        self.leaves_number = data['leaves_paramters']['leaves_number']
        self.new_leaf_size = data['leaves_paramters']['new_leaf_size']
        self.new_petioles_size = data['leaves_paramters']['new_petioles_size']
        self.new_rachid_size = self.new_leaf_size
        self.leaf_y_angle = data['leaves_paramters']['leaf_y_angle']
        self.leaf_bending_rate = data['leaves_paramters']['leaf_bending_rate']
        self.leaf_style = data['leaves_paramters']['leaf_style']
        if self.leaf_style == "simple":
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
            self.leaf_function = leaf_function_standard
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
            self.leaf_function = leaf_function_lingulate

        # stem data
        self.new_stem_lenght = data['stem_parameters']['new_stem_lenght']
        self.new_stem_radius = data['stem_parameters']['new_stem_radius']

        # root data
        self.new_root_lenght = data['root_parameters']['new_root_lenght']
        self.new_root_radius = data['root_parameters']['new_root_radius']
        
        
        # initial data
        self.initial_stem_lenght = data['initial_data']['initial_stem_lenght']
        self.initial_stem_radius = data['initial_data']['initial_stem_radius']
        self.initial_leaf_number = data['initial_data']['initial_leaf_number']
        self.initial_leaflets_number = data['initial_data']['initial_leaflets_number']
        self.initial_root_lenght = data['initial_data']['initial_root_lenght']
        self.initial_root_radius = data['initial_data']['initial_root_radius']



        # growth data

        if 'growth_data_for_tuning' not in data:
            print("No growth data for tuning found.")
            print("Looking for pre-trained regressors.")
            if 'regressors' not in data:
                raise ValueError("No regressors found.")
            else:
                self.model_stem = data['regressors']['stem']
                self.model_leaf = data['regressors']['leaf']
        else:            
            data_path = data['growth_data_for_tuning']
            
            data = pd.read_csv(data_path)
            data = data.to_dict(orient='records')
            self.tune(data)

            # as of now, we will use pre-trained models
            # linear regression is not the best choice, but it is simple

           



        
    def tune(self, data):
        
        # start tuning the scalers
        print("Tuning...")

        # data is temperature and rank, output is stem lenght and leaf size
        # we need to tune a regression model for each output
        # use something easy to interchange, beahaviour is likely non-linear

        # we can use a simple neural network, or a decision tree, or a random forest
        
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split

        X = np.array([[d['Temp'], d['rank']] for d in data])
        
        # let's separate the outputs
        y_stem = np.array([d['len'] for d in data])
        y_leaf = np.array([d['len'] for d in data])

        # split the data
        X_train, X_test, y_train_stem, y_test_stem = train_test_split(X, y_stem, test_size=0.33, random_state=42)
        X_train, X_test, y_train_leaf, y_test_leaf = train_test_split(X, y_leaf, test_size=0.33, random_state=42)

        # create the models
        model_stem = MLPRegressor(random_state=1, max_iter=500)
        model_leaf = MLPRegressor(random_state=1, max_iter=500)

        # fit the models
        model_stem.fit(X_train, y_train_stem)
        model_leaf.fit(X_train, y_train_leaf)

        # predict the test data
        y_pred_stem = model_stem.predict(X_test)
        y_pred_leaf = model_leaf.predict(X_test)

        # evaluate the models
        from sklearn.metrics import mean_squared_error
        mse_stem = mean_squared_error(y_test_stem, y_pred_stem)
        mse_leaf = mean_squared_error(y_test_leaf, y_pred_leaf)

        print(f"MSE Stem: {mse_stem}")
        print(f"MSE Leaf: {mse_leaf}")
        print("Tuning completed.")

        self.model_stem = model_stem
        self.model_leaf = model_leaf


        self.LAR_model = lambda height: 6 if height < 70 else np.inf
        

    def read_csv(self,path):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        return data

    def __str__(self):
        
        with open(self.file_name, 'r') as f:
            data = json.load(f)

        # print nicely the JSON

        return json.dumps(data, indent=4)
    

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
