import numpy as np
import json
from plotter import Plotter 


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
        # COSE DA TUNARE
        self.l_max = 10
        self.s_max = 10
        self.LAR = 2
        optimal_rank = 1
        alpha = 0.1
        
        beta = 0.01
        optimal_temp = 21

        self.K = 0.1

        scaler = lambda alpha, optimal_val, val: np.exp(-alpha * (val - optimal_val)**2)
        
        self.rank_scaler = lambda rank: scaler(alpha, optimal_rank, rank)
        self.temp_scaler = lambda temp: scaler(beta, optimal_temp, temp)
       
        self.stem_lenght_scaler = lambda rank,temp: self.rank_scaler(rank) #* self.temp_scaler(temp)
        self.leaf_size_scaler = lambda rank,temp: self.rank_scaler(rank)

    def read_csv(self,path):
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        return data

    def __str__(self):
        
        with open(self.file_name, 'r') as f:
            data = json.load(f)

        # print nicely the JSON

        return json.dumps(data, indent=4)
    

if __name__ == "__main__":
    gr = GrowthRegulation('botanicode/tomato_data/tomato.json')

    # plot the scaler 

    rank = np.linspace(0,20,20)
    temp = np.linspace(0,40,80)

    rank_scaler = np.vectorize(gr.rank_scaler)
    temp_scaler = np.vectorize(gr.temp_scaler)

    rank_scaler = rank_scaler(rank)
    temp_scaler = temp_scaler(temp)

    import matplotlib.pyplot as plt

    # subplots
    fig, ax = plt.subplots(1,2, figsize=(15,8))

    ax[0].plot(rank, gr.l_max*rank_scaler)
    ax[0].set_title("Rank scaler")
    ax[0].set_xlabel("Rank")
    ax[0].set_ylabel("Factor")
    ax[0].grid()

    ax[1].plot(temp, gr.l_max*temp_scaler)
    ax[1].set_title("Temperature scaler")
    ax[1].set_xlabel("Temperature")
    ax[1].set_ylabel("Factor")
    ax[1].grid()


    plt.show()

