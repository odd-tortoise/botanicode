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

        # stem data
        self.new_stem_lenght = data['stem_parameters']['new_stem_lenght']
        self.new_stem_radius = data['stem_parameters']['new_stem_radius']

        # root data
        self.new_root_lenght = data['root_parameters']['new_root_lenght']
        self.new_root_radius = data['root_parameters']['new_root_radius']
        

        # growth data
        self.growth_data = data['growth_data']
        
        # initial data
        self.initial_stem_lenght = data['initial_data']['initial_stem_lenght']
        self.initial_stem_radius = data['initial_data']['initial_stem_radius']
        self.initial_leaf_number = data['initial_data']['initial_leaf_number']
        self.initial_leaflets_number = data['initial_data']['initial_leaflets_number']
        self.initial_root_lenght = data['initial_data']['initial_root_lenght']
        self.initial_root_radius = data['initial_data']['initial_root_radius']

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

    print(gr)