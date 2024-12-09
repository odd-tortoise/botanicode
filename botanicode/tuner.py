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

        # leaf data
        self.leaf_y_angle = data['leaves_paramters']['leaf_y_angle']
        

        # growth data
        self.growth_data = data['growth_data']

        for key in self.growth_data.keys():
            self.growth_data[key] = self.read_csv(self.growth_data[key])


        
        # provisional
        self.length_to_shoot = 4


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