import json
import numpy as np
from light import Sky
from soil import Soil
from air import Air


class Environment:
    def __init__(self, sky = None, soil = None, air = None):
        self.sky = sky
        self.soil = soil
        self.air = air

    def measure(self,point, var,t):

        if var == "light":
            return self.sky.measure_light(point, t)
        elif var == "temp":
            return self.air.measure_temperature(point,t)
        elif var == "humidity":
            return self.air.measure_humidity(point,t)
        elif var == "water":
            return self.soil.measure_moisture(point,t)
        else:
            raise ValueError("Variable not recognized.")
        
    def set_env(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        
        self.sky = Sky(data["sky"]["PAR"])
        self.soil = Soil(data["soil"]["moisture"])
        self.air = Air(data["air"]["temperature"],data["air"]["humidity"])

        return self