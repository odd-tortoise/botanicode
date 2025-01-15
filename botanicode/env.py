class Environment:
    def __init__(self, sky, soil,air):
        self.sky = sky
        self.soil = soil
        self.air = air

        self.clock = None 

    def set_clock(self, clock):
        self.clock = clock

    def measure(self,point, var):
        if var == "light":
            return self.sky.measure_light(point)
        elif var == "temp":
            return self.air.measure_temperature(point)
        elif var == "humidity":
            return self.air.measure_water_concentration(point)
        elif var == "water":
            return self.soil.measure_moisture(point)
        else:
            raise ValueError("Variable not recognized.")