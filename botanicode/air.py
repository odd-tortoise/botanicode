class Air:
    def __init__(self, temperature, water_concentration):
        self.temperature = temperature
        self.water_concentration = water_concentration

    def measure_temperature(self, point):
        return self.temperature + 0.5*point[2]
    
    def measure_water_concentration(self, point):
        return self.water_concentration