class Air:
    def __init__(self, temperature, humidity):
        self.temperature = temperature
        self.humidity = humidity

    def measure_temperature(self, point,t):
        return self.temperature + 0.5*point[2]
    
    def measure_humidity(self, point,t):
        return self.humidity