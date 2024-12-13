class Environment:
    def __init__(self, sky, soil,air):
        self.sky = sky
        self.soil = soil
        self.air = air

    def measure(self,point):
        if point[2]>0:
            data = {
                "light": self.sky.measure_light(point),
                "moisture": None,
                "temperature": self.air.measure_temperature(point),
                "water_concentration": self.air.measure_water_concentration(point)
            }
        else:
            data = {
                "light": None,
                "moisture": self.soil.measure_moisture(point),
                "temperature": None,
                "water_concentration": None
            }

        return data