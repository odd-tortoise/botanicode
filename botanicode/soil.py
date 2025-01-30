class Soil:
    def __init__(self, moisture = 0.5):
        self.moisture = moisture

    def measure_moisture(self, point, t):
        return self.moisture