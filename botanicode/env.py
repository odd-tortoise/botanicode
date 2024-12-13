class Environment:
    def __init__(self, sky, soil):
        self.sky = sky
        self.soil = soil

    def get_sky(self):
        return self.sky

    def get_soil(self):
        return self.soil
