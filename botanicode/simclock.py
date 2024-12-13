class SimClock:
    def __init__(self, start_time=0):
        self.elapsed_time = start_time
        self.photo_period = 12
        self.current_time = start_time

    def tick(self, dt):
        self.elapsed_time += dt

    def is_day(self):
        return self.current_time % 24 < self.photo_period