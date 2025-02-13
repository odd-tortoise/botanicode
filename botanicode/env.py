
import numpy as np
from env_components import Air, Soil, Sky

from typing import Tuple

class Environment:
    """Class representing the environment with sky, soil, and air components."""

    def __init__(self, sky : Sky, soil : Soil, air : Air):
        """
        Initialize the Environment.

        Args:
            sky (Optional[Sky]): The sky component.
            soil (Optional[Soil]): The soil component.
            air (Optional[Air]): The air component.
        """
        self.sky = sky
        self.soil = soil
        self.air = air

    def measure(self, point: np.ndarray, var: str, t: float) -> float:
        """
        Measure an environmental variable at a given point and time.

        Args:
            point (np.ndarray): The point at which to measure.
            var (str): The variable to measure ("light", "temp", "humidity", "water").
            t (float): The time at which to measure.

        Returns:
            float: The measured value.

        Raises:
            ValueError: If the variable is not recognized.
        """
        if var == "light":
            return self.sky.measure_light(point, t)
        elif var == "temp":
            return self.air.measure_temperature(point, t)
        elif var == "humidity":
            return self.air.measure_humidity(point, t)
        elif var == "water":
            return self.soil.measure_moisture(point, t)
        else:
            raise ValueError("Variable not recognized.")

        



class Clock:
    """Class representing the simulation clock."""
    
    def __init__(self, photo_period: Tuple[int, int] = (8, 18), step: str = "hour") -> None:
        """
        Initialize the simulation clock.

        Args:
            photo_period (Tuple[int, int]): Tuple defining the start and end of the daylight period in hours.
            step (str): Simulation step mode, either "hour" or "day".
        """
        self.elapsed_time: float = 0  # Total elapsed time in hours always
        self.photo_period: Tuple[int, int] = photo_period
        self.step: str = step  # Step mode: "hour" or "day"

    
    def tick(self, dt: float) -> None:
        """
        Advance the clock by a given time step.

        :param dt: Time increment (in hours if step="hour", or days if step="day").
        """
        if self.step == "hour":
            self.elapsed_time += dt
        elif self.step == "day":
            self.elapsed_time += dt * 24
        else:
            raise ValueError("Unsupported step mode. Use 'hour' or 'day'.")
        

    def get_hour(self) -> float:
        """
        Get the current hour of the day.

        Returns:
            int: Integer representing the current hour (0-23).

        Raises:
            ValueError: If the step mode is not "hour".
        """
        if self.step == "hour":
            return self.elapsed_time % 24
        else:
            raise ValueError("Use hour step mode to get info on the current hour.")
    
    def get_day(self) -> float:
        """
        Get the current day of the simulation.

        Returns:
            int: Integer representing the current day.

        Raises:
            ValueError: If the step mode is not "hour" or "day".
        """
        if self.step in ["hour", "day"]:
            return self.elapsed_time // 24
        else:
            raise ValueError("Unsupported step mode. Use 'hour' or 'day'.")
        

    def is_day(self) -> bool:
        """
        Check if it's currently daytime based on the photo period.

        Returns:
            bool: Boolean indicating if it's currently day.

        Raises:
            ValueError: If the step mode is not "hour".
        """
        if self.step == "hour":
            current_hour = self.get_hour()
            return self.photo_period[0] <= current_hour < self.photo_period[1]
        else:
            raise ValueError("Use hour step mode to get info on day/night.")
        

    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time in hours since the simulation started.

        Returns:
            float: Float representing the elapsed time in hours.
        """
        if self.step == "hour":
            return self.elapsed_time
        elif self.step == "day":
            return self.elapsed_time / 24
        else:
            raise ValueError("Unsupported step mode. Use 'hour' or 'day'.")