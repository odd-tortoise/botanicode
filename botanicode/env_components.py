import numpy as np


# In these classes one can add more methods to measure the environmental variables
# for example one can integrate sensor readings to get the values of the variables

# Future:
# It can be nice to have the Air data coming from a weather API
# It can be nice to have the Air data computed with a diffussion model

class Air:
    """Class representing the air component of the environment."""
    
    def __init__(self, temperature: float, humidity: float) -> None:
        """
        Initialize the Air component.

        Args:
            temperature (float): The temperature of the air.
            humidity (float): The humidity of the air.
        """
        self.temperature = temperature
        self.humidity = humidity

    def measure_temperature(self, point: np.ndarray, t: float) -> float:
        """
        Measure the temperature at a given point and time.

        Args:
            point (np.ndarray): The point at which to measure.
            t (float): The time at which to measure.

        Returns:
            float: The measured temperature.
        """
        return self.temperature + 0.5 * point[2]
    
    def measure_humidity(self, point: np.ndarray, t: float) -> float:
        """
        Measure the humidity at a given point and time.

        Args:
            point (np.ndarray): The point at which to measure.
            t (float): The time at which to measure.

        Returns:
            float: The measured humidity.
        """
        return self.humidity

class Soil:
    """Class representing the soil component of the environment."""
    
    def __init__(self, moisture: float) -> None:
        """
        Initialize the Soil component.

        Args:
            moisture (float): The moisture level of the soil.
        """
        self.moisture = moisture

    def measure_moisture(self, point: np.ndarray, t: float) -> float:
        """
        Measure the moisture at a given point and time.

        Args:
            point (np.ndarray): The point at which to measure.
            t (float): The time at which to measure.

        Returns:
            float: The measured moisture.
        """
        return self.moisture

class Sky:
    """Class representing the sky component of the environment."""
    
    def __init__(self, light_intensity: float) -> None:
        """
        Initialize the Sky component.

        Args:
            light_intensity (float): The light intensity of the sky.
        """
        self.light_intensity = light_intensity

    def measure_light(self, point: np.ndarray, t: float) -> float:
        """
        Measure the light intensity at a given point and time.

        Args:
            point (np.ndarray): The point at which to measure.
            t (float): The time at which to measure.

        Returns:
            float: The measured light intensity.
        """
        return self.light_intensity