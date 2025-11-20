import numpy as np
import json

class SensorPositionCalculator:
    def __init__(self, config_file="sensor_positions.json"):
        """
        Initializes the SensorPositionCalculator by loading sensor positions from a JSON file.
        """
        self.config_file = config_file
        self.positions = self.load_sensor_positions()
        self.current_position = np.array([0, 0, 0])  # Default reference position

    def load_sensor_positions(self):
        """
        Loads the predefined sensor positions from a JSON file.

        Returns:
            np.array: The stored sensor positions.
        """
        try:
            with open(self.config_file, "r") as f:
                sensor_data = json.load(f)
            return np.array(sensor_data["positions"])
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_file}' not found!")
            return np.zeros((32, 3))  # Default empty positions if file is missing

    def set_current_position(self, x, y, z):
        """
        Sets the current position of the reference point.

        Args:
            x (float): X coordinate of the current position.
            y (float): Y coordinate of the current position.
            z (float): Z coordinate of the current position.
        """
        self.current_position = np.array([x, y, z])

    def calculate_absolute_positions(self, position):
        """
        Computes the absolute sensor positions by adding the current position.

        Returns:
            np.array: The absolute positions of all sensors.
        """
        return self.positions + position

    def save_absolute_positions(self, output_file="absolute_sensor_positions.json"):
        """
        Saves the computed absolute sensor positions to a JSON file.

        Args:
            output_file (str): Filename to save the absolute positions.
        """
        absolute_positions = self.calculate_absolute_positions().tolist()
        with open(output_file, "w") as f:
            json.dump({"absolute_positions": absolute_positions}, f, indent=4)
        print(f"Absolute sensor positions saved to '{output_file}'")

    def display_positions(self):
        """
        Prints the absolute sensor positions.
        """
        abs_positions = self.calculate_absolute_positions()
        print("Absolute Sensor Positions:")
        print(abs_positions)

# Example Usage
if __name__ == "__main__":
    sensor_calculator = SensorPositionCalculator("sensor_positions.json")
    sensor_calculator.set_current_position(0.1, -0.05, 0.02)  # Set reference position
    sensor_calculator.display_positions()  # Print absolute sensor positions
    sensor_calculator.save_absolute_positions()  # Save to a JSON file
