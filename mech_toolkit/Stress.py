import math
import numpy as np
import matplotlib.pyplot as plt

class Stress:

    @staticmethod
    def calculate_stress(s_x: float, s_y: float, t_xy: float, show_values: bool = False) -> tuple:
        """
        Calculate the average normal stress, maximum shear stress, principal stresses, and principal plane angle.
        Parameters:
        s_x (float): Normal stress in the x-direction.
        s_y (float): Normal stress in the y-direction.
        t_xy (float): Shear stress in the xy-plane.
        show_values (bool): If True, prints the calculated values.
        Returns:
        tuple: A tuple containing the average normal stress, maximum shear stress, principal stresses, and principal plane angle.
        """
        if not isinstance(s_x, (int, float)) or not isinstance(s_y, (int, float)) or not isinstance(t_xy, (int, float)):
            raise TypeError("s_x, s_y, and t_xy must be numeric values.")
        if s_x == 0 and s_y == 0 and t_xy == 0:
            raise ValueError("Seriously? All stresses are zero? What are you trying to calculate?")
        avg_stress = (s_x + s_y) / 2
        radius=math.sqrt(((s_x - s_y) / 2) ** 2 + t_xy ** 2)
        stress_1 = avg_stress + radius
        stress_2 = avg_stress - radius

        theta_rad=math.atan2(2 * t_xy, s_x - s_y)
        theta_deg=math.degrees(theta_rad) / 2

        principal_points = [(stress_1,0.), (stress_2, 0.)] # This will be used for a plotting feature later
    

        if show_values:
            print(f"\nAverage normal stress: {avg_stress:.2f}")
            print(f"Maximum shear stress: {radius:.2f}")
            print(f"Principal stresses:\n\tσ1 = {stress_1:.2f}\n\tσ2 = {stress_2:.2f}")
            print(f"Principal plane angle: {theta_deg:.2f}°\n")

        return (avg_stress, radius, stress_1, stress_2, theta_deg)