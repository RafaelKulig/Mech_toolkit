import numpy as np
import matplotlib.pyplot as plt

class Mohr_Circle:
    """
    Class to create and plot Mohr's Circle for 2D stress analysis.
    Attributes:
        sigma_x (float): Normal stress in the x-direction.
        sigma_y (float): Normal stress in the y-direction.
        tau_xy (float): Shear stress.
    """
    def __init__(self, sigma_x: float, sigma_y: float, tau_xy: float):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.tau_xy = tau_xy
        self.center = (sigma_x + sigma_y) / 2
        self.radius = np.sqrt(((sigma_x - sigma_y) / 2) ** 2 + tau_xy ** 2)
        self.theta_p = 0.5 * np.arctan2(2 * tau_xy, sigma_x - sigma_y)  # in radians

    def principal_stresses(self) -> tuple:
        """
        Calculate the principal stresses.
        Returns:
            Tuple containing principal stresses (sigma_1, sigma_2).
        """
        sigma_1 = self.center + self.radius
        sigma_2 = self.center - self.radius
        return (sigma_1, sigma_2)
    
    def plot(self):
        """
        Plot Mohr's Circle.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        circle = plt.Circle((self.center, 0), self.radius, color='b', fill=False, label="Mohr's Circle")  # type: ignore
        ax.add_artist(circle)

        # Plot principal stresses
        sigma_1, sigma_2 = self.principal_stresses()
        ax.plot([sigma_1, sigma_2], [0, 0], 'go', label='Principal Stresses')

        # Plot original stress state
        ax.plot([self.sigma_x, self.sigma_y], [self.tau_xy, -self.tau_xy], 'ro--', label='Original Stress State')

        # Axes settings
        ax.set_xlim(self.center - self.radius - 10, self.center + self.radius + 10)
        ax.set_ylim(-self.radius - 10, self.radius + 10)
        ax.set_xlabel('Normal Stress (σ)')
        ax.set_ylabel('Shear Stress (τ)')
        ax.axhline(0, color='black',linewidth=0.5, ls='--')
        ax.axvline(0, color='black',linewidth=0.5, ls='--')
        ax.set_aspect('equal', adjustable='box')
        ax.grid()
        ax.legend()
        plt.title("Mohr's Circle")
        plt.show()
    
    def __str__(self) -> str:
        sigma_1, sigma_2 = self.principal_stresses()
        theta_p_deg = np.degrees(self.theta_p)
        return (f"Principal Stress σ1: {sigma_1:.2f}, Principal Stress σ2: {sigma_2:.2f}, "
                f"Angle of Principal Planes: {theta_p_deg:.2f} degrees")