import math
from typing import List, Tuple
class Beam_Calculator:
    """
    Initialy for 2D beam with only one load applied.

    """
    def __init__(self,lenght: float, load: float, load_position: float, load_angle: float = 90, support_A: str = "fixed", support_B: str = "roller", support_A_position: float = 0, support_B_position: float = None):
        self.lenght = lenght
        self.load = load
        self.load_position = load_position
        self.load_angle = math.radians(load_angle)
        self.support_A = support_A
        self.support_B = support_B
        self.support_A_position = support_A_position
        self.support_B_position = support_B_position if support_B_position is not None else lenght
        self.reactions = self.calculate_reactions()

    def calculate_reactions(self) -> Tuple[float, float, float, float]:
        """
        Calculate the reactions at the supports.
        Returns:
            Tuple containing reactions (R_A_x, R_A_y, R_B_y, R_B_x).
        """
        Px = self.load * math.cos(self.load_angle)
        Py = self.load * math.sin(self.load_angle)
        Py_abs = abs(Py)

        L = self.support_B_position - self.support_A_position
        a = self.load_position - self.support_A_position
        d_left = self.support_A_position - self.load_position  # distance from load to support A
        d_right = self.load_position - self.support_B_position  # distance from load to support B

        # Identify if both supports are bearings (pin, roller, bearing)
        bearings = (self.support_A in ("pin", "roller", "bearing")) and (self.support_B in ("pin", "roller", "bearing"))

        if bearings:
            # Static determinate case
            if self.load_position < self.support_A_position:
                # Left of the span
                RBy = Py_abs * d_left / L
                RAy = Py_abs - RBy
                RBx, RAx = 0.0, 0.0
                
            elif self.load_position > self.support_B_position:
                # Right of the span
                RAy = Py_abs * d_right / L
                RBy = Py_abs - RAy
                RBx, RAx = 0.0, 0.0

            else:
                # Load within the span
                RBy = Py_abs * a / L
                RAy = Py_abs - RBy
                RBx, RAx = 0.0, 0.0

            return (RAx, RAy, RBx, RBy)

        # General case with at least one fixed support
        if self.load_position < self.support_A_position:
            RAy = Py_abs
            RBy = 0.0
            RAx = -Px if self.support_A == "fixed" else 0.0
            RBx = -Px if self.support_B == "fixed" else 0.0
            
            
        elif self.load_position > self.support_B_position:
            RAy = 0.0
            RBy = Py_abs
            RAx = -Px if self.support_A == "fixed" else 0.0
            RBx = -Px if self.support_B == "fixed" else 0.0
            
        else:
            RBy = -Py * a / L
            RAy = -Py - RBy
            RAx = -Px if self.support_A == "fixed" else 0.0
            RBx = -Px if self.support_B == "fixed" else 0.0
            

        return (RAx, RAy, RBx, RBy)

    def __str__(self) -> str:
        RAx, RAy, RBx, RBy = self.reactions
        return (f"Beam Length: {self.lenght} m\n"
                f"Load: {self.load} N at {self.load_position} m with angle {math.degrees(self.load_angle)}°\n"
                f"Support A: {self.support_A} at {self.support_A_position} m\n"
                f"Support B: {self.support_B} at {self.support_B_position} m\n"
                f"Reactions:\n"
                f"  R_A_x: {RAx:.2f} N\n"
                f"  R_A_y: {RAy:.2f} N\n"
                f"  R_B_x: {RBx:.2f} N\n"
                f"  R_B_y: {RBy:.2f} N\n")