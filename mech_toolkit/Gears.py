from math import pi
class Gears:

    @staticmethod
    def speed_ratio(Z_g:int,Z_p:int) -> float:
        """
        Calculate the speed ratio between gear and pinion.

        Parameters:
        Z_g (int): Number of teeth on the gear.
        Z_p (int): Number of teeth on the pinion.

        Returns:
        float: The gear mesh frequency.
        """
        if (Z_p == 0) or (Z_g == 0):
            raise ValueError("Number of teeth must be greater than zero.")
        if not isinstance(Z_g, int) or not isinstance(Z_p, int):
            raise TypeError("Number of teeth must be integers.")
        if Z_g < 0 or Z_p < 0:
            raise ValueError("Number of teeth must be non-negative.")
        if (Z_g%2==0) and (Z_p%2==0):
            print("Warning: Both gear and pinion have even number of teeth, which may lead to resonance issues.")
        return (Z_g / Z_p)
    
    @staticmethod
    def gear_mesh_frequency(Z_g:int, RPM:int) -> float:
        """
        Calculate the gear mesh frequency.

        Parameters:
        Z_g (int): Number of teeth on the gear.
        RPM (int): Rotational speed in revolutions per minute.

        Returns:
        float: The gear mesh frequency in Hz.
        """
        if Z_g <= 0 or RPM < 0:
            raise ValueError("Number of teeth must be greater than zero and RPM must be non-negative.")
        if not isinstance(Z_g, int) or not isinstance(RPM, int):
            raise TypeError("Number of teeth and RPM must be integers.")
        
        return (Z_g * RPM) / 60.0
    
    @staticmethod
    def circular_pitch(diameter:float, Z:int) -> float:
        """
        Calculate the circular pitch of a gear.

        Parameters:
        diameter (float): Pitch diameter of the gear.
        Z (int): Number of teeth on the gear.

        Returns:
        float: The circular pitch in meters.
        """
        if diameter <= 0 or Z <= 0:
            raise ValueError("Diameter and number of teeth must be greater than zero.")
        if not isinstance(diameter, (int, float)) or not isinstance(Z, int):
            raise TypeError("Diameter must be a number and number of teeth must be an integer.")
        if Z % 2 == 0:
            print("Warning: Gear has an even number of teeth, which may lead to resonance issues.")
        return (pi * diameter) / Z
    
    @staticmethod
    def pitch_diameter(module:float, Z:int) -> float:
        """
        Calculate the pitch diameter of a gear.

        Parameters:
        module (float): Module of the gear.
        Z (int): Number of teeth on the gear.

        Returns:
        float: The pitch diameter in meters.
        """
        if module <= 0 or Z <= 0:
            raise ValueError("Module and number of teeth must be greater than zero.")
        if not isinstance(module, (int, float)) or not isinstance(Z, int):
            raise TypeError("Module must be a number and number of teeth must be an integer.")
        if Z % 2 == 0:
            print("Warning: Gear has an even number of teeth, which may lead to resonance issues.")
        return module * Z
    
    @staticmethod
    def addendum(diameter:float, Z:int) -> float:
        """
        Calculate the addendum of a gear.

        Parameters:
        diameter (float): Pitch diameter of the gear.
        Z (int): Number of teeth on the gear.

        Returns:
        float: The addendum in meters.
        """
        if diameter <= 0 or Z <= 0:
            raise ValueError("Diameter and number of teeth must be greater than zero.")
        if not isinstance(diameter, (int, float)) or not isinstance(Z, int):
            raise TypeError("Diameter must be a number and number of teeth must be an integer.")
        if Z % 2 == 0:
            print("Warning: Gear has an even number of teeth, which may lead to resonance issues.")
        return diameter / Z
    