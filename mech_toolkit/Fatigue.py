from typing import Literal
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

def load_json(filename: str) -> dict:
    """
    Load a JSON file from the specified path.

    Parameters:
    filename (str): Name of the JSON file to load.

    Returns:
    dict: Parsed JSON data.
    """
    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

class Fatigue:

    @staticmethod
    def fatigue_stress(load_c:float, size_c:float, surface_c:float, temperature_c:float, reliability_c:float, endurance_limit:float) -> float:
        """
        Calculate the fatigue stress of a material given various correction factors.

        Parameters:
        load_c (float): Load correction factor.
        size_c (float): Size correction factor.
        surface_c (float): Surface finish correction factor.
        temperature_c (float): Temperature correction factor.
        reliability_c (float): Reliability correction factor.
        endurance_limit (float): Endurance limit of the material.

        Returns:
        float: The calculated fatigue stress.
        """
        return load_c * size_c * surface_c * temperature_c * reliability_c * endurance_limit
    
    @staticmethod
    def load_factor(load_type:str) -> float:
        """
        Calculate the load correction factor based on the type of load applied.

        Parameters:
        load_type (str): Type of load ('tension', 'compression', 'bending', 'torsion').

        Returns:
        float: The calculated load correction factor.
        """
        load_factors = load_json("load_factors.json")
        if load_type in load_factors:
            return load_factors[load_type]
        else:
            raise ValueError("Invalid load type. Choose from 'tension', 'compression', 'bending', or 'torsion'.")

    @staticmethod
    def size_factor(d:float=0,a_95:float=0) -> float:
        """
        Calculate the size correction factor based on the diameter of the component.

        Parameters:
        d (float) in mm: Diameter of the component.
        a_95 (float): Portion of cross-sectional area of a non-cylindrical part tensioned at 95% to 100% of maximum stress

        Returns:
        float: The calculated size correction factor.
        """
        if (d>0) and (a_95>0):
            raise ValueError("Choose either d (cylindrical component) or a_95(non-cylindrical component), not both.")
        if d >= 250:
            a_95,d=0.0766*d**2,0
        if d <= 0 and a_95 > 0:
            d = (a_95/0.0766)**0.5
        if d < 8:
            return 1.0
        elif 8 <= d < 250:
            return 1.189 * d**-0.097
        else:
            return 0.6   
            
        
    @staticmethod
    def surface_factor(surface_finish:str, ultimate_strength:float) -> float:
        """
        Calculate the surface finish correction factor based on the type of surface finish and the ultimate strength of the material.

        Parameters:
        surface_finish (str): Type of surface finish ('ground', 'machined', 'hot-rolled', 'as-forged').
        ultimate_strength (float) in MPa: Ultimate strength of the material.

        Returns:
        float: The calculated surface finish correction factor.
        """
        finish = load_json("surface_finish.json")
        if surface_finish in finish:
            a, b = finish[surface_finish]
            if a * ultimate_strength**b <1:
                return a * ultimate_strength**b
            else:
                return 1
        else:
            raise ValueError("Invalid surface finish type. Choose from 'ground', 'machined', 'hot-rolled', 'as-forged'.")

    @staticmethod        
    def temperature_factor(temperature:float) -> float:
        """
        Calculate the temperature correction factor based on the operating temperature of the steel.

        Parameters:
        temperature (float) in ºC: Operating temperature of the material in degrees Celsius.

        Returns:
        float: The calculated temperature correction factor.
        """
        if temperature <= -273.13:
            raise ValueError("Seriously? Below absolute zero?")
        elif temperature <=450:
            return 1.0
        elif 450 < temperature <= 550:
            return 1-0.0058*(temperature-450)
        else:
            return 1-0.0032*(temperature-840)
    
    @staticmethod
    def reliability_factor(reliability:float) -> float:
        """
        Calculate the reliability correction factor based on the desired reliability level.

        Parameters:
        reliability (float): Desired reliability level (e.g., 0.5 for 50%, 0.9 for 90%).

        Returns:
        float: The calculated reliability correction factor.
        """
        reliab = load_json("reliability.json")
        key = str(reliability)
        if key in reliab:
            return reliab[key]
        raise ValueError("Reliability must be one of the following values: 0.5, 0.9, 0.99, 0.999, or 0.9999.")

    @staticmethod
    def endurance_limit_factor(ultimate_strength:float, material:Literal['steel', 'aluminum', 'titanium']) -> float:
        """
        Calculate the endurance limit of a material based on its ultimate strength and type.

        Parameters:
        ultimate_strength (float): Ultimate strength of the material.
        material (str): Type of material ('steel', 'aluminum', 'titanium').

        Returns:
        float: The calculated endurance limit.
        """
        if material == 'steel':
            return 0.5 * ultimate_strength
        elif material == 'aluminum':
            return 0.4 * ultimate_strength
        elif material == 'titanium':
            return 0.3 * ultimate_strength
        else:
            raise ValueError("Invalid material type. Choose from 'steel', 'aluminum', or 'titanium'.")