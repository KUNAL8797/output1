import numpy as np
import pandas as pd
from typing import Dict, Any, Union

# Assuming Parameters class is available for import from the parameters.py module.
# This relative import assumes the file structure where analyzer.py is in the same
# directory level as parameters.py.
from parameters import Parameters 

class Analyzer:
    """
    The Analyzer class performs post-processing and analysis of the simulation results.
    It includes methods for quantifying deviations between model and experimental data,
    calculating hydrogen crossover percentages, and breaking down overpotential
    and ohmic loss contributions.
    """
    def __init__(self, params: Parameters):
        """
        Initializes the Analyzer with a Parameters object to access global constants
        and configuration settings.

        Args:
            params (Parameters): An instance of the Parameters class containing all
                                 model constants, material properties, and
                                 geometric parameters loaded from config.yaml.
        """
        self.params = params
        self.F: float = self.params.get_value("F") # Faraday constant in C mol^-1

    def calculate_deviation(self, model_data: np.ndarray, exp_data: np.ndarray) -> Dict[str, float]:
        """
        Quantifies the difference between model predictions (e.g., cell voltage)
        and experimental data. It calculates Maximum Absolute Percentage Deviation (MAPD),
        Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).

        Args:
            model_data (np.ndarray): NumPy array of model-predicted values (e.g., cell voltage).
            exp_data (np.ndarray): NumPy array of experimental values (e.g., cell voltage).

        Returns:
            Dict[str, float]: A dictionary containing calculated deviation metrics:
                              'max_abs_deviation_pct', 'rmse', 'mape'.
        """
        # Ensure input arrays are not empty
        if model_data.size == 0 or exp_data.size == 0:
            return {
                "max_abs_deviation_pct": 0.0,
                "rmse": 0.0,
                "mape": 0.0
            }
        
        # Ensure arrays have the same shape for element-wise operations
        if model_data.shape != exp_data.shape:
            raise ValueError(
                f"Model data shape {model_data.shape} does not match "
                f"experimental data shape {exp_data.shape}."
            )

        # Filter out zero or near-zero experimental values for percentage calculations
        # to prevent division by zero errors. A small epsilon is used to avoid issues
        # with floating-point comparisons to zero.
        epsilon: float = 1e-9 # A small value to consider 'zero' for division purposes
        non_zero_exp_indices = np.abs(exp_data) > epsilon
        
        # If no valid experimental points for percentage calculation, return 0 for percentage errors
        if not np.any(non_zero_exp_indices):
            rmse = np.sqrt(np.mean((model_data - exp_data)**2)) # RMSE is always calculable
            return {
                "max_abs_deviation_pct": 0.0,
                "rmse": rmse,
                "mape": 0.0
            }

        exp_data_filtered = exp_data[non_zero_exp_indices]
        model_data_filtered = model_data[non_zero_exp_indices]

        # Calculate absolute differences for all points (used for RMSE)
        abs_diff_all = np.abs(model_data - exp_data)
        
        # Calculate absolute differences for filtered points (used for percentage errors)
        abs_diff_filtered = np.abs(model_data_filtered - exp_data_filtered)

        # Calculate Root Mean Square Error (RMSE)
        rmse: float = np.sqrt(np.mean(abs_diff_all**2))

        # Calculate Maximum Absolute Percentage Deviation (MAPD)
        # np.divide handles division by zero by returning np.inf or np.nan, then np.max handles it.
        # However, filtering handles this more explicitly for robustness.
        max_abs_deviation_pct: float = np.max(abs_diff_filtered / exp_data_filtered) * 100.0

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape: float = np.mean(abs_diff_filtered / exp_data_filtered) * 100.0

        return {
            "max_abs_deviation_pct": max_abs_deviation_pct,
            "rmse": rmse,
            "mape": mape
        }

    def calculate_h2_crossover_percentage(self, N_en_H2: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
        Calculates the percentage of hydrogen content at the anode side due to gas crossover,
        as defined by Equation (57) in the paper.

        Equation (57): H2 in O2 (%) = (N_en_H2 / (J/(4F) + N_en_H2)) * 100

        Logic:
        1.  The term J/(4F) represents the stoichiometric molar flux of oxygen produced at the anode
            based on Faraday's law, assuming 4 electrons per O2 molecule.
        2.  The denominator (J/(4F) + N_en_H2) thus represents the total molar flux of gases
            at the anode side (produced oxygen plus crossover hydrogen).
        3.  Handles division by zero by identifying where the denominator is zero or very small.
            For these points, the crossover percentage is set to 0.0 to ensure numerical stability
            and physical realism (e.g., if there's no gas flow at all).
        4.  Clips the final percentage between 0 and 100 to ensure physical bounds.

        Args:
            N_en_H2 (np.ndarray): Total hydrogen flux density crossing over to the anode (mol m^-2 s^-1).
                                  This is the output of Model.calculate_n_total_h2_crossover.
            J (np.ndarray): Current density (A m^-2).

        Returns:
            np.ndarray: Percentage of hydrogen in the oxygen stream at the anode (%).
        """
        # Calculate the molar flux of oxygen produced at the anode (mol m^-2 s^-1)
        # J_over_4F can be close to zero or negative if J is. Clamp J to ensure non-negative current.
        J_clamped = np.maximum(J, 0.0)
        J_over_4F: np.ndarray = J_clamped / (4 * self.F)

        # Calculate the total molar flux in the denominator
        denominator: np.ndarray = J_over_4F + N_en_H2

        # Initialize the result array with zeros
        h2_crossover_pct: np.ndarray = np.zeros_like(J, dtype=float)

        # Identify valid indices where denominator is not zero or extremely close to zero
        # Use a small epsilon to avoid floating point issues.
        valid_indices = np.abs(denominator) > 1e-15

        # Calculate percentage only for valid indices
        # Ensure N_en_H2 is also non-negative, as it's a crossover flux.
        N_en_H2_clamped = np.maximum(N_en_H2, 0.0) # Crossover flux should be non-negative

        h2_crossover_pct[valid_indices] = (N_en_H2_clamped[valid_indices] / denominator[valid_indices]) * 100.0

        # Ensure the percentage is within physically meaningful bounds [0, 100]
        h2_crossover_pct = np.clip(h2_crossover_pct, 0.0, 100.0)

        return h2_crossover_pct

    def calculate_overpotential_breakdown(
        self, V_rev: np.ndarray, V_act_a: np.ndarray, V_act_c: np.ndarray,
        V_ohmic: np.ndarray, V_con: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Breaks down the total cell voltage into its contributing components:
        reversible voltage, total activation overpotential, ohmic overpotential,
        and concentration overpotential.

        This method returns the absolute voltage values for each component over the J range.
        The plotting function will use these arrays to visualize their contributions.

        Args:
            V_rev (np.ndarray): Reversible cell voltage (V).
            V_act_a (np.ndarray): Anode activation overpotential (V).
            V_act_c (np.ndarray): Cathode activation overpotential (V).
            V_ohmic (np.ndarray): Ohmic overpotential (V).
            V_con (np.ndarray): Concentration overpotential (V).

        Returns:
            Dict[str, np.ndarray]: A dictionary containing NumPy arrays for each voltage component:
                                   'V_rev', 'V_act_total', 'V_ohmic', 'V_con'.
        """
        # Calculate the total activation overpotential by summing anode and cathode contributions
        V_act_total: np.ndarray = V_act_a + V_act_c

        return {
            "V_rev": V_rev,
            "V_act_total": V_act_total,
            "V_ohmic": V_ohmic,
            "V_con": V_con
        }

    def calculate_ohmic_loss_breakdown(
        self, J: np.ndarray, R_el: np.ndarray, R_mem: np.ndarray, R_pl: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the individual ohmic voltage losses due to electrode, membrane,
        and bipolar plate resistances. These losses are computed using Ohm's law (V = J * R).

        This method returns the absolute ohmic loss values for each component over the J range.
        The plotting function will use these arrays to visualize their contributions.

        Args:
            J (np.ndarray): Current density (A m^-2).
            R_el (np.ndarray): Total electrode resistance (Ohm m^2).
            R_mem (np.ndarray): Membrane resistance (Ohm m^2).
            R_pl (np.ndarray): Bipolar plate resistance (Ohm m^2).

        Returns:
            Dict[str, np.ndarray]: A dictionary containing NumPy arrays for each ohmic loss component:
                                   'V_ohmic_electrode', 'V_ohmic_membrane', 'V_ohmic_bipolar_plate'.
        """
        # Ensure current density is non-negative for ohmic loss calculation.
        # Negative current density is not physically meaningful in this context for power consumption.
        J_clamped = np.maximum(J, 0.0)

        # Calculate individual ohmic voltage drops using Ohm's law (V = J * R)
        V_ohmic_electrode: np.ndarray = J_clamped * R_el
        V_ohmic_membrane: np.ndarray = J_clamped * R_mem
        V_ohmic_bipolar_plate: np.ndarray = J_clamped * R_pl

        return {
            "V_ohmic_electrode": V_ohmic_electrode,
            "V_ohmic_membrane": V_ohmic_membrane,
            "V_ohmic_bipolar_plate": V_ohmic_bipolar_plate
        }

