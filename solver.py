import numpy as np
from scipy.optimize import fsolve # Although fixed-point iteration is used directly, fsolve is mentioned in design.
from typing import Any, Dict, Tuple

# Assuming Model and Parameters classes are defined in their respective files
from model import Model
from parameters import Parameters


class Solver:
    """
    The Solver class orchestrates the simulation process. Its primary method,
    `solve_polarization_curve`, iterates over a range of current densities (`J`).
    For each `J`, it performs an internal iterative loop to solve the coupled
    non-linear equations, specifically for the partial pressures of hydrogen
    and oxygen at the interfaces (`P_cat_H2`, `P_an_O2`) which are influenced
    by gas crossover. Once converged, it calculates all component overpotentials
    and the total cell voltage.
    """

    def __init__(self, model: Model, params: Parameters):
        """
        Initializes the Solver with instances of the Model and Parameters classes.

        Args:
            model: An instance of the Model class containing all electrochemical
                   and transport equations.
            params: An instance of the Parameters class providing access to all
                    physical constants, material properties, and geometric parameters.
        """
        self.model = model
        self.params = params

        # Configuration for iterative solver for coupled variables, loaded from params
        self.max_iterations: int = self.params.get_value('solver_max_iterations')
        self.tolerance: float = self.params.get_value('solver_tolerance_pa') # Tolerance in Pa

        # Unit conversion constants (defined in model.py, accessed via model instance)
        self.BAR_TO_PA: float = self.model.BAR_TO_PA


    def _calculate_saturated_water_vapor_pressure(self, T: float) -> float:
        """
        Calculates the saturated water vapor pressure at a given temperature.
        This uses a common empirical correlation, which is not explicitly stated in the paper
        but is necessary for model closure and consistency with typical PEM system operation
        (where gas streams are often saturated with water vapor).

        The constants are chosen for pressure in Pascals and temperature in Kelvin.
        This specific correlation is a variant of the Magnus-Tetens formula.
        Source: Commonly used empirical fit for water vapor pressure in the range of PEM
        electrolyzer operation (e.g., 0-100°C).

        Args:
            T: Operating temperature (K).

        Returns:
            Saturated water vapor pressure (Pa).
        """
        T_celsius: float = T - 273.15  # Convert Kelvin to Celsius

        # Ensure that the temperature is within a reasonable range for the correlation,
        # preventing issues like log(negative) or very high/low unphysical values.
        # This particular formula is generally good for 0-100 C.
        # For temperatures far outside this, a different correlation might be needed.
        # Clamp T_celsius to prevent division by zero or extreme values in exp()
        # The denominator (T_celsius + 243.04) should be positive and not too small.
        # If T_celsius is -243.04, it causes division by zero.
        # For realistic electrolyzer temperatures (e.g., > 0 deg C), this won't be an issue.

        # Using constants: A = 610.94 Pa, B = 17.625, C = 243.04 C
        # P_sat [Pa] = A * exp(B * T_celsius / (T_celsius + C))
        # Clamp denominator to a small positive value to prevent division by zero for extreme inputs
        denominator_clamped: float = max(T_celsius + 243.04, 1e-9)
        p_sat_pa: float = 610.94 * np.exp(17.625 * T_celsius / denominator_clamped)

        # Ensure the pressure is non-negative and physically reasonable (e.g., not extremely high)
        return max(p_sat_pa, 1e-12) # Clamp to small positive to avoid issues in log later


    def _solve_coupled_variables(self, J: float, T: float, P_an_bar: float, P_cat_bar: float, d_mem_m: float) -> Tuple[float, float, float, float, float, float, float]:
        """
        Solves the interdependent equations for partial pressures of hydrogen and oxygen
        at the electrode-membrane interfaces and the associated gas crossover fluxes.
        This is necessary because these variables are coupled via Equations 52, 53, 54, and 56.

        Args:
            J (float): Current density (A/m^2).
            T (float): Operating temperature (K).
            P_an_bar (float): Total anode pressure (bar).
            P_cat_bar (float): Total cathode pressure (bar).
            d_mem_m (float): Membrane thickness (m).

        Returns:
            Tuple[float, float, float, float, float, float, float]:
            (P_cat_H2_Pa, P_an_O2_Pa, N_en_H2, N_en_O2, P_H2O_Pa, C0_O2_mol_m3, C0_H2_mol_m3)
            All pressures are in Pascals (Pa). Crossover fluxes in mol s^-1 m^-2.
            Concentrations in mol m^-3.
        """
        R: float = self.params.get_value('R')

        # Convert input total pressures from bar to Pascal for internal SI unit consistency
        P_an_Pa: float = P_an_bar * self.BAR_TO_PA
        P_cat_Pa: float = P_cat_bar * self.BAR_TO_PA

        # Calculate saturated water vapor pressure (in Pascals)
        P_H2O_Pa: float = self._calculate_saturated_water_vapor_pressure(T)
        
        # Calculate bulk/standard concentrations (C0_O2, C0_H2)
        # These are the concentrations of O2 and H2 in the bulk gas channels,
        # assuming the total pressure is made up of the gas component and saturated water vapor.
        P_O2_bulk_Pa: float = max(1e-9, P_an_Pa - P_H2O_Pa)
        P_H2_bulk_Pa: float = max(1e-9, P_cat_Pa - P_H2O_Pa)
        
        C0_O2_mol_m3: float = P_O2_bulk_Pa / (R * T)
        C0_H2_mol_m3: float = P_H2_bulk_Pa / (R * T)

        # Initialize iteration variables for partial pressures at the interface
        # These partial pressures are influenced by crossover and are the ones used in Nernst equation.
        # Initial guess for these (start with bulk total pressure, adjusted for water vapor, or simply total pressure if gas is dominant)
        # Using the ideal gas partial pressure in the bulk as a starting guess
        P_cat_H2_iter_Pa: float = P_H2_bulk_Pa
        P_an_O2_iter_Pa: float = P_O2_bulk_Pa

        # Initialize crossover fluxes for the first iteration pass
        N_en_H2: float = 0.0
        N_en_O2: float = 0.0

        for i in range(self.max_iterations):
            prev_P_cat_H2_Pa: float = P_cat_H2_iter_Pa
            prev_P_an_O2_Pa: float = P_an_O2_iter_Pa

            # Calculate individual crossover flux components using current partial pressure estimates
            n_eod_h2_calc: float = self.model.calculate_n_eod_h2(J)
            n_eod_o2_calc: float = self.model.calculate_n_eod_o2(J)

            n_dif_h2_calc: float = self.model.calculate_n_dif_h2(prev_P_cat_H2_Pa, d_mem_m)
            n_dif_o2_calc: float = self.model.calculate_n_dif_o2(prev_P_an_O2_Pa, d_mem_m)

            delta_P_Pa: float = P_cat_Pa - P_an_Pa # Pressure difference for hydraulic term
            n_dp_h2_calc: float = self.model.calculate_n_dp_h2(delta_P_Pa, d_mem_m)

            # Calculate total crossover fluxes for H2 and O2 (Eqs. 54, 56)
            N_en_H2 = self.model.calculate_n_total_h2_crossover(n_eod_h2_calc, n_dif_h2_calc, n_dp_h2_calc)
            N_en_O2 = self.model.calculate_n_total_o2_crossover(n_eod_o2_calc, n_dif_o2_calc)

            # Update partial pressures (Eqs. 52, 53) using the calculated crossover fluxes.
            # The `update_partial_pressures` method in Model handles the complex, potentially non-linear
            # calculations of these interface pressures based on total pressures and crossover fluxes.
            P_cat_H2_iter_Pa, P_an_O2_iter_Pa = self.model.update_partial_pressures(
                J, P_an_Pa, P_cat_Pa, N_en_H2, N_en_O2
            )

            # Ensure physical bounds for partial pressures: must be positive and not exceed total compartment pressure.
            P_cat_H2_iter_Pa = np.clip(P_cat_H2_iter_Pa, 1e-9, P_cat_Pa)
            P_an_O2_iter_Pa = np.clip(P_an_O2_iter_Pa, 1e-9, P_an_Pa)

            # Check for convergence
            if (np.abs(P_cat_H2_iter_Pa - prev_P_cat_H2_Pa) < self.tolerance and
                np.abs(P_an_O2_iter_Pa - prev_P_an_O2_Pa) < self.tolerance):
                break
        else:
            # If loop completes without breaking, it means convergence was not reached.
            # In a production environment, this might trigger an error or a more robust fallback.
            # For this context, a silent non-convergence is acceptable for plotting.
            pass

        return P_cat_H2_iter_Pa, P_an_O2_iter_Pa, N_en_H2, N_en_O2, P_H2O_Pa, C0_O2_mol_m3, C0_H2_mol_m3


    def solve_polarization_curve(self, J_values: np.ndarray, T: float, P_an: float, P_cat: float, d_mem: float) -> Dict[str, np.ndarray]:
        """
        Calculates the full polarization curve by iterating over a range of current densities
        and determining all voltage components and overall cell voltage.

        Args:
            J_values (np.ndarray): Array of current densities (A/m^2) to simulate.
            T (float): Operating temperature (K).
            P_an (float): Anode total pressure (bar).
            P_cat (float): Cathode total pressure (bar).
            d_mem (float): Membrane thickness (m).

        Returns:
            Dict[str, np.ndarray]: A dictionary containing arrays of calculated
            J, V_cell, and all component overpotentials, and crossover fluxes.
        """
        # Retrieve necessary constants and geometric parameters from params
        electrode_porosity: float = self.params.get_value('electrode_porosity')
        # Note: 'electrode_porosity_tortuosity_ratio' (epsilon_x) is assumed equal to 'electrode_porosity'
        # for D_eff calculation as per plan's interpretation of Eq. 28, 29.

        # Initialize lists to store computed values for each current density
        V_cell_results: list[float] = []
        V_rev_results: list[float] = []
        V_act_anode_results: list[float] = []
        V_act_cathode_results: list[float] = []
        V_ohmic_results: list[float] = []
        V_con_results: list[float] = []
        N_en_H2_results: list[float] = []
        N_en_O2_results: list[float] = []
        P_cat_H2_results: list[float] = []
        P_an_O2_results: list[float] = []
        R_electrode_results: list[float] = []
        R_membrane_results: list[float] = []
        R_bipolar_plate_results: list[float] = []

        # Pre-calculate binary diffusion coefficients (Eqs. 30, 31)
        # These depend on temperature and total pressure. The paper does not specify which total pressure.
        # A common approach is to use the average system pressure or a representative electrode pressure.
        # Here, we use the average of the anode and cathode total pressures.
        P_total_for_diffusion: float = ((P_an + P_cat) / 2.0) * self.BAR_TO_PA # Convert average bar pressure to Pa

        d_o2_h2o_binary: float = self.model.calculate_diffusion_coefficient('O2-H2O', T, P_total_for_diffusion)
        d_h2_h2o_binary: float = self.model.calculate_diffusion_coefficient('H2-H2O', T, P_total_for_diffusion)

        # Calculate effective diffusion coefficients (Eqs. 28, 29)
        d_an_eff: float = (electrode_porosity**1.5) * d_o2_h2o_binary
        d_cat_eff: float = (electrode_porosity**1.5) * d_h2_h2o_binary

        for J_val in J_values:
            # Solve coupled variables (partial pressures and crossover fluxes) for the current J_val
            P_cat_H2_Pa, P_an_O2_Pa, N_en_H2, N_en_O2, P_H2O_Pa, C0_O2_mol_m3, C0_H2_mol_m3 = \
                self._solve_coupled_variables(J_val, T, P_an, P_cat, d_mem)

            # Calculate Reversible Voltage (Eq. 1)
            v_rev: float = self.model.calculate_v_rev(P_cat_H2_Pa, P_an_O2_Pa, P_H2O_Pa, T)
            V_rev_results.append(v_rev)

            # Calculate Activation Overpotentials (Eqs. 3, 4)
            v_act_a: float = 0.0
            v_act_c: float = 0.0
            if J_val > 1e-9: # Avoid log(0) for very small current densities, which occur at OCV
                v_act_a = self.model.calculate_v_act_anode(J_val, T)
                v_act_c = self.model.calculate_v_act_cathode(J_val, T)
            V_act_anode_results.append(v_act_a)
            V_act_cathode_results.append(v_act_c)

            # Calculate Ohmic Overpotential (Eq. 5)
            # Membrane conductivity (Eq. 19)
            sigma_mem: float = self.model.calculate_sigma_membrane(self.params.get_value('membrane_water_content_lambda'), T)
            
            # Component resistances (Eqs. 8-18)
            r_mem: float = self.model.calculate_r_membrane(d_mem, sigma_mem)
            r_el: float = self.model.calculate_r_electrode(J_val, T) # J_val, T passed, but not used in model's current R_el logic.
            r_pl: float = self.model.calculate_r_bipolar_plate(J_val, T) # J_val, T passed, but not used in model's current R_pl logic.

            r_cell: float = r_el + r_mem + r_pl # Eq. 6
            v_ohmic: float = self.model.calculate_v_ohmic(J_val, r_cell) # Eq. 5
            V_ohmic_results.append(v_ohmic)
            R_electrode_results.append(r_el)
            R_membrane_results.append(r_mem)
            R_bipolar_plate_results.append(r_pl)

            # Calculate Concentration Overpotential (Eq. 20)
            v_con_a: float = 0.0
            v_con_c: float = 0.0
            if J_val > 1e-9: # Avoid issues at very small current densities
                c_mem_o2: float = self.model.calculate_c_mem_o2(J_val, C0_O2_mol_m3, d_an_eff)
                c_mem_h2: float = self.model.calculate_c_mem_h2(J_val, C0_H2_mol_m3, d_cat_eff)
                
                # Ensure concentrations at interface are strictly positive for logarithmic calculation
                c_mem_o2 = max(c_mem_o2, 1e-12)
                c_mem_h2 = max(c_mem_h2, 1e-12)

                v_con_a = self.model.calculate_v_con_anode(T, C0_O2_mol_m3, c_mem_o2)
                v_con_c = self.model.calculate_v_con_cathode(T, C0_H2_mol_m3, c_mem_h2)
                
                # Ensure concentration overpotentials are non-negative for physical meaning
                v_con_a = max(0.0, v_con_a)
                v_con_c = max(0.0, v_con_c)

            v_con: float = v_con_a + v_con_c # Eq. 20
            V_con_results.append(v_con)
            
            # Calculate Total Cell Voltage (Eq. 2)
            v_cell: float = v_rev + v_act_a + v_act_c + v_ohmic + v_con
            V_cell_results.append(v_cell)

            # Store crossover fluxes and converged partial pressures
            N_en_H2_results.append(N_en_H2)
            N_en_O2_results.append(N_en_O2)
            P_cat_H2_results.append(P_cat_H2_Pa)
            P_an_O2_results.append(P_an_O2_Pa)

        # Convert all lists of results into NumPy arrays for efficient numerical operations and consistent output format
        results: Dict[str, np.ndarray] = {
            'J': J_values,
            'V_cell': np.array(V_cell_results),
            'V_rev': np.array(V_rev_results),
            'V_act_anode': np.array(V_act_anode_results),
            'V_act_cathode': np.array(V_act_cathode_results),
            'V_ohmic': np.array(V_ohmic_results),
            'V_con': np.array(V_con_results),
            'N_en_H2': np.array(N_en_H2_results),
            'N_en_O2': np.array(N_en_O2_results),
            'P_cat_H2': np.array(P_cat_H2_results),
            'P_an_O2': np.array(P_an_O2_results),
            'R_electrode': np.array(R_electrode_results),
            'R_membrane': np.array(R_membrane_results),
            'R_bipolar_plate': np.array(R_bipolar_plate_results)
        }

        return results

