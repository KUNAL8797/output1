import yaml
import numpy as np
import os
from typing import Dict, Any, List, Tuple

# Import all necessary modules from the project structure
from parameters import Parameters
from model import Model
from solver import Solver
from data_loader import DataLoader
from analyzer import Analyzer
from plotter import Plotter

class Main:
    """
    The main execution script `main.py` serves as the entry point and orchestrator
    of the entire simulation workflow for the PEM electrolyzer model.
    It loads configuration, initializes modules, runs simulations for validation
    and sensitivity analyses, and generates plots.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the Main orchestrator by loading configuration and setting up
        all necessary simulation components.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # ...existing code...

        # Process all parameter sections, including solver_parameters
        for section in ['physical_constants', 'model_parameters', 'assumed_parameters', 'solver_parameters']:
            params = config.get(section, {})
            for key, value in params.items():
                if key not in self._parameters:
                    self._parameters[key] = value

# ...existing code...
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()

        # Print identified unclear points/assumptions from the configuration
        self._print_unclear_points()

        # Combine all parameter sections for a unified Parameters object
        all_parameters_dict = {}
        for section in ['physical_constants', 'model_parameters', 'assumed_parameters']:
            all_parameters_dict.update(self.config.get(section, {}))

        # Initialize core components
        self.params = Parameters("/Users/kunalnandeshwar/Developer/outputs/2021_repo/config.yaml")
        self.model = Model(self.params)
        self.solver = Solver(self.model, self.params)
        self.data_loader = DataLoader()
        self.analyzer = Analyzer(self.params)
        self.plotter = Plotter()

        # Define unit conversion constants for clarity in main
        self.UM_TO_M: float = 1e-6    # Micrometers to meters
        self.BAR_TO_PA: float = 1e5   # Bar to Pascals (for consistency with model.py's internal units)

        # Retrieve base operating conditions from config
        self.base_temp_K: float = self.config['operating_conditions']['base_temperature_K']
        self.base_anode_p_bar: float = self.config['operating_conditions']['base_anode_pressure_bar']
        self.base_cathode_p_bar: float = self.config['operating_conditions']['base_cathode_pressure_bar']
        self.base_d_mem_um: float = self.config['operating_conditions']['base_membrane_thickness_um']

        # Generate current density array for simulations
        j_range_cfg: Dict[str, float] = self.config['operating_conditions']['current_density_range_Am2']
        self.J_values_Am2: np.ndarray = np.linspace(
            j_range_cfg['start'], j_range_cfg['end'], int(j_range_cfg['num_points'])
        )

    def _load_config(self) -> Dict[str, Any]:
        """Loads the configuration from the specified YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _print_unclear_points(self) -> None:
        """Prints the identified unclear points and assumptions from the config."""
        print("--- Identified Unclear Points / Assumptions (from config.yaml) ---")
        unclear_points = self.config.get('unclear_points', {})
        if not unclear_points:
            print("No specific unclear points or assumptions documented in config.yaml.")
        else:
            for key, value in unclear_points.items():
                print(f"- {key.replace('_', ' ').capitalize()}: {value}")
        print("---------------------------------------------------\n")

    def run_simulation(self) -> None:
        """
        Orchestrates the entire simulation process:
        1. Runs validation cases against experimental data.
        2. Performs sensitivity analyses for key parameters.
        3. Generates plots for all scenarios.
        """
        print("Starting PEM Electrolyzer Model Simulation...\n")

        self._run_validation_cases()
        self._run_sensitivity_analyses()
        self._run_overpotential_breakdown_analysis()
        self._run_ohmic_loss_breakdown_analysis()

        print("\nSimulation complete. Plots saved to 'results/plots' directory.")

    def _run_validation_cases(self) -> None:
        """
        Executes simulations for defined validation cases and compares with
        experimental data.
        """
        print("--- Running Validation Cases ---")
        validation_cases_cfg = self.config.get('validation_cases', {})

        for case_key, case_details in validation_cases_cfg.items():
            case_name = case_details['name']
            print(f"\nValidating against: {case_name}")

            # Extract and convert case-specific parameters
            current_temp_K: float = case_details['temperature_K']
            current_d_mem_m: float = case_details['membrane_thickness_um'] * self.UM_TO_M
            current_anode_p_bar: float = case_details['anode_pressure_bar']
            current_cathode_p_bar: float = case_details['cathode_pressure_bar']
            data_file: str = case_details['data_file']
            expected_deviation_pct: float = case_details['expected_deviation_pct']

            try:
                # Load experimental data
                exp_data_df = self.data_loader.load_experimental_data(
                    os.path.join("data", data_file), case_name
                )
                exp_J = exp_data_df['J'].values
                exp_V = exp_data_df['V_exp'].values

                # Run model simulation for validation conditions
                model_results = self.solver.solve_polarization_curve(
                    self.J_values_Am2,
                    current_temp_K,
                    current_anode_p_bar,
                    current_cathode_p_bar,
                    current_d_mem_m
                )

                # Interpolate model results to experimental J values for deviation calculation
                model_V_at_exp_J = np.interp(
                    exp_J, model_results['J'], model_results['V_cell']
                )

                # Calculate deviation
                deviation_metrics = self.analyzer.calculate_deviation(model_V_at_exp_J, exp_V)
                max_abs_dev_pct = deviation_metrics['max_abs_deviation_pct']
                rmse = deviation_metrics['rmse']
                mape = deviation_metrics['mape']

                print(f"  Model Deviation: Max Abs. Pct. {max_abs_dev_pct:.2f}% (Expected <{expected_deviation_pct:.1f}%)")
                print(f"  RMSE: {rmse:.4f} V, MAPE: {mape:.2f}%")

                # Plot polarization curve
                plot_filename = f"{case_key}_polarization_curve.png"
                self.plotter.plot_polarization_curve(
                    model_results['J'], model_results['V_cell'], exp_data_df,
                    f"Polarization Curve: {case_name}", plot_filename
                )
                print(f"  Plot saved: {plot_filename}")

            except FileNotFoundError as e:
                print(f"  Error: {e}. Skipping validation for this case.")
            except ValueError as e:
                print(f"  Error processing data for '{case_name}': {e}. Skipping validation.")
            except Exception as e:
                print(f"  An unexpected error occurred for '{case_name}': {e}. Skipping validation.")
        print("--- Validation Cases Complete ---\n")

    def _run_sensitivity_analyses(self) -> None:
        """
        Executes simulations for various sensitivity analyses (membrane thickness,
        cathode pressure, temperature).
        """
        print("--- Running Sensitivity Analyses ---")
        sensitivity_ranges_cfg = self.config.get('sensitivity_analysis_ranges', {})

        # --- Sensitivity to Membrane Thickness (Figs. 3, 6a, 6b) ---
        print("\n- Analyzing sensitivity to Membrane Thickness...")
        d_mem_range_cfg = sensitivity_ranges_cfg['membrane_thickness_um']
        d_mem_values_um = np.linspace(
            d_mem_range_cfg['min'], d_mem_range_cfg['max'], int(d_mem_range_cfg['num_points'])
        )

        h2_crossover_by_d_mem: Dict[str, np.ndarray] = {}
        v_con_by_d_mem: Dict[str, np.ndarray] = {}
        v_ohmic_by_d_mem: Dict[str, np.ndarray] = {}

        for d_mem_um in d_mem_values_um:
            d_mem_m = d_mem_um * self.UM_TO_M
            label = f"{int(d_mem_um)} um"
            
            # Run simulation with varied d_mem, others at base conditions
            results = self.solver.solve_polarization_curve(
                self.J_values_Am2, self.base_temp_K, self.base_anode_p_bar,
                self.base_cathode_p_bar, d_mem_m
            )

            h2_crossover_by_d_mem[label] = self.analyzer.calculate_h2_crossover_percentage(
                results['N_en_H2'], results['J']
            )
            v_con_by_d_mem[label] = results['V_con']
            v_ohmic_by_d_mem[label] = results['V_ohmic']

        # Plotting Membrane Thickness Sensitivity
        self.plotter.plot_h2_crossover_sensitivity(
            self.J_values_Am2, h2_crossover_by_d_mem,
            "Impact of Membrane Thickness on Anodic Hydrogen Content (Fig. 3)",
            "Current density (A/cm$^2$)", "H$_2$ in O$_2$ (%)", "Membrane Thickness",
            "h2_crossover_membrane_thickness_sensitivity.png"
        )
        self.plotter.plot_concentration_ohmic_sensitivity(
            self.J_values_Am2, v_con_by_d_mem,
            "Membrane Thickness", "um", "Concentration Overpotential (V)",
            "Effect of Membrane Thickness on Concentration Overpotential (Fig. 6a)",
            "concentration_membrane_thickness_sensitivity.png"
        )
        self.plotter.plot_concentration_ohmic_sensitivity(
            self.J_values_Am2, v_ohmic_by_d_mem,
            "Membrane Thickness", "um", "Ohmic Overpotential (V)",
            "Effect of Membrane Thickness on Ohmic Overpotential (Fig. 6b)",
            "ohmic_membrane_thickness_sensitivity.png"
        )
        print("  Membrane Thickness sensitivity plots generated.")

        # --- Sensitivity to Cathode Pressure (Figs. 4, 7) ---
        print("\n- Analyzing sensitivity to Cathode Pressure...")
        p_cat_range_cfg = sensitivity_ranges_cfg['cathode_pressure_bar']
        p_cat_values_bar = np.linspace(
            p_cat_range_cfg['min'], p_cat_range_cfg['max'], int(p_cat_range_cfg['num_points'])
        )

        h2_crossover_by_p_cat: Dict[str, np.ndarray] = {}
        v_con_by_p_cat: Dict[str, np.ndarray] = {}

        for p_cat_bar in p_cat_values_bar:
            label = f"{int(p_cat_bar)} bar"
            # Run simulation with varied P_cat, others at base conditions
            results = self.solver.solve_polarization_curve(
                self.J_values_Am2, self.base_temp_K, self.base_anode_p_bar,
                p_cat_bar, self.base_d_mem_um * self.UM_TO_M
            )

            h2_crossover_by_p_cat[label] = self.analyzer.calculate_h2_crossover_percentage(
                results['N_en_H2'], results['J']
            )
            v_con_by_p_cat[label] = results['V_con']

        # Plotting Cathode Pressure Sensitivity
        self.plotter.plot_h2_crossover_sensitivity(
            self.J_values_Am2, h2_crossover_by_p_cat,
            "Impact of Cathode Pressure on Anodic Hydrogen Content (Fig. 4)",
            "Current density (A/cm$^2$)", "H$_2$ in O$_2$ (%)", "Cathode Pressure",
            "h2_crossover_cathode_pressure_sensitivity.png"
        )
        self.plotter.plot_concentration_ohmic_sensitivity(
            self.J_values_Am2, v_con_by_p_cat,
            "Cathode Pressure", "bar", "Concentration Overpotential (V)",
            "Effect of Cathode Pressure on Concentration Overpotential (Fig. 7)",
            "concentration_cathode_pressure_sensitivity.png"
        )
        print("  Cathode Pressure sensitivity plots generated.")

        # --- Sensitivity to Temperature (Fig. 5) ---
        print("\n- Analyzing sensitivity to Temperature...")
        temp_range_cfg = sensitivity_ranges_cfg['temperature_K']
        temp_values_K = np.linspace(
            temp_range_cfg['min'], temp_range_cfg['max'], int(temp_range_cfg['num_points'])
        )

        h2_crossover_by_temp: Dict[str, np.ndarray] = {}

        for temp_K in temp_values_K:
            label = f"{int(temp_K)} K"
            # Run simulation with varied T, others at base conditions
            results = self.solver.solve_polarization_curve(
                self.J_values_Am2, temp_K, self.base_anode_p_bar,
                self.base_cathode_p_bar, self.base_d_mem_um * self.UM_TO_M
            )

            h2_crossover_by_temp[label] = self.analyzer.calculate_h2_crossover_percentage(
                results['N_en_H2'], results['J']
            )

        # Plotting Temperature Sensitivity
        self.plotter.plot_h2_crossover_sensitivity(
            self.J_values_Am2, h2_crossover_by_temp,
            "Impact of Temperature on Anodic Hydrogen Content (Fig. 5)",
            "Current density (A/cm$^2$)", "H$_2$ in O$_2$ (%)", "Temperature",
            "h2_crossover_temperature_sensitivity.png"
        )
        print("  Temperature sensitivity plots generated.")
        print("--- Sensitivity Analyses Complete ---\n")

    def _run_overpotential_breakdown_analysis(self) -> None:
        """
        Analyzes and plots the contribution of different overpotentials to the
        total cell voltage (reproducing Fig. 9).
        """
        print("--- Running Overpotential Breakdown Analysis ---")
        # Use base operating conditions for this analysis as per paper's Fig. 9
        # (353 K, 1 bar cathode pressure, 50 mm membrane thickness, implicitly)
        # We will use the base parameters defined in config.yaml which are 353K, 1 bar, 100um.
        # To strictly reproduce Fig. 9 conditions from paper, we'd use d_mem = 50 um as stated in the Fig. 9 caption text.
        # Let's use 50um for this specific plot, overriding base d_mem.
        d_mem_for_plot_9 = 50.0 * self.UM_TO_M

        results = self.solver.solve_polarization_curve(
            self.J_values_Am2, self.base_temp_K, self.base_anode_p_bar,
            self.base_cathode_p_bar, d_mem_for_plot_9
        )

        overpotentials_breakdown = self.analyzer.calculate_overpotential_breakdown(
            results['V_rev'], results['V_act_anode'], results['V_act_cathode'],
            results['V_ohmic'], results['V_con']
        )

        self.plotter.plot_overpotential_breakdown(
            self.J_values_Am2, overpotentials_breakdown,
            "Contribution of Different Overpotentials (Fig. 9)",
            "overpotential_contributions.png"
        )
        print("  Overpotential contribution plot generated.")
        print("--- Overpotential Breakdown Analysis Complete ---\n")

    def _run_ohmic_loss_breakdown_analysis(self) -> None:
        """
        Analyzes and plots the breakdown of ohmic losses into electrode,
        bipolar plate, and membrane resistances (reproducing Fig. 8).
        """
        print("--- Running Ohmic Loss Breakdown Analysis ---")
        # Use base operating conditions for this analysis as per paper's Fig. 8 caption
        # (353 K, 1 bar, 50 mm membrane thickness, implicitly)
        d_mem_for_plot_8 = 50.0 * self.UM_TO_M

        # Re-run simulation to get individual resistance contributions if not already available for these specific conditions
        results_ohmic = self.solver.solve_polarization_curve(
            self.J_values_Am2, self.base_temp_K, self.base_anode_p_bar,
            self.base_cathode_p_bar, d_mem_for_plot_8
        )

        # Prepare data structure for plotting Ohmic breakdown, as plotter expects cumulative values
        # V_base_non_ohmic = V_rev + V_act_total + V_con
        # V_cumulative_el_bp = V_base_non_ohmic + V_ohmic_electrode + V_ohmic_bipolar_plate
        # V_cell_total = V_cumulative_el_bp + V_ohmic_membrane (This is the total V_cell)

        V_rev_local = results_ohmic['V_rev']
        V_act_total_local = results_ohmic['V_act_anode'] + results_ohmic['V_act_cathode']
        V_con_local = results_ohmic['V_con']
        
        V_base_non_ohmic = V_rev_local + V_act_total_local + V_con_local

        # Calculate individual ohmic losses for plotting
        ohmic_losses_component_breakdown = self.analyzer.calculate_ohmic_loss_breakdown(
            results_ohmic['J'], results_ohmic['R_electrode'], results_ohmic['R_membrane'], results_ohmic['R_bipolar_plate']
        )

        V_ohmic_electrode_local = ohmic_losses_component_breakdown['V_ohmic_electrode']
        V_ohmic_bipolar_plate_local = ohmic_losses_component_breakdown['V_ohmic_bipolar_plate']
        V_ohmic_membrane_local = ohmic_losses_component_breakdown['V_ohmic_membrane']

        V_cumulative_el_bp = V_base_non_ohmic + V_ohmic_electrode_local + V_ohmic_bipolar_plate_local
        V_cell_total = V_cumulative_el_bp + V_ohmic_membrane_local

        ohmic_plot_data = {
            'V_base_non_ohmic': V_base_non_ohmic,
            'V_cumulative_el_bp': V_cumulative_el_bp,
            'V_cell_total': V_cell_total
        }

        self.plotter.plot_ohmic_loss_breakdown(
            self.J_values_Am2, ohmic_plot_data,
            "Contribution of Ohmic Losses (Fig. 8)",
            "ohmic_loss_contributions.png"
        )
        print("  Ohmic loss contribution plot generated.")
        print("--- Ohmic Loss Breakdown Analysis Complete ---\n")


if __name__ == "__main__":
    main_app = Main()
    main_app.run_simulation()

