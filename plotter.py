import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from typing import Dict, Any, Union, List, Tuple

class Plotter:
    """
    The Plotter class handles all data visualization. It implements methods to
    reproduce all figures presented in the paper, ensuring appropriate labels,
    units, and legends for clarity and comparison. All current densities on
    plots will be converted to A/cm^2 to match the paper's figures.
    """

    def __init__(self, output_dir: str = "results/plots"):
        """
        Initializes the Plotter instance, setting up the output directory
        for saving figures and configuring default matplotlib parameters.

        Args:
            output_dir (str): The directory where generated plots will be saved.
                              Defaults to "results/plots".
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure default matplotlib parameters for consistent plot aesthetics
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'figure.figsize': (8, 6),
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.75
        })

    def _convert_j_to_acm2(self, J_Am2: np.ndarray) -> np.ndarray:
        """
        A private helper method to convert current density values from A/m^2
        (internal unit) to A/cm^2 (plotting unit as per paper figures).

        Args:
            J_Am2 (np.ndarray): Current density values in A/m^2.

        Returns:
            np.ndarray: Current density values in A/cm^2.
        """
        return J_Am2 / 10000.0

    def _extract_numeric_value(self, s: str) -> float:
        """
        A private helper method used to extract a numeric value from string
        labels (e.g., "50 um", "10 bar", "333 K") for proper sorting of
        sensitivity curves in legends.

        Args:
            s (str): The input string label.

        Returns:
            float: The first numeric value found in the string. Defaults to 0.0
                   if no number is found.
        """
        match: Union[re.Match[str], None] = re.search(r"[-+]?\d*\.?\d+", s)
        if match:
            return float(match.group(0))
        return 0.0 # Default value if no number is found

    def plot_polarization_curve(
        self,
        model_J_Am2: np.ndarray,
        model_V: np.ndarray,
        exp_data: pd.DataFrame,
        title: str,
        filename: str
    ) -> None:
        """
        Reproduces the polarization curve validation figures (Fig. 2a and 2b).

        Args:
            model_J_Am2 (np.ndarray): Modeled current densities (in A/m^2).
            model_V (np.ndarray): Modeled cell voltages (in V).
            exp_data (pd.DataFrame): Experimental data containing at least 'J'
                                     (Current Density in A/m^2) and 'V_exp'
                                     (Experimental Cell Voltage in V) columns.
            title (str): The title for the plot.
            filename (str): The filename (e.g., "ioroi_polarization.png") to save the plot.
        """
        fig, ax = plt.subplots()

        # Convert model current density to A/cm^2 for plotting
        model_J_Acm2: np.ndarray = self._convert_j_to_acm2(model_J_Am2)

        # Convert experimental current density to A/cm^2 for plotting, assuming it's in A/m^2
        exp_J_Acm2: np.ndarray = self._convert_j_to_acm2(exp_data['J'].values)
        
        ax.plot(model_J_Acm2, model_V, label="Model", color='blue')
        ax.plot(exp_J_Acm2, exp_data['V_exp'].values, 'o', label="Experimental", color='red', markersize=5)

        ax.set_xlabel("Current density (A/cm$^2$)")
        ax.set_ylabel("Cell voltage (V)")
        ax.set_title(title)
        ax.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_h2_crossover_sensitivity(
        self,
        J_values_Am2: np.ndarray,
        h2_crossover_data_map: Dict[str, np.ndarray],
        title: str,
        xlabel: str, # This should always be "Current density (A/cm$^2$)"
        ylabel: str, # This should always be "H$_2$ in O$_2$ (%)"
        legend_title: str, # Added for clarity, as this indicates the varying parameter
        filename: str
    ) -> None:
        """
        Visualizes the percentage of hydrogen crossover as a function of current
        density for varying parameters (membrane thickness, cathode pressure, temperature).
        This method is designed to reproduce Figs. 3, 4, and 5.

        Args:
            J_values_Am2 (np.ndarray): Current densities (in A/m^2).
            h2_crossover_data_map (Dict[str, np.ndarray]): Keys are string labels
                                   for the varying parameter values (e.g., "50 um",
                                   "1 bar", "333 K"), and values are NumPy arrays
                                   of the corresponding H2 crossover percentages.
            title (str): The title for the plot.
            xlabel (str): The label for the x-axis (e.g., "Current density (A/cm$^2$)").
            ylabel (str): The label for the y-axis (e.g., "H$_2$ in O$_2$ (%)").
            legend_title (str): Title for the legend (e.g., "Membrane Thickness",
                                "Cathode Pressure", "Temperature").
            filename (str): The filename to save the plot.
        """
        fig, ax = plt.subplots()

        J_values_Acm2: np.ndarray = self._convert_j_to_acm2(J_values_Am2)

        # Sort the keys of the data map based on their numeric value for consistent legend ordering
        sorted_labels: List[str] = sorted(h2_crossover_data_map.keys(), key=self._extract_numeric_value)

        for label in sorted_labels:
            ax.plot(J_values_Acm2, h2_crossover_data_map[label], label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title=legend_title, loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_overpotential_breakdown(
        self,
        J_values_Am2: np.ndarray,
        overpotentials: Dict[str, np.ndarray],
        title: str,
        filename: str
    ) -> None:
        """
        Illustrates the cumulative contribution of different overpotentials
        to the total cell voltage. Reproduces Fig. 9.

        Args:
            J_values_Am2 (np.ndarray): Current densities (in A/m^2).
            overpotentials (Dict[str, np.ndarray]): Contains NumPy arrays for each
                                   overpotential component. Expected keys include
                                   'V_rev', 'V_act_total', 'V_ohmic', and 'V_con'.
            title (str): The title for the plot.
            filename (str): The filename to save the plot.
        """
        fig, ax = plt.subplots()

        J_values_Acm2: np.ndarray = self._convert_j_to_acm2(J_values_Am2)

        # Retrieve components, defaulting to zeros if missing (should not happen with proper Analyzer output)
        V_rev: np.ndarray = overpotentials.get('V_rev', np.zeros_like(J_values_Am2))
        V_act_total: np.ndarray = overpotentials.get('V_act_total', np.zeros_like(J_values_Am2))
        V_ohmic: np.ndarray = overpotentials.get('V_ohmic', np.zeros_like(J_values_Am2))
        V_con: np.ndarray = overpotentials.get('V_con', np.zeros_like(J_values_Am2))

        # Calculate cumulative voltage curves
        V_base_rev: np.ndarray = V_rev
        V_cumulative_act: np.ndarray = V_base_rev + V_act_total
        V_cumulative_ohmic: np.ndarray = V_cumulative_act + V_ohmic
        V_cumulative_con: np.ndarray = V_cumulative_ohmic + V_con # This is the total V_cell

        ax.plot(J_values_Acm2, V_base_rev, label="Open Circuit Voltage", linestyle='--', color='brown')
        ax.plot(J_values_Acm2, V_cumulative_act, label="Activation Overpotential Added", color='purple')
        ax.plot(J_values_Acm2, V_cumulative_ohmic, label="Ohmic Overpotential Added", color='orange')
        ax.plot(J_values_Acm2, V_cumulative_con, label="Concentration Overpotential Added (Final Voltage)", color='blue')

        ax.set_xlabel("Current density (A/cm$^2$)")
        ax.set_ylabel("Cell voltage (V)")
        ax.set_title(title)
        ax.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_ohmic_loss_breakdown(
        self,
        J_values_Am2: np.ndarray,
        ohmic_losses_breakdown: Dict[str, np.ndarray],
        title: str,
        filename: str
    ) -> None:
        """
        Shows the cumulative contribution of ohmic losses from electrodes,
        bipolar plates, and membrane to the total cell voltage. Reproduces Fig. 8.

        Args:
            J_values_Am2 (np.ndarray): Current densities (in A/m^2).
            ohmic_losses_breakdown (Dict[str, np.ndarray]): This dictionary
                                   should contain pre-calculated cumulative
                                   voltage arrays representing the ohmic loss
                                   breakdown. Expected keys:
                                   'V_base_non_ohmic': V_rev + V_act_total + V_con
                                   'V_cumulative_el_bp': V_base_non_ohmic + V_ohmic_electrode + V_ohmic_bipolar_plate
                                   'V_cell_total': V_cumulative_el_bp + V_ohmic_membrane (This is the total V_cell)
            title (str): The title for the plot.
            filename (str): The filename to save the plot.
        """
        fig, ax = plt.subplots()

        J_values_Acm2: np.ndarray = self._convert_j_to_acm2(J_values_Am2)

        # Retrieve components from the pre-calculated breakdown, defaulting to zeros
        V_base_non_ohmic: np.ndarray = ohmic_losses_breakdown.get('V_base_non_ohmic', np.zeros_like(J_values_Am2))
        V_cumulative_el_bp: np.ndarray = ohmic_losses_breakdown.get('V_cumulative_el_bp', np.zeros_like(J_values_Am2))
        V_cell_total: np.ndarray = ohmic_losses_breakdown.get('V_cell_total', np.zeros_like(J_values_Am2))

        ax.plot(J_values_Acm2, V_base_non_ohmic, label="Without ohmic loss", color='green')
        ax.plot(J_values_Acm2, V_cumulative_el_bp, label="Electrodes & BP ohmic loss added", color='red')
        ax.plot(J_values_Acm2, V_cell_total, label="Membrane ohmic loss added (Total Cell Voltage)", color='blue')

        ax.set_xlabel("Current density (A/cm$^2$)")
        ax.set_ylabel("Cell voltage (V)")
        ax.set_title(title)
        ax.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_concentration_ohmic_sensitivity(
        self,
        J_values_Am2: np.ndarray,
        data_map: Dict[str, np.ndarray],
        varying_param_label: str,
        value_unit: str,
        y_axis_label: str,
        plot_title: str,
        filename: str
    ) -> None:
        """
        Displays the sensitivity of either concentration overpotential (Fig. 6a, Fig. 7)
        or ohmic overpotential (Fig. 6b) to varying parameters (membrane thickness,
        cathode pressure). This method is designed to plot multiple lines on a single graph.

        Args:
            J_values_Am2 (np.ndarray): Current densities (in A/m^2).
            data_map (Dict[str, np.ndarray]): Keys are string labels for the varying
                                   parameter values (e.g., "50 um", "1 bar"), and
                                   values are NumPy arrays of the corresponding overpotential
                                   (V_con or V_ohmic).
            varying_param_label (str): The descriptive label for the parameter being varied
                                       (e.g., "Membrane Thickness", "Cathode Pressure").
                                       Used for legend title.
            value_unit (str): The unit of the varying parameter (e.g., "um", "bar").
                              Used in legend labels.
            y_axis_label (str): The label for the y-axis (e.g., "Concentration Overpotential (V)",
                                "Ohmic Overpotential (V)").
            plot_title (str): The title for the plot.
            filename (str): The filename to save the plot.
        """
        fig, ax = plt.subplots()

        J_values_Acm2: np.ndarray = self._convert_j_to_acm2(J_values_Am2)

        # Sort the keys of the data map based on their numeric value for consistent legend ordering
        sorted_labels: List[str] = sorted(data_map.keys(), key=self._extract_numeric_value)

        for label in sorted_labels:
            ax.plot(J_values_Acm2, data_map[label], label=f"{label} {value_unit}")

        ax.set_xlabel("Current density (A/cm$^2$)")
        ax.set_ylabel(y_axis_label)
        ax.set_title(plot_title)
        ax.legend(title=varying_param_label, loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

