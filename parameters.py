import yaml
from typing import Any, Dict

class Parameters:
    """
    Centralized class for managing all model constants, material properties,
    and geometric parameters loaded from a configuration file.

    It handles unit conversions to ensure internal consistency (SI units)
    and provides methods to access individual or all parameters.
    """

    def __init__(self, config_path: str):
        """
        Initializes the Parameters object by loading and processing parameters
        from the specified YAML configuration file.

        Args:
            config_path: The file path to the config.yaml file.
        """
        # ...existing code...

        # Process Assumed Parameters
        assumed_params = config.get('assumed_parameters', {})
        for key, value in assumed_params.items():
            self._parameters[key] = value

        # Process Solver Parameters (add this block)
        solver_params = config.get('solver_parameters', {})
        for key, value in solver_params.items():
            self._parameters[key] = value

        # Calculate active_surface_area after mea_width and mea_length are processed
        # These are assumed to be in meters from config.yaml.
        mea_width = self._parameters.get('mea_width', 0.0)
        mea_length = self._parameters.get('mea_length', 0.0)
        self._parameters['active_surface_area'] = mea_width * mea_length

# ...existing code...
        self._parameters: Dict[str, Any] = {}
        self._unclear_points: Dict[str, str] = {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")

        # Process Physical Constants
        # These are assumed to be correctly defined in SI units in config.yaml
        self._parameters.update(config.get('physical_constants', {}))

        # Process Model Parameters (from Paper's Table 1)
        model_params = config.get('model_parameters', {})
        
        # Apply specific unit conversions as identified in the plan (bar to Pa conversions)
        # For values already pre-converted in config.yaml (e.g., A/cm^2 to A/m^2),
        # they are directly assigned.
        self._parameters['h2_solubility'] = model_params.get('h2_solubility', 0.0) / 1e5  # mol m^-3 bar^-1 to mol m^-3 Pa^-1
        self._parameters['o2_solubility'] = model_params.get('o2_solubility', 0.0) / 1e5  # mol m^-3 bar^-1 to mol m^-3 Pa^-1
        self._parameters['h2_partial_pressure_enhancement_factor'] = model_params.get('h2_partial_pressure_enhancement_factor', 0.0) / 1e5  # bar m^2 A^-1 to Pa m^2 A^-1
        self._parameters['o2_partial_pressure_enhancement_factor'] = model_params.get('o2_partial_pressure_enhancement_factor', 0.0) / 1e5  # bar m^2 A^-1 to Pa m^2 A^-1
        self._parameters['h2_diffusion_constant_conc_diff'] = model_params.get('h2_diffusion_constant_conc_diff', 0.0) / 1e5  # mol m^-1 s^-1 bar^-1 to mol m^-1 s^-1 Pa^-1
        self._parameters['o2_diffusion_constant_conc_diff'] = model_params.get('o2_diffusion_constant_conc_diff', 0.0) / 1e5  # mol m^-1 s^-1 bar^-1 to mol m^-1 s^-1 Pa^-1
        self._parameters['h2_diffusion_constant_press_diff'] = model_params.get('h2_diffusion_constant_press_diff', 0.0) / 1e5  # mol m^-1 s^-1 bar^-1 to mol m^-1 s^-1 Pa^-1

        # Add all other model parameters directly. They are assumed to be either already in SI units
        # in config.yaml (e.g., A/m^2, Ohm m, m, J/mol, K) or dimensionless.
        for key, value in model_params.items():
            if key not in self._parameters: # Only add if not already handled by explicit conversion above
                self._parameters[key] = value
            # Strong type default values (though config is expected complete)
            # Example for robustness, if config was truly missing values:
            # if key == 'anode_charge_transfer_coefficient' and key not in self._parameters: self._parameters[key] = 0.8
            # ... and so on for other keys if they could be genuinely missing.

        # Process Assumed Parameters
        assumed_params = config.get('assumed_parameters', {})

        # All assumed parameters are copied directly. They are assumed to be either already in SI units
        # in config.yaml (e.g., m, m^2/s, kg/mol, Pa s) or dimensionless.
        for key, value in assumed_params.items():
            self._parameters[key] = value
            # Strong type default values (though config is expected complete)
            # Example: if key == 'electrode_porosity' and key not in self._parameters: self._parameters[key] = 0.5
            # ... and so on for other keys.

        # Calculate active_surface_area after mea_width and mea_length are processed
        # These are assumed to be in meters from config.yaml.
        mea_width = self._parameters.get('mea_width', 0.0)
        mea_length = self._parameters.get('mea_length', 0.0)
        self._parameters['active_surface_area'] = mea_width * mea_length

        # Store Unclear Points for documentation and reference in README.md
        self._unclear_points = config.get('unclear_points', {})

        # Explicitly note the J0,i temperature dependence assumption as per detailed plan
        self._parameters['J0_i_temperature_dependence_assumption'] = (
            "The J_0,an and J_0,cat values are treated as fixed reference exchange current densities "
            "at the typical operating temperature (353 K), as no explicit Arrhenius-type equation "
            "for their temperature dependence is provided alongside the activation overpotential "
            "equations (Eqs. 3 & 4). The k_0,i and E_act,i values are noted but not currently "
            "utilized for J_0,i calculation based on the paper's direct use of J_0,i in formulas."
        )

    def get_value(self, param_name: str) -> Any:
        """
        Retrieves the value of a specific parameter.

        Args:
            param_name: The name of the parameter to retrieve.

        Returns:
            The value of the parameter.

        Raises:
            KeyError: If the parameter name is not found.
        """
        if param_name not in self._parameters:
            raise KeyError(f"Parameter '{param_name}' not found in the loaded configuration.")
        return self._parameters[param_name]

    def get_all(self) -> Dict[str, Any]:
        """
        Returns a copy of all loaded and processed parameters.

        Returns:
            A dictionary containing all parameters.
        """
        return self._parameters.copy()

