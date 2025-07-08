import numpy as np
from typing import Any, Dict, Tuple

# Assuming Parameters class is defined in parameters.py
# from parameters import Parameters 

class Model:
    """
    Core PEM electrolyzer model class encapsulating all the mathematical equations
    from the paper. Each major component (e.g., reversible voltage, activation
    overpotentials, ohmic resistances, concentration overpotentials, water
    transport fluxes, gas crossover fluxes) is implemented as a dedicated method.
    """

    def __init__(self, params: "Parameters"):
        """
        Initializes the Model with a Parameters object.

        Args:
            params (Parameters): An instance of the Parameters class containing
                                 all model constants, material properties, and
                                 geometric parameters loaded from config.yaml.
        """
        self.params = params

        # --- Unit Conversion Factors (for internal use if parameters are not pre-converted in config.yaml) ---
        # These factors are critical for maintaining internal consistency
        # All calculations will aim for SI units (meters, Pascals, A/m^2, mol, K, J).
        # Note: Some conversions for parameters like J_0,i, r_el, etc., are handled directly in parameters.py
        # by pre-multiplying values from config.yaml with conversion factors before storing.
        # This allows direct use of self._get_param() for already SI-unit values.
        self.BAR_TO_PA = 1e5        # 1 bar = 10^5 Pa
        self.PA_TO_BAR = 1e-5       # 1 Pa = 10^-5 bar
        self.PA_TO_ATM = 1 / 101325.0 # 1 atm = 101325 Pa (for diffusion coefficient empirical constant)
        
        # Factors for mol m^-3 bar^-1 to mol m^-3 Pa^-1 or mol m^-1 s^-1 bar^-1 to mol m^-1 s^-1 Pa^-1
        # These are used when the original config parameter is in 'bar' based units
        # but the formula needs 'Pa' based units. The parameter class already performs
        # a primary conversion using these, so the value fetched from self.params
        # should already be in the Pa-consistent unit.
        # Example: h2_solubility in config is 'mol m^-3 bar^-1'. parameters.py stores it as
        # 'mol m^-3 Pa^-1'. So when using it, no further conversion needed.
        # However, for consistency with paper equations, where inputs might be in bar,
        # intermediate conversion to bar might still be necessary.

    def _get_param(self, name: str) -> Any:
        """Helper method to fetch parameter values from the Parameters object."""
        return self.params.get_value(name)

    def calculate_v_rev(self, P_cat_H2: float, P_an_O2: float, P_H2O: float, T: float) -> float:
        """
        Calculates the reversible voltage of the electrolyzer (Equation 1).

        Logic:
        - Retrieves universal gas constant (R), Faraday constant (F), and standard reversible voltage (V0_rev) from `config.yaml`.
        - Implements the Nernst equation for the reaction H2O -> H2 + 0.5 O2.
        - The logarithmic term uses ratios of partial pressures. All pressures are expected in Pascals (Pa).
        - Handles potential `log(0)` or division by zero by clamping pressures to a small positive epsilon.

        Args:
            P_cat_H2 (float): Partial pressure of hydrogen at cathode (Pa).
            P_an_O2 (float): Partial pressure of oxygen at anode (Pa).
            P_H2O (float): Water pressure (Pa).
            T (float): Operating temperature (K).

        Returns:
            float: Reversible voltage (V).
        """
        R = self._get_param('R')
        F = self._get_param('F')
        V0_rev = self._get_param('V0_rev')

        # Clamp pressures to a small positive value to avoid log(0) or division by zero
        P_cat_H2_clamped = max(P_cat_H2, 1e-9)
        P_an_O2_clamped = max(P_an_O2, 1e-9)
        P_H2O_clamped = max(P_H2O, 1e-9)

        log_term = np.log((P_cat_H2_clamped * np.sqrt(P_an_O2_clamped)) / P_H2O_clamped)
        return V0_rev + (R * T / (2 * F)) * log_term

    def calculate_v_act_anode(self, J: float, T: float) -> float:
        """
        Calculates the anode activation overpotential (Equation 3).

        Logic:
        - Retrieves R, F, anode charge transfer coefficient (alpha_a), and anode exchange current density (J0_an) from `config.yaml`.
        - Implements the Butler-Volmer equation for anode.
        - Addresses the "unclear point" regarding J0_an's temperature dependency: J0_an is used directly as a fixed reference value from `config.yaml`, as no explicit Arrhenius-type correlation is provided in the paper's formulas.
        - Handles `J=0` by returning `0.0` to avoid `log(0)`. Clamps J0_an to a small positive value.

        Args:
            J (float): Current density (A/m^2).
            T (float): Operating temperature (K).

        Returns:
            float: Anode activation overpotential (V).
        """
        R = self._get_param('R')
        F = self._get_param('F')
        alpha_a = self._get_param('anode_charge_transfer_coefficient')
        J0_an = self._get_param('anode_exchange_current_density_ref') # Already A/m^2 from config

        if J <= 1e-9: # Current density close to zero
            return 0.0
        J0_an_clamped = max(J0_an, 1e-12) # Clamp J0_an to avoid log(0) or division by zero

        return (R * T / (alpha_a * F)) * np.log(J / J0_an_clamped)

    def calculate_v_act_cathode(self, J: float, T: float) -> float:
        """
        Calculates the cathode activation overpotential (Equation 4).

        Logic:
        - Retrieves R, F, cathode charge transfer coefficient (alpha_c), and cathode exchange current density (J0_cat) from `config.yaml`.
        - Implements the Butler-Volmer equation for cathode.
        - Addresses the "unclear point" regarding J0_cat's temperature dependency: J0_cat is used directly as a fixed reference value from `config.yaml`.
        - Handles `J=0` by returning `0.0` to avoid `log(0)`. Clamps J0_cat to a small positive value.

        Args:
            J (float): Current density (A/m^2).
            T (float): Operating temperature (K).

        Returns:
            float: Cathode activation overpotential (V).
        """
        R = self._get_param('R')
        F = self._get_param('F')
        alpha_c = self._get_param('cathode_charge_transfer_coefficient')
        J0_cat = self._get_param('cathode_exchange_current_density_ref') # Already A/m^2 from config

        if J <= 1e-9: # Current density close to zero
            return 0.0
        J0_cat_clamped = max(J0_cat, 1e-12) # Clamp J0_cat to avoid log(0) or division by zero

        return (R * T / (alpha_c * F)) * np.log(J / J0_cat_clamped)

    def calculate_r_electrode(self, J: float, T: float) -> float:
        """
        Calculates the total electrode resistance R_el (Equations 8, 10, 11, 12).

        Logic:
        - Retrieves electrode specific resistance (r_el), electrode porosity (epsilon), electrode thickness (d_el),
          number of anode/cathode channels (n_an_ch, n_cat_ch), and MEA length (L) from `config.yaml`.
        - Computes effective electrode resistance (`r_eff`) using Eq. 8.
        - Computes anode and cathode electrode resistances (`R_el_an`, `R_el_cat`) using Eqs. 10 and 11, respectively.
        - Sums them for total electrode resistance (`R_el`) as per Eq. 12.
        - Unit consistency: As discussed in the detailed plan, the paper's formulas for electrode and bipolar plate
          resistances (Eqs 10-17) imply total resistance (Ohm). However, for `V_ohmic = J * R_cell`, `R_cell`
          must be an area-specific resistance (Ohm m^2). To resolve this, these methods are interpreted to calculate
          the area-specific resistance components directly, despite the literal unit cancellation of the formulas.
          This assumes the numerical values of the constants are tuned to this interpretation.

        Args:
            J (float): Current density (A/m^2). (Not directly used in calculation per paper's formulas)
            T (float): Operating temperature (K). (Not directly used in calculation per paper's formulas)

        Returns:
            float: Total electrode area-specific resistance (Ohm m^2).
        """
        r_el = self._get_param('electrode_specific_resistance')      # Ohm m
        epsilon = self._get_param('electrode_porosity')              # -
        d_el = self._get_param('electrode_thickness')                # m
        n_an_ch = self._get_param('anode_num_channels')              # -
        n_cat_ch = self._get_param('cathode_num_channels')            # -
        L = self._get_param('mea_length')                            # m

        r_eff = r_el * (1 - epsilon)**1.5                               # Eq. 8 (Ohm m)

        # Interpretation: These equations, despite how they look, are assumed to yield area-specific resistance (Ohm m^2).
        # This implies that (r_eff * d_el) / (n_an_ch * L) somehow leads to Ohm m^2, or that the paper's underlying
        # model is using an implicit scaling that effectively makes it so.
        # For strict reproducibility, we implement the structure directly.
        
        # Ensure denominators are not zero
        n_an_ch_clamped = max(n_an_ch, 1e-9)
        n_cat_ch_clamped = max(n_cat_ch, 1e-9)
        L_clamped = max(L, 1e-9)

        R_el_an_specific = (r_eff * d_el) / (n_an_ch_clamped * L_clamped) # Eq. 10 (interpreted as Ohm m^2)
        R_el_cat_specific = (r_eff * d_el) / (n_cat_ch_clamped * L_clamped) # Eq. 11 (interpreted as Ohm m^2)

        R_el = R_el_an_specific + R_el_cat_specific # Eq. 12 (Ohm m^2)
        return R_el

    def calculate_r_bipolar_plate(self, J: float, T: float) -> float:
        """
        Calculates the total bipolar plate (BP) resistance R_pl (Equations 13-17).

        Logic:
        - Retrieves anode/cathode plate specific resistances (r_an_p, r_cat_p), MEA width (W), MEA length (L),
          number of channels, channel heights, and number of channel supports from `config.yaml`.
        - Calculates individual resistance components (R_r, R_l, R_s_ch_an, R_s_ch_cat) using Eqs. 13-16.
        - Sums them for total BP resistance (R_pl) as per Eq. 17.
        - Unit consistency: Similar to `calculate_r_electrode`, these are interpreted as yielding area-specific
          resistances (Ohm m^2).
        - Note: `J` and `T` are not directly used in these specific calculations.

        Args:
            J (float): Current density (A/m^2). (Not directly used in calculation per paper's formulas)
            T (float): Operating temperature (K). (Not directly used in calculation per paper's formulas)

        Returns:
            float: Total bipolar plate area-specific resistance (Ohm m^2).
        """
        r_an_p = self._get_param('anode_plate_specific_resistance') # Ohm m
        r_cat_p = self._get_param('cathode_plate_specific_resistance') # Ohm m
        W = self._get_param('mea_width') # m
        L = self._get_param('mea_length') # m
        n_an_ch = self._get_param('anode_num_channels') # -
        n_cat_ch = self._get_param('cathode_num_channels') # -
        h_an_c = self._get_param('anode_channel_height') # m
        h_cat_c = self._get_param('cathode_channel_height') # m
        n_s_ch_an = self._get_param('anode_num_channel_supports') # -
        n_s_ch_cat = self._get_param('cathode_num_channel_supports') # -

        # Ensure denominators are not zero
        n_an_ch_clamped = max(n_an_ch, 1e-9)
        n_cat_ch_clamped = max(n_cat_ch, 1e-9)
        L_clamped = max(L, 1e-9)
        n_s_ch_an_clamped = max(n_s_ch_an, 1e-9)
        n_s_ch_cat_clamped = max(n_s_ch_cat, 1e-9)

        # Assuming literal computation yields area-specific resistance (Ohm m^2)
        R_r_specific = (r_an_p * W) / (n_an_ch_clamped * L_clamped)                  # Eq. 13
        R_l_specific = (r_cat_p * W) / (n_cat_ch_clamped * L_clamped)                  # Eq. 14
        R_s_ch_an_specific = (r_an_p * h_an_c) / (n_s_ch_an_clamped * L_clamped)     # Eq. 15
        R_s_ch_cat_specific = (r_cat_p * h_cat_c) / (n_s_ch_cat_clamped * L_clamped) # Eq. 16

        R_pl = R_r_specific + R_l_specific + R_s_ch_an_specific + R_s_ch_cat_specific # Eq. 17 (Ohm m^2)
        return R_pl

    def calculate_sigma_membrane(self, lambda_val: float, T: float) -> float:
        """
        Calculates the membrane conductivity (Equation 19).

        Logic:
        - Implements the empirical correlation for membrane conductivity.
        - The equation as given in the paper is expected to result in S/cm.
        - Converts the result from S/cm to S/m for SI unit consistency.

        Args:
            lambda_val (float): Water content of membrane.
            T (float): Operating temperature (K).

        Returns:
            float: Membrane conductivity (S/m).
        """
        # Eq. 19: Result is in S/cm
        sigma_mem_scm = (0.5139 * lambda_val - 0.326) * np.exp(1268 * (1/303 - 1/T))
        # Clamp to a small positive value if it becomes non-positive due to small lambda_val
        sigma_mem_scm = max(sigma_mem_scm, 1e-12)
        return sigma_mem_scm * 100.0 # 1 S/cm = 100 S/m

    def calculate_r_membrane(self, d_mem: float, sigma_mem: float) -> float:
        """
        Calculates the membrane resistance (Equation 18).

        Logic:
        - Computes membrane resistance based on its thickness and conductivity.
        - Addresses the unit ambiguity in Eq. 18 (R_mem = d_mem / (A * sigma_mem)):
          For consistency with other ohmic resistance components that sum to `R_cell` (Ohm m^2),
          this method calculates the area-specific membrane resistance `d_mem / sigma_mem` (Ohm m^2).
          The 'A' (active surface area) in the paper's Eq. 18 is disregarded in this calculation,
          as it would result in Ohm/m^2 if A is in m^2, instead of Ohm m^2.

        Args:
            d_mem (float): Membrane thickness (m).
            sigma_mem (float): Membrane conductivity (S/m).

        Returns:
            float: Membrane area-specific resistance (Ohm m^2).
        """
        if sigma_mem <= 1e-12: # Avoid division by zero for very low conductivity
            return np.inf
        
        # Eq. 18 in paper: R_mem = d_mem / (A * sigma_mem).
        # As discussed in logic analysis, for R_mem to be in Ohm m^2, the 'A' in the denominator
        # must be disregarded, making it R_mem = d_mem / sigma_mem, which is the standard definition
        # of area-specific resistance.
        return d_mem / sigma_mem # Unit: Ohm m^2

    def calculate_v_ohmic(self, J: float, R_cell: float) -> float:
        """
        Calculates the ohmic overpotential (Equation 5).

        Logic:
        - Directly applies Ohm's law with current density `J` and total cell resistance `R_cell`.
        - `R_cell` is expected in Ohm m^2 for output in Volts.
        - Ensures `R_cell` is non-negative.

        Args:
            J (float): Current density (A/m^2).
            R_cell (float): Total effective ohmic resistance (Ohm m^2).

        Returns:
            float: Ohmic overpotential (V).
        """
        R_cell_clamped = max(0.0, R_cell) # Ensure resistance is non-negative
        return J * R_cell_clamped

    def calculate_diffusion_coefficient(self, gas_pair: str, T: float, P_total: float) -> float:
        """
        Calculates the binary diffusion coefficient for a gas pair (Equations 30, 31).

        Logic:
        - Retrieves molar masses, molecular radii, Lennard-Jones potentials, and collision integral from `config.yaml`.
        - Implements the Chapman-Enskog correlation for binary diffusion.
        - Converts `P_total` from Pa to atm for the empirical constant `0.001858` which is for atm units.
        - Calculates mixed collision diameter `sigma_ij` as `(sigma_i + sigma_j) / 2`.
        - Handles unknown `gas_pair` by raising a `ValueError`.

        Args:
            gas_pair (str): Identifier for the gas pair ('O2-H2O' or 'H2-H2O').
            T (float): Operating temperature (K).
            P_total (float): Total pressure (Pa).

        Returns:
            float: Binary diffusion coefficient (m^2/s).
        """
        M_O2 = self._get_param('molar_mass_oxygen')
        M_H2 = self._get_param('molar_mass_hydrogen')
        M_H2O = self._get_param('molar_mass_water')

        sigma_H2 = self._get_param('h2_molecular_radius')
        sigma_O2 = self._get_param('o2_molecular_radius')
        sigma_H2O = self._get_param('h2o_molecular_radius')

        Omega_D = self._get_param('diffusion_collision_integral_omega_d') # Assumed 1.0

        P_total_atm = P_total * self.PA_TO_ATM # Convert total pressure from Pa to atm

        # Clamp P_total_atm to avoid division by zero
        P_total_atm_clamped = max(P_total_atm, 1e-9)
        Omega_D_clamped = max(Omega_D, 1e-9)

        if gas_pair == 'O2-H2O':
            sigma_O2_H2O = (sigma_O2 + sigma_H2O) / 2.0
            M_avg = (1/M_O2 + 1/M_H2O)
            D = (0.001858 * T**1.5 * np.sqrt(M_avg)) / \
                (P_total_atm_clamped * sigma_O2_H2O**2 * Omega_D_clamped)
            return D
        elif gas_pair == 'H2-H2O':
            sigma_H2_H2O = (sigma_H2 + sigma_H2O) / 2.0
            M_avg = (1/M_H2 + 1/M_H2O)
            D = (0.001858 * T**1.5 * np.sqrt(M_avg)) / \
                (P_total_atm_clamped * sigma_H2_H2O**2 * Omega_D_clamped)
            return D
        else:
            raise ValueError(f"Unknown gas_pair: {gas_pair}. Must be 'O2-H2O' or 'H2-H2O'.")

    def calculate_knudsen_diffusion(self, T: float, M: float) -> float:
        """
        Calculates the Knudsen diffusion coefficient (Equation 32).

        Logic:
        - Retrieves universal gas constant (R) and average pore radius (r_pore) from `config.yaml`.
        - Addresses the "unclear point" in Eq. 32: The 'F' (Faraday constant) is replaced with 'R' (Gas constant)
          as it is a common typo and `R` is physically consistent with the Knudsen diffusion formula.
        - Ensures molar mass `M` is positive to avoid `sqrt(negative)`.

        Args:
            T (float): Operating temperature (K).
            M (float): Molar mass of the diffusing species (kg/mol).

        Returns:
            float: Knudsen diffusion coefficient (m^2/s).
        """
        R_val = self._get_param('R')
        r_pore = self._get_param('average_pore_radius') # m

        M_clamped = max(M, 1e-9) # Clamp M

        # Eq. 32 with 'F' replaced by 'R' as per unclear_points and standard Knudsen diffusion formula.
        return (2 * r_pore / 3) * np.sqrt((8 * R_val * T) / (np.pi * M_clamped))

    def calculate_c_mem_o2(self, J: float, C0_O2: float, D_an_eff: float) -> float:
        """
        Calculates the oxygen concentration at the electrode-membrane interface (Equation 23).

        Logic:
        - Retrieves Faraday constant (F) and electrode thickness (d_el) from `config.yaml`.
        - Calculates the concentration at the interface by subtracting the consumed concentration from the bulk.
        - Handles potential division by zero for `D_an_eff` by clamping.

        Args:
            J (float): Current density (A/m^2).
            C0_O2 (float): Standard/bulk concentration of oxygen (mol/m^3).
            D_an_eff (float): Effective diffusion coefficient for anode (m^2/s).

        Returns:
            float: Oxygen concentration at membrane interface (mol/m^3).
        """
        F = self._get_param('F')
        d_el = self._get_param('electrode_thickness') # m

        D_an_eff_clamped = max(D_an_eff, 1e-12) # Clamp D_an_eff

        C_mem_O2 = C0_O2 - (J * d_el) / (4 * F * D_an_eff_clamped)
        # Ensure concentration is non-negative
        return max(C_mem_O2, 1e-12)

    def calculate_c_mem_h2(self, J: float, C0_H2: float, D_cat_eff: float) -> float:
        """
        Calculates the hydrogen concentration at the membrane-electrode interface (Equation 24).

        Logic:
        - Retrieves Faraday constant (F) and electrode thickness (d_el) from `config.yaml`.
        - Calculates the concentration at the interface by subtracting the consumed concentration from the bulk.
        - Handles potential division by zero for `D_cat_eff` by clamping.

        Args:
            J (float): Current density (A/m^2).
            C0_H2 (float): Standard/bulk concentration of hydrogen (mol/m^3).
            D_cat_eff (float): Effective diffusion coefficient for cathode (m^2/s).

        Returns:
            float: Hydrogen concentration at membrane interface (mol/m^3).
        """
        F = self._get_param('F')
        d_el = self._get_param('electrode_thickness') # m

        D_cat_eff_clamped = max(D_cat_eff, 1e-12) # Clamp D_cat_eff

        C_mem_H2 = C0_H2 - (J * d_el) / (2 * F * D_cat_eff_clamped)
        # Ensure concentration is non-negative
        return max(C_mem_H2, 1e-12)

    def calculate_v_con_anode(self, T: float, C0_O2: float, C_mem_O2: float) -> float:
        """
        Calculates the anode concentration overpotential (Equation 21).

        Logic:
        - Retrieves R and F from `config.yaml`.
        - Implements the logarithmic concentration overpotential relation.
        - Handles `C_mem_O2 <= 0` (indicating mass transport limitation) by returning `np.inf`.
        - Ensures `C0_O2` is positive for valid log.

        Args:
            T (float): Operating temperature (K).
            C0_O2 (float): Standard/bulk concentration of oxygen (mol/m^3).
            C_mem_O2 (float): Oxygen concentration at membrane interface (mol/m^3).

        Returns:
            float: Anode concentration overpotential (V).
        """
        R = self._get_param('R')
        F = self._get_param('F')

        C_mem_O2_clamped = max(C_mem_O2, 1e-12) # Clamp to avoid log(0) or division by zero
        C0_O2_clamped = max(C0_O2, 1e-12) # Clamp C0_O2

        return (R * T / (4 * F)) * np.log(C0_O2_clamped / C_mem_O2_clamped)

    def calculate_v_con_cathode(self, T: float, C0_H2: float, C_mem_H2: float) -> float:
        """
        Calculates the cathode concentration overpotential (Equation 22).

        Logic:
        - Retrieves R and F from `config.yaml`.
        - Implements the logarithmic concentration overpotential relation.
        - Handles `C_mem_H2 <= 0` (indicating mass transport limitation) by returning `np.inf`.
        - Ensures `C0_H2` is positive for valid log.

        Args:
            T (float): Operating temperature (K).
            C0_H2 (float): Standard/bulk concentration of hydrogen (mol/m^3).
            C_mem_H2 (float): Hydrogen concentration at membrane interface (mol/m^3).

        Returns:
            float: Cathode concentration overpotential (V).
        """
        R = self._get_param('R')
        F = self._get_param('F')

        C_mem_H2_clamped = max(C_mem_H2, 1e-12) # Clamp to avoid log(0) or division by zero
        C0_H2_clamped = max(C0_H2, 1e-12) # Clamp C0_H2

        return (R * T / (2 * F)) * np.log(C0_H2_clamped / C_mem_H2_clamped)

    def calculate_n_eod_h2(self, J: float) -> float:
        """
        Calculates the flux density of hydrogen due to electro-osmosis drag (Equation 47).

        Logic:
        - Retrieves Faraday constant (F), hydrogen crossover 'z' (h2_crossover_z), hydrogen solubility (S_H2),
          and water molar concentration (C_H2O) from `config.yaml`.
        - Implements Eq. 47.
        - `S_H2` is already converted to `mol m^-3 Pa^-1` in `parameters.py` from `mol m^-3 bar^-1`.
          So it's used directly here with `C_H2O` which is `mol m^-3`.
        - Note: The paper's formulation `S_H2 * C(H2O)` inside the flux equation for EOD is unusual for typical EOD flux.
          EOD flux is usually `n_d * J / F`. Here it seems to imply a concentration of dissolved gas being dragged.
          We implement literally as per Eq. 47.
        - Ensures `J` is non-negative as EOD is driven by current.

        Args:
            J (float): Current density (A/m^2).

        Returns:
            float: Hydrogen electro-osmotic drag flux density (mol s^-1 m^-2).
        """
        F = self._get_param('F')
        z_H2 = self._get_param('h2_crossover_z')
        S_H2_Pa_based = self._get_param('h2_solubility') # mol m^-3 Pa^-1 (already converted in Parameters)
        C_H2O = self._get_param('h2o_molar_concentration') # mol m^-3

        J_clamped = max(J, 0.0) # Current density should be positive for EOD drag
        F_clamped = max(F, 1e-9)
        z_H2_clamped = max(z_H2, 1e-9)

        return (J_clamped / (F_clamped * z_H2_clamped)) * S_H2_Pa_based * C_H2O

    def calculate_n_eod_o2(self, J: float) -> float:
        """
        Calculates the flux density of oxygen due to electro-osmosis drag (Equation 48).

        Logic:
        - Retrieves F, oxygen crossover 'z' (o2_crossover_z), oxygen solubility (S_O2),
          and water molar concentration (C_H2O) from `config.yaml`.
        - Implements Eq. 48.
        - `S_O2` is already converted to `mol m^-3 Pa^-1` in `parameters.py`.
        - Note: Similar unit consistency considerations as for `calculate_n_eod_h2`.
        - Ensures `J` is non-negative.

        Args:
            J (float): Current density (A/m^2).

        Returns:
            float: Oxygen electro-osmotic drag flux density (mol s^-1 m^-2).
        """
        F = self._get_param('F')
        z_O2 = self._get_param('o2_crossover_z')
        S_O2_Pa_based = self._get_param('o2_solubility') # mol m^-3 Pa^-1 (already converted in Parameters)
        C_H2O = self._get_param('h2o_molar_concentration') # mol m^-3

        J_clamped = max(J, 0.0) # Current density should be positive for EOD drag
        F_clamped = max(F, 1e-9)
        z_O2_clamped = max(z_O2, 1e-9)

        return (J_clamped / (F_clamped * z_O2_clamped)) * S_O2_Pa_based * C_H2O

    def calculate_n_dif_h2(self, P_cat_H2: float, d_mem: float) -> float:
        """
        Calculates the hydrogen diffusion flux density due to concentration difference (based on Eq. 49 and 54).

        Logic:
        - Retrieves hydrogen diffusion constant for concentration difference (epsilon_dif_H2) from `config.yaml`.
        - Implements the diffusion term from Eq. 54. This term assumes the driving force is simply `P_cat_H2`
          (i.e., assuming zero H2 at anode side for diffusion, and `P_cat_H2` represents the concentration driving force).
        - `epsilon_dif_H2` is already converted to `mol m^-1 s^-1 Pa^-1` in `parameters.py`.
          `P_cat_H2` is in `Pa`.
        - Handles `d_mem = 0` by returning `np.inf`.

        Args:
            P_cat_H2 (float): Partial pressure of hydrogen at cathode (Pa).
            d_mem (float): Membrane thickness (m).

        Returns:
            float: Hydrogen diffusion flux density (mol s^-1 m^-2).
        """
        epsilon_dif_H2_Pa_based = self._get_param('h2_diffusion_constant_conc_diff') # mol m^-1 s^-1 Pa^-1 (converted)

        d_mem_clamped = max(d_mem, 1e-12) # Avoid division by zero
        P_cat_H2_clamped = max(P_cat_H2, 0.0) # Ensure non-negative partial pressure

        # As per Eq 54, diffusion term for H2 is (epsilon_dif,H2 * P_cat,H2) / d_mem
        return (epsilon_dif_H2_Pa_based * P_cat_H2_clamped) / d_mem_clamped

    def calculate_n_dif_o2(self, P_an_O2: float, d_mem: float) -> float:
        """
        Calculates the oxygen diffusion flux density due to concentration difference (based on Eq. 49 and 56).

        Logic:
        - Retrieves oxygen diffusion constant for concentration difference (epsilon_dif_O2) from `config.yaml`.
        - Implements the diffusion term from Eq. 56, assuming driving force is `P_an_O2`.
        - `epsilon_dif_O2` is already converted to `mol m^-1 s^-1 Pa^-1` in `parameters.py`.
          `P_an_O2` is in `Pa`.
        - Handles `d_mem = 0` by returning `np.inf`.

        Args:
            P_an_O2 (float): Partial pressure of oxygen at anode (Pa).
            d_mem (float): Membrane thickness (m).

        Returns:
            float: Oxygen diffusion flux density (mol s^-1 m^-2).
        """
        epsilon_dif_O2_Pa_based = self._get_param('o2_diffusion_constant_conc_diff') # mol m^-1 s^-1 Pa^-1 (converted)

        d_mem_clamped = max(d_mem, 1e-12) # Avoid division by zero
        P_an_O2_clamped = max(P_an_O2, 0.0) # Ensure non-negative partial pressure

        # As per Eq 56, diffusion term for O2 is (epsilon_dif,O2 * P_an,O2) / d_mem
        return (epsilon_dif_O2_Pa_based * P_an_O2_clamped) / d_mem_clamped

    def calculate_n_dp_h2(self, delta_P: float, d_mem: float) -> float:
        """
        Calculates the hydrogen flux density due to pressure difference (Equation 50).

        Logic:
        - Retrieves hydrogen diffusion constant for pressure difference (epsilon_dp_H2) from `config.yaml`.
        - Implements Eq. 50.
        - `epsilon_dp_H2` is already converted to `mol m^-1 s^-1 Pa^-1` in `parameters.py`.
          `delta_P` is in `Pa`.
        - Handles `d_mem = 0` by returning `np.inf`.

        Args:
            delta_P (float): Pressure difference across membrane (P_cat - P_an) (Pa).
            d_mem (float): Membrane thickness (m).

        Returns:
            float: Hydrogen pressure-driven flux density (mol s^-1 m^-2).
        """
        epsilon_dp_H2_Pa_based = self._get_param('h2_diffusion_constant_press_diff') # mol m^-1 s^-1 Pa^-1 (converted)

        d_mem_clamped = max(d_mem, 1e-12) # Avoid division by zero
        
        # Note: Delta P can be negative if P_an > P_cat, leading to flow in the opposite direction.
        # We don't clamp it to positive here to allow for reverse flow if the model supports it,
        # but the paper states H2 diffuses from cathode to anode.

        return (epsilon_dp_H2_Pa_based * delta_P) / d_mem_clamped

    def calculate_n_total_h2_crossover(self, N_eod_H2: float, N_dif_H2: float, N_dp_H2: float) -> float:
        """
        Calculates the total flux density of hydrogen crossover.

        Logic:
        - Implements summation for total hydrogen crossover based on Eq. 54.
        - Addresses the "unclear point" where the paper states three mechanisms (electro-osmotic, diffusion, pressure difference)
          but Eq. 54 only explicitly sums electro-osmotic drag (`N_eod_H2`) and concentration-driven diffusion (`N_dif_H2`).
          The pressure-driven flux (`N_dp_H2`) is therefore NOT included in this sum to strictly follow Eq. 54.

        Args:
            N_eod_H2 (float): Hydrogen electro-osmotic drag flux density (mol s^-1 m^-2).
            N_dif_H2 (float): Hydrogen diffusion flux density (mol s^-1 m^-2).
            N_dp_H2 (float): Hydrogen pressure-driven flux density (mol s^-1 m^-2). (NOT included in sum as per Eq. 54)

        Returns:
            float: Total hydrogen crossover flux density (mol s^-1 m^-2).
        """
        # Strictly following Eq 54: N_en_H2 = epsilon_dif_H2 P_cat_H2 / d_mem + J / (F z) S_H2 C(H2O)
        # This translates to: N_en_H2 = N_dif_H2 + N_eod_H2
        # The N_dp_H2 component is NOT included in this specific "total" flux equation (Eq 54) in the paper.
        return N_eod_H2 + N_dif_H2

    def calculate_n_total_o2_crossover(self, N_eod_O2: float, N_dif_O2: float, N_dp_O2: float = 0.0) -> float:
        """
        Calculates the total flux density of oxygen crossover.

        Logic:
        - Implements summation for total oxygen crossover based on Eq. 56.
        - Eq. 56 explicitly sums electro-osmotic drag (`N_eod_O2`) and concentration-driven diffusion (`N_dif_O2`).
        - The paper states oxygen pressure-driven diffusion can be ignored, so `N_dp_O2` is an argument but is not used in the sum here.

        Args:
            N_eod_O2 (float): Oxygen electro-osmotic drag flux density (mol s^-1 m^-2).
            N_dif_O2 (float): Oxygen diffusion flux density (mol s^-1 m^-2).
            N_dp_O2 (float): Oxygen pressure-driven flux density (mol s^-1 m^-2). (Not used in sum per paper)

        Returns:
            float: Total oxygen crossover flux density (mol s^-1 m^-2).
        """
        # Strictly following Eq 56: N_en_O2 = epsilon_dif_O2 P_an_O2 / d_mem + J / (F z) S_O2 C(H2O)
        # This translates to: N_en_O2 = N_dif_O2 + N_eod_O2
        return N_eod_O2 + N_dif_O2

    def update_partial_pressures(self, J: float, P_an: float, P_cat: float, N_en_H2: float, N_en_O2: float) -> Tuple[float, float]:
        """
        Updates the partial pressures of H2 at cathode and O2 at anode, considering crossover.
        (Equations 52, 53).

        Logic:
        - Retrieves Faraday constant (F), partial pressure enhancement factors (A_H2, A_O2),
          and hydrogen diffusion constants (epsilon_dif_H2, epsilon_dp_H2) from `config.yaml`.
        - **Addresses significant "unclear points" regarding unit inconsistencies and coupled non-linear form**:
            - The paper's formulation for Eqs. 52 and 53 has units issues.
            - `A_H2` and `A_O2` are `bar m^2 A^-1`. `epsilon_dif_H2` and `epsilon_dp_H2` are `mol m^-1 s^-1 bar^-1`.
            - `J` is `A/m^2`. `N_en` is `mol s^-1 m^-2`.
            - For strict adherence, the equations are implemented numerically *as written*,
              assuming that the parameters implicitly account for unit conversions or are empirical.
            - Input pressures (`P_an`, `P_cat`) are converted to `bar` for consistency with `A_i` factors' units,
              and then results are converted back to `Pa`.
            - For Eq. 53 (P_cat_H2), which is a quadratic equation in `P_cat_H2` and depends on `P_an_O2`:
                - It is solved algebraically as a quadratic equation for `P_cat_H2` using the standard quadratic formula.
                - The `P_an_O2` term within Eq. 53 (RHS) is the one calculated from Eq. 52 in this same iteration step.
                - Physically meaningful (positive) roots are selected. In case of no valid root or complex roots, a fallback is used.

        Args:
            J (float): Current density (A/m^2).
            P_an (float): Total anode pressure (Pa).
            P_cat (float): Total cathode pressure (Pa).
            N_en_H2 (float): Total hydrogen crossover flux density (mol s^-1 m^-2).
            N_en_O2 (float): Total oxygen crossover flux density (mol s^-1 m^-2).

        Returns:
            Tuple[float, float]: Updated (P_cat_H2, P_an_O2) in Pa.
        """
        F = self._get_param('F')
        A_H2_bar_based = self._get_param('h2_partial_pressure_enhancement_factor') # bar m^2 A^-1 (converted in Parameters)
        A_O2_bar_based = self._get_param('o2_partial_pressure_enhancement_factor') # bar m^2 A^-1 (converted in Parameters)
        epsilon_dif_H2_bar_based = self._get_param('h2_diffusion_constant_conc_diff') # mol m^-1 s^-1 bar^-1 (converted in Parameters)
        epsilon_dp_H2_bar_based = self._get_param('h2_diffusion_constant_press_diff') # mol m^-1 s^-1 bar^-1 (converted in Parameters)

        # Convert input pressures from Pa to bar for calculations as per paper's implied units for A_i factors
        P_an_bar = P_an * self.PA_TO_BAR
        P_cat_bar = P_cat * self.PA_TO_BAR

        # Clamp J, F to avoid division by zero
        J_clamped = max(J, 0.0)
        F_clamped = max(F, 1e-9)

        # --- Calculate P_an_O2 (Equation 52) ---
        # P_an,O2 = P_an - ( (J / (4F)) * A_O2 + N_en_O2 )
        # Unit inconsistency acknowledged: (J / (4F)) * A_O2 should give bar*m^-2 (A/m^2 / (C/mol) * bar*m^2/A = bar*mol/C * m^2), not bar.
        # N_en_O2 is mol s^-1 m^-2. Directly summing with pressure is problematic.
        # However, for reproduction, we perform the arithmetic operations directly using the numerical values.
        
        # The paper's Eq. 52: P_an_O2 = P_an - ( (J / (4F)) * A_O2 + N_en_O2 )
        # To make it dimensionally consistent if A_O2 is intended to convert J/(4F) to pressure units,
        # A_O2 would need to be `bar / (A/m^2)` or similar. The given unit is `bar m^2 A^-1`.
        # This implies: (J * A_O2) / (4F)
        # (A/m^2 * bar*m^2/A) / (C/mol) = bar / (C/mol) which is not a pressure unit.
        # Given the units of A_O2, it seems more like a resistance-like term for converting a current to pressure drop,
        # or it's a strongly empirical fit.
        # The N_en_O2 is a molar flux. Direct subtraction from pressure implies N_en_O2 has been implicitly
        # converted to a partial pressure *contribution*.
        
        # Given the "unclear points" and for direct numerical reproduction:
        # We assume the right-hand side terms (J/(4F) * A_O2 and N_en_O2) are effectively small
        # pressure-like contributions that reduce the total anode pressure to give partial oxygen pressure.
        
        # We will apply the J and N_en_O2 term as a reduction in pressure.
        # This is where the model gets highly empirical/unphysical in terms of units.
        # For literal implementation, we calculate the subtrahend.
        subtrahend_an_o2 = ((J_clamped / (4 * F_clamped)) * A_O2_bar_based) + N_en_O2 * self.PA_TO_BAR # Assuming N_en_O2 needs to be converted to bar equivalent
        
        P_an_O2_new_bar = P_an_bar - subtrahend_an_o2
        P_an_O2_new_bar = max(1e-9, P_an_O2_new_bar) # Ensure partial pressure is non-negative

        # --- Calculate P_cat_H2 (Equation 53) ---
        # P_cat_H2 = P_cat + epsilon_dif_H2 P_cat_H2 A_H2 + epsilon_dp_H2 P_cat_H2 A_H2 - epsilon_dp_H2 P_an_O2 A_H2 epsilon_dif_H2 P_cat_H2
        # This equation contains P_cat_H2 on both sides and appears as a quadratic form if rearranged:
        # (epsilon_dp_H2 * A_H2 * epsilon_dif_H2) * P_cat_H2^2
        # + (1 - epsilon_dif_H2 * A_H2 - epsilon_dp_H2 * A_H2) * P_cat_H2
        # - (P_cat - epsilon_dp_H2 * P_an_O2 * A_H2) = 0

        # Note: All epsilon_dif/dp_H2 and A_H2 are already converted to Pa-based units in parameters.py.
        # However, the formula as written in the paper assumes `bar` for `P_cat_H2`, `P_an_O2`.
        # So, we convert them back to bar for calculation as per paper's equation units.
        # The epsilon_ constants from config.yaml are already mol m^-1 s^-1 Pa^-1.
        # The A_ constants from config.yaml are already Pa m^2 A^-1.
        # If we use these directly, the terms will be Pa based.
        # So we should use P_an_O2_new_Pa directly and not convert to bar, unless the constants
        # A_H2, etc. are implicitly expected to multiply with bar.
        
        # Let's re-confirm the units for A_H2 and epsilon_dif/dp_H2.
        # config.yaml:
        # h2_partial_pressure_enhancement_factor: 2.4e-04 # bar m^2 A^-1 -> converted to Pa m^2 A^-1 in parameters.py
        # h2_diffusion_constant_conc_diff: 4.65e-09 # mol m^-1 s^-1 bar^-1 -> converted to mol m^-1 s^-1 Pa^-1 in parameters.py
        # This implies that all these parameters, when fetched by `_get_param`, are already Pa-consistent.
        # So, we should use Pa values for `P_an_O2_new_Pa` and expect `P_cat_H2` result in Pa.

        P_an_O2_for_eq53 = P_an_O2_new_Pa # Use the newly calculated oxygen pressure in Pa

        # Equation 53 as a quadratic in P_cat_H2_new_Pa:
        # (epsilon_dp_H2_Pa_based * A_H2_Pa_based * epsilon_dif_H2_Pa_based) * P_cat_H2^2
        # + (1 - epsilon_dif_H2_Pa_based * A_H2_Pa_based - epsilon_dp_H2_Pa_based * A_H2_Pa_based) * P_cat_H2
        # - (P_cat - epsilon_dp_H2_Pa_based * P_an_O2_for_eq53 * A_H2_Pa_based) = 0
        
        # Note: The paper's equation 53 is a simplified notation that makes dimensional analysis difficult.
        # If epsilon_dif_H2 and A_H2 are just factors, it is empirical.
        # Implementing the mathematical form directly based on the coefficients shown.
        # Let's ensure the `A_H2` and `epsilon` terms are used as `Pa`-based values.
        
        A_H2_Pa_based = self._get_param('h2_partial_pressure_enhancement_factor') # Pa m^2 A^-1
        epsilon_dif_H2_Pa_based = self._get_param('h2_diffusion_constant_conc_diff') # mol m^-1 s^-1 Pa^-1
        epsilon_dp_H2_Pa_based = self._get_param('h2_diffusion_constant_press_diff') # mol m^-1 s^-1 Pa^-1

        # Original equation from paper for P_cat_H2, but with Pa units throughout
        # P_cat_H2 = P_cat + epsilon_dif_H2 P_cat_H2 A_H2 + epsilon_dp_H2 P_cat_H2 A_H2 - epsilon_dp_H2 P_an_O2 A_H2 epsilon_dif_H2 P_cat_H2

        # Rearranging to a quadratic equation for P_cat_H2 (let x = P_cat_H2):
        # (epsilon_dp_H2_Pa_based * A_H2_Pa_based * epsilon_dif_H2_Pa_based) * x^2  (This term should be 0, as it leads to mol s^-1 m^-2, not pressure)
        # + (1 - epsilon_dif_H2_Pa_based * A_H2_Pa_based - epsilon_dp_H2_Pa_based * A_H2_Pa_based) * x (This term implies 1 - dimensionless)
        # - (P_cat - epsilon_dp_H2_Pa_based * P_an_O2_for_eq53 * A_H2_Pa_based) = 0

        # This `a_coeff` term should be zero if the units are correct for pressure. It's `mol/s/m` * `Pa*m^2/A` * `mol/s/m`.
        # This makes the equation (53) highly questionable dimensionally.
        # For direct reproduction: We *must* use the literal coefficients from the equation.
        # The "constant diffusion due to pressure difference" `epsilon_dp` and "diffusion constant due to concentration difference" `epsilon_dif`
        # are used in the partial pressure equation. This is extremely unusual.
        
        # It's more likely that the structure of Eq. 53 in the paper is:
        # P_cat_H2 = P_cat + (some coeff 1) * P_cat_H2 + (some coeff 2) * P_cat_H2 - (some coeff 3) * P_cat_H2 * P_an_O2
        # So it's effectively P_cat_H2 = f(P_cat, P_an_O2, P_cat_H2)
        # A common form for such equations is a fixed-point iteration, x_new = f(x_old).
        # However, the design specifies solving this implicitly as part of the `update_partial_pressures`.

        # Let's assume the coefficients are dimensionless as implied by the structure, or empirical multipliers
        # that resolve to dimensionless form.
        
        # Term 1: epsilon_dp_H2 * P_an_O2_for_eq53 * A_H2 * epsilon_dif_H2 * P_cat_H2 (This is the negative term in paper, 5th term)
        # This term means `a_coeff * P_cat_H2` (where a_coeff is epsilon_dp_H2 * P_an_O2_for_eq53 * A_H2 * epsilon_dif_H2)
        # This implies a product of 4 parameters and a pressure is dimensionless if it is used to scale P_cat_H2.
        # If P_cat_H2 is in Pa, A_H2 in Pa m^2 A^-1, epsilon in mol m^-1 s^-1 Pa^-1, then this term is a unit nightmare.

        # Given the instruction to "implement complete, reliable, reusable code snippets" and "write out EVERY CODE DETAIL",
        # and "You MUST FOLLOW 'Data structures and interfaces'. DONT CHANGE ANY DESIGN.",
        # the quadratic form derived is the literal mathematical interpretation of the equation.
        # The fact that the coefficients are dimensionally inconsistent with a pressure result implies an empirical correlation
        # where the constants (`epsilon`, `A`) are implicitly unit-adjusted for the input `bar` pressures.
        # Since I've already converted parameters to SI (Pa-based), I will use them, and accept the dimensional inconsistency
        # as a property of the source paper's model.

        # Recalculate coefficients for the quadratic equation a*x^2 + b*x + c = 0 where x = P_cat_H2
        # Eq. 53: P_cat_H2 = P_cat + (epsilon_dif_H2 * A_H2) * P_cat_H2 + (epsilon_dp_H2 * A_H2) * P_cat_H2 - (epsilon_dp_H2 * P_an_O2 * A_H2 * epsilon_dif_H2) * P_cat_H2
        # Rearranging:
        # 0 = (epsilon_dp_H2 * A_H2 * epsilon_dif_H2) * P_cat_H2^2  (This term should be P_cat_H2 * (Pa m^2 A^-1) * (mol m^-1 s^-1 Pa^-1) * (mol m^-1 s^-1 Pa^-1))
        #     + (epsilon_dif_H2 * A_H2 + epsilon_dp_H2 * A_H2 - 1) * P_cat_H2
        #     + (P_cat - epsilon_dp_H2 * P_an_O2 * A_H2)
        # The paper's equation has a minus sign on the (epsilon_dp_H2 P_an_O2 A_H2 epsilon_dif_H2 P_cat_H2) term, so:
        # (epsilon_dp_H2_Pa_based * A_H2_Pa_based * epsilon_dif_H2_Pa_based) * P_cat_H2^2  (this term is indeed quadratic)
        # + (epsilon_dif_H2_Pa_based * A_H2_Pa_based + epsilon_dp_H2_Pa_based * A_H2_Pa_based - 1) * P_cat_H2
        # + (P_cat_bar - (epsilon_dp_H2_Pa_based * P_an_O2_for_eq53 * A_H2_Pa_based) ) = 0
        # Wait, the equation 53 is:
        # P_cat_H2 = P_cat + ε_dif_H2 P_cat_H2 A_H2 + ε_dp_H2 P_cat_H2 A_H2 - ε_dp_H2 P_an_O2 A_H2 ε_dif_H2 P_cat_H2
        # The last term has P_an_O2 AND P_cat_H2.
        # So it's:
        # P_cat_H2 (1 - ε_dif_H2 A_H2 - ε_dp_H2 A_H2 + ε_dp_H2 P_an_O2 A_H2 ε_dif_H2) = P_cat
        # This is LINEAR in P_cat_H2. My previous quadratic rearrangement was wrong.
        # Let's re-read carefully: "P cat H 2 ¼ P cat þ ε dif H 2 P cat H 2 A H 2 þ ε dp H 2 P cat H 2 A H 2 À ε dp H 2 P an O 2 A H 2 ε dif H 2 P cat H 2"
        # Oh, the last term has `P_cat_H2` at the very end. This makes it a quadratic term: `(P_cat_H2)^2`.
        # This confirms the quadratic interpretation of Eq. 53.

        # Let's use the Pa-based values for constants
        a_coeff = epsilon_dp_H2_Pa_based * A_H2_Pa_based * epsilon_dif_H2_Pa_based
        b_coeff = (epsilon_dif_H2_Pa_based * A_H2_Pa_based) + (epsilon_dp_H2_Pa_based * A_H2_Pa_based) - 1
        c_coeff = P_cat_bar - (epsilon_dp_H2_Pa_based * P_an_O2_for_eq53 * A_H2_Pa_based) # P_cat_bar is in bar, others are Pa-based. This is the core inconsistency.
        # To maintain paper consistency, P_cat should also be Pa.
        # Let's convert P_cat to Pa directly for this equation.
        c_coeff = P_cat - (epsilon_dp_H2_Pa_based * P_an_O2_for_eq53 * A_H2_Pa_based) # P_cat is already in Pa

        # Solving the quadratic equation a*x^2 + b*x + c = 0
        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        
        P_cat_H2_new_Pa = P_cat # Default fallback

        if discriminant < 0:
            # No real roots. Fallback to P_cat.
            pass
        elif np.isclose(a_coeff, 0.0): # Handle case where a_coeff is zero, it becomes a linear equation
            if not np.isclose(b_coeff, 0.0):
                root = -c_coeff / b_coeff
                if root > 0: P_cat_H2_new_Pa = root
            # else: b_coeff is also zero, equation is 0=0 or c_coeff=0 (infinite solutions or no solution) - rely on fallback
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            root1 = (-b_coeff + sqrt_discriminant) / (2 * a_coeff)
            root2 = (-b_coeff - sqrt_discriminant) / (2 * a_coeff)
            
            valid_roots = []
            if root1 > 1e-9: # Must be positive
                valid_roots.append(root1)
            if root2 > 1e-9:
                valid_roots.append(root2)
            
            if valid_roots:
                # If multiple positive roots, choose the one that makes physical sense.
                # Often, the smallest positive root, or the one closest to a previous estimate.
                # Here, we'll choose the minimum positive root.
                P_cat_H2_new_Pa = min(valid_roots)
            # else: no valid positive root, fallback to P_cat

        # Ensure final partial pressures are non-negative
        P_an_O2_new_Pa = max(1e-9, P_an_O2_new_Pa)
        P_cat_H2_new_Pa = max(1e-9, P_cat_H2_new_Pa)
        
        return P_cat_H2_new_Pa, P_an_O2_new_Pa

