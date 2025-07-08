# PEM Electrolyzer Performance Model Reproduction

This repository contains a Python-based computational model designed to reproduce the findings of the research paper:

**Paper ID:** 2021_test
**Title:** "Performance assessment of gas crossover phenomenon and water transport mechanism in high pressure PEM electrolyzer"

The model implements the mathematical framework, electrochemical equations, and transport phenomena described in the paper, focusing on gas crossover and water management in a high-pressure Proton Exchange Membrane (PEM) electrolyzer. It aims to achieve computational reproducibility of the paper's polarization curves, overpotential breakdowns, and sensitivity analyses.

## Table of Contents

1.  [Overview](#1-overview)
2.  [Core Model Description](#2-core-model-description)
3.  [Mathematical Model & Equations (Key Implementations)](#3-mathematical-model--equations-key-implementations)
    *   [3.1. Reversible Voltage](#31-reversible-voltage)
    *   [3.2. Activation Overpotential](#32-activation-overpotential)
    *   [3.3. Ohmic Overpotential](#33-ohmic-overpotential)
    *   [3.4. Concentration Overpotential](#34-concentration-overpotential)
    *   [3.5. Water Transfer Mechanisms](#35-water-transfer-mechanisms)
    *   [3.6. Gas Crossover Phenomena](#36-gas-crossover-phenomena)
4.  [Parameters and Configuration](#4-parameters-and-configuration)
5.  [Simulation Workflow](#5-simulation-workflow)
6.  [Reproducing Figures](#6-reproducing-figures)
7.  [Validation Strategy](#7-validation-strategy)
8.  [Debugging Strategy](#8-debugging-strategy)
9.  [Setup and Installation](#9-setup-and-installation)
10. [How to Run](#10-how-to-run)
11. [Output Interpretation](#11-output-interpretation)
12. [Limitations and Future Work](#12-limitation-and-future-work)

---

## 1. Overview

This project provides a modular, reproducible implementation of the PEM electrolyzer model from the specified paper. It is designed to:
*   Accurately implement all mathematical equations presented in the paper's methodology section.
*   Manage parameters and operating conditions through a centralized `config.yaml` file.
*   Solve the coupled system of equations numerically.
*   Validate the model against experimental data cited in the paper.
*   Perform sensitivity analyses to investigate the impact of key design and operating parameters.
*   Generate plots mirroring those in the original publication.

## 2. Core Model Description

The model calculates the overall cell voltage ($V_{cell}$) of a PEM electrolyzer as a sum of the reversible voltage ($V_{rev}$) and various overpotentials: activation ($V_{act}$), ohmic ($V_{ohmic}$), and concentration ($V_{con}$). It specifically incorporates detailed sub-models for gas crossover mechanisms (electro-osmotic drag, diffusion, and pressure difference) and water transport across the membrane. The model operates under steady-state, single-phase, and ideal gas assumptions, as stated in the paper.

## 3. Mathematical Model & Equations (Key Implementations)

All equations are implemented within the `Model` class, with parameters sourced from `config.yaml`. Unit consistency is maintained by converting all relevant quantities to SI units (meters, seconds, kilograms, moles, Kelvin, Pascal, Amperes, Volts) internally, unless specific formulas from the paper dictate other units (e.g., bar for solubility constants) which are then handled with appropriate conversions.

### 3.1. Reversible Voltage

The reversible voltage is calculated using a modified Nernst equation, accounting for partial pressures of reactants and products, and temperature.

*   **Equation (1):** $V_{rev} = V^0_{rev} + \frac{RT}{2F} \ln\left(\frac{P_{cat,H2} \cdot P^{0.5}_{an,O2}}{P_{H2O}}\right)$
    *   **Implemented in:** `Model.calculate_v_rev(P_cat_H2, P_an_O2, P_H2O, T)`
    *   **Inputs**: `P_cat_H2` (partial pressure of hydrogen at cathode, Pa), `P_an_O2` (partial pressure of oxygen at anode, Pa), `P_H2O` (partial pressure of water, Pa), `T` (operating temperature, K).
    *   **Parameter Sourcing**: `R` (gas constant) and `F` (Faraday constant) are from `config.yaml['physical_constants']`. `V0_rev` (standard reversible voltage, 1.229 V) is also from `config.yaml['physical_constants']` as it's typically used in such models.

### 3.2. Activation Overpotential

Activation overpotentials at the anode and cathode are modeled using Tafel-like equations, dependent on current density and exchange current densities.

*   **Equation (3) (Anode):** $V_{act,a} = \frac{RT}{\alpha_a F} \ln\left(\frac{J}{J_{0,an}}\right)$
*   **Equation (4) (Cathode):** $V_{act,c} = \frac{RT}{\alpha_c F} \ln\left(\frac{J}{J_{0,cat}}\right)$
    *   **Implemented in:** `Model.calculate_v_act_anode(J, T)` and `Model.calculate_v_act_cathode(J, T)`
    *   **Inputs**: `J` (Current density, A/m²), `T` (Operating temperature, K).
    *   **Parameter Sourcing**: `alpha_a` (anode charge transfer coefficient), `alpha_c` (cathode charge transfer coefficient), `anode_exchange_current_density_ref` (`J_0,an`), `cathode_exchange_current_density_ref` (`J_0,cat`) are from `config.yaml['model_parameters']`.
    *   **Ambiguity Addressed**: The paper lists activation energies (`E_act,i`) and reaction rate coefficients (`k_0,i`) which typically imply an Arrhenius-type temperature dependence for `J_0,i`. However, no explicit Arrhenius equation for `J_0,i` is provided alongside Equations (3) and (4). **For strict adherence to the provided equations, our implementation uses the `J_0,i` values from Table 1 directly, treating them as fixed reference values applicable at the operating temperatures investigated (e.g., 353 K).** This assumption is crucial and noted in `config.yaml['unclear_points']['j0i_temperature_dependence']`.

### 3.3. Ohmic Overpotential

Ohmic losses arise from the resistance of the membrane, electrodes, and bipolar plates.

*   **Equation (5):** $V_{ohmic} = J \cdot R_{cell}$
    *   **Implemented in:** `Model.calculate_v_ohmic(J, R_cell)`
    *   **Inputs**: `J` (Current density, A/m²), `R_cell` (Total effective ohmic resistance, Ohm m²). `R_cell` is calculated as the sum of electrode, membrane, and bipolar plate resistances (Eq. 6).

    *   **3.3.1. Electrode Resistance**
        *   **Implemented in:** `Model.calculate_r_electrode()`
        *   **Equations (8-12):** These equations define the effective resistance of the electrode material, average electron path length, and the total resistance contributions from anode and cathode electrodes.
        *   **Parameter Sourcing**: `electrode_specific_resistance` (`r_el`) is from `config.yaml['model_parameters']`.
        *   **Assumed Parameters**: `electrode_porosity` ($\epsilon$), `electrode_thickness` ($d_{el}$), `channel_width` ($w_c$), `channel_support_width` ($w_s$), `anode_num_channels` ($n_{an,ch}$), `cathode_num_channels` ($n_{cat,ch}$), `mea_length` ($L$), and `active_surface_area` ($A$) are **assumed parameters** taken from `config.yaml['assumed_parameters']`, as they are not explicitly provided in Table 1 or the text.

    *   **3.3.2. Bipolar Plate Resistance**
        *   **Implemented in:** `Model.calculate_r_bipolar_plate()`
        *   **Equations (13-17):** These describe the resistances contributed by the plate material at the right/left sides and channel supports.
        *   **Parameter Sourcing**: `anode_plate_specific_resistance` (`r_an,p`) and `cathode_plate_specific_resistance` (`r_cat,p`) are from `config.yaml['model_parameters']`.
        *   **Assumed Parameters**: `mea_width` ($W$), `anode_channel_height` ($h_{an,c}$), `cathode_channel_height` ($h_{cat,c}$), `anode_num_channel_supports` ($n_{s,ch,an}$), `cathode_num_channel_supports` ($n_{s,ch,cat}$), and `mea_length` ($L$) are **assumed parameters** from `config.yaml['assumed_parameters']`.

    *   **3.3.3. Membrane Resistance**
        *   **Implemented in:** `Model.calculate_r_membrane(d_mem, sigma_mem)`
        *   **Equation (18):** $R_{mem} = d_{mem} / (A \cdot \sigma_{mem})$
        *   **Inputs**: `d_mem` (membrane thickness, m), `sigma_mem` (membrane conductivity, S/m).
        *   **Parameter Sourcing**: `active_surface_area` ($A$) is an `assumed_parameter`.
        *   **Membrane Conductivity**:
            *   **Implemented in:** `Model.calculate_sigma_membrane(lambda_val, T)`
            *   **Equation (19):** $\sigma_{mem} = (0.5139 \cdot \lambda - 0.326) \exp\left(1268\left(\frac{1}{303} - \frac{1}{T}\right)\right)$ (Units $\Omega^{-1}$ cm$^{-1}$, converted to S/m internally)
            *   **Inputs**: `lambda_val` (water content of membrane), `T` (temperature, K).
            *   **Assumed Parameter**: `membrane_water_content_lambda` (`lambda_val`) is a **crucial assumed parameter** in `config.yaml`. The paper does not provide an explicit model for `lambda`'s dependency on water activity or temperature. A constant typical value for Nafion is used.

### 3.4. Concentration Overpotential

Concentration losses occur due to changes in reactant concentrations at the electrode surface.

*   **Equation (20):** $V_{con} = V_{an,con} + V_{cat,con}$
    *   **Implemented in:** `Model.calculate_v_con_anode(T, C0_O2, C_mem_O2)` and `Model.calculate_v_con_cathode(T, C0_H2, C_mem_H2)`
    *   **Inputs**: `T` (K), `C0_i` (standard concentration, mol/m³), `C_mem_i` (concentration at membrane interface, mol/m³).
    *   **Logic**: Direct application of Equations (21) and (22).
    *   **Standard Concentrations (`C0_O2`, `C0_H2`)**: These are calculated internally within the `Solver` from the bulk partial pressures of oxygen and hydrogen in the channels and temperature using the ideal gas law.
    *   **Concentrations at Interface (`C_mem_O2`, `C_mem_H2`)**:
        *   **Implemented in:** `Model.calculate_c_mem_o2(J, C0_O2, D_an_eff)` and `Model.calculate_c_mem_h2(J, C0_H2, D_cat_eff)`
        *   **Equations (23, 24):** $C_{mem,O2} = C^0_{O2} - \frac{J \cdot d_{el}}{4F \cdot D_{an,eff}}$ and $C_{mem,H2} = C^0_{H2} - \frac{J \cdot d_{el}}{2F \cdot D_{cat,eff}}$
        *   **Inputs**: `J` (A/m²), `C0_i` (mol/m³), `D_eff` (effective diffusion coefficient, m²/s).
        *   `electrode_thickness` (`d_el`) is an `assumed_parameter`.
    *   **Effective Diffusion Coefficients (`D_an,eff`, `D_cat,eff`)**:
        *   **Equations (28, 29):** $D_{eff} = \epsilon_x^{1.5} \cdot D_{binary}$. `electrode_porosity_tortuosity_ratio` ($\epsilon_x$) is assumed equal to `electrode_porosity` ($\epsilon$) as per plan's interpretation.
        *   **Binary Diffusion Coefficients**:
            *   **Implemented in:** `Model.calculate_diffusion_coefficient(gas_pair, T, P_total)`
            *   **Equations (30, 31):** These formulas, based on kinetic theory, describe binary diffusion.
            *   **Parameter Sourcing**: `h2_molecular_radius`, `o2_molecular_radius`, `h2o_molecular_radius` (molecular radii for `sigma_i`), `o2_h2o_lennard_jones_potential_k`, `h2_h2o_lennard_jones_potential_k` (Lennard-Jones potentials) are from `config.yaml['model_parameters']`.
            *   **Assumed Parameters**: `molar_mass_oxygen`, `molar_mass_hydrogen`, `molar_mass_water` (standard molar masses) and `diffusion_collision_integral_omega_d` (`Omega_D`) are **assumed parameters**. `Omega_D` is assumed to be 1.0 due to lack of explicit correlation in the paper.
        *   **Knudsen Diffusion Coefficient**:
            *   **Implemented in:** `Model.calculate_knudsen_diffusion(T, M)`
            *   **Equation (32):** $D_{H2O,K} = \frac{2 \cdot r}{3} \sqrt{\frac{8 R T}{\pi M_{H2O}}}$.
            *   **Parameter Sourcing**: `average_pore_radius` (`r`) is from `config.yaml['model_parameters']`.
            *   **Ambiguity Addressed**: Equation (32) in the paper contains `F` (Faraday constant) in its numerator, which is physically incorrect for a diffusion coefficient formula. **This has been corrected to `R` (Gas constant) in our implementation** (as per `config.yaml['unclear_points']['knudsen_diffusion_eq_32_typo']`), as `R` is physically consistent with Knudsen diffusion theory. Note that while this function is implemented, the paper's equations for `D_an,eff` and `D_cat,eff` (Eqs. 28, 29) do not explicitly show `D_H2O,K` being combined into the overall effective diffusion coefficient.

### 3.5. Water Transfer Mechanisms

The paper describes water transfer across the membrane via three mechanisms: diffusion, electro-osmosis, and hydraulic pressure difference. The net water molar flow rate is the sum of these contributions.

*   **Equation (39):** $N^{mem}_{H2O} = N^{diff}_{H2O} + N^{eod}_{H2O} + N^{pe}_{H2O}$
    *   **Implemented in:** `Model.calculate_n_net_water_flow(...)` (indirectly via calls to component functions)
    *   **Water Diffusion Molar Flow Rate (Eq. 40):** Dependent on `water_diffusion_coefficient_membrane` (`D_w`, an **assumed parameter**).
    *   **Electro-osmosis Water Molar Flow Rate (Eq. 42):** Dependent on `electro_osmosis_coefficient` (`n_d`, an **assumed parameter**) and current density.
    *   **Pressure Water Molar Flow Rate (Eq. 43):** Dependent on `darcy_permeability_membrane` (`K_Darcy`, an **assumed parameter**), `active_surface_area` ($A$), pressure gradient (`Delta_P`), `water_viscosity_ref` (`mu_H2O`, an **assumed parameter**), and `membrane_thickness` (`d_mem`).

### 3.6. Gas Crossover Phenomena

Gas crossover (H₂ from cathode to anode, O₂ from anode to cathode) is explicitly modeled. These fluxes directly impact the partial pressures of hydrogen and oxygen used in the Nernst equation and determine the hydrogen content in the anode outlet.

*   **Electro-osmotic Drag Flux**:
    *   **Implemented in:** `Model.calculate_n_eod_h2(J)` and `Model.calculate_n_eod_o2(J)`
    *   **Equations (47, 48):** $N_{eod,i} = \frac{J}{F z} S_i C(H_2O)$
    *   **Inputs**: `J` (Current density, A/m²).
    *   **Parameter Sourcing**: `h2_solubility` (`S_H2`), `o2_solubility` (`S_O2`), `h2o_molar_concentration` (`C(H_2O)`) are from `config.yaml['model_parameters']`.
    *   **Assumed Parameters**: `h2_crossover_z` (for H₂) and `o2_crossover_z` (for O₂) are the number of electrons for each species (typically 2 for H₂, 4 for O₂) and are **assumed parameters**.

*   **Diffusion Flux**:
    *   **Implemented in:** `Model.calculate_n_dif_h2(P_cat_H2, d_mem)` and `Model.calculate_n_dif_o2(P_an_O2, d_mem)`
    *   **Equation (49):** $N_{dif} = \epsilon_{dif} \Delta P / d_{mem}$
    *   **Inputs**: `P_cat_H2` (partial pressure of H₂ at cathode, Pa), `P_an_O2` (partial pressure of O₂ at anode, Pa), `d_mem` (membrane thickness, m).
    *   **Logic**: This equation uses a partial pressure difference as the driving force for diffusion.
    *   **Parameter Sourcing**: `h2_diffusion_constant_conc_diff` ($\epsilon_{dif,H2}$) and `o2_diffusion_constant_conc_diff` ($\epsilon_{dif,O2}$) are from `config.yaml['model_parameters']`.

*   **Pressure Difference Flux**:
    *   **Implemented in:** `Model.calculate_n_dp_h2(delta_P, d_mem)`
    *   **Equation (50):** $N_{dp,H2} = \epsilon_{dp,H2} \Delta P / d_{mem}$
    *   **Inputs**: `delta_P` (total pressure difference, Pa), `d_mem` (membrane thickness, m).
    *   **Parameter Sourcing**: `h2_diffusion_constant_press_diff` ($\epsilon_{dp,H2}$) is from `config.yaml['model_parameters']`.

*   **Partial Pressures Accounting for Crossover**:
    *   The partial pressures of hydrogen and oxygen at the electrode-membrane interfaces (`P_cat,H2`, `P_an,O2`) are influenced by gas crossover. These are defined by a coupled system of equations that requires an iterative solution.
    *   **Implemented in:** `Model.update_partial_pressures(J, P_an_Pa, P_cat_Pa, N_en_H2, N_en_O2)`
    *   **Equation (52):** $P_{an,O2} = P_{an} - \left(\frac{J}{4F} \cdot A_{O2} + N_{en,O2}\right)$ (Note: This partial pressure is for the gas phase excluding water vapor, at the interface)
    *   **Equation (53):** $P_{cat,H2} = P_{cat} + \epsilon_{dif,H2} P_{cat,H2} A_{H2} + \epsilon_{dp,H2} P_{cat,H2} A_{H2} - \epsilon_{dp,H2} P_{an,O2} A_{H2} \epsilon_{dif,H2} P_{cat,H2}$
    *   **Parameter Sourcing**: `h2_partial_pressure_enhancement_factor` ($A_{H2}$) and `o2_partial_pressure_enhancement_factor` ($A_{O2}$) are from `config.yaml['model_parameters']`.
    *   **Ambiguity Addressed**: Equation (53) for `P_cat,H2` is highly convoluted and appears to have an unusual algebraic structure, with the term $P_{cat,H2}$ appearing multiple times and also in a product with $P_{an,O2}$. **It is implemented exactly as written in the paper for faithful reproduction**, and its solution will rely on the iterative solver to converge values. This complexity is noted in `config.yaml['unclear_points']['pcath2_eq_53_complexity']`.

*   **Total Crossover Flux Density of Hydrogen and Oxygen**:
    *   **Implemented in:** `Model.calculate_n_total_h2_crossover(N_eod_H2, N_dif_H2, N_dp_H2)` and `Model.calculate_n_total_o2_crossover(N_eod_O2, N_dif_O2)`
    *   **Equation (54) (Hydrogen):** $N_{en,H2} = \epsilon_{dif,H2} \frac{P_{cat,H2}}{d_{mem}} + \frac{J}{F z_{H2}} S_{H2} C(H_2O)$
    *   **Equation (56) (Oxygen):** $N_{en,O2} = \epsilon_{dif,O2} \frac{P_{an,O2}}{d_{mem}} + \frac{J}{F z_{O2}} S_{O2} C(H_2O)$
    *   **Ambiguity Addressed**: The paper states gas crossover occurs due to electro-osmotic drag, concentration difference (diffusion), and differential pressure. However, these "total flux" equations (54, 56) only explicitly sum the electro-osmotic drag term and a term resembling diffusion from concentration difference. The distinct pressure difference flux (`N_{dp,H2}` from Eq. 50) is not directly included in the summation for `N_{en,H2}`. **Our implementation replicates the formulas as written, noting this potential omission or specific definition of "total flux" for those equations** (`config.yaml['unclear_points']['total_gas_flux_equations_54_56_completeness']`).

*   **Percentage of Hydrogen Content at the Anode Side**:
    *   **Implemented in:** `Analyzer.calculate_h2_crossover_percentage(N_en_H2, J)`
    *   **Equation (57):** $H_{2,in,O2}(\%) = \frac{N_{en,H2}}{J/(4F) + N_{en,H2}} \times 100$
    *   **Inputs**: `N_en_H2` (total hydrogen crossover flux, mol m⁻² s⁻¹), `J` (current density, A m⁻²).
    *   **Ambiguity Addressed**: The term `J/(4F)` typically represents the molar flow rate of oxygen produced at the anode. Its presence in the denominator of a hydrogen percentage calculation is unusual if the intent is to normalize by total hydrogen produced. **This term is implemented exactly as written in the paper for fidelity**, assuming it represents the molar equivalent of oxygen generated at the anode or total gas at anode for the calculation (`config.yaml['unclear_points']['h2_crossover_percentage_eq_57_denominator']`).

## 4. Parameters and Configuration

All model parameters, physical constants, operating conditions, and experimental validation settings are managed through `config.yaml`. This file is parsed by the `Parameters` class (`parameters.py`).

*   **`physical_constants`**: Universal constants like `R` (Gas universal constant) and `F` (Faraday constant), and the assumed standard reversible voltage (`V0_rev`).
*   **`model_parameters`**: Values directly extracted from Table 1 or the text of the paper. All units are converted to SI (e.g., A cm⁻² to A m⁻², Ohm cm to Ohm m, Angstrom to m) for internal consistency.
*   **`assumed_parameters`**: This is a critical section. The paper lacks explicit values for many geometric and material properties commonly required for such models. For these, **typical values for PEM electrolyzers have been assumed** and are clearly listed in `config.yaml`. These include:
    *   Electrode porosity, thickness, and channel dimensions.
    *   MEA width and length (to define active surface area).
    *   Membrane water content (`lambda`), water diffusion coefficient (`D_w`), electro-osmosis coefficient (`n_d`), Darcy permeability (`K_Darcy`), and water viscosity (`mu_H2O`).
    *   Molar masses of H₂, O₂, H₂O.
    *   Collision integral for diffusion (`Omega_D`).
    *   It is important to note that these assumed values can significantly impact the quantitative results.
*   **`operating_conditions`**: Defines default or base conditions for simulations, including current density ranges for polarization curves.
*   **`validation_cases`**: Specifies the exact operating and geometric conditions for the two experimental validation cases (Ioroi et al. and Deb et al.), including their respective data file paths (e.g., `ioroi_exp_data.csv`).
*   **`sensitivity_analysis_ranges`**: Defines the parameter ranges for the sensitivity studies as performed in the paper (e.g., membrane thickness, cathode pressure, temperature).
*   **`safety_thresholds`**: Includes relevant safety limits, such as the flammability limit for hydrogen in oxygen.
*   **`unclear_points`**: A dedicated section documenting all ambiguities, typos, or missing information identified during the paper's analysis, along with the chosen interpretation or assumption for their implementation. This is crucial for transparency and future development.

## 5. Simulation Workflow

The simulation is orchestrated by `main.py` and primarily executed by the `Solver` class.

1.  **Initialization**: `main.py` loads the `config.yaml` and initializes `Parameters`, `Model`, `Solver`, `DataLoader`, `Analyzer`, and `Plotter` objects.
2.  **Data Loading**: `DataLoader` reads experimental validation data (current density, cell voltage) from specified CSV files (e.g., `data/ioroi_exp_data.csv`, `data/deb_exp_data.csv`).
3.  **Polarization Curve Simulation**:
    *   For each validation case or sensitivity scenario, `Solver.solve_polarization_curve` is called with specific operating conditions (Temperature, Anode Pressure, Cathode Pressure, Membrane Thickness) and a range of current densities (`J_values`).
    *   **Coupled Variables Solution**: Within `solve_polarization_curve`, for each `J`, a crucial iterative fixed-point loop is performed by `Solver` (specifically `_solve_coupled_variables`) to find the converged values for the coupled partial pressures (`P_cat_H2`, `P_an_O2`) and the associated gas crossover fluxes. This loop repeatedly calls `Model.update_partial_pressures` until convergence criteria are met (based on `solver_max_iterations` and `solver_tolerance_pa` from `config.yaml`).
    *   Once converged, the `Solver` calculates all individual overpotentials (`V_act_a`, `V_act_c`, `V_ohmic`, `V_con`), `V_rev`, and the total `V_cell` by calling the respective methods in the `Model` class.
    *   All calculated intermediate and final values are stored for analysis and plotting.
4.  **Analysis**: `Analyzer` processes the simulation results.
    *   `calculate_deviation`: Quantifies the model's accuracy against experimental data (using Maximum Absolute Percentage Deviation, Root Mean Square Error, Mean Absolute Percentage Error).
    *   `calculate_h2_crossover_percentage`: Computes the anodic hydrogen content (Equation 57).
    *   `calculate_overpotential_breakdown`: Breaks down the total cell voltage into its contributing overpotential components (reversible, activation, ohmic, concentration).
    *   `calculate_ohmic_loss_breakdown`: Further dissects the ohmic loss into contributions from electrode, membrane, and bipolar plate resistances.
5.  **Plotting**: `Plotter` generates various figures based on the analyzed data, mirroring those in the original publication and saving them to the `results/plots` directory.

## 6. Reproducing Figures

The `Plotter` class is designed to reproduce the following figures from the paper:

*   **`plot_polarization_curve`**: Reproduces Fig. 2a & 2b (Polarization Curves) comparing model predictions against Ioroi et al. [30] and Deb et al. [31] experimental data.
*   **`plot_h2_crossover_sensitivity`**: Visualizes H₂ crossover percentage as a function of current density for varying membrane thickness (Fig. 3), cathode pressure (Fig. 4), and operating temperature (Fig. 5).
*   **`plot_concentration_ohmic_sensitivity`**: Displays the sensitivity of concentration overpotential to membrane thickness (Fig. 6a) and cathode pressure (Fig. 7), and ohmic overpotential to membrane thickness (Fig. 6b).
*   **`plot_ohmic_loss_breakdown`**: Shows the contribution of ohmic losses from electrodes, bipolar plates, and membrane resistances (Fig. 8).
*   **`plot_overpotential_breakdown`**: Illustrates the cumulative contribution of different overpotentials and reversible voltage to the total cell performance (Fig. 9).

All plots standardize the x-axis to "Current density (A/cm$^2$)" to match the paper's representation.

## 7. Validation Strategy

The model's accuracy is validated by comparing its predicted polarization curves against the two experimental datasets presented in the paper (Ioroi et al. [30] and Deb et al. [31]).

1.  **Data Loading**: Experimental `J-V` data (current density vs. cell voltage) is loaded via `DataLoader` from CSV files specified in `config.yaml`.
2.  **Simulation**: The model is run under the exact operating and geometric conditions specified for each experimental case in `config.yaml`.
3.  **Comparison**: The `Analyzer` calculates the deviation metrics (max. absolute percentage deviation, RMSE, MAPE) between the simulated and interpolated experimental cell voltages.
4.  **Visual Verification**: `Plotter` generates comparative plots (e.g., `ioroi_polarization_curve.png`, `deb_polarization_curve.png`) to visually assess the agreement. The expected deviations (2% for Ioroi, 5% for Deb) stated in the paper serve as benchmarks printed to the console.

## 8. Debugging Strategy

A multi-faceted debugging strategy is employed to ensure correctness and robustness throughout development:

*   **Modular Testing**: Each class (`Parameters`, `Model`, `Solver`, `DataLoader`, `Analyzer`, `Plotter`) is developed and tested independently. Unit-level verification ensures the correctness of individual equations and methods before integration.
*   **Clear Interfaces**: Emphasis is placed on well-defined function signatures and explicit input/output types between modules. Input validation is implemented at key interfaces to catch incorrect data early in the workflow.
*   **Step-by-Step Verification**: The `Solver` can be configured to output intermediate calculation results (e.g., individual overpotential contributions, converged partial pressures, crossover fluxes) for a single current density step, allowing for detailed manual verification against expected physical trends or values.
*   **Data Validation**: The `DataLoader` includes robust checks for file existence, correct column names, and appropriate data types to prevent errors stemming from malformed input files.
*   **Visual Debugging**: The `Plotter` is extensively used during development to visualize intermediate results and directly compare them against figures in the paper, which quickly highlights discrepancies in trends or magnitudes.
*   **Parameter Sensitivity Trace**: For ambiguous or assumed parameters, small-scale sensitivity studies are performed to observe their impact on the final polarization curve and crossover behavior, aiding in debugging and informing potential parameter tuning or literature search efforts.

## 9. Setup and Installation

### Required Python Packages

The following Python packages are required:

*   `numpy>=1.21.0`: For numerical operations and array manipulation.
*   `scipy>=1.7.0`: Specifically `scipy.optimize` for solving coupled non-linear equations.
*   `pandas>=1.3.0`: For efficient data loading and handling of tabular data (experimental results).
*   `matplotlib>=3.4.0`: For generating all plots and visualizations.
*   `pyyaml>=5.4.0`: For parsing the `config.yaml` file.

### Installation Instructions

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Required Packages**:
    ```bash
    pip install numpy scipy pandas matplotlib pyyaml
    ```
    Alternatively, create a `requirements.txt` file with the above packages and run `pip install -r requirements.txt`.

## 10. How to Run

To run the simulation and generate the results and plots:

1.  Ensure all dependencies are installed (see [Setup and Installation](#9-setup-and-installation)).
2.  Place your experimental data CSV files (e.g., `ioroi_exp_data.csv`, `deb_exp_data.csv`) into a `data/` directory at the root of the project. Example CSV content for experimental data:
    ```csv
    Current Density (A/m^2),Cell Voltage (V)
    1000,1.50
    2000,1.55
    3000,1.60
    ...
    ```
    Ensure column names exactly match `Current Density (A/m^2)` and `Cell Voltage (V)`.
3.  Ensure the `config.yaml` file is correctly configured with all necessary parameters and paths to experimental data files.
4.  Execute the main script from the terminal:
    ```bash
    python main.py
    ```

The script will execute the simulation scenarios (validation and sensitivity analyses) as defined in `config.yaml` and save the generated plots to the `results/plots` directory.

## 11. Output Interpretation

Upon successful execution, the program will:

*   Print status messages to the console indicating the progress of simulations (e.g., "Running Validation Cases", "Analyzing sensitivity to Membrane Thickness").
*   Display the calculated percentage deviations (Max Abs. Pct., RMSE, MAPE) for the validation cases, allowing for direct comparison with the paper's reported accuracy.
*   Generate several `.png` plot files in the `results/plots` directory. These plots will include:
    *   Polarization curves comparing model predictions with experimental data (e.g., `ioroi_polarization_curve.png`).
    *   Plots showing hydrogen crossover percentage as a function of current density for varying membrane thickness, cathode pressure, and temperature (e.g., `h2_crossover_membrane_thickness_sensitivity.png`).
    *   Breakdown plots illustrating the contribution of different overpotentials (`overpotential_contributions.png`) and ohmic resistance components (`ohmic_loss_contributions.png`) to the total cell voltage.
    *   Plots detailing the sensitivity of concentration and ohmic overpotentials (e.g., `concentration_membrane_thickness_sensitivity.png`).
*   The plot titles and labels will indicate the specific conditions and parameters being varied, matching the figures in the original paper.

## 12. Limitations and Future Work

This reproduction adheres strictly to the mathematical formulations provided in the paper. However, certain limitations and areas for future work stem directly from ambiguities or missing information in the original publication:

*   **Assumed Parameters**: A significant number of geometric and fundamental transport parameters (e.g., electrode porosity, channel dimensions, membrane water content, water diffusion coefficient) had to be **assumed** based on typical values from PEM electrolyzer literature. While these choices enable model execution, they represent potential sources of quantitative deviation from the paper's exact results if the original authors used different values. A more extensive sensitivity analysis on these assumed parameters could quantify their impact.
*   **Equation Ambiguities**:
    *   **J₀,ᵢ Temperature Dependence**: The paper lists kinetic parameters (`k₀,ᵢ`, `E_act,ᵢ`) alongside fixed exchange current densities (`J₀,ᵢ`), without explicitly defining their relationship or how `J₀,ᵢ` changes with temperature. Our implementation uses the fixed `J₀,ᵢ` values. Future work could explore incorporating an Arrhenius-type dependency to enhance model robustness across wider temperature ranges.
    *   **Knudsen Diffusion (Eq. 32) Typo**: The `F` (Faraday constant) in Equation (32) for Knudsen diffusion was identified as a likely typo and replaced with `R` (Gas constant). While this aligns with standard physics, it's an interpretation.
    *   **P_cat,H₂ (Eq. 53) Complexity**: Equation (53) defining the partial pressure of hydrogen at the cathode is highly convoluted and implicitly defined. Its exact physical derivation and full implications are challenging to parse directly from the text. Our implementation follows the mathematical expression strictly, relying on numerical solvers for convergence.
    *   **Total Crossover Fluxes (Eqs. 54, 56) Completeness**: The paper states gas crossover occurs due to electro-osmotic drag, concentration difference (diffusion), and differential pressure. However, the provided "total flux" equations (54, 56) for H₂ and O₂ explicitly sum only the electro-osmotic drag term and a term resembling diffusion from concentration difference. The distinct pressure-driven flux (`N_dp,H₂`) is not directly included in this summation. Our implementation replicates the formulas as written, implying the authors' specific definition of "total flux" for those equations.
    *   **H₂ Crossover Percentage Denominator (Eq. 57)**: The `J/(4F)` term in the denominator for calculating H₂ crossover percentage (Eq. 57) is ambiguous regarding its physical meaning (it usually represents O₂ production, not total hydrogen). Our implementation strictly uses `J/(4F)` as written.

Future work could involve:
*   **Parameter Sensitivity Analysis**: Systematically studying the impact of all assumed parameters on model outputs.
*   **External Data Sourcing**: Attempting to find more precise values for missing parameters from the cited references or industry benchmarks.
*   **Model Refinement**: Incorporating more sophisticated sub-models (e.g., explicit water activity dependence for `lambda`, detailed multi-component diffusion models) if inconsistencies arise or for extended applicability.
*   **Full Validation**: Beyond polarization curves, comparing other reported values (e.g., specific overpotential contributions at given current densities) if extractable from the paper's figures and discussion.
