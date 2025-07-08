import pandas as pd
import os
from typing import Any, Dict # Not strictly needed for this file, but good for future proofing/consistency

class DataLoader:
    """
    A utility class responsible for loading experimental validation data from
    specified CSV files. It parses current density and cell voltage columns,
    ensures proper data types, handles potential missing values, and standardizes
    column names for consistency with the simulation results.
    """

    def __init__(self):
        """
        Initializes the DataLoader. No parameters are needed for initialization
        as data loading is handled by the method call.
        """
        pass

    def load_experimental_data(self, filepath: str, case_name: str) -> pd.DataFrame:
        """
        Loads experimental data from a specified CSV file, validates its content,
        and prepares it for comparison with model outputs.

        The method expects the CSV file to contain columns named 'Current Density (A/m^2)'
        and 'Cell Voltage (V)'. It renames these columns to 'J' and 'V_exp' respectively.

        Args:
            filepath (str): The absolute or relative path to the CSV file containing
                            the experimental data.
            case_name (str): A descriptive name for the experimental case (e.g.,
                             "Ioroi et al. [30] (Fig. 2a)"). Used for error messages.

        Returns:
            pd.DataFrame: A DataFrame with 'J' (Current Density in A/m^2) and
                          'V_exp' (Experimental Cell Voltage in V) columns.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If the loaded DataFrame is empty, or if essential columns
                        are missing, or if all data becomes invalid after cleaning.
        """
        # 1. File Existence Check
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Experimental data file not found for '{case_name}': {filepath}"
            )

        try:
            # 2. CSV Reading
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(
                f"Error reading CSV file for '{case_name}' at '{filepath}': {e}"
            )

        # 3. Empty File/DataFrame Check
        if df.empty:
            raise ValueError(
                f"The loaded experimental data file for '{case_name}' at '{filepath}' is empty."
            )

        # Define expected original column names and their standardized equivalents
        expected_j_col = 'Current Density (A/m^2)'
        expected_v_col = 'Cell Voltage (V)'
        standard_j_col = 'J'
        standard_v_col = 'V_exp'

        # 4. Column Identification and Renaming
        # 4.1. Column Existence Check
        if expected_j_col not in df.columns:
            raise ValueError(
                f"Missing expected current density column '{expected_j_col}' "
                f"in experimental data for '{case_name}' at '{filepath}'."
            )
        if expected_v_col not in df.columns:
            raise ValueError(
                f"Missing expected cell voltage column '{expected_v_col}' "
                f"in experimental data for '{case_name}' at '{filepath}'."
            )

        # 4.2. Standardization: Select and rename columns
        df_processed = df[[expected_j_col, expected_v_col]].copy()
        df_processed = df_processed.rename(columns={
            expected_j_col: standard_j_col,
            expected_v_col: standard_v_col
        })

        # 5. Data Type Conversion and Cleaning
        # 5.1. Numeric Conversion (with coercion for non-numeric values)
        df_processed[standard_j_col] = pd.to_numeric(df_processed[standard_j_col], errors='coerce')
        df_processed[standard_v_col] = pd.to_numeric(df_processed[standard_v_col], errors='coerce')

        # 5.2. Missing Value Handling
        initial_rows = len(df_processed)
        df_processed.dropna(subset=[standard_j_col, standard_v_col], inplace=True)
        dropped_rows = initial_rows - len(df_processed)

        if dropped_rows > 0:
            print(f"Warning: Dropped {dropped_rows} rows with non-numeric or missing values "
                  f"in 'J' or 'V_exp' for '{case_name}'.")

        # 5.3. Data Validity Check after cleaning
        if df_processed.empty:
            raise ValueError(
                f"All rows were dropped from experimental data for '{case_name}' at '{filepath}' "
                f"due to missing or invalid 'J' or 'V_exp' values."
            )

        # 6. Unit Consistency:
        # As per the instructions and config.yaml, it's assumed that 'Current Density (A/m^2)'
        # in the input CSV is already in A/m^2. No explicit conversion is performed here.
        # Cell Voltage is standard V.

        return df_processed

