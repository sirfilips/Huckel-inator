# Hückel Molecular Orbital Calculator

This Python script calculates the Hückel molecular orbitals for a given molecule, based on its SMILES representation. It uses the RDKit library for cheminformatics and NumPy for numerical calculations. Matplotlib is used for plotting the energy levels.

## Features

* Hückel Matrix Construction: Builds the Hückel matrix for a molecule based on its SMILES input.
* Energy Level Calculation: Solves the Hückel equations to compute molecular orbital energy levels and coefficients.
* Charge and Bond Order Calculation: Calculates atomic charges, bond orders, and electron density.
* Visualization:
  - Displays the molecular structure.
  - Plots molecular orbital energy levels in an Aufbau-like diagram.
*   Results Export: Saves the computed results (charges, bond orders, energy levels, etc.) to a text file.
*   Solves the Hückel equation to obtain eigenvalues (energy levels) and eigenvectors (molecular orbital coefficients).
*   Visualizes the energy levels using a Matplotlib plot, highlighting degenerate orbitals.
*   Handles molecules containing C, N, and O atoms (can be extended to other atom types).

## Requirements

*   Python 3.x
*   NumPy
*   RDKit
*   Matplotlib
*   PubChemPy
*   Colorama

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```
## Start

To use the program open a terminal in the folder of the python file and type

```bash
python Huckel.py
```
## How to Use

1. **Run the Program**:
   Execute the script in your Python environment.

2. **Input the SMILES Code**:
   When prompted, enter the SMILES representation of the molecule you want to analyze.

3. **View Results**:
   - The program will display molecular properties such as charges, bond orders, and energy levels in the terminal.
   - Use the interactive menu to:
     - Visualize the molecular structure. (1)
     - Plot the molecular orbital energy levels. (2)
     - Save the results to a text file. (3)

4. **Exit**:
   Select the `q` option in the menu to close the program.

## Example Workflow

1. Run the program.
2. Input the SMILES code when prompted, e.g., `C1=CC=CC=C1` for benzene.
3. View the calculated properties in the terminal.
4. Use the menu options to visualize the molecule or save the results.

## Output

The program saves results in a text file located in the `output` directory. The filename includes the date and time of the analysis, e.g., `risultati_huckel_2025-03-25_12-30-45.txt`.
