# Hückel Molecular Orbital Calculator

This Python script calculates the Hückel molecular orbitals for a given molecule, based on its SMILES representation. It uses the RDKit library for cheminformatics and NumPy for numerical calculations. Matplotlib is used for plotting the energy levels.

## Features

*   Calculates the Hückel matrix based on the molecule's SMILES string.
*   Converts the Hückel matrix to Hartree notation.
*   Solves the Hückel equation to obtain eigenvalues (energy levels) and eigenvectors (molecular orbital coefficients).
*   Visualizes the energy levels using a Matplotlib plot, highlighting degenerate orbitals.
*   Handles molecules containing C, N, and O atoms (can be extended to other atom types).

## Requirements

*   Python 3.x
*   NumPy
*   RDKit
*   Matplotlib
*   PubChemPy

You can install the required packages using pip:

```bash
pip install numpy rdkit matplotlib pubchempy
```
## Start

To use the program open a terminal in the folder of the python file and type

```bash
python Huckel.py
```
