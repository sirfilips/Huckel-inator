import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pubchempy as pcp

# Definire i valori di α e β per ciascun tipo di atomo e gruppo
alpha_values = {
    'C': 0.0,  # Carbonio
    'N': -0.5,  # Azoto (esempio)
    'O': -1.0,  # Ossigeno (esempio)
    'CH3': -2  # Gruppo metilico
}

beta_values = {
    ('C', 'C'): -1.0,  # Interazione C-C
    ('C', 'N'): -0.8,  # Interazione C-N (esempio)
    ('C', 'O'): -0.7,  # Interazione C-O (esempio)
    ('N', 'N'): -0.9,  # Interazione N-N (esempio)
    ('N', 'O'): -0.6,  # Interazione N-O (esempio)
    ('O', 'O'): -0.5,  # Interazione O-O (esempio)
    ('C', 'CH3'): 0.7  # Interazione C-CH3
}

def build_huckel_matrix(smiles):
    """
    Costruisce la matrice di Hückel estesa per una molecola data dal suo codice SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")

    # Conta gli idrogeni impliciti per ciascun atomo
    num_atoms = mol.GetNumAtoms()
    hydrogen_counts = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        num_hydrogens = atom.GetTotalNumHs()  # Conta gli idrogeni impliciti
        hydrogen_counts.append(num_hydrogens)
        print(f"Atom {i} ({atom.GetSymbol()}) has {num_hydrogens} hydrogen neighbors")

    mol = Chem.RemoveHs(mol)  # Rimuove gli atomi di idrogeno espliciti
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    num_atoms = adj_matrix.shape[0]

    # Creazione della matrice di Hückel
    huckel_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        atom_i = mol.GetAtomWithIdx(i).GetSymbol()
        num_hydrogens = hydrogen_counts[i]

        if atom_i == 'C' and num_hydrogens == 3:
            atom_i = 'CH3'
            print(f"Carbon atom {i} is part of a methyl group (CH3)")

        alpha_value = alpha_values.get(atom_i, 0.0)
        if callable(alpha_value):
            huckel_matrix[i, i] = alpha_value(alpha=0, beta=-1)
        else:
            huckel_matrix[i, i] = alpha_value
        print(f"Alpha value for atom {i} ({atom_i}): {huckel_matrix[i, i]}")

        for j in range(num_atoms):
            if adj_matrix[i, j] == 1:
                atom_j = mol.GetAtomWithIdx(j).GetSymbol()
                num_hydrogens_j = hydrogen_counts[j]

                if atom_j == 'C' and num_hydrogens_j == 3:
                    atom_j = 'CH3'
                    print(f"Carbon atom {j} is part of a methyl group (CH3)")

                beta_value = beta_values.get((atom_i, atom_j), beta_values.get((atom_j, atom_i), -1.0))
                huckel_matrix[i, j] = beta_value
                print(f"Beta value for interaction ({atom_i}, {atom_j}) between atoms {i} and {j}: {beta_value}")

    return huckel_matrix

def convert_to_hartree(huckel_matrix, alpha=0, beta=-1):
    """
    Converte la matrice di Hückel in notazione di Hartree.
    """
    x = 1  # Definiamo x come il fattore di conversione
    hartree_matrix = alpha * np.eye(huckel_matrix.shape[0]) + beta * huckel_matrix * x
    return hartree_matrix

def convert_to_hartree(huckel_matrix, alpha=0, beta=-1):
    """
    Converte la matrice di Hückel in notazione di Hartree.
    """
    hartree_matrix = alpha * np.eye(huckel_matrix.shape[0]) + beta * huckel_matrix
    return hartree_matrix

def solve_huckel(hartree_matrix):
    """ Risolve l'equazione di Hückel restituendo gli autovalori e autovettori. """
    eigenvalues, eigenvectors = np.linalg.eigh(hartree_matrix)
    return eigenvalues, eigenvectors

def plot_energy_levels(eigenvalues):
    """
    Visualizza i livelli energetici degli orbitali molecolari in modo simile a un diagramma di Aufbau,
    con una migliore visualizzazione degli orbitali degeneri.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ordina gli autovalori in ordine crescente
    sorted_eigenvalues = np.sort(eigenvalues)

    # Grafico dei livelli energetici
    x_offset = 0.1  # Offset orizzontale per separare gli orbitali degeneri
    current_energy = None
    degenerate_count = 0
    degenerate_positions = []

    for i, energy in enumerate(sorted_eigenvalues):
        if current_energy is None or not np.isclose(energy, current_energy):
            # Posiziona gli orbitali degeneri al centro
            if degenerate_count > 1:
                for j in range(degenerate_count):
                    x_position = (j - (degenerate_count - 1) / 2) * x_offset
                    ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
                    ax.text(x_position + 0.06, current_energy, f'OM {i - degenerate_count + j + 1}', verticalalignment='center', fontsize=10, color='black')
            elif degenerate_count == 1:
                x_position = 0
                ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
                ax.text(x_position + 0.06, current_energy, f'OM {i}', verticalalignment='center', fontsize=10, color='black')

            current_energy = energy
            degenerate_count = 0
            degenerate_positions = []

        degenerate_positions.append(i)
        degenerate_count += 1

    # Gestisce l'ultimo gruppo di orbitali degeneri
    if degenerate_count > 1:
        for j in range(degenerate_count):
            x_position = (j - (degenerate_count - 1) / 2) * x_offset
            ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
            ax.text(x_position + 0.06, current_energy, f'OM {len(sorted_eigenvalues) - degenerate_count + j + 1}', verticalalignment='center', fontsize=10, color='black')
    elif degenerate_count == 1:
        x_position = 0
        ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
        ax.text(x_position + 0.06, current_energy, f'OM {len(sorted_eigenvalues)}', verticalalignment='center', fontsize=10, color='black')

    # Imposta i limiti dell'asse y in base ai valori minimi e massimi degli autovalori
    ax.set_ylim(min(sorted_eigenvalues) - 0.5, max(sorted_eigenvalues) + 0.5)

    # Imposta i limiti dell'asse x in modo dinamico
    ax.set_xlim(-x_offset * len(sorted_eigenvalues) / 2, x_offset * len(sorted_eigenvalues) / 2)

    # Etichette e titolo
    ax.set_ylabel('Energia (unità arbitrarie)')
    ax.set_title('Diagramma di Aufbau dei Livelli Energetici degli Orbitali Molecolari')
    ax.set_xticks([])  # Rimuove i tick sull'asse x
    ax.invert_yaxis()  # Inverte l'asse y

    plt.show()

def visualize_molecule(smiles):
    """
    Visualizza la molecola data dal codice SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")
    mol = Chem.RemoveHs(mol)  # Rimuove gli atomi di idrogeno espliciti
    img = Draw.MolToImage(mol, size=(300, 300))
    img.show()

def get_iupac_name(smiles):
    compounds = pcp.get_compounds(smiles, 'smiles')
    if compounds:
        return compounds[0].iupac_name
    else:
        return None

def main():
    print(r"""
 _   _            _        _       _             _             
| | | |_   _  ___| | _____| |     (_)_ __   __ _| |_ ___  _ __
| |_| | | | |/ __| |/ / _ \ |_____| | '_ \ / _` | __/ _ \| '__|
|  _  | |_| | (__|   <  __/ |_____| | | | | (_| | || (_) | |
|_| |_|\__,_|\___|_|\_\___|_|     |_|_| |_|\__,_|\__\___/|_|
    """)
    smiles = input("Inserisci il codice SMILES della molecola: ")
    try:
        # Visualizza la molecola
        visualize_molecule(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("SMILES non valido")

        iupac_name = get_iupac_name(smiles)
        if iupac_name:
            print(f"Nome IUPAC della molecola: {iupac_name}")
        else:
            print("Nome IUPAC non trovato.")
        print(f"Codice SMILES della molecola: {smiles}")

        huckel_matrix = build_huckel_matrix(smiles)
        hartree_matrix = convert_to_hartree(huckel_matrix)
        eigenvalues, eigenvectors = solve_huckel(hartree_matrix)
        print("Matrice di Hückel (in notazione di Hartree):")
        print(huckel_matrix)
        print("\nAutovalori (Livelli energetici):")
        print(eigenvalues)
        print("\nAutovettori (Coefficienti degli orbitali molecolari):")
        for i, eigenvector in enumerate(eigenvectors.T):
            rounded_vector = np.round(eigenvector, 3)
            print(f"OM {i + 1}: {rounded_vector}")

        # Visualizzazione grafica dei livelli energetici
        plot_energy_levels(eigenvalues)

    except ValueError as e:
        print(f"Errore: {e}")

if __name__ == "__main__":
    main()
