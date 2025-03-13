import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Definire i valori di α e β per ciascun tipo di atomo
alpha_values = {
    'C': 0.0,  # Carbonio
    'N': -0.5,  # Azoto (esempio)
    'O': -1.0,  # Ossigeno (esempio)
    # Aggiungere altri atomi se necessario
}

beta_values = {
    ('C', 'C'): -1.0,  # Interazione C-C
    ('C', 'N'): -0.8,  # Interazione C-N (esempio)
    ('C', 'O'): -0.7,  # Interazione C-O (esempio)
    ('N', 'N'): -0.9,  # Interazione N-N (esempio)
    ('N', 'O'): -0.6,  # Interazione N-O (esempio)
    ('O', 'O'): -0.5,  # Interazione O-O (esempio)
    # Aggiungere altre interazioni se necessario
}

def build_huckel_matrix(smiles):
    """
    Costruisce la matrice di Hückel estesa per una molecola data dal suo codice SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")

    mol = Chem.RemoveHs(mol)  # Rimuove gli atomi di idrogeno espliciti
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    num_atoms = adj_matrix.shape[0]

    # Creazione della matrice di Hückel
    huckel_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        atom_i = mol.GetAtomWithIdx(i).GetSymbol()
        huckel_matrix[i, i] = alpha_values.get(atom_i, 0.0)  # Energia alfa per l'atomo i
        for j in range(num_atoms):
            if adj_matrix[i, j] == 1:
                atom_j = mol.GetAtomWithIdx(j).GetSymbol()
                huckel_matrix[i, j] = beta_values.get((atom_i, atom_j), -1.0)  # Interazione tra atomi connessi

    return huckel_matrix

def convert_to_hartree(huckel_matrix, alpha=0, beta=-1):
    """
    Converte la matrice di Hückel in notazione di Hartree.
    """
    x = 1  # Definiamo x come il fattore di conversione
    hartree_matrix = alpha * np.eye(huckel_matrix.shape[0]) + beta * huckel_matrix * x
    return hartree_matrix

def solve_huckel(hartree_matrix):
    """ Risolve l'equazione di Hückel restituendo gli autovalori e autovettori. """
    eigenvalues, eigenvectors = np.linalg.eigh(hartree_matrix)
    sorted_indices = np.argsort(eigenvalues)
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def plot_energy_levels(eigenvalues):
    """
    Visualizza i livelli energetici degli orbitali molecolari in modo simmetrico.
    """
    # Creazione del grafico
    fig, ax = plt.subplots(figsize=(8, 6))

    # Livelli energetici
    levels = np.arange(len(eigenvalues))

    # Ordina gli autovalori in ordine decrescente e ottieni gli indici originali
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Posizione simmetrica degli orbitali
    positions = np.concatenate([-levels[::-1], levels + 1])

    # Grafico a barre orizzontali simmetriche
    for i, energy in enumerate(sorted_eigenvalues):
        ax.plot([-i-1, i+1], [energy, energy], 'o-', color='skyblue', markersize=8, markeredgecolor='gray')
        ax.text(i+1, energy, f'OM {sorted_indices[i] + 1}', verticalalignment='center', fontsize=10, color='black')

    # Etichette e titolo
    ax.set_ylabel('Energia (unità arbitrarie)')
    ax.set_title('Livelli Energetici degli Orbitali Molecolari')
    ax.set_xticks([])  # Rimuove i tick sull'asse x
    ax.invert_yaxis()  # Inverte l'asse y

    # Evidenziazione degli orbitali degeneri
    degenerate_levels = {}
    for i, energy in enumerate(sorted_eigenvalues):
        if energy in degenerate_levels:
            degenerate_levels[energy].append(i)
        else:
            degenerate_levels[energy] = [i]

    for energy, indices in degenerate_levels.items():
        if len(indices) > 1:
            for idx in indices:
                ax.plot([-idx-1, idx+1], [energy, energy], 'o-', color='lightcoral', markersize=8, markeredgecolor='gray')

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