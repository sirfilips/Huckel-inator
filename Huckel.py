import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import pubchempy as pcp
from tabulate import tabulate
import os  # Per gestire la creazione della cartella output
from datetime import datetime  # Per ottenere la data e l'ora
from colorama import Fore, Style
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO

# Definire i valori di α e β per ciascun tipo di atomo e gruppo (valori di letteratura adattati)
alpha_values = {
    'C': 0.0,       # Carbonio (standard)
    'N': -0.5,      # Azoto (esempio adattato)
    'O': -1.0,      # Ossigeno (esempio adattato)
    'CH3': -2.0,    # Gruppo metilico
    'F': -1.5,      # Fluoro
    'Cl': -1.3,     # Cloro
    'Br': -1.2,     # Bromo
    'I': -1.1       # Iodio
}

beta_values = {
    ('C', 'C'): -1.0,  # Interazione C-C (standard)
    ('C', 'N'): -0.8,  # Interazione C-N
    ('C', 'O'): -0.7,  # Interazione C-O
    ('C', 'F'): -0.6,  # Interazione C-F
    ('C', 'Cl'): -0.5, # Interazione C-Cl
    ('C', 'Br'): -0.4, # Interazione C-Br
    ('C', 'I'): -0.3,  # Interazione C-I
    ('N', 'N'): -0.9,  # Interazione N-N
    ('N', 'O'): -0.6,  # Interazione N-O
    ('O', 'O'): -0.5,  # Interazione O-O
    ('C', 'CH3'): -0.7 # Interazione C-CH3
}

def build_huckel_matrix(smiles):
    """
    Costruisce la matrice di Hückel estesa per una molecola data dal suo codice SMILES.
    Tiene conto della carica totale e dei radicali.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")

    # Calcola la carica totale e il numero di radicali
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    num_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

    print(f"Carica totale della molecola: {total_charge}")
    print(f"Numero di elettroni spaiati (radicali): {num_radicals}")

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

    return huckel_matrix, total_charge, num_radicals

def convert_to_hartree(huckel_matrix, alpha=0, beta=-1):
    """
    Converte la matrice di Hückel in notazione di Hartree.
    """
    hartree_matrix = alpha * np.eye(huckel_matrix.shape[0]) + beta * huckel_matrix
    return hartree_matrix

def solve_huckel(hartree_matrix):
    """ Risolve l'equazione di Hückel restituendo gli autovalori e autovettori. """
    eigenvalues, eigenvectors = np.linalg.eigh(hartree_matrix)

    # Ordina gli autovalori in ordine decrescente e riordina gli autovettori di conseguenza
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Indici per ordine decrescente
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors

def plot_energy_levels(eigenvalues):
    """
    Visualizza i livelli energetici degli orbitali molecolari in modo simile a un diagramma di Aufbau,
    con una migliore visualizzazione degli orbitali degeneri.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Ordina gli autovalori in ordine decrescente
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

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
                    ax.text(x_position + 0.06, current_energy, f'OM {i - degenerate_count + j + 1}',
                            verticalalignment='center', fontsize=10, color='black')
            elif degenerate_count == 1:
                x_position = 0
                ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
                ax.text(x_position + 0.06, current_energy, f'OM {i}',
                        verticalalignment='center', fontsize=10, color='black')

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
            ax.text(x_position + 0.06, current_energy, f'OM {len(sorted_eigenvalues) - degenerate_count + j + 1}',
                    verticalalignment='center', fontsize=10, color='black')
    elif degenerate_count == 1:
        x_position = 0
        ax.plot([x_position, x_position + 0.05], [current_energy, current_energy], 'k-', lw=2)
        ax.text(x_position + 0.06, current_energy, f'OM {len(sorted_eigenvalues)}',
                verticalalignment='center', fontsize=10, color='black')

    # Imposta i limiti dell'asse y in base ai valori minimi e massimi degli autovalori
    ax.set_ylim(min(sorted_eigenvalues) - 0.5, max(sorted_eigenvalues) + 0.5)

    # Imposta i limiti dell'asse x in modo dinamico
    ax.set_xlim(-x_offset * len(sorted_eigenvalues) / 2, x_offset * len(sorted_eigenvalues) / 2)

    # Etichette e titolo
    ax.set_ylabel('Energia (unità arbitrarie di energia)')
    ax.set_title('Diagramma dei Livelli Energetici degli Orbitali Molecolari')
    ax.set_xticks([])  # Rimuove i tick sull'asse x
    ax.invert_yaxis()  # Inverte l'asse y

    plt.show()

def visualize_molecule(smiles):
    """
    Visualizza la molecola data dal codice SMILES con gli atomi numerati in modo coerente con i calcoli.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")
    
    # Rimuove gli idrogeni espliciti per mantenere coerenza con i calcoli
    mol = Chem.RemoveHs(mol)

    # Crea un dizionario per etichettare gli atomi con i loro indici
    atom_labels = {atom.GetIdx(): str(atom.GetIdx()) for atom in mol.GetAtoms()}

    # Genera l'immagine con le etichette degli atomi
    img = Draw.MolToImage(mol, size=(300, 300), atomLabels=atom_labels)
    img.show()

def get_iupac_name(smiles):
    """
    Ottiene il nome IUPAC della molecola utilizzando PubChem.
    Gestisce eventuali errori, come la mancanza di connessione a Internet.
    """
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds:
            return compounds[0].iupac_name
        else:
            return None
    except Exception as e:
        print(Fore.RED + f"Errore durante il recupero del nome IUPAC: {e}" + Style.RESET_ALL)
        return "Errore: Nome IUPAC non disponibile"

def calculate_charges(eigenvectors, num_electrons):
    """
    Calcola la carica su ogni atomo di carbonio.
    """
    num_atoms = eigenvectors.shape[0]
    charges = np.zeros(num_atoms)
    num_occupied_orbitals = num_electrons // 2

    # Calcola la somma dei quadrati dei coefficienti per gli orbitali occupati
    for i in range(num_atoms):
        for j in range(num_occupied_orbitals):
            charges[i] += eigenvectors[i, j] ** 2

    # La carica è 1 - somma dei quadrati dei coefficienti
    charges = 1 - charges

    # Normalizza le cariche per garantire che la somma sia coerente con la carica totale
    total_charge = 0  # Cambia se la molecola ha una carica netta diversa
    charge_correction = (total_charge - np.sum(charges)) / num_atoms
    charges += charge_correction

    return charges

def calculate_bond_orders(eigenvectors, num_electrons):
    """
    Calcola l'ordine di legame di tutti i legami.
    """
    num_atoms = eigenvectors.shape[0]
    bond_orders = np.zeros((num_atoms, num_atoms))
    num_occupied_orbitals = num_electrons // 2

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            for k in range(num_occupied_orbitals):
                bond_orders[i, j] += eigenvectors[i, k] * eigenvectors[j, k]
            bond_orders[j, i] = bond_orders[i, j]

    return bond_orders

def calculate_electron_density(eigenvectors, num_electrons):
    """
    Calcola la densità elettronica totale su ogni atomo.
    """
    num_atoms = eigenvectors.shape[0]
    density = np.zeros(num_atoms)
    num_occupied_orbitals = num_electrons // 2

    for i in range(num_atoms):
        for j in range(num_occupied_orbitals):
            density[i] += eigenvectors[i, j] ** 2

    return density

def save_results_to_txt(smiles, iupac_name, charges, electron_density, bond_orders, eigenvalues, eigenvectors):
    """
    Salva i risultati in un file di testo con un nome che include data e ora.
    """
    # Creazione della cartella "output" se non esiste
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Genera il nome del file con data e ora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"risultati_huckel_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as file:
        file.write(f"Codice SMILES: {smiles}\n")
        file.write(f"Nome IUPAC: {iupac_name if iupac_name else 'Non trovato'}\n\n")
        file.write("Cariche sugli atomi di carbonio:\n")
        for i, charge in enumerate(charges):
            file.write(f"Atomo {i}: {charge:.3f}\n")
        file.write("\nDensità elettronica su ogni atomo:\n")
        for i, density in enumerate(electron_density):
            file.write(f"Atomo {i}: {density:.3f}\n")
        file.write("\nOrdini di legame:\n")
        for i in range(bond_orders.shape[0]):
            for j in range(i + 1, bond_orders.shape[1]):
                if bond_orders[i, j] != 0:
                    file.write(f"Legame {i}-{j}: {bond_orders[i, j]:.3f}\n")
        file.write("\nAutovalori (Livelli energetici):\n")
        file.write(", ".join(f"{val:.3f}" for val in eigenvalues) + "\n")
        file.write("\nAutovettori (Coefficienti degli orbitali molecolari):\n")
        for i, eigenvector in enumerate(eigenvectors.T):
            file.write(f"OM {i + 1}: {np.round(eigenvector, 3)}\n")

    print(f"Risultati salvati in: {filepath}")

def plot_orbital_phases(eigenvectors):
    """
    Visualizza graficamente le fasi degli orbitali molecolari.
    Le fasi positive e negative sono rappresentate con colori diversi.
    I grafici sono ordinati in modo che il primo corrisponda all'orbital molecolare con l'energia più bassa.
    """
    num_orbitals = eigenvectors.shape[1]
    num_atoms = eigenvectors.shape[0]

    # Inverti l'ordine degli orbitali per partire dal più basso
    eigenvectors = eigenvectors[:, ::-1]

    fig, axes = plt.subplots(num_orbitals, 1, figsize=(8, 2 * num_orbitals), sharex=True)
    if num_orbitals == 1:
        axes = [axes]  # Assicura che `axes` sia sempre una lista

    for i, ax in enumerate(axes):
        orbital = eigenvectors[:, i]
        colors = ['blue' if coeff >= 0 else 'red' for coeff in orbital]
        ax.bar(range(num_atoms), orbital, color=colors, edgecolor='black')
        # Aggiorna il titolo per riflettere l'ordine corretto
        ax.set_title(f"Orbital Molecolare OM {num_orbitals - i} (OM{num_orbitals - i})")
        ax.set_ylabel("Coefficiente")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.xlabel("Indice dell'atomo")
    plt.tight_layout()
    plt.show()

def visualize_orbitals_on_molecule(smiles, eigenvectors, orbital_index):
    """
    Visualizza la molecola con le fasi degli orbitali molecolari sovrapposte.
    Le fasi positive e negative sono rappresentate con gradienti di colore sugli atomi.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")
    
    # Rimuove gli idrogeni espliciti per mantenere coerenza con i calcoli
    mol = Chem.RemoveHs(mol)
    
    # Genera le coordinate 2D per la molecola
    rdDepictor.Compute2DCoords(mol)
    
    # Ottieni i coefficienti dell'orbitale selezionato
    orbital_coefficients = eigenvectors[:, orbital_index]
    
    # Normalizza i coefficienti per mappare i valori su una scala di colori
    max_coeff = max(abs(orbital_coefficients))
    normalized_coefficients = orbital_coefficients / max_coeff
    
    # Crea un dizionario per mappare i colori agli atomi
    atom_colors = {}
    for i, coeff in enumerate(normalized_coefficients):
        if coeff > 0:
            atom_colors[i] = (1.0, 0.0, 0.0)  # Rosso per fasi positive
        else:
            atom_colors[i] = (0.0, 0.0, 1.0)  # Blu per fasi negative
    
    # Disegna la molecola con i colori degli atomi
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    
    # Converti l'immagine in un formato visualizzabile senza salvarla su disco
    img_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    img.show()

def main():
    print(Fore.GREEN + r"""
 _   _            _        _       _             _             
| | | |_   _  ___| | _____| |     (_)_ __   __ _| |_ ___  _ __ 
| |_| | | | |/ __| |/ / _ \ |_____| | '_ \ / _` | __/ _ \| '__|
|  _  | |_| | (__|   <  __/ |_____| | | | | (_| | || (_) | |   
|_| |_|\__,_|\___|_|\_\___|_|     |_|_| |_|\__,_|\__\___/|_|   
    """ + Style.RESET_ALL)
    smiles = input(Fore.YELLOW + "Inserisci il codice SMILES della molecola: " + Style.RESET_ALL)
    try:
        print(Fore.MAGENTA + "\n--- Costruzione della matrice di Hückel ---" + Style.RESET_ALL)
        huckel_matrix, total_charge, num_radicals = build_huckel_matrix(smiles)
        hartree_matrix = convert_to_hartree(huckel_matrix)
        eigenvalues, eigenvectors = solve_huckel(hartree_matrix)

        print(Fore.MAGENTA + "\n--- Informazioni sulla molecola ---" + Style.RESET_ALL)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("SMILES non valido")

        iupac_name = get_iupac_name(smiles)
        if iupac_name:
            print(Fore.CYAN + f"Nome IUPAC della molecola: {iupac_name}" + Style.RESET_ALL)
        else:
            print(Fore.RED + "Nome IUPAC non trovato." + Style.RESET_ALL)
        print(Fore.CYAN + f"Codice SMILES della molecola: {smiles}" + Style.RESET_ALL)

        num_electrons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

        print(Fore.MAGENTA + "\n--- Cariche sugli atomi di carbonio ---" + Style.RESET_ALL)
        charges = calculate_charges(eigenvectors, num_electrons)
        charge_table = [[f"Atomo {i}", f"{charge:.3f}"] for i, charge in enumerate(charges)]
        print(tabulate(charge_table, headers=["Atomo", "Carica"], tablefmt="grid"))

        print(Fore.MAGENTA + "\n--- Densità elettronica su ogni atomo ---" + Style.RESET_ALL)
        electron_density = calculate_electron_density(eigenvectors, num_electrons)
        density_table = [[f"Atomo {i}", f"{density:.3f}"] for i, density in enumerate(electron_density)]
        print(tabulate(density_table, headers=["Atomo", "Densità elettronica"], tablefmt="grid"))

        print(Fore.MAGENTA + "\n--- Ordini di legame ---" + Style.RESET_ALL)
        bond_orders = calculate_bond_orders(eigenvectors, num_electrons)
        for i in range(bond_orders.shape[0]):
            for j in range(i + 1, bond_orders.shape[1]):
                if bond_orders[i, j] != 0:
                    print(f"Legame {i}-{j}: {bond_orders[i, j]:.3f}")

        print(Fore.MAGENTA + "\n--- Matrice di Hückel (in notazione di Hartree) ---" + Style.RESET_ALL)
        print(hartree_matrix)

        eigenvalues, eigenvectors = np.linalg.eigh(hartree_matrix)

        # Ordina gli autovalori in ordine decrescente e riordina gli autovettori di conseguenza
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Indici per ordine decrescente
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        print(Fore.MAGENTA + "\n--- Autovalori (Livelli energetici) ---" + Style.RESET_ALL)
        print(eigenvalues)

        print(Fore.MAGENTA + "\n--- Autovettori (Coefficienti degli orbitali molecolari) ---" + Style.RESET_ALL)
        for i, eigenvector in enumerate(eigenvectors.T):
            rounded_vector = np.round(eigenvector, 3)  # Arrotonda i valori a 3 cifre decimali
            print(f"OM {i + 1}: {rounded_vector}")

        # Menù interattivo
        while True:
            print(Fore.YELLOW + "\n--- Menù ---" + Style.RESET_ALL)
            print("1. Visualizza la struttura della molecola")
            print("2. Visualizza il diagramma degli orbitali molecolari (OM)")
            print("3. Salva i risultati in un file")
            print("4. Visualizza le fasi degli orbitali molecolari")
            print("5. Visualizza gli orbitali molecolari sulla molecola")
            print("q. Chiudi il programma")
            choice = input(Fore.YELLOW + "Scegli un'opzione: " + Style.RESET_ALL)

            if choice == "1":
                print(Fore.MAGENTA + "\n--- Visualizzazione della struttura della molecola ---" + Style.RESET_ALL)
                visualize_molecule(smiles)
            elif choice == "2":
                print(Fore.MAGENTA + "\n--- Diagramma degli orbitali molecolari (OM) ---" + Style.RESET_ALL)
                plot_energy_levels(eigenvalues)
                plt.show(block=False)  # Evita il blocco del programma
            elif choice == "3":
                print(Fore.MAGENTA + "\n--- Salvataggio dei risultati ---" + Style.RESET_ALL)
                save_results_to_txt(
                    smiles,
                    iupac_name,
                    charges,
                    electron_density,
                    bond_orders,
                    eigenvalues,
                    eigenvectors,
                )
            elif choice == "4":
                print(Fore.MAGENTA + "\n--- Visualizzazione delle fasi degli orbitali molecolari ---" + Style.RESET_ALL)
                plot_orbital_phases(eigenvectors)
            elif choice == "5":
                print(Fore.MAGENTA + "\n--- Visualizzazione degli orbitali molecolari sulla molecola ---" + Style.RESET_ALL)
                try:
                    orbital_index = int(input(Fore.YELLOW + "Inserisci il numero dell'orbitale molecolare (OM) da visualizzare (1-based): " + Style.RESET_ALL)) - 1
                    if 0 <= orbital_index < eigenvectors.shape[1]:
                        visualize_orbitals_on_molecule(smiles, eigenvectors, orbital_index)
                    else:
                        print(Fore.RED + "Indice non valido. Riprova." + Style.RESET_ALL)
                except ValueError:
                    print(Fore.RED + "Inserisci un numero valido." + Style.RESET_ALL)
            elif choice == "q":
                print(Fore.RED + "Chiusura del programma..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Opzione non valida. Riprova." + Style.RESET_ALL)

    except ValueError as e:
        print(Fore.RED + f"Errore: {e}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
