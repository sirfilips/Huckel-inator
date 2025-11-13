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
    Costruisce la matrice di Hückel estesa per il sistema π di interesse in una molecola data dal suo codice SMILES.
    I gruppi sostituenti sono trattati come unità singole con valori di α e β definiti.
    Gli atomi radicalici adiacenti a doppi legami o legami aromatici sono inclusi nel sistema π.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES non valido")

    mol = Chem.RemoveHs(mol)
    debug_atom_connections(mol)

    # Identifica gli atomi del sistema π (doppi legami coniugati o aromatici)
    pi_system_atoms = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC]:
            pi_system_atoms.add(bond.GetBeginAtomIdx())
            pi_system_atoms.add(bond.GetEndAtomIdx())

    # Includi atomi radicalici adiacenti a doppi legami o aromatici nel sistema π
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            if any(neigh.GetIdx() in pi_system_atoms for neigh in atom.GetNeighbors()):
                pi_system_atoms.add(atom.GetIdx())

    # --- NUOVO: Includi atomi carichi adiacenti al sistema π ---
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            if any(neigh.GetIdx() in pi_system_atoms for neigh in atom.GetNeighbors()):
                pi_system_atoms.add(atom.GetIdx())

    # Creazione della matrice di Hückel per il sistema π
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
    num_atoms = len(pi_system_atoms)
    huckel_matrix = np.zeros((num_atoms, num_atoms))
    pi_system_atoms = sorted(pi_system_atoms)  # Ordina gli indici degli atomi

    atom_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(pi_system_atoms)}

    for i, old_idx_i in enumerate(pi_system_atoms):
        atom_i = mol.GetAtomWithIdx(old_idx_i).GetSymbol()
        alpha_value = alpha_values.get(atom_i, 0.0)
        huckel_matrix[i, i] = alpha_value

        for j, old_idx_j in enumerate(pi_system_atoms):
            if adj_matrix[old_idx_i, old_idx_j] == 1 and i != j:
                atom_j = mol.GetAtomWithIdx(old_idx_j).GetSymbol()
                beta_value = beta_values.get((atom_i, atom_j), beta_values.get((atom_j, atom_i), -1.0))
                huckel_matrix[i, j] = beta_value

    # Aggiunge l'effetto dei sostituenti come correzione ai valori di α
    for i, old_idx in enumerate(pi_system_atoms):
        atom = mol.GetAtomWithIdx(old_idx)
        substituents = []
        for neighbor in atom.GetNeighbors():
            # Non considera come sostituente un atomo radicalico incluso nel sistema π
            if neighbor.GetIdx() not in pi_system_atoms:
                substituent = neighbor.GetSymbol()
                substituent_alpha = alpha_values.get(substituent, 0.0)
                huckel_matrix[i, i] += substituent_alpha
                substituents.append(f"{substituent} (α = {substituent_alpha})")
        if substituents:
            print(f"Atomo {old_idx} ({atom.GetSymbol()}) ha i seguenti sostituenti: {', '.join(substituents)}")

    return huckel_matrix

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
    Ora usa la densità elettronica corretta (2 elettroni per orbitale occupato,
    +1 per orbitale singolarmente occupato se num_electrons è dispari).
    """
    # Usa la funzione di densità per ottenere le popolazioni atomiche π
    population = calculate_electron_density(eigenvectors, num_electrons)

    # Presa di riferimento: 1 elettrone π per atomo (se vuoi altro riferimento, cambiare qui)
    num_atoms = eigenvectors.shape[0]
    reference = 1.0

    charges = reference - population

    # Correzione per carica totale (se necessario)
    total_charge = 0  # cambiare se la molecola ha carica netta diversa
    if num_atoms > 0:
        charge_correction = (total_charge - np.sum(charges)) / num_atoms
        charges += charge_correction

    return charges

def calculate_bond_orders(eigenvectors, num_electrons):
    """
    Calcola l'ordine di legame di tutti i legami.
    Usa l'espressione: BO_ij = 2 * sum_{occupati} c_i,k * c_j,k
    Se num_electrons è dispari aggiunge il contributo 1 * c_i,last * c_j,last.
    """
    num_atoms = eigenvectors.shape[0]
    bond_orders = np.zeros((num_atoms, num_atoms))

    num_full_orbitals = num_electrons // 2
    has_half = (num_electrons % 2) == 1
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            bo = 0.0
            # contributo orbitalmente doppi
            for k in range(num_full_orbitals):
                bo += 2.0 * eigenvectors[i, k] * eigenvectors[j, k]
            # contributo eventuale orbitale singolarmente occupato
            if has_half:
                bo += 1.0 * eigenvectors[i, num_full_orbitals] * eigenvectors[j, num_full_orbitals]
            bond_orders[i, j] = bo
            bond_orders[j, i] = bo

    return bond_orders

def calculate_electron_density(eigenvectors, num_electrons):
    """
    Calcola la densità elettronica totale su ogni atomo.
    Densità = 2 * sum_{orbitali doppi occupati} c_i,k^2 + 1 * (se c'è un orbitale singolarmente occupato) c_i,last^2
    """
    num_atoms = eigenvectors.shape[0]
    density = np.zeros(num_atoms)

    num_full_orbitals = num_electrons // 2
    has_half = (num_electrons % 2) == 1

    # contributo degli orbitali doppi
    for i in range(num_atoms):
        for j in range(num_full_orbitals):
            density[i] += 2.0 * eigenvectors[i, j] ** 2

    # contributo dell'eventuale orbitale singolarmente occupato
    if has_half:
        half_idx = num_full_orbitals
        for i in range(num_atoms):
            density[i] += 1.0 * eigenvectors[i, half_idx] ** 2

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
            file.write(f"Atomo {i+1}: {charge:.3f}\n")  # <-- parte da 1
        file.write("\nDensità elettronica su ogni atomo:\n")
        for i, density in enumerate(electron_density):
            file.write(f"Atomo {i+1}: {density:.3f}\n")  # <-- parte da 1
        file.write("\nOrdini di legame:\n")
        for i in range(bond_orders.shape[0]):
            for j in range(i + 1, bond_orders.shape[1]):
                if bond_orders[i, j] != 0:
                    file.write(f"Legame {i+1}-{j+1}: {bond_orders[i, j]:.3f}\n")  # <-- parte da 1
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
    
    mol = Chem.RemoveHs(mol)
    
    # Identifica gli atomi del sistema π (stessa logica di build_huckel_matrix)
    pi_system_atoms = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC]:
            pi_system_atoms.add(bond.GetBeginAtomIdx())
            pi_system_atoms.add(bond.GetEndAtomIdx())
    # Includi atomi radicalici adiacenti a doppi legami o aromatici nel sistema π
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            if any(neigh.GetIdx() in pi_system_atoms for neigh in atom.GetNeighbors()):
                pi_system_atoms.add(atom.GetIdx())
    # --- NUOVO: Includi atomi carichi adiacenti al sistema π ---
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            if any(neigh.GetIdx() in pi_system_atoms for neigh in atom.GetNeighbors()):
                pi_system_atoms.add(atom.GetIdx())
    pi_system_atoms = sorted(pi_system_atoms)  # Ordina gli indici degli atomi

    # Genera le coordinate 2D per la molecola
    rdDepictor.Compute2DCoords(mol)
    
    # Ottieni i coefficienti dell'orbitale selezionato
    orbital_coefficients = eigenvectors[:, orbital_index]
    
    # Normalizza i coefficienti per mappare i valori su una scala di colori
    max_coeff = max(abs(orbital_coefficients))
    normalized_coefficients = orbital_coefficients / max_coeff if max_coeff != 0 else orbital_coefficients
    
    # Crea un dizionario per mappare i colori agli atomi
    atom_colors = {}
    for i, coeff in enumerate(normalized_coefficients):
        original_idx = pi_system_atoms[i]  # Mappa l'indice del sistema π all'indice originale
        if coeff > 0:
            atom_colors[original_idx] = (1.0, 0.0, 0.0)  # Rosso per fasi positive
        else:
            atom_colors[original_idx] = (0.0, 0.0, 1.0)  # Blu per fasi negative
    
    # Disegna la molecola con i colori degli atomi
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    
    # Converti l'immagine in un formato visualizzabile senza salvarla su disco
    img_data = drawer.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    img.show()

def debug_atom_connections(mol):
    """
    Stampa le connessioni di tutti gli atomi nella molecola, inclusi gli idrogeni impliciti,
    i valori di α e β utilizzati, la carica totale e il numero di elettroni spaiati.
    Ora mostra una tabella per gli atomi di carbonio.
    """
    print("\n--- Connessioni degli atomi e valori di α e β ---")
    # Calcola la carica totale della molecola
    total_formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    print(f"Carica formale della molecola: {total_formal_charge}")
    # Calcola il numero totale di elettroni spaiati
    total_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    print(f"Numero di elettroni spaiati (radicali): {total_radicals}")

    # --- 1. DEBUG: tabella degli atomi di carbonio ---
    # Prepara dati per la tabella degli atomi di carbonio
    carbon_table = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C":
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            formal_charge = atom.GetFormalCharge()
            num_radical = atom.GetNumRadicalElectrons()
            hydrogen_count = atom.GetTotalNumHs()
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors() if n.GetSymbol() != "H"]
            carbon_table.append([
                atom_idx + 1,  # <-- parte da 1
                atom_symbol,
                formal_charge,
                num_radical,
                hydrogen_count,
                ", ".join(neighbors)
            ])
    if carbon_table:
        print(tabulate(
            carbon_table,
            headers=["Indice", "Simbolo", "Carica formale", "Elettroni spaiati", "H legati", "Vicini"],
            tablefmt="grid"
        ))

    # Stampa dettagli per tutti gli atomi (come prima)
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        neighbors = []
        hydrogen_count = atom.GetTotalNumHs()
        for neighbor in atom.GetNeighbors():
            neighbor_symbol = neighbor.GetSymbol()
            if neighbor_symbol != "H":
                neighbors.append(neighbor_symbol)
        print(f"Atomo {atom_idx + 1} ({atom_symbol}) è legato a: {', '.join(neighbors)}")  # <-- parte da 1
        if hydrogen_count > 0:
            print(f"  {atom_symbol} ha {hydrogen_count} idrogeni legati.")
        alpha_value = alpha_values.get(atom_symbol, 0.0)
        print(f"  Valore di α per {atom_symbol}: {alpha_value}")
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            print(f"  {atom_symbol} ha {num_radical} elettrone/i spaiato/i (radicale).")
        for neighbor in atom.GetNeighbors():
            neighbor_symbol = neighbor.GetSymbol()
            if neighbor_symbol != "H":
                beta_value = beta_values.get((atom_symbol, neighbor_symbol), beta_values.get((neighbor_symbol, atom_symbol), None))
                if beta_value is not None:
                    print(f"  Valore di β per legame {atom_symbol}-{neighbor_symbol}: {beta_value}")
    print("--- Fine connessioni e valori di α e β ---\n")

def print_ascii_energy_diagram(eigenvalues):
    """
    Stampa un diagramma ASCII delle energie degli orbitali molecolari,
    con trattini tra le etichette OM degeneri sulla stessa riga.
    Le etichette OM partono dall'energia più alta (OM1) e scendono.
    """
    import numpy as np
    from colorama import Fore, Style

    # Ordina dal più alto al più basso
    rounded = np.round(eigenvalues, 4)
    idx_sorted = np.argsort(rounded)[::-1]
    rounded = rounded[idx_sorted]

    # Raggruppa OM degeneri
    unique_energies = []
    om_groups = []
    om_counter = 1
    for i, e in enumerate(rounded):
        found = False
        for idx, ue in enumerate(unique_energies):
            if np.isclose(e, ue):
                om_groups[idx].append(f"OM{om_counter}")
                found = True
                break
        if not found:
            unique_energies.append(e)
            om_groups.append([f"OM{om_counter}"])
        om_counter += 1

    print(Fore.CYAN + "\nDiagramma ASCII delle energie degli orbitali molecolari:" + Style.RESET_ALL)
    for energy, oms in zip(unique_energies, om_groups):
        om_line = " ----- ".join(oms)
        print(f"{energy:7.3f} ----- {om_line}")

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
        huckel_matrix = build_huckel_matrix(smiles)
        if huckel_matrix.shape[0] == 0:
            raise ValueError("Nessun atomo π trovato nella molecola. Controlla il codice SMILES o la selezione degli atomi π.")
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

        # --- 2. MAIN: tabelle cariche e densità elettronica ---
        print(Fore.MAGENTA + "\n--- Cariche sugli atomi di carbonio ---" + Style.RESET_ALL)
        charges = calculate_charges(eigenvectors, num_electrons)
        charge_table = [[f"Atomo {i+1}", f"{charge:.3f}"] for i, charge in enumerate(charges)]  # <-- parte da 1
        print(tabulate(charge_table, headers=["Atomo", "Carica"], tablefmt="grid"))

        print(Fore.MAGENTA + "\n--- Densità elettronica su ogni atomo ---" + Style.RESET_ALL)
        electron_density = calculate_electron_density(eigenvectors, num_electrons)
        density_table = [[f"Atomo {i+1}", f"{density:.3f}"] for i, density in enumerate(electron_density)]  # <-- parte da 1
        print(tabulate(density_table, headers=["Atomo", "Densità elettronica"], tablefmt="grid"))

        # --- 3. MAIN: stampa ordini di legame ---
        print(Fore.MAGENTA + "\n--- Ordini di legame ---" + Style.RESET_ALL)
        bond_orders = calculate_bond_orders(eigenvectors, num_electrons)
        for i in range(bond_orders.shape[0]):
            for j in range(i + 1, bond_orders.shape[1]):
                if bond_orders[i, j] != 0:
                    print(f"Legame {i+1}-{j+1}: {bond_orders[i, j]:.3f}")  # <-- parte da 1

        print(Fore.MAGENTA + "\n--- Matrice di Hückel (in notazione di Hartree) ---" + Style.RESET_ALL)
        print(hartree_matrix)

        eigenvalues, eigenvectors = np.linalg.eigh(hartree_matrix)

        # Ordina gli autovalori in ordine decrescente e riordina gli autovettori di conseguenza
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Indici per ordine decrescente
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        print_ascii_energy_diagram(eigenvalues)
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
            print("0. Chiudi il programma")
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
            elif choice == "0":
                print(Fore.RED + "Chiusura del programma..." + Style.RESET_ALL)
                break
            else:
                print(Fore.RED + "Opzione non valida. Riprova." + Style.RESET_ALL)

    except ValueError as e:
        print(Fore.RED + f"Errore: {e}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
