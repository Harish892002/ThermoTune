# Install ViennaRNA Python bindings
# !pip install viennarna

# Import the RNA module
import RNA

import numpy as np
import random

def generate_random_rna(n_sequences=100000, length=500, seed=42):
    random.seed(seed)
    bases = ["A", "C", "G", "U"]
    sequences = []

    for _ in range(n_sequences):
        seq = "".join(random.choices(bases, k=length))
        sequences.append(seq)

    return sequences

# Generate
sequences = generate_random_rna()

# Save to file (fast, compressed)
np.save("rna_100k_sequences.npy", sequences)

len(sequences), sequences[0][:50]

# Calculate minimum free energy structure and value
import numpy as np
import RNA
from tqdm import tqdm

# Load sequences (generated earlier)
sequences = np.load("rna_100k_sequences.npy", allow_pickle=True)

mfe_values = []
structures = []

# Loop over each RNA sequence and compute MFE
for seq in tqdm(sequences, desc="Computing MFE"):
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()   # returns (structure, mfe_energy)
    structures.append(struct)
    mfe_values.append(mfe)

# Save results
np.save("rna_100k_mfe.npy", np.array(mfe_values))
np.save("rna_100k_structures.npy", np.array(structures))

len(mfe_values), mfe_values[:5]