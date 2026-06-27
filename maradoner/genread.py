import numpy as np
from pyfaidx import Fasta
from Bio.Seq import Seq
from tqdm import tqdm

def seq2onehot(sequences):
    """
    Converts a list of strings into a 3D one-hot encoded NumPy array.
    Output shape: (num_seqs, max_seq_len, 4)
    """
    num_seqs = len(sequences)
    # Find the maximum sequence length to ensure uniform tensor shape (zero-padding shorter ones)
    max_len = max(len(seq) for seq in sequences)
    
    # Pre-allocate a 3D dense array filled with zeros. 
    # Float32 is required by JAX/Flax for neural network weights.
    onehot = np.zeros((num_seqs, max_len, 4), dtype=np.float32)
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i, seq in tqdm(enumerate(sequences), total=num_seqs, desc="One-hot encoding"):
        for j, char in enumerate(seq):
            if char in mapping:
                onehot[i, j, mapping[char]] = 1.0
            # Characters like 'N' are ignored, leaving the row as [0, 0, 0, 0]
                
    return onehot

def read_seqs(input_intervals: list[str], genome_path: str, len_down_override: int = 10, 
              len_up_override: int = 250):

    # 1. Load Genome and Promoter Metadata
    genes = Fasta(genome_path)
    
    seqs_list = []

    # 2. Extract Nucleotide Sequences
    print("Extracting sequences...")
    for entry in tqdm(input_intervals):
        parts = entry.split('_')
        chrom = parts[0]
        pos = int(parts[1])
        direction = parts[-1] 
        
        # Determine Window
        start = pos - len_down_override
        if len(parts) < 4:
            end = pos + len_up_override
        else:
            end = int(parts[2])
        
        try:
            # Fetch sequence and handle reverse complement
            raw_seq = genes[chrom][start:end].seq.upper()
            
            if direction == "rc":
                final_seq = str(Seq(raw_seq).reverse_complement())
            else:
                final_seq = raw_seq
                
            seqs_list.append(final_seq)
            
        except KeyError:
            # Fill with 'N' if chromosome is missing
            expected_len = end - start
            seqs_list.append('N' * expected_len)

    # 3. Convert to 3D One-Hot NumPy Array
    seqs_3d = seq2onehot(seqs_list)
    
    print(f"Final tensor shape: {seqs_3d.shape}") # Should be (p, seq_len, 4)
    return seqs_3d