import os
from Bio import SeqIO
import random
from p_tqdm import p_map

def cat_fasta_shuffle(fasta_dir, output_file):
    '''
     cat all the fasta files and shuffle sequences
    '''
    sequences = []
    for fasta_file in os.listdir(fasta_dir):
        seqs = [str(record.seq) for record in SeqIO.parse(f"{fasta_dir}/{fasta_file}", "fasta")]
        print(f"Number of sequences in {fasta_file}: {len(seqs)}")
        sequences.extend(seqs)
    random.shuffle(sequences)

    print(f"Total number of sequences: {len(sequences)}")

    with open(output_file, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f'>test_{i}\n')
            f.write(f'{seq}\n')

def split_fasta(fasta_file, n, output_dir, name = None):
    """Split a FASTA file into multiple smaller files with 'n' sequences each.

    Args:
    fasta_file (str): Path to the input FASTA file.
    n (int): Number of sequences per output file.
    output_dir (str): Directory to save the split files.
    """
    # Create a sequence iterator
    sequence_iterator = SeqIO.parse(fasta_file, 'fasta')

    if name is None:
        name = fasta_file.split('/')[-1].split('.')[0]

    # Keep track of file and sequence counts
    file_count = 0
    sequence_count = 0
    #
    non_atgc_file_count = 0
    non_atgc_count = 0

    # Create the first output file
    os.makedirs(output_dir,exist_ok=True)
    current_output_file = open(f"{output_dir}/{name}_split_{file_count}.fasta", 'w')
    current_output_file_non_atgc = open(f"{output_dir}/{name}_split_{non_atgc_file_count}_non_atgc.fasta", 'w')

    for record in sequence_iterator:
        # Write the current record to the output file
        # filter out the sequences with not in ATCG
        if not set(record.seq).issubset(set('ATCGN')):
            SeqIO.write(record, current_output_file_non_atgc, 'fasta')
            non_atgc_count +=1
            if non_atgc_count % 1000 == 0:
                print(f"non_atgc_count: {non_atgc_count}")

            # If we hit the limit, start a new file
            if non_atgc_count >= n:
                current_output_file_non_atgc.close()
                non_atgc_file_count += 1
                non_atgc_count = 0
                current_output_file_non_atgc = open(f"{output_dir}/{name}_split_{non_atgc_file_count}_non_atgc.fasta", 'w')

        else:
            SeqIO.write(record, current_output_file, 'fasta')
            sequence_count += 1

            # If we hit the limit, start a new file
            if sequence_count >= n:
                current_output_file.close()
                file_count += 1
                sequence_count = 0
                current_output_file = open(f"{output_dir}/{name}_split_{file_count}.fasta", 'w')

    # Close the last output file if it's open
    if not current_output_file.closed:
        current_output_file.close()

    if not current_output_file_non_atgc.closed:
        current_output_file_non_atgc.close()

    print(f"Total number of sequences: {sequence_count}")
    print(f"Total number of non-ATGC sequences: {non_atgc_count}")

# Example usage:
# fasta_dir = '/public/home/taoshen/data/rna/sequence_database/kaggle/fastas'
# output_file = '/public/home/taoshen/data/rna/sequence_database/kaggle/all_shuffle.fasta'
# cat_fasta_shuffle(fasta_dir, output_file)

def process_one(param):
    fasta_file, save_root = param
    split_fasta(fasta_file,100000, save_root)

def split_txt(txt_file, n, output_dir, name = None):
    """Split a FASTA file into multiple smaller files with 'n' sequences each.

    Args:
    fasta_file (str): Path to the input FASTA file.
    n (int): Number of sequences per output file.
    output_dir (str): Directory to save the split files.
    """
    # Create a sequence iterator

    f = open(txt_file, 'r')
    lines = f.readlines()
    f.close()

    # Keep track of file and sequence counts
    file_count = 0
    sequence_count = 0
    non_atgc_count = 0

    # Create the first output file
    os.makedirs(output_dir,exist_ok=True)
    current_output_file = open(f"{output_dir}/split_{file_count}.fasta", 'w')

    for seq in lines:
        id = f'{sequence_count}'
        current_output_file.write(f'>{id}\n')
        current_output_file.write(f'{seq.strip()}\n')

        sequence_count += 1

        # If we hit the limit, start a new file
        if sequence_count >= n:
            current_output_file.close()
            file_count += 1
            sequence_count = 0
            current_output_file = open(f"{output_dir}/split_{file_count}.fasta", 'w')

    # Close the last output file if it's open
    if not current_output_file.closed:
        current_output_file.close()

def split_mars():

    root = '/public/home/taoshen/data/rna/sequence_database/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta'
    save_root = '/public/home/taoshen/data/rna/sequence_database/MARS/remove_sim_0.8_8192_clstr/rep_seq/rep_seqs/fasta_0.1m'
    os.makedirs(save_root, exist_ok=True)

    params = []
    for i in range(200):
        idx_str = str(i).zfill(3)
        if os.path.exists(f'{root}/MARS_{idx_str}.cluster_rep_seq.fasta'):
            params.append((f'{root}/MARS_{idx_str}.cluster_rep_seq.fasta', save_root))

    p_map(process_one, params)

def split_cds():

    fasta_file = '/public/home/taoshen/data/rna/sequence_database/utrdb/all_0.8_rep_seq.fasta'
    save_root = '/public/home/taoshen/data/rna/sequence_database/utrdb/all_0.8_rep_seq_split_fastas'
    os.makedirs(save_root, exist_ok=True)

    split_fasta(fasta_file,100000, save_root)

if __name__ == '__main__':
    # split_mars()
    # split_rnacentral()
    split_cds()