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

def split_fasta(fasta_file, n, output_dir):
    """Split a FASTA file into multiple smaller files with 'n' sequences each.

    Args:
    fasta_file (str): Path to the input FASTA file.
    n (int): Number of sequences per output file.
    output_dir (str): Directory to save the split files.
    """
    # Create a sequence iterator

    f = open(fasta_file, 'r')
    lines = f.readlines()
    f.close()

    # Keep track of file and sequence counts
    file_count = 0
    sequence_count = 0

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
            print(sequence_count, file_count)
            current_output_file = open(f"{output_dir}/split_{file_count}.fasta", 'w')

    # Close the last output file if it's open
    if not current_output_file.closed:
        current_output_file.close()

# def split_uniref():
#     fasta_file = '/public/home/taoshen/data/protein/uniref/uniref90.fasta'
#     n = 100000
#     output_dir = '/public/home/taoshen/data/protein/uniref/uniref90'
#     os.makedirs(output_dir,exist_ok=True)
#     split_fasta(fasta_file, n, output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    split_fasta(args.fasta_file, args.n, args.output_dir)


