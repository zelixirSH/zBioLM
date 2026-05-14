import os

# Linclust output
#  - result_rep_seq.fasta: Representatives
#  - result_all_seq.fasta: FASTA-like per cluster
#  - result_cluster.tsv:   Adjecency list

# Important parameter: --min-seq-id, --cov-mode and -c
#                  --cov-mode
#                  0    1    2
# Q: MAVGTACRPA  60%  IGN  60%
# T: -AVGTAC---  60% 100%  IGN
#        -c 0.7    -    +    -
#        -c 0.6    +    +    +

# Cluster nucleotide sequences
# mmseqs easy-linclust examples/DB.fasta result tmp --kmer-per-seq-scale 0.3

bin_dir = '/public/home/taoshen/tools/mmseqs/bin'

def run_one(param):
    fasta, clusterRes, tmp, seq_sim = param
    cmd = f'{bin_dir}/mmseqs easy-linclust {fasta} {clusterRes} {tmp} --min-seq-id {seq_sim} -c {seq_sim} --cov-mode 1'
    os.system(cmd)

if __name__ == '__main__':
    fasta = '/public/home/taoshen/data/rna/sequence_database/utrdb/fasta/all.fasta'
    seq_sim = 0.8
    clusterRes = f'all_{seq_sim}'
    tmp_dir = './tmp'
    param = (fasta, clusterRes, tmp_dir, seq_sim)
    run_one(param)




