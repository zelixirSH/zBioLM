import json
import glob
import os
from pathlib import Path
import sys
from typing import List
from multiprocessing import Process

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import biolm.dataset.packed_dataset as packed_dataset
from biolm.dataset.mars_new.tokenizer import Tokenizer
from Bio import SeqIO
import random
from tqdm import tqdm
import os
import numpy as np
import torch
import pandas as pd
import random
from multiprocessing import Pool, cpu_count
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
from tqdm import tqdm
import numpy as np

tokenizer = Tokenizer()

def get_ss_np(input_text, ss, length):

    if len(ss) < 512:
        assert len(ss) == length - 1, print(len(input_text), len(ss), length, input_text,  ss)  # 长度要减1，因为末尾有特殊标记

    ss_array = np.zeros(length, dtype=int)

    stack = []
    for i, char in enumerate(ss):
        if char == '(':
            stack.append(i)
            ss_array[i] = 1      # 遇到 '(' 存储 1

        elif char == ')':
            if stack:
                stack.pop()
                ss_array[i] = 2  # 遇到 ')' 存储 2

    return ss_array


def process_sequence(input):
    '''

    '''

    input_text, ss = input

    try:
        text_ids = tokenizer.encode(input_text,
                                    bos = False,
                                    eos = True,
                                    mode = 'char')

        ss_np = get_ss_np(input_text, ss, text_ids.shape[0])  # 确保ss_np的长度与tokenized序列一致
        return text_ids, ss_np.reshape(-1)
    except:
        return None, None


def process_csvs(
              destination_path: Path,
              chunk_size: int,
              split: str = "train",
              filenames: List[str] = None,
              process_id: int = 0,
              mode='char',
):
    '''

    '''

    tokenizer = Tokenizer()

    if not filenames:
        raise RuntimeError(f"No files matching")

    if isinstance(filenames, list) and len(filenames) == 0:
        return

    destination_path.mkdir(parents=True, exist_ok=True)

    sequences_all = []
    for csv_file in tqdm(filenames):
        df = pd.read_csv(csv_file)
        sequences = list(zip(df['sequence'].tolist(), df['structure'].tolist()))
        sequences = [(seq[0], seq[1] ) for seq in sequences]
        sequences_all.extend(sequences)

    random.shuffle(sequences_all)

    n = len(sequences_all)
    sequences_all = sequences_all[:n//2]  # 取一半的数据

    # Use multiprocessing to process sequences
    with Pool(cpu_count()) as pool:
        contents_all = list(tqdm(pool.imap(process_sequence, sequences_all), total=len(sequences_all)))

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_{mode}_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
        is_ss_token=True,
    )

    # shuffle the sequences
    random.shuffle(contents_all)

    for text_ids, ss_ids in contents_all:
        if text_ids is not None:
            builder.add_array(np.array(text_ids, dtype=builder.dtype), arr_ss=np.array(ss_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()
    print('done')


def prepare(
    source_path: Path = Path("/public/home/taoshen/data/rna/sequence_database/rnacentral_2023-02-24/ss/eternafold_512_csv_rep"),
    destination_path: Path = Path("/public/home/taoshen/data/rna/sequence_database/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_half_bin"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
    seed: int = 42,  # Add a new parameter for the random seed
    n_jobs=4,
    num_files_per_chunk = 1,  # 每个子列表中的文件数量
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, "*.csv"), recursive=True)

    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]

    # Set the random seed and shuffle the filenames
    random.seed(seed)
    random.shuffle(filenames)

    filenames = filenames[:int(len(filenames) * percentage)]

    num_chunks = -(-len(filenames) // num_files_per_chunk)  # 计算需要的子列表数量，向上取整
    chunked_filenames = np.array_split(filenames, num_chunks)

    print(f"Preparing {len(filenames)} files in {len(chunked_filenames)} chunks")

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):

        if len(processes) >= n_jobs:  # Don't start more than n_jobs processes at once
            p = processes.pop(0)
            p.join()

        p = Process(target=process_csvs, args=(destination_path, chunk_size, split, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


def unitest():
    '''

    '''

    root = '/public/home/taoshen/data/rna/sequence_database/rnacentral_2023-02-24/ss/eternafold_512_csv_all'

    filenames = [f'{root}/{f}' for f in os.listdir(root) if f.endswith('.csv')]

    process_csvs(
        destination_path=Path(f"{root}_bin_debug"),
        chunk_size=2049 * 1024,
        split="train",
        filenames=filenames,
        process_id=0,
    )

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)



