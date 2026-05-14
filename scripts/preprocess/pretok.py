import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import biolm.dataset.packed_dataset as packed_dataset
from Bio import SeqIO
import random

def prepare_full(
    source_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0,
    mode = 'char',
) -> None:

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset

    if isinstance(filenames, list) and len(filenames) == 0:
        return

    # destination_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(destination_path, exist_ok=True)

    tokenizer = Tokenizer()

    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    # get all sequences from fasta file
    contents_all = []
    print(f"Processing {filenames}")
    for filepath in tqdm(filenames):
        contents = [str(record.seq) for record in SeqIO.parse(filepath, "fasta")]
        contents_all.extend(contents)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_{mode}_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    # shuffle the sequences
    random.shuffle(contents_all)
    for text in contents_all:
        text_ids = tokenizer.encode(text, mode=mode)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids,
    # see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()

def prepare(
    source_path: Path = Path("./split_fastas"),
    destination_path: Path = Path("./split_fastas_bin"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
    modes = ['char'],
    seed: int = 42,  # Add a new parameter for the random seed
    n_jobs=64,
    num_files_per_chunk = 64,  # 每个子列表中的文件数量
    **kwargs,
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, "*.fasta"), recursive=True)

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

    for mode in modes:
        for i, subset in enumerate(chunked_filenames):

            if len(processes) >= n_jobs:  # Don't start more than n_jobs processes at once
                p = processes.pop(0)
                p.join()

            p = Process(target=prepare_full, args=(source_path, destination_path, chunk_size, split, list(subset), i, mode))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default="./split_fastas")
    parser.add_argument('--destination_path', type=str, default="./split_fastas_bin")
    parser.add_argument('--chunk_size', type=int, default=2049 * 1024)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--percentage', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=64)
    parser.add_argument('--num_files_per_chunk', type=int, default=64)
    parser.add_argument("--data_type", type=str, default='Protein')  # NA / Protein
    args = parser.parse_args()

    assert args.data_type in ['Protein', 'NA']

    if args.data_type == 'Protein':
        from biolm.dataset.uniref.tokenizer import Tokenizer
    elif args.data_type == 'NA':
        from biolm.dataset.mars_new.tokenizer import Tokenizer
    else:
        raise NotImplementedError

    prepare(**vars(args))
