# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct
import torch.distributed as dist
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id
        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
                filenames=filenames,
                n_chunks=self._n_chunks,
                block_size=self._block_size,
                seed=self._seed,
                shuffle=self._shuffle,
                wrap=self._wrap,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None,
                 is_ss_token = False):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype

        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token

        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)

        # for ss tokens
        self.is_ss_token = is_ss_token
        if self.is_ss_token:
            self._arr_ss = np.zeros(self._chunk_size, dtype=self._dtype)
            self._arr_ss.fill(self._sep_token)

        self._idx = 0
        self._version = 1
        self._filenames = []

    def _write_chunk(self):
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)

        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        if self.is_ss_token:
            filename_ss = f"{self._prefix}_{self._counter:010d}_ss.bin"
            filename_ss = os.path.join(self._outdir, filename_ss)
            with open(filename_ss, "wb") as f:
                f.write(HDR_MAGIC)
                f.write(struct.pack("<Q", self._version))
                f.write(struct.pack("<B", code(self._dtype)))
                f.write(struct.pack("<Q", self._chunk_size))
                f.write(self._arr_ss.tobytes(order="C"))

        self._filenames.append(filename)
        self._counter += 1

        self._arr.fill(self._sep_token)
        if self.is_ss_token:
            self._arr_ss.fill(self._sep_token)

        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr, arr_ss = None):

        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx

            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            if self.is_ss_token:
                self._arr_ss[self._idx : self._idx + part_len] = arr_ss[:part_len]

            self._write_chunk()

            arr = arr[part_len:]
            if self.is_ss_token:
                arr_ss = arr_ss[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        if self.is_ss_token:
            self._arr_ss[self._idx : self._idx + arr_len] = arr_ss

        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()



class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap = False, is_ss_token = False):
        self._seed = seed
        self._shuffle = shuffle
        self._block_idxs = None

        self._rng = np.random.default_rng(seed) if shuffle else None
        self._wrap = wrap

        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.

        self._filenames = filenames
        # exclude ss file
        self._filenames = [f for f in self._filenames if not f.endswith('_ss.bin')]

        # check ss file exists
        self.is_ss_token = is_ss_token
        # self.ss_mask_mode = ss_mask_mode
        if self.is_ss_token:
            self._filenames_ss = [f for f in self._filenames if os.path.exists(f.replace('.bin', '_ss.bin'))]
            print(f'Found {len(self._filenames_ss)} ss files')
            assert len(self._filenames_ss) == len(self._filenames), 'ss file not found'

        if self._shuffle:
            print(f"Shuffling {len(self._filenames)} files")
            self._rng.shuffle(self._filenames)

        self._file_idx = 0
        self._n_chunks = n_chunks

        if self._n_chunks > len(self._filenames):
            print(f"Warning: n_chunks ({self._n_chunks}) > n_files ({len(self._filenames)}).")
            self._n_chunks = len(self._filenames)

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._mmaps_ss = []
        self._buffers = []
        self._buffers_ss = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()


    def _read_header(self, path):

        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))

        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

        for mmap in self._mmaps_ss:
            mmap._mmap.close()

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._mmaps_ss
        del self._buffers
        del self._buffers_ss

    def test_load_n_chunks(self):
        while True:
            if self._n_chunks > len(self._filenames[self._file_idx:]):
                break
            self._load_n_chunks()
            self._close_mmaps()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._mmaps_ss = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            # if not self._wrap:
            #     raise StopIteration

            self._file_idx = 0
            if self._shuffle:
                print(f"Shuffling {len(self._filenames)} files")
                self._rng.shuffle(self._filenames)

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]

            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size

            # TODO: check header matches with previous files
            # print(f"worker_id: {self.worker_id} rank:{self.rank}, Loading {filename}")
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

            if self.is_ss_token:
                filename_ss = filename.replace('.bin', '_ss.bin')
                # print(f"worker_id: {self.worker_id} rank:{self.rank}, Loading {filename_ss}")
                mmap_ss = np.memmap(filename_ss, mode="r", order="C", offset=HDR_SIZE)
                self._mmaps_ss.append(mmap_ss)
                self._buffers_ss.append(memoryview(mmap_ss))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)
        self._curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote

        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)

        if self.is_ss_token:
            buffer_ss = self._buffers_ss[chunk_id]
            arr_ss = np.frombuffer(buffer_ss, dtype=self._dtype, count=self._block_size, offset=offset)
            self._curr_idx += 1
            return torch.from_numpy(arr.astype(np.int64)), torch.from_numpy(arr_ss.astype(np.int64))

        else:
            self._curr_idx += 1
            return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

def unitest():

    root = '/public/home/taoshen/data/rna/sequence_database/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin'
    is_ss_token = True
    filenames = [f'{root}/{f}' for f in os.listdir(root) if f.endswith('.bin')]
    filenames = [f for f in filenames if not f.endswith('_ss.bin')]

    n_chunks = 128
    block_size = 1024
    seed = 121
    shuffle = False
    wrap = False

    dataset = PackedDatasetIterator(filenames, n_chunks, block_size, seed, shuffle, is_ss_token=is_ss_token)
    dataset.test_load_n_chunks()

    # dataset = iter(PackedDatasetIterator(filenames, n_chunks, block_size, seed, shuffle, is_ss_token=is_ss_token))
    # while True:
    #     data, data_ss = next(dataset)
    #     print(data.shape, data_ss.shape)
    #     print(data)
    #     print(list(data_ss))
    #     exit()


if __name__ == '__main__':
    unitest()