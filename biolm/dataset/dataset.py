"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""
from biolm.utils.t5_utils import DataCollatorForT5MLM, random_spans_noise_mask
from biolm.utils.mlm_utils import mask_tokens

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from biolm.dataset.packed_dataset import PackedDatasetIterator
from torch.utils.data import get_worker_info

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def _shift_right(input_ids, decoder_start_token_id, pad_token_id):
    # decoder_start_token_id = 20
    # pad_token_id = 0

    assert decoder_start_token_id is not None and pad_token_id is not None
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self,
                 filenames,
                 n_chunks = 8,
                 seq_len = 512,
                 shuffle = True,
                 mlm_probability = 0.15,
                 mode = 'GLM',
                 is_ss_token=False,
                 ss_mask_mode = 'v1',
                 seed = 0,
                 num_processes = 1,
                 process_rank = 0,
                 tokenizer = None,
                 ):
        super().__init__()

        assert mode in ['GLM', 'MLM', 'T5']

        assert tokenizer is not None
        self.tokenizer = tokenizer

        self._num_processes = num_processes
        self._process_rank = process_rank
        self._filenames = filenames

        self.n_chunks = n_chunks
        self.shuffle = shuffle
        self.seed = seed

        self.is_ss_token = is_ss_token
        self.max_seq_len = seq_len
        self.mode = mode

        self.mlm_probability = mlm_probability
        self.prob_replace_mask = 0.8
        self.prob_replace_rand = 0.1
        self.mean_noise_span_length = 3.0

        self.ss_mask_mode = ss_mask_mode

        if mode == 'T5':
            self.target_length = 128
            # Adding 100 sentinel tokens
            self.tokenizer.add_tokens([f"<extra_id_{i}>" for i in range(100)])

            self.data_collator = DataCollatorForT5MLM(
                                                        tokenizer=self.tokenizer,
                                                        noise_density=mlm_probability,
                                                        mean_noise_span_length=self.mean_noise_span_length,
                                                        input_length=self.max_seq_len,
                                                        target_length=self.target_length,
                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                        ss_mask_mode=ss_mask_mode,
            )

    def __iter__(self):

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id
        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        data_iterator = iter(PackedDatasetIterator(filenames,
                                                    self.n_chunks,
                                                    self.max_seq_len,
                                                    self.seed,
                                                    self.shuffle,
                                                    is_ss_token=self.is_ss_token,
                                                   ))

        while True:
            ss_tok = None
            if self.is_ss_token:
                seq_tok, ss_tok = next(data_iterator)
            else:
                seq_tok = next(data_iterator)

            if self.mode == 'MLM':
                # x, y = mask_tokens(seq_tok, self.tokenizer,
                #                    self.mlm_probability, self.prob_replace_mask, self.prob_replace_rand)

                x = seq_tok
                attn_mask = (x != self.tokenizer.pad_token_id).long()
                length = attn_mask.sum(dim=-1)

                is_noise = random_spans_noise_mask(length,
                                                   ss_tok.unsqueeze(0).numpy() if ss_tok is not None else None,
                                                   self.mlm_probability,
                                                   self.mean_noise_span_length,
                                                   ss_mask_mode=self.ss_mask_mode,
                                                   )

                is_noise = None if np.sum(is_noise) == 0 else is_noise
                is_noise = torch.LongTensor(is_noise).bool() if is_noise is not None else None

                x, y = mask_tokens(seq_tok,
                                   self.tokenizer,
                                   self.mlm_probability,
                                   self.prob_replace_mask,
                                   self.prob_replace_rand,
                                   is_noise,)

                # x_dec = _shift_right(y, self.tokenizer.bos_id, self.tokenizer.pad_token_id)
                # dec_attn_mask = (x_dec != self.tokenizer.pad_token_id).long()

                yield x, y, attn_mask #, x_dec, dec_attn_mask

            elif self.mode == 'T5':

                input_ids = list((seq_tok.data.cpu().numpy()).astype(np.int64))
                input_ss = list((ss_tok.data.cpu().numpy()).astype(np.int64)) if self.is_ss_token else None

                batch = self.data_collator([{'input_ids': input_ids,'input_ss': input_ss}])
                x, y = batch['input_ids'].squeeze(0), batch['labels'].squeeze(0)

                attn_mask = (x != self.tokenizer.pad_token_id).long()
                x_dec = _shift_right(y, self.tokenizer.bos_id, self.tokenizer.pad_token_id)
                dec_attn_mask = (x_dec != self.tokenizer.pad_token_id).long()

                yield x, y, attn_mask, x_dec, dec_attn_mask

            elif self.mode == 'GLM':
                # GLM is next token prediction
                y = seq_tok
                x = _shift_right(y, self.tokenizer.bos_id, self.tokenizer.pad_token_id)
                attn_mask = (x != self.tokenizer.pad_token_id).long()
                yield x, y, attn_mask

            else:
                raise NotImplementedError


def unitest_iter():
    seq_db_root = '/public/home/taoshen/data/rna/sequence_database'
    root = f"{seq_db_root}/rnacentral_2023-02-24/ss/eternafold_512_csv_rep_TinyLlamaBin"
    is_ss_token = True
    filenames = [f'{root}/{f}' for f in os.listdir(root) if f.endswith('.bin')]
    from biolm.dataset.mars_new.tokenizer import Tokenizer
    # mode = 'T5'
    mode = 'MLM'
    dataset_iter = iter(PretokDataset(filenames,
                                      mode=mode,
                                      is_ss_token=is_ss_token,
                                      ss_mask_mode='v4',
                                      tokenizer=Tokenizer()))

    while True:
        x, y, attn_mask, x_dec, dec_attn_mask = next(dataset_iter)
        print('x shape', x.shape)
        print('y shape', y.shape)

def unitest_uniref_iter():
    from biolm.dataset.uniref.tokenizer import Tokenizer

    seq_db_root = '/public/home/taoshen/data/protein/uniref/'
    root = f"{seq_db_root}/uniref50_bin"
    is_ss_token = False
    filenames = [f'{root}/{f}' for f in os.listdir(root) if f.endswith('.bin')]
    # mode = 'MLM'
    mode = 'GLM'
    dataset_iter = iter(PretokDataset(filenames,
                                      mode=mode,
                                      is_ss_token=is_ss_token,
                                      tokenizer=Tokenizer()))

    while True:
        x, y, attn_mask = next(dataset_iter) #, x_dec, dec_attn_mask
        print('x shape', x.shape)
        print('y shape', y.shape)
        print(x)
        print(y)
        exit()

if __name__ == "__main__":
    # unitest_iter()
    unitest_uniref_iter()








