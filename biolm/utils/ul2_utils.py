import os
import torch
import random
import numpy as np
from collections.abc import Mapping
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from typing import Dict, List
from dataclasses import dataclass


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def random_spans_noise_mask(length, mean_noise_span_length, noise_density):
    """
    A copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L230 (inception)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans
    )

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]

def compute_input_and_target_lengths(
        inputs_length, noise_density, mean_noise_span_length
    ):
    """
    A copy of copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L76 (shits getting meta)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length


    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

@dataclass
class DataCollatorForUL2(DataCollatorMixin):
    """

    Data collator used for UL2

    """
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0.25
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0.5
    x_denoising: bool = True
    x_probability: float = 0.25
    x_denoising_config: Tuple[Tuple] = ((32, 0.5), (64, 0.2))
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id

        assert sum(task_prob) == 1.0

    def assign_task_type(self, batch_size: int):
        '''
            Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task, weights=self.task_prob, k=batch_size)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)
        task_ids = self.assign_task_type(len(examples))
        # print('task id', task_ids)
        task_type = torch.tensor(task_ids)
        lengths = torch.tensor([len(e['input_ids']) for e in examples], dtype=torch.long)
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt",
                                       pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer,
                                                  pad_to_multiple_of=self.pad_to_multiple_of)
            }
        max_length = batch['input_ids'].shape[-1]

        new_batch = {
            "input_ids": torch.zeros(batch['input_ids'].shape, dtype=torch.long),
            "labels": torch.zeros(batch['input_ids'].shape, dtype=torch.long)
        }

        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            # union of different denoising settings
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            diff = max_length - _labels.shape[-1]
            _labels = np.pad(_labels, [(0, 0), (0, diff)], 'constant',
                             constant_values=self.label_pad_token_id)
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0, 0), (0, diff)], 'constant')
            new_batch['input_ids'][r_denoising_idx] = torch.from_numpy(_sub_input_ids).long()
            new_batch['labels'][r_denoising_idx] = torch.from_numpy(_labels).long()

        s_denoising_idx = task_type == 1
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            _labels = []
            _input_ids = []
            for input_id, len_ in zip(sub_input_ids, lengths[s_denoising_idx]):
                split = max(len_ // 2, 2)
                diff = expanded_length - split
                _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
                past_seq = input_id[split:]
                if past_seq[-1] != self.tokenizer.eos_token_id:
                    past_seq[-1] = self.tokenizer.eos_token_id
                _labels.append(F.pad(past_seq, (0, split), 'constant', self.label_pad_token_id))

            new_batch['input_ids'][s_denoising_idx] = torch.stack(_input_ids)
            new_batch['labels'][s_denoising_idx] = torch.stack(_labels)

        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[x_denoising_idx]
            for (mean_span, noise) in self.x_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            labels_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)
            diff = max_length - _labels.shape[-1]
            _labels = np.pad(_labels, [(0, 0), (0, diff)], 'constant',
                             constant_values=self.label_pad_token_id)
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0, 0), (0, diff)], 'constant')
            new_batch['input_ids'][x_denoising_idx] = torch.from_numpy(_sub_input_ids).long()
            new_batch['labels'][x_denoising_idx] = torch.from_numpy(_labels).long()

        return self.prepare_decoder_inputs_from_labels(new_batch)

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = []
        for row in input_ids_full:
            collapsed_id = row[row >= 0]
            diff = len(row) - len(collapsed_id)
            collapsed_id = np.pad(collapsed_id, (0, diff), 'constant')
            input_ids.append(collapsed_id)
        return np.array(input_ids)

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def prepare_decoder_inputs_from_labels(self, batch):
        # decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id.
        # See T5 docs for more information
        batch["labels"][batch["labels"] == self.pad_token_id] = self.label_pad_token_id
        shifted_labels = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].clone()
        shifted_labels[..., 0] = self.decoder_start_token_id  # decoder_start_token_id

        batch["decoder_input_ids"] = torch.masked_fill(
            shifted_labels,
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            shifted_labels == self.label_pad_token_id,
            0,
            torch.ones_like(shifted_labels),
        )
        return batch

    def np_prepare_decoder_inputs_from_labels(self, batch):
        batch["labels"][batch["labels"] == self.pad_token_id] = self.label_pad_token_id
        shifted_labels = np.zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].copy()
        shifted_labels[..., 0] = self.decoder_start_token_id

        batch["decoder_input_ids"] = np.where(
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id,
            shifted_labels
        )
        batch["decoder_attention_mask"] = np.where(
            shifted_labels == self.label_pad_token_id,
            0,
            np.ones_like(shifted_labels)
        )
        return batch


def unitest():
    from transformers import EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(f"{script_dir}/vocab_esm_mars.txt")

    input_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    tokenizer.add_special_tokens({'bos_token': '</s>'})

    data_collator = DataCollatorForUL2(tokenizer)

    batch = data_collator([{'input_ids':input_ids}])

    print(batch.keys())

    for key in batch:
        print(key, batch[key].shape)
        # print(batch[key])

if __name__ == '__main__':
    unitest()