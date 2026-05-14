import os
import numpy as np
import torch
import numpy as np
from omegaconf import OmegaConf
from omegaconf import open_dict
from typing import Dict, List
from transformers import BatchEncoding
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    ss_mask_mode: str


    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:

        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]

        #TODO
        def find_last_pad_token_id(arr):
            # Flipped array to find the last occurrence of a non-zero element
            flipped_arr = np.flip(arr)
            # Get the index of the first non-zero element (which is actually the last non-zero element in the original array)
            last_nonzero_index = np.argmax(flipped_arr != self.tokenizer.pad_token_id)
            # Calculate the index of the last non-zero element in the original array
            return arr.size - last_nonzero_index - 1

        last_nonzero_idx = find_last_pad_token_id(input_ids[0])
        input_ids = input_ids[0][:last_nonzero_idx + 1]
        input_ids = input_ids.reshape((1,-1))

        input_ss = batch["input_ss"]
        # TODO
        input_ss = None if (input_ss[0] is None or np.sum(input_ss[0]) == 0) else input_ss

        batch_size, expandend_input_length = input_ids.shape

        # align input_ss with input_ids
        if input_ss is not None:
            input_ss = input_ss[:, :expandend_input_length]

        mask_indices = np.asarray(
            [
                random_spans_noise_mask(expandend_input_length, input_ss, self.noise_density, self.mean_noise_span_length,
                                        ss_mask_mode=self.ss_mask_mode)
                for i in range(batch_size)
            ]
        )

        # TODO
        if np.sum(mask_indices) == 0:
            mask_indices = np.asarray(
                [
                    random_spans_noise_mask(expandend_input_length, None, self.noise_density, self.mean_noise_span_length,
                                            ss_mask_mode=self.ss_mask_mode)
                    for i in range(batch_size)
                ]
            )

        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        # if input_ss is None:
        #     if batch["input_ids"].shape[-1] != self.input_length:
        #         raise ValueError(
        #             f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
        #             f" should be {self.input_length}."
        #         )
        #
        #     if batch["labels"].shape[-1] != self.target_length:
        #         raise ValueError(
        #             f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
        #             f" {self.target_length}."
        #         )

        # Pad to max len, tokenizer.pad for input_ids and -100 for labels
        padded_input_ids = np.full((batch_size, self.input_length), fill_value=self.tokenizer.pad_token_id)
        padded_labels = np.full((batch_size, self.target_length), fill_value=-100)

        for i, (input_id, label) in enumerate(zip(batch["input_ids"], batch["labels"])):
            padded_input_ids[i, :len(input_id)] = input_id[:self.input_length]  # Crop if needed
            padded_labels[i, :len(label)] = label[:self.target_length]  # Crop if needed
            # Assign the padded (and possibly cropped) arrays back to the batch

        new_batch = {}
        new_batch["input_ids"] = padded_input_ids.astype(np.int64)
        new_batch["labels"] = padded_labels.astype(np.int64)

        new_batch = {k: torch.from_numpy(v) for k, v in new_batch.items()}
        return new_batch

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

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))

        # already handled in pretok
        """
        # input_ids = np.concatenate(
        #     [
        #         input_ids,
        #         np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
        #     ],
        #     axis=-1,
        # )
        """
        return input_ids



def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper
    <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
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

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


def unitest():
    from transformers import EsmTokenizer

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    tokenizer = EsmTokenizer.from_pretrained(f"{script_dir}/vocab_esm_mars.txt")

    input_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    args = OmegaConf.load('./nanoT5/configs/default.yaml')
    args.data.input_length = len(input_ids)

    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=args.data.input_length,
        noise_density=args.data.mlm_probability,
        mean_noise_span_length=args.data.mean_noise_span_length,
    )

    with open_dict(args):
        args.data.before_mask_input_length = before_mask_input_length
        args.data.target_length = target_length

    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=args.data.mlm_probability,
        mean_noise_span_length=args.data.mean_noise_span_length,
        input_length=args.data.input_length,
        target_length=args.data.target_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    batch = data_collator([{'input_ids':input_ids}])
    print(batch.keys())


def mask_ss_v1(addition_mask, length, noise_density):
    # Convert addition_mask to a numpy array for processing
    addition_mask = np.array(addition_mask, dtype=np.bool_)

    # Initialize mask, all values are False
    is_noise = np.zeros(length, dtype=np.bool_)

    # Find the start and end indices of consecutive 1s in addition_mask
    one_indices = np.where(addition_mask[0])[0]  # Index into the first dimension to get a 1-D array

    if one_indices.size > 0:
        # Compute the consecutive regions of 1s
        breaks = np.where(np.diff(one_indices) > 1)[0]
        ends = np.split(one_indices, breaks + 1)

        # Apply the mask for each consecutive region
        for segment in ends:
            # Randomly decide to mask this span or not
            r_ = np.random.rand()

            if r_ < noise_density:
                is_noise[segment[0]:segment[-1] + 1] = True

    return is_noise

def mask_ss_v2(addition_mask, mask_probability):
    """
    根据概率，掩盖 RNA 二级结构中连续括号配对的一侧。

    :param addition_mask: 一个表示 RNA 二级结构的数组，其中 1 和 2 分别表示 ( 和 )。
    :param mask_probability: 掩盖的概率。
    :return: 一个布尔数组，表示是否掩盖了对应位置的碱基。
    """

    addition_mask = addition_mask[0]
    is_masked = np.zeros(len(addition_mask), dtype=bool)
    stack = []
    pair_indices = []

    # 寻找所有配对的括号
    for i, val in enumerate(addition_mask):
        if val == 1:
            stack.append(i)
        elif val == 2 and stack:
            start = stack.pop()
            pair_indices.append((start, i))

    # 对所有配对按照左侧括号的位置进行排序
    pair_indices.sort(key=lambda x: x[0])

    # find all consecutive pairs
    i = 0
    # print(pair_indices)
    consecutive_pairs = []
    while i < len(pair_indices):
        c_pairs_index = [i]
        j = i + 1
        # 寻找连续的配对
        while j < len(pair_indices) and pair_indices[j][0] == pair_indices[j-1][0] + 1:
            c_pairs_index.append(j)
            j += 1
        i = j
        # print(c_pairs_index)
        consecutive_pairs.append(c_pairs_index)

    # check
    for c_pairs_index in consecutive_pairs:
        # 决定是否掩盖这一段括号
        if np.random.rand() < mask_probability:
            # 掩盖左侧或右侧
            is_left_mask = np.random.rand() < 0.5
            for i in c_pairs_index:
                start, end = pair_indices[i]
                if is_left_mask:
                    is_masked[start] = True
                else:
                    is_masked[end] = True

    return is_masked


def random_spans_noise_mask(length, addition_mask, noise_density, mean_noise_span_length, ss_mask_mode = 'v1'):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    Args:
        length: an int32 scalar (length of the incoming token sequence)
        addition_mask: , shape(1, length), is SS in RNA seq, 0,1,2 denoes no SS, (, )
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number

    Returns:
        a boolean tensor with shape [length]
    """

    if addition_mask is not None:

        # Ensure the addition_mask's length matches the input length
        if addition_mask.shape[1] != length:
            raise ValueError(f"addition_mask length {addition_mask.shape[1]} does not match input length {length}")

        if ss_mask_mode == 'v1':

            return mask_ss_v1(addition_mask, length, noise_density)

        elif ss_mask_mode == 'v2':

            return mask_ss_v2(addition_mask, noise_density)

        elif ss_mask_mode == 'v3':

            if np.random.rand() < 0.5:
                is_noise = mask_ss_v2(addition_mask, noise_density)

            else:
                # for regions with no base pairing, apply default T5 mask
                is_noise = random_spans_noise_mask(length, None, noise_density, mean_noise_span_length)
                is_noise = is_noise * (addition_mask[0] == 0)

            return is_noise

        elif ss_mask_mode == 'v4':

            if np.random.rand() < 0.5:
                is_noise = mask_ss_v2(addition_mask, noise_density)
                if not np.sum(is_noise) > 0:
                    # if no ss is masked, apply default T5 mask
                    is_noise = random_spans_noise_mask(length, None, noise_density, mean_noise_span_length)

            else:
                # for regions with no base pairing, apply default T5 mask
                is_noise = random_spans_noise_mask(length, None, noise_density, mean_noise_span_length)
                is_noise = is_noise * (addition_mask[0] == 0)

            # if np.sum(is_noise) == 0:
            #     print('is_noise', np.sum(is_noise) > 0)

            return is_noise

        else:
            raise NotImplementedError

    else:
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

def unitest_mask_ss():

    # 示例使用
    addition_mask = [1, 1, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]  # 示例 RNA 二级结构
    mask_probability = 1.  # 掩盖概率
    length = len(addition_mask)
    masked = mask_ss_v2(addition_mask, mask_probability)
    print(masked.astype(np.int64))

def unitest_spans():
    '''

    '''

    # Example usage:
    length = 100
    # addition_mask = np.array([[False, True, True, True, False, True, True, False, False, False]])
    noise_density = 0.9  # 50% of the span will be masked
    mean_noise_span_length = 3.0

    addition_mask = None
    # addition_mask should be the length of the input sequence, set a random addition_mask w consecutive 1s
    addition_mask = np.zeros((1, length), dtype=np.int64)
    addition_mask[0, 0:10] = 1
    addition_mask[0, 20:30] = 1
    addition_mask[0, 60:70] = 2
    addition_mask[0, 80:90] = 2

    # Create a random noise mask
    print(addition_mask[0])
    for i in range(10):
        noise_mask = random_spans_noise_mask(length, addition_mask, noise_density, mean_noise_span_length)
        print(noise_mask.astype(np.int64))


if __name__ == '__main__':
    # unitest()
    # unitest_spans()

    unitest_spans()

