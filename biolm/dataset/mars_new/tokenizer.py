import json
from pathlib import Path
from typing import Optional
import torch
from transformers.models.esm.tokenization_esm import *

class MarsTokenizer(PreTrainedTokenizer):
    """
    Constructs an ESM tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        bos_token="<bos>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        self.tokenizer_char = EsmTokenizer.from_pretrained(f"{script_dir}/vocab_na.txt")
        self.tokenizer_3mer = EsmTokenizer.from_pretrained(f"{script_dir}/vocab_3mer.txt")

        # TODO: add bpe tokenizer
        special_tokens_dict = {'unk_token': unk_token,
                               'cls_token': cls_token,
                               'pad_token': pad_token,
                               'mask_token': mask_token,
                               'eos_token': eos_token,
                               'bos_token': bos_token}

        self.all_tokens = self.tokenizer_char.all_tokens  + self.tokenizer_3mer.all_tokens + list(special_tokens_dict.values())
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        self.unk_token = unk_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.unique_no_split_tokens = self.all_tokens

        # self._create_trie(self.unique_no_split_tokens)

    def __len__(self):
        return len(self.all_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def tokenize(self, text, mode='3mer', **kwargs):

        if mode == '3mer':
            return self.tokenizer_3mer.tokenize(text)
        elif mode == 'char':
            return self.tokenizer_char.tokenize(text)
        else:
            raise NotImplementedError

    def get_vocab_size(self, with_added_tokens=False):
        return len(self._id_to_token)

    def get_vocab(self):
        try:
            data = {token: i for i, token in enumerate(self.all_tokens)}
        except:
            data = {}
        return data

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size(with_added_tokens=False)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:

        # update
        try:
            self.all_tokens = self.all_tokens + new_tokens
            self._id_to_token = dict(enumerate(self.all_tokens))
            self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

            # add special tokens
            self.tokenizer_char.add_tokens(new_tokens, special_tokens=special_tokens)
            self.tokenizer_3mer.add_tokens(new_tokens, special_tokens=special_tokens)
        except:
            print('new_tokens', new_tokens)

        return super()._add_tokens(new_tokens, special_tokens=True)


class Tokenizer:
    def __init__(self,) -> None:
        # some checkpoints have both files, `.model` takes precedence
        self.tokenizer = MarsTokenizer()
        self.bos_id = self.tokenizer.token_to_id(self.tokenizer.bos_token)
        self.pad_token_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
        # for MLM
        self.mask_token_id = self.tokenizer.token_to_id(self.tokenizer.mask_token)


    def add_tokens(self, new_tokens, special_tokens = False):
        self.tokenizer.add_tokens(new_tokens, special_tokens)

    def __len__(self):
        return len(self.tokenizer)
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def token_to_id(self, token: str, mode = 'char') -> int:
        if mode == '3mer':
            return self.tokenizer.token_to_id(token)
        elif mode == 'char':
            return self.tokenizer.token_to_id(token)
        else:
            raise NotImplementedError

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = True,
        max_length: int = -1,
        mode = '3mer',
    ) -> torch.Tensor:

        string = string.upper()
        string = string.replace('U', 'T')

        tokens = self.tokenizer.tokenize(string, mode=mode)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if bos:
            bos_id = self.tokenizer.token_to_id(self.tokenizer.bos_token)
            if bos_id is None:
                raise NotImplementedError("This tokenizer does not defined a bos token")
            tokens = [bos_id] + tokens

        if eos:
            tokens = tokens + [self.tokenizer.token_to_id(self.tokenizer.eos_token)]

        if max_length > 0:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.tokenizer.decode(tokens)

def unitext_tokenizer():
    '''

    '''
    return

if __name__ == '__main__':
    import os
    from transformers import EsmTokenizer

    # Create a dummy dna sequence and tokenize it
    sequences = ["ATTCCG"]
    tokenizer = MarsTokenizer()
    print(tokenizer.vocab_size)
    print(tokenizer._token_to_id)

    # print(tokenizer.tokenize(sequences[0], mode='3mer'))
    # print(tokenizer.tokenize(sequences[0], mode='char'))
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequences[0], mode='3mer')))
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequences[0], mode='char')))
    #
    # tokenizer.add_tokens(['<xxxx>', '<yyyy>','<zzzz>'], special_tokens=True)
    # print(tokenizer.vocab_size)
    # sequences = ["ATTCCG<xxxx><yyyy><zzzz>"]
    # print(tokenizer.tokenize(sequences[0], mode='char'))
    # print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequences[0], mode='char')))

    # tok = Tokenizer()
    # print(tok.vocab_size)
    # print(tok.encode(sequences[0], mode='3mer'))
    # print(tok.encode(sequences[0], mode='char'))
