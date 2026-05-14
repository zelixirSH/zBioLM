import random
import math
import inspect
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from biolm.utils.attn_utils import get_extended_attention_mask
from biolm.llama2 import (Attention,
                                precompute_freqs_cis,
                                TransformerBlock,
                                RMSNorm,
                                )
from torch import nn
from torch.nn import CrossEntropyLoss
from biolm.t5_model import Seq2SeqLMOutput, EncoderOutput
from biolm.esm2 import EsmModel
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class ModelArgs:
    esmconfig = None
    dim: int = 288
    # n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    vocab_size: int = 21   # defined later by tokenizer
    multiple_of: int = 32  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.15
    layer_dropout: float = 0.0
    is_decoder: bool = True
    # pretrain_mode: str = 'MLM'  # MLM|GLM|MLM+GLM|
    n_dec_layers: int = 1 # this is for encoder-decoder

from transformers.models.esm.configuration_esm import EsmConfig

def get_model_args(model_size, vocab_size, dropout):

    multiple_of = 32

    if model_size == "s":
        # 8m
        esmconfig = EsmConfig.from_pretrained(f'./configs/config_esm2/esm2_t6_8M_UR50D')  ## from scratch
        #
    elif model_size == "m":
        # 35m
        esmconfig = EsmConfig.from_pretrained(f'./configs/config_esm2/esm2_t12_35M_UR50D')  ## from scratch
        # dim, n_layers, n_heads = 512,8,8
    else:
        raise ValueError(f"Unknown model size {model_size}")

    esmconfig.vocab_size = vocab_size
    esmconfig.hidden_dropout_prob = dropout
    esmconfig.attention_probs_dropout_prob = dropout

    dim, n_heads = esmconfig.hidden_size, esmconfig.num_attention_heads

    # model init
    model_args = dict(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        dropout=dropout,
    )   # start with model_args from command line

    return model_args, esmconfig

def get_esm2t5_model_args(model_size, vocab_size = 21, dropout = 0.15, n_dec_layers = 1):

    model_args, esmconfig = get_model_args(model_size, vocab_size=vocab_size, dropout=dropout)

    model_args = ModelArgs(**model_args)

    model_args.esmconfig = esmconfig
    model_args.n_dec_layers = n_dec_layers

    return model_args

class ESM2T5(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.params = params
        self.vocab_size = params.vocab_size

        self.esm = EsmModel(params.esmconfig)
        self.dropout = nn.Dropout(params.dropout)

        decdoer_params = deepcopy(params)
        decdoer_params.is_decoder = True
        self.dec_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_dec_layers):
            self.dec_layers.append(TransformerBlock(layer_id, decdoer_params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the esm2 word embedding with the transformer word embedding
        # share the unembedding parameters with the embedding parameters
        self.esm.embeddings.word_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)

        # # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call.
        # This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_layers(self,
                       input_ids,
                       attention_mask,
                       layers,
                       is_decoder,
                       encoder_h,):

        ckpt_inter = 999

        _bsz, seqlen = input_ids.shape

        h = self.esm.embeddings(input_ids, attention_mask)
        if attention_mask is not None:
            h = (h * attention_mask.unsqueeze(-1)).to(h.dtype)

        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # extend attention mask to get the score
        if attention_mask is None:
            attention_mask = torch.ones((_bsz, seqlen), device=h.device, dtype=h.dtype)

        extended_attn_mask = get_extended_attention_mask(attention_mask,(_bsz, seqlen), h.device, h.dtype, is_decoder=is_decoder)

        for c, layer in enumerate(layers):

            if (c+1) % ckpt_inter == 0:
                h,_ = checkpoint(layer, h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)
            else:
                h,_ = layer(h, extended_attn_mask, freqs_cos, freqs_sin, None, encoder_h)

        return h

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> (torch.Tensor, torch.Tensor):
        '''
        :param tokens: (bsz, seqlen) LongTensor of input token indices
        :param attention_mask: (bsz, seqlen) padding mask for attention
        '''

        # encoder_output = self.forward_layers(input_ids, attention_mask, self.enc_layers, is_decoder = False, encoder_h=None)

        encoder_output = self.esm(input_ids, attention_mask)

        h = self.forward_layers(decoder_input_ids, decoder_attention_mask, self.dec_layers, is_decoder = True, encoder_h=encoder_output)
        # LM head
        h = self.norm(h)

        logits = self.output(h)

        self.last_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            self.last_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=self.last_loss,
            logits=logits,
            encoder_outputs=EncoderOutput(hidden_states=encoder_output, attention_mask=attention_mask),
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0.


def cal_param_ESM2T5():
    args = get_esm2t5_model_args(model_size='s')
    model = ESM2T5(args)
    print(model)
    print(f'number of encoder parameters: {sum(p.numel() for p in model.enc_layers.parameters())}')
    print(f'number of decoder parameters: {sum(p.numel() for p in model.dec_layers.parameters())}')
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

if __name__ == '__main__':
    # ESM2T5()
    import numpy as np
    model_size, vocab_size, dropout = 's', 21, 0.15
    model_args = get_esm2t5_model_args(model_size, vocab_size=21, dropout=0.15, n_dec_layers=1)
    # print(model_args)
    model = ESM2T5(model_args)
    print(model)

    # forward pseudo input
    bs = 1
    seq_len = 128
    rand_array = np.random.randint(0, 11, (bs, seq_len))
    input = torch.LongTensor(rand_array)
    attn_mask = torch.ones((bs, seq_len))
    target = torch.ones((bs, seq_len)).long()
    print(f'input shape: {input.shape}')
    print(f'input: {input}')
    print(f'target: {target.shape}')
    device = 'cpu'
    model.to(device)
    input = input.to(device)
    attn_mask = attn_mask.to(device)
    target = target.to(device)
    output = model.forward(input_ids=input, attention_mask=attn_mask, labels=target, decoder_input_ids=input, decoder_attention_mask=attn_mask)
    print(output)