import argparse
from biolm.llama2 import Transformer
from biolm.llama2_t5 import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='/public_new/taoshen/biolm_ckpts')
parser.add_argument("--data_type", type=str, default="Protein")
parser.add_argument("--model_type", type=str, default="decoder")
parser.add_argument("--model_size", type=str, default="es")
args = parser.parse_args()
root = args.root

if args.model_type == "encoder":
    # get llama model ckpt path
    if args.model_size == "eg": # 35m params model
        model = 'Protein-encoder-MLMv1-eg-d0.0-s1024-nc32-nw8_uniref5090_mlm0.15_smv1-fromPretrained-2024_03_31_10_13_42'
    else:
        raise NotImplementedError

    ckpt_path = f'{root}/{model}/final.pth'

    vocab_size = 221
    dropout = 0.1              # this is important for downstream tasks
    ckpt_inter = None          # this is for gradient checkpointing, None for no checkpointing

    extractor = load_model(ckpt_path,
                                   model_size=args.model_size,
                                   vocab_size=vocab_size,
                                   device='cpu',
                                   dropout=dropout,
                                   strict=True,
                                   ckpt_inter=ckpt_inter,
                                   nn_model=Transformer,
                                   model_type='encoder',
                                   )
    extractor.eval()

elif args.model_type == "decoder":

    if args.model_size == "es": # 8M params
        model = 'Protein-decoder-GLMv1-es-d0.0-s512-nc32-nw8_uniref5090_mlm0.15_smv1-2024_03_31_22_34_03'
    else:
        raise NotImplementedError

    ckpt_path = f'{root}/{model}/iter-175000-ckpt.pth'

    vocab_size = 221
    dropout = 0.1              # this is important for downstream tasks
    ckpt_inter = None          # this is for gradient checkpointing, None for no checkpointing

    extractor = load_model(ckpt_path,
                                   model_size=args.model_size,
                                   vocab_size=vocab_size,
                                   device='cpu',
                                   dropout=dropout,
                                   strict=True,
                                   ckpt_inter=ckpt_inter,
                                   nn_model=Transformer,
                                   model_type='decoder',
                                   )
    extractor.eval()

else:
    raise NotImplementedError

if args.data_type == "Protein":
    from biolm.dataset.uniref.tokenizer import Tokenizer
    tokenizer = Tokenizer()
    example_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQI"

elif args.data_type == "NA":
    raise NotImplementedError

else:
    raise NotImplementedError

if args.model_type == "encoder":
    # test the encoder model for embedding extraction
    text_ids = tokenizer.encode(example_seq, mode='char')
    tokens = text_ids.unsqueeze(0)
    attn_mask = (tokens != tokenizer.pad_token_id).long()
    logits, h = extractor(tokens, attn_mask)
    print(logits.shape, h.shape)

elif args.model_type == "decoder":
    # test the decoder model for generation
    seq_len = len(example_seq)
    # use the first 32 tokens for prompt
    text_ids = tokenizer.encode(example_seq[:32], mode='char')
    tokens = text_ids.unsqueeze(0)
    # remove the eos token
    tokens = tokens[:, :-1]
    attn_mask = (tokens != tokenizer.pad_token_id).long()
    logits, h = extractor(tokens, attn_mask)

    print('ref seq:',example_seq)
    for i in range(10):
        gen_idx = extractor.generate(tokens, max_new_tokens=seq_len, temperature=1.0, top_k=None)
        gen_seq = tokenizer.decode(gen_idx[0])
        print('gen seq:',gen_seq.replace(' ',''))
