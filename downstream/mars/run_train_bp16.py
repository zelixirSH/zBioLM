"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=1 run_train_ss_bp16.py
# torchrun --standalone --nproc_per_node=1 ./downstream/mars/run_train_bp16.py
# torchrun --standalone --nproc_per_node=8 ./downstream/mars/run_train_bp16.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import yaml
import time
from contextlib import nullcontext
from datetime import datetime
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from biolm.predictors.utils import get_lr
from biolm.utils.utils import step_csv_logger

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, default="encoder")
parser.add_argument("--model_size", type=str, default="s")
parser.add_argument("--data_mode", type=str, default='ss_pred')

parser.add_argument("--out_dir", type=str, default='./out_ckpts')
#add from scratch
parser.add_argument("--from_scratch", action='store_true')
#add layerdrop
parser.add_argument("--layer_dropout", type=float, default=0.1)
#add layerdrop
parser.add_argument("--dropout", type=float, default=None)
parser.add_argument("--max_iters", type=int, default=None)
#extra args
parser.add_argument("--species", type=str, default='mel')
#
parser.add_argument("--cls_task", type=str, default='sirna')
#add gradient accumulation
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
# add is ema model, store_true
parser.add_argument("--is_ema_model", action='store_true')
# add is checkpoint
parser.add_argument("--ckpt",  action='store_true')
parser.add_argument("--fsdp",  action='store_true') # sth is wrong with fsdp
parser.add_argument("--config_file", type=str, default=f'config_new_pipline.yaml')
parser.add_argument("--is_wandb", action="store_true")

parser.add_argument("--eval_only", action="store_true")
parser.add_argument("--pretrained_ckpt", type=str, default=None)
parser.add_argument("--is_freeze", action="store_true")

parser.add_argument("--data_dir", type=str, default= '/public_new/taoshen/data/mars_fm_data/downstream')

args = parser.parse_args()

ckpt_inter = None if not args.ckpt else 1
eval_only = args.eval_only # if True, script exits right after the first eval
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
num_gpu = torch.cuda.device_count()

wandb_log = args.is_wandb  # disabled by default
#-----------------------------------------------------------------------------------------------------------------------
compile = False    # use PyTorch 2.0 to compile the model to be faster

# get current script dir path
script_dir = os.path.dirname(os.path.realpath(__file__))
config_file = args.config_file

def get_ckpt_path(args, config_fpath = f'{script_dir}/configs/{config_file}'):
    '''
    '''

    with open(config_fpath, 'r') as f:
        config = yaml.safe_load(f)

    root = config["root_paths"].get(args.model_type, None)
    model_path = config["model_paths"][args.model_type].get(args.model_size, None)
    if not model_path:
        raise NotImplementedError

    ckpt_path = f'{root}/{model_path}/ckpt.pt'

    if not os.path.exists(ckpt_path):
        ckpt_path = f'{root}/{model_path}/ckpt_175000.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'{root}/{model_path}/iter-175000-ckpt.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'{root}/{model_path}/iter-125000-ckpt.pth'

    vocab_size = config["vocab_sizes"].get(args.model_type, config["vocab_sizes"]["default"])

    return ckpt_path, vocab_size

ckpt_path, vocab_size = get_ckpt_path(args)

learning_rate = 3e-4  # max learning rate
log_interval = 10
num_workers = 8
dataset = "mars"
data_mode = args.data_mode
layer_dropout = args.layer_dropout

extra_args = {}
extra_args_predictor = {}

if config_file == f'config_new_pipline.yaml':
    extra_args['tok_mode'] = 'MarsTok'

if data_mode == 'ss_pred':
    from biolm.predictors.ss_pred import SSCNNPredictor as Predictor
    from biolm.predictors.ss_pred import estimate_metrics, get_data_loader

    best_metrics = {
        'best_val_loss_l': 1e9,
        'best_val_f1_h': 0.0,
    }

    batch_size = 16 // num_gpu
    max_iters = 175000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 1000
    dropout = 0.10 if args.dropout is None else args.dropout

elif data_mode == 'en_pred':
    from biolm.predictors.en_pred import Predictor, estimate_metrics, get_data_loader
    # 'rice' 'Arabidopsis' 'mel'
    extra_args ['species']: args.species
    best_metrics = {
        'best_val_PCC_h': 0,
    }
    learning_rate = 3e-4  # max learning rate
    batch_size = 128 // num_gpu
    max_iters = 100000  if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 500
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

elif data_mode == 'cls_pred':
    from biolm.predictors.cls_pred import Predictor, estimate_metrics

    if args.cls_task == 'sirna':
        from biolm.predictors.cls_pred import get_data_loader_sirna as get_data_loader
    else:
        from biolm.predictors.cls_pred import get_data_loader

    extra_args[ 'cls_task'] = args.cls_task,
    best_metrics = {'best_val_f1_h': 0}

    learning_rate = 3e-4  # max learning rate
    batch_size = 128 // num_gpu
    max_iters = 30000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 250
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

elif data_mode == 'pa_pred':
    from biolm.predictors.pa_pred import Predictor, estimate_metrics, get_data_loader

    best_metrics = {
        'best_val_PCC_h': 0,
    }
    learning_rate = 1e-4  # max learning rate
    batch_size = 64 // num_gpu
    max_iters = 50000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 250
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

elif data_mode == 'sirna_pred':
    from biolm.predictors.sirna_pred import Predictor, estimate_metrics, get_data_loader

    best_metrics = {
        'best_val_PCC_h': 0,
    }
    learning_rate = 1e-4  # max learning rate
    batch_size = 64 // num_gpu
    max_iters = 10000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 250
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

elif data_mode == 'utr_pred':
    from biolm.predictors.utr_pred import Predictor, estimate_metrics, get_data_loader

    best_metrics = {
        'best_val_PCC_h': 0,
    }
    learning_rate = 1e-4  # max learning rate
    batch_size = 64 // num_gpu
    max_iters = 10000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 250
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

elif data_mode == 'mrna_pred':
    from biolm.predictors.mrna_pred import Predictor, estimate_metrics, get_data_loader
    best_metrics = {
        'best_val_MCRMSE_l': 1e9,
    }
    batch_size = 128 // num_gpu
    max_iters = 50000 if args.max_iters is None else args.max_iters  # total number of training iterations
    eval_interval = 100
    dropout = 0.15 if args.dropout is None else args.dropout
    num_workers = 2

else:
    raise NotImplementedError

# moe dropout
if "MOE" in args.model_type:
    extra_args["is_moe"] = True
    extra_args["moe_dropout"] = 0.50

    if 'expert16_4' in args.model_size:
        extra_args["num_experts"] = 16
        extra_args["num_selects"] = 4

    elif 'expert32_4' in args.model_size:
        extra_args["num_experts"] = 32
        extra_args["num_selects"] = 4

# adamw optimizer
gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size /= gradient_accumulation_steps
batch_size = int(batch_size)

tag = f''
if ddp:
    tag += f'-ddp{num_gpu}'

#SS pred
if 'mode' in extra_args_predictor and extra_args_predictor['mode'] == 'cnn1':
    tag += f'-cnn1'

name_clean = True
if not name_clean:
    if len(extra_args.keys())>0:
        for k, v in extra_args.items():
            tag += f'-{k}_{v}'

if args.dropout is not None:
    tag += f'-dp{args.dropout}'

# wandb logging
wandb_project = f"llamac-{dataset}-{data_mode}"
wandb_run_name = f"{data_mode}-{dataset}-{args.model_type}-{args.model_size}-{tag}-"

if args.is_ema_model:
    wandb_run_name += "ema-"

if args.from_scratch:
    wandb_run_name += "fromScratch-"

if args.is_freeze:
    wandb_run_name += "freeze-"

wandb_run_name += datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

out_dir = f"{args.out_dir}/{dataset}-{data_mode}/{dataset}_{wandb_run_name}"
print(f"out_dir: {out_dir}")

if args.pretrained_ckpt is not None:
    learning_rate = 1e-4

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True               # whether to decay the learning rate
warmup_iters = 1000           # how many steps to warm up for
lr_decay_iters = max_iters    # should be ~= max_iters per Chinchilla
min_lr = 0.0                  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = "cuda"               # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"            # float32|bfloat16|float16

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

# exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32,
           "bfloat16": torch.bfloat16,
           "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

from biolm.llama2_t5 import load_encoder_model

if "encoder" in args.model_type:
    from biolm.llama2 import Transformer
    extractor = load_encoder_model(ckpt_path,
                                   model_size=args.model_size.split('-')[0],
                                   vocab_size = vocab_size,
                                   device='cpu',
                                   dropout=dropout,
                                   layer_dropout=layer_dropout,
                                   strict=True,
                                   is_ema_model=args.is_ema_model,
                                   ckpt_inter=ckpt_inter,
                                   nn_model=Transformer,
                                   **extra_args
                                   )
else:
    raise NotImplementedError

# get predictor
model = Predictor(extractor, is_freeze = args.is_freeze, **extra_args_predictor)

# load pretrained ckpt
if args.pretrained_ckpt is not None:
    print(f'load pretrained ckpt from {args.pretrained_ckpt}')
    ckpt = torch.load(args.pretrained_ckpt)
    model.load_state_dict(ckpt['model'])

if master_process:
    print(model)

model.to(device)
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}

    if args.fsdp:
        model = FSDP(model, device_id=ddp_local_rank)
    else:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# dataloaders
train_sampler, train_loader, valid_loader, test_loader = \
    get_data_loader(args.data_dir, ddp, batch_size, num_workers, **extra_args)

# init wandb and create output dir
if master_process:
    os.makedirs(out_dir, exist_ok=True)

if wandb_log and master_process:
    import wandb
    os.makedirs(f'./wandb_bf16/{args.model_type}_{args.model_size}', exist_ok=True)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config,
               dir = f'./wandb_bf16/{args.model_type}_{args.model_size}',
               id=wandb_run_name)

if master_process:
    csv_logger = step_csv_logger(f"{args.out_dir}/{dataset}-{data_mode}/", f"{dataset}_{wandb_run_name}")

# main training loop
iter_num = 0
micro_step = 0

t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed

def save_ckpt(log_dict, best_metrics, raw_model, optimizer, iter_num, config, out_dir):

    for b_metric in best_metrics.keys():
        split, m, mode = b_metric.split('_')[1],  b_metric.split('_')[2], b_metric.split('_')[3]

        if (mode == 'l' and log_dict[f"{split}/{m}"] < best_metrics[b_metric]) or \
            (mode == 'h' and log_dict[f"{split}/{m}"] > best_metrics[b_metric]):
            best_metrics[b_metric] = log_dict[f"{split}/{m}"]

            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_metrics[b_metric],
                    "config": config,
                }
                print(f"saving best_{m} {best_metrics[b_metric]} checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f"best_{split}_{m}_ckpt.pt"))

    return best_metrics


while True and not eval_only:

    if ddp:
        train_sampler.set_epoch(iter_num)

    for data_dict in train_loader:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process and iter_num > 0:
            log_dict = { "iter": iter_num, "lr": lr,}
            log_dict.update(estimate_metrics(ctx, model, raw_model, 'val', valid_loader, device))
            log_dict.update(estimate_metrics(ctx, model, raw_model, 'test', test_loader, device))
            best_metrics = save_ckpt(log_dict, best_metrics, raw_model, optimizer, iter_num, config, out_dir)
            log_dict.update(best_metrics)

            if wandb_log:
                wandb.log(log_dict)

            csv_logger.log_metrics(log_dict, iter_num)
            csv_logger.save()

        X, attn_mask, Y = data_dict["input_ids"], data_dict["attention_mask"], data_dict["label"]
        X, attn_mask, Y  = X.to(device), attn_mask.to(device), Y.to(device)

        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

        with ctx:
            logits = model(X, attn_mask, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
            micro_step += 1

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if micro_step == gradient_accumulation_steps:
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            iter_num += 1
            local_iter_num += 1

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % log_interval == 0 and master_process:
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item()
                print(f"{wandb_project}/{wandb_run_name}: {iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms")

                log_dict = {"iter": iter_num,
                            "lr": lr,
                            "train/loss": lossf,
                            "iter_time": dt,
                            }

                if wandb_log:
                    wandb.log(log_dict)

                csv_logger.log_metrics(log_dict, iter_num)
                csv_logger.save()

            # termination conditions
            if iter_num > max_iters:
                if ddp:
                    destroy_process_group()
                exit()

# final evaluation
if master_process:
    log_dict = {}
    log_dict.update(estimate_metrics(ctx, model, raw_model, 'val', valid_loader, device))
    log_dict.update(estimate_metrics(ctx, model, raw_model, 'test', test_loader, device))
    best_metrics = save_ckpt(log_dict, best_metrics, raw_model, optimizer, iter_num, config, out_dir)
    log_dict.update(best_metrics)
    print(log_dict)

if ddp:
    destroy_process_group()