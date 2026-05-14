import glob
import os.path
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from biolm.dataset.packed_dataset import CombinedDataset

#lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 ./scripts/pretrain.py --devices 8 --model_size es --is_wandb

from pytorch_lightning.loggers import WandbLogger
import random
from datetime import datetime

from biolm.llama2_t5 import Llama2T5, get_transformer_model_args
from biolm.utils.speed_monitor import SpeedMonitorFabric as Monitor
from biolm.utils.utils import get_default_supported_precision, num_parameters
from biolm.utils.utils import step_csv_logger
from pathlib import Path
from io import StringIO
import pandas as pd
import functools
class BlankMeter():
    def __init__(self,*args,**kwargs) -> None:
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        #Exception handling here
        pass
def perf2csv(f):
    file=Path(f)
    out_dir=file.parent
    stem=file.stem
    table=file.read_text()
    # Parse table to save as CSV
    f = StringIO(table)
    data = []

    # Use the dashes on the first line to know the section indices
    sections = f.readline().split()

    def parse_line(line):
        line_data = []
        i = 0
        for section in sections:
            start = i
            end = i + len(section)
            line_data.append(line[start:end].strip().replace(",", ";"))

            # Skip two spaces after each section
            i += len(section) + 2
        return line_data

    # Parse rest of header
    data.append(parse_line(f.readline()))
    _ = f.readline()

    # Parse body
    for line in f:
        # Stop when we reach the end of the table
        if "-----" in line.strip():
            break
        data.append(parse_line(line))

    # Write to timings file
    with open(out_dir/f'{stem}.csv', "w") as fw:
        for line in data:
            fw.write(",".join(line) + "\n")

    return f'{stem}.csv'
def trace_handler(prof):    
    # 打印一个 name 全名的 op+kernel list
    name=my_perf_name
    info=prof.key_averages().table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit = -1)
    fname=f'{name}_timing_kernel.log'
    out_dir=Path(fname).parent
    out_dir.mkdir(parents=True,exist_ok=True)
    with open(fname, "w") as f:
        f.write(info)
    out=perf2csv(fname)
    #  打印包含shape信息的list
    info=prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", max_src_column_width=10000, max_name_column_width=10000, row_limit = -1)
    fname=f'{name}_timing_shape.log'
    with open(fname, "w") as f:
        f.write(info)
    out=perf2csv(fname)
    prof.export_chrome_trace(f"{name}_perf.json")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit=-1))
    # prof.export_chrome_trace("A100.trace_pytorch.json")



import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_size", type=str, default="es")  # s | m | l
parser.add_argument("--n_dec_layers", type=int, default=1)   # 1 / 6 / 12
parser.add_argument("--ckpt_inter", type=int, default=None)
parser.add_argument("--micro_batch_size", type=int, default=128)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
#
parser.add_argument("--lr", type=float, default=4e-4)
parser.add_argument("--dropout", type=float, default=0.)
parser.add_argument("--max_step", type=int, default=175)
parser.add_argument("--seq_len", type=int, default=512)
# model mode
parser.add_argument("--model_mode", type=str, default='encoder') #T5
parser.add_argument("--data_type", type=str, default='Protein' )
parser.add_argument("--data_mode", type=str, default='MLM' )
# moe config
parser.add_argument("--is_moe", action="store_true")
parser.add_argument("--tracer", action="store_true")
parser.add_argument("--num_experts", type=int, default=8)
parser.add_argument("--num_selects", type=int, default=2)
#
parser.add_argument("--out_dir", type=str, default="./out_ckpts")
parser.add_argument("--pretrained_ckpt", type=str, default=None)
parser.add_argument("--is_wandb", action="store_true")
parser.add_argument("--db", type=str, default='demo')

parser.add_argument("--data_dir", type=str, default=f"./data/demo_bin/")

args = parser.parse_args()
tracer=args.tracer
# TODO
# this leads to a bug, resulting in higher loss and lower accuracy in downstream tasks
att_mask_set_to_none = False
learning_rate = args.lr          # 4e-4
max_step = args.max_step * 1  # 175
seq_len = args.seq_len           # 512

micro_batch_size = args.micro_batch_size  # 128
is_moe = args.is_moe
rm_keys = []
pretrained_ckpt = args.pretrained_ckpt
is_wandb = args.is_wandb
ckpt_inter = args.ckpt_inter
model_size = args.model_size
n_dec_layers = args.n_dec_layers
dropout = args.dropout
model_mode = args.model_mode
data_mode = args.data_mode
ckpt_inter_str=f'ckpt{ckpt_inter}' if ckpt_inter else 'ckpt0'
my_perf_name=f'logs/perf_{model_size}_{seq_len}_{ckpt_inter_str}/A100_{model_size}_{seq_len}_{ckpt_inter_str}'
assert model_mode in ['encoder', 'T5', 'encoderT5']

if args.data_type == 'RNA':
    from biolm.dataset.mars_new.tokenizer import Tokenizer
    # from biolm.dataset.mars_new.data_config import get_data_config
    vocab_size = 21 + 200
    db = args.db
    ss_mask_mode = 'v4'

elif args.data_type == 'Protein':
    from biolm.dataset.uniref.tokenizer import Tokenizer
    # from biolm.dataset.uniref.data_config import get_data_config
    vocab_size = 21 + 200
    db = args.db
    ss_mask_mode = 'v1'
    train_data_config = [
        (args.data_dir, "train_char", 1.0, False),
    ]
    val_data_config = None
else:
    raise NotImplementedError

# dataset config
mlm_prob = 0.15
# Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
torch.backends.cuda.enable_flash_sdp(True)

torch.set_float32_matmul_precision("high")
eval_iters = 100
eval_step_interval = 1000

n_chunks = 32
num_workers = 8
num_of_devices = 8

out_dir_ = f'{args.out_dir}/{db}'
name = f'{args.data_type}-{model_mode}-{data_mode}{ss_mask_mode}-{model_size}-d{dropout}-s{seq_len}-nc{n_chunks}-nw{num_workers}_{db}_mlm{mlm_prob}_sm{ss_mask_mode}-'

if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
    name += 'fromPretrained-'
    max_step = 50 * 1000
    learning_rate = 1e-4

if is_moe:
    name += 'MoE-'
    name += f'ne{args.num_experts}-{args.num_selects}-'

name += datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir = Path(out_dir_) / name

# Hyperparameters
warmup_steps = 2000
log_step_interval = 10
save_step_interval = 25000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

gradient_accumulation_steps = args.gradient_accumulation_steps
batch_size = micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps
max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
print('log_iter_interval', log_iter_interval)
csv_logger = step_csv_logger(out_dir_, name, flush_logs_every_n_steps=log_iter_interval)

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}   # will be useful for logging
print(config)
# -----------------------------------------------------------------------------

loggers = []
if is_wandb:
    wandb_logger = WandbLogger(name=name,
                               id = name,
                               config=config,
                               )
    loggers = [wandb_logger]

def setup(
    devices: int = num_of_devices,
    train_data_dir: Path = Path(),
    val_data_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
) -> None:

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                # auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=loggers)
    fabric.print(hparams)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    if fabric.global_rank == 0:
        if tracer:
            prof_func=torch.profiler.profile
        else:
            prof_func=BlankMeter
        prof=prof_func(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=trace_handler,
            record_shapes=True)
        prof.__enter__()
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=seq_len,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )

    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()

    # define model here
    with fabric.init_module(empty_init=False):
        model_args = get_transformer_model_args(vocab_size=vocab_size,
                                               model_size=model_size,
                                               dropout=dropout,
                                               n_dec_layers=n_dec_layers,
                                               ckpt_inter=ckpt_inter,
                                               mask_token_id = Tokenizer().mask_token_id,
                                               pretrain_mode = data_mode,
                                               is_moe=is_moe,
                                               num_experts=args.num_experts,
                                               num_selects=args.num_selects,
                                               model_type=model_mode,
                                               )

        if model_mode == 'encoder' or model_mode == 'decoder':
            from biolm.llama2 import Transformer
            model = Transformer(model_args)
        elif model_mode == 'T5':
            model = Llama2T5(model_args)
        else:
            raise NotImplementedError

        if pretrained_ckpt is not None and os.path.exists(pretrained_ckpt):
            fabric.print(f'load from {pretrained_ckpt}')
            device = 'cpu'
            strict = False if model_mode == 'encoderT5' else True
            checkpoint = torch.load(pretrained_ckpt, map_location=device)
            state_dict = checkpoint['model']

            # remove embedding layer
            if rm_keys is not None:
                for key in rm_keys:
                    if key in state_dict:
                        print(key)
                        state_dict.pop(key)
                strict = False

            model.load_state_dict(state_dict, strict=strict)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]

    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")

    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    if fabric.global_rank == 0:
        prof.__exit__(None,None,None)


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        validate(fabric, model, val_dataloader)  # sanity check

    estimated_flops = 0.
    total_lengths = 0
    total_t0 = time.perf_counter()
    iter_t_last = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    
    initial_iter = state["iter_num"]
    curr_iter = 0

    for train_data in train_dataloader:
        # resume loader state. This is not elegant but it works. Should rewrite it in the future.
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))

        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, Y, attn_mask = train_data
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            model(tokens=input_ids, attn_mask=attn_mask, labels=Y)
            loss = model.last_loss
            last_losses = model.last_losses
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        elif fabric.device.type == "xla":
            xm.mark_step()

        state["iter_num"] += 1
        # input_id: B L 
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()

        fabric.print(
                f"{name} iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''},"
                f" iter time (include data): {(t1 - iter_t_last) * 1000:.2f}ms"
                f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                # print days as well
                f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
            )

        if state["step_count"] % log_step_interval == 0:
            log_dict = {"metric/loss": loss.item(),
                                     "total_tokens": seq_len * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size,
                                     "lr": lr,
                                     "total_lengths": total_lengths,
                                     }
            # add training losses
            log_dict.update(last_losses)
            csv_logger.log_metrics(log_dict, state["step_count"])
            csv_logger.save()

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss = loss.item()
        )

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item(), "total_tokens": seq_len * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.log_dict({"metric/val_ppl": math.exp(val_loss.item()), "total_tokens": seq_len * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            # log learning rate
            fabric.log_dict({"metric/lr": lr, "total_tokens": seq_len * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
            fabric.barrier()
            # log to csv logger
            log_dict = {"metric/val_loss": val_loss.item(),
                        "total_tokens": seq_len * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size,
                        "lr": lr,
                        "total_lengths": total_lengths,
                        }
            csv_logger.log_metrics(log_dict,state["step_count"])
            csv_logger.save()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        iter_t_last = time.perf_counter()

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        input_ids, Y, attn_mask = val_data
        model(tokens=input_ids, attn_mask=attn_mask, labels=Y)
        losses[k] = model.last_loss.item()
        
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:

    datasets = []
    data_config = train_data_config if split == "train" else val_data_config

    from biolm.dataset.dataset import PretokDataset

    for _config in data_config:
        data_dir = _config[0]
        prefix = _config[1]
        is_ss_token = _config[3]
        mlm_prob = _config[4] if len(_config) > 4 else 0.15 # default to 0.15

        filenames = sorted(glob.glob(str( Path(data_dir) / f"{prefix}*")))
        # if len(filenames) < wordsize * rank * num_workers, duplicate the filenames
        while len(filenames) < fabric.world_size * num_of_devices * num_workers * n_chunks:
            filenames += filenames
        print(f"rank {fabric.global_rank} num_workers {num_workers} len(filenames) {len(filenames)}")
        # exclude _ss.bin files
        filenames = [f for f in filenames if not f.endswith('_ss.bin')]
        print(str(Path(data_dir) / f"{prefix}*"), len(filenames))
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PretokDataset(filenames,
                               n_chunks=n_chunks,
                               seq_len=block_size,
                               shuffle=shuffle,
                               mode=data_mode,
                               is_ss_token=is_ss_token,
                               ss_mask_mode=ss_mask_mode,
                               mlm_probability=mlm_prob,
                               seed=seed+fabric.global_rank,
                               num_processes=fabric.world_size,
                               process_rank=fabric.global_rank,
                               tokenizer=Tokenizer(),
                              )

        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [config[2] for config in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("xxx"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_config is not None
        else None
    )
    return train_dataloader, val_dataloader

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    setup()
