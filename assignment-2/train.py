import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import tiktoken
from alive_progress import alive_it

from gpt2 import GPT2Config, GPT2Model, GPT2_PRESETS
from dataset import get_wikitext103_dataloader


def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def build_model(args, vocab_size, device):
    if args.model not in GPT2_PRESETS:
        raise ValueError(f"Unknown model size '{args.model}'. Valid: {list(GPT2_PRESETS.keys())}")

    preset = GPT2_PRESETS[args.model]
    config = GPT2Config(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        model_dim=preset["model_dim"],
        feed_forward_hidden_dim=preset["feed_forward_hidden_dim"],
        dropout=0.1
    )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype '{args.dtype}'. Use 'float32', 'float16', or 'bfloat16'.")

    model = GPT2Model(config)
    model = model.to(device=device, dtype=dtype_map[args.dtype])
    return model


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    else:
        raise ValueError(f"Unsupported optimizer '{args.optimizer}'. Must be 'sgd' or 'adamw'.")


def train(
    model,
    optimizer,
    dataloader,
    tokenizer,
    warmup_steps,
    total_steps,
    accum_steps,
    device
):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    step_times = []
    backward_times = []
    mem_allocated = []
    mem_reserved = []
    dl_iter = iter(dataloader)

    optimizer.zero_grad(set_to_none=True)

    for step in range(total_steps):
        model.train()
        start_time = time.time()

        # Fetch the next batch
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        # Move input to device
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Forward
        logits = model(X)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / accum_steps

        torch.cuda.synchronize(device=device)
        backward_start_time = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device=device)
        backward_end_time = time.perf_counter()
        backward_time = backward_end_time - backward_start_time
        backward_times.append(backward_time)

        # Gradient Accumulation
        if (step + 1) % accum_steps == 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device=device)
        end_time = time.time()

        # Exclude warmup steps from timing
        if step >= warmup_steps:
            step_times.append(end_time - start_time)
            mem_allocated.append(torch.cuda.memory_allocated(device=device) / 1e9)
            mem_reserved.append(torch.cuda.memory_reserved(device=device) / 1e9)

    # Compute average step time
    if len(step_times) > 0:
        avg_time = np.mean(step_times)
        std_dev_time = np.std(step_times)
        print(f"Rank {dist.get_rank()}: Avg time per batch (excluding warmup): {avg_time:.4f}s ± {std_dev_time:.4f}s")

        avg_backward_time = np.mean(backward_times)
        std_dev_backward_time = np.std(backward_times)
        print(f"Rank {dist.get_rank()}: Avg backward time per batch: {avg_backward_time:.4f}s ± {std_dev_backward_time:.4f}s")

        avg_mem_reserved = np.mean(mem_reserved)
        std_dev_mem_reserved = np.std(mem_reserved)
        avg_mem_allocated = np.mean(mem_allocated)
        std_dev_mem_allocated = np.std(mem_allocated)
        print(f"Rank {dist.get_rank()}: Avg memory allocated: {avg_mem_allocated:.4f}GB ± {std_dev_mem_allocated:.4f}GB")
        print(f"Rank {dist.get_rank()}: Avg memory reserved: {avg_mem_reserved:.4f}GB ± {std_dev_mem_reserved:.4f}GB")
    else:
        print(f"Rank {dist.get_rank()}: No steps timed because total_steps <= warmup_steps.")


def mb_to_mib(mb):
    return (mb * 1e6) / (2**20)


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on a dataset (no distributed).")
    parser.add_argument("--dataset", type=str, default="my_data")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--model", type=str, default="gpt2-small", choices=list(GPT2_PRESETS.keys()))
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--bucket-size-mb", type=int, default=25, help="DDP bucket size in MB")
    args = parser.parse_args()

    # setup distributed training environment
    slurm_rank = int(os.getenv("SLURM_PROCID", default=0))
    slurm_world_size = int(os.getenv("SLURM_NTASKS", default=1))
    slurm_local_rank = int(os.getenv("SLURM_LOCALID", default=0))

    device = torch.device("cuda", slurm_local_rank)
    assert slurm_local_rank < torch.cuda.device_count(), f"SLURM_LOCALID={slurm_local_rank} is greater than the number of GPUs"

    dist.init_process_group(backend="nccl", rank=slurm_rank, world_size=slurm_world_size, device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # create tokenizer
    # zaratan compute nodes do not have internet access, so use cached tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # load in dataset
    dataloader = get_wikitext103_dataloader(args.dataset, tokenizer, seq_len=args.max_seq_len, batch_size=args.batch_size, distributed=slurm_world_size > 1)

    # initalize model
    model = build_model(args, dataloader.dataset.vocab_size, device)
    bucket_size = mb_to_mib(args.bucket_size_mb)   # convert to MiB
    print0(f"Using bucket size of {bucket_size:.1f} MiB")
    model = DDP(model, device_ids=[slurm_local_rank], bucket_cap_mb=bucket_size, gradient_as_bucket_view=True)

    optimizer = build_optimizer(args, model)

    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params = f"{num_params/1e9:.1f}B" if num_params >= 1e9 else f"{num_params/1e6:.1f}M"
    print(f"Rank {dist.get_rank()}: Training {args.model} with {num_params} parameters on {device} device.")

    # train model
    train(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        tokenizer=tokenizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        accum_steps=args.gradient_accumulation,
        device=device
    )

    print0("Training complete!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
