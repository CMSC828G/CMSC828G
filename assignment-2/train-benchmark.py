import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import tiktoken
from alive_progress import alive_it

from gpt2 import GPT2Config, GPT2Model, GPT2_PRESETS
from dataset import get_wikitext103_dataloader


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

    model = GPT2Model(config, profile_grads=True)
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
    criterion = nn.CrossEntropyLoss()

    step_times = []
    mem_allocated = []
    mem_reserved = []
    dl_iter = iter(dataloader)

    optimizer.zero_grad(set_to_none=True)

    for step in alive_it(range(total_steps), total=total_steps, title="Training"):
        model.train()
        start_time = time.time()

        if step == warmup_steps:
            torch.cuda.cudart().cudaProfilerStart()

        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        X, y = batch
        X, y = X.to(device), y.to(device)

        # Forward
        if step >= warmup_steps:
            torch.cuda.nvtx.range_push("forward")
        logits = model(X)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / accum_steps
        if step >= warmup_steps:
            torch.cuda.nvtx.range_pop()

        # Backward
        if step >= warmup_steps:
            torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if step >= warmup_steps:
            torch.cuda.nvtx.range_pop()

        # Gradient Accumulation
        if (step + 1) % accum_steps == 0:
            if step >= warmup_steps:
                torch.cuda.nvtx.range_push("step")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step >= warmup_steps:
                torch.cuda.nvtx.range_pop()
            optimizer.zero_grad(set_to_none=True)

        # End timing
        torch.cuda.synchronize()
        end_time = time.time()

        # Exclude warmup steps from timing
        if step >= warmup_steps:
            step_times.append(end_time - start_time)
            mem_allocated.append(torch.cuda.memory_allocated() / 1024**3)
            mem_reserved.append(torch.cuda.memory_reserved() / 1024**3)
    

    # Compute average step time
    if len(step_times) > 0:
        avg_time = np.mean(step_times)
        std_dev_time = np.std(step_times)
        print(f"Avg time per batch (excluding warmup): {avg_time:.4f}s ± {std_dev_time:.4f}s")

        avg_mem_reserved = np.mean(mem_reserved)
        std_dev_mem_reserved = np.std(mem_reserved)
        avg_mem_allocated = np.mean(mem_allocated)
        std_dev_mem_allocated = np.std(mem_allocated)
        print(f"Avg memory allocated: {avg_mem_allocated:.4f}GB ± {std_dev_mem_allocated:.4f}GB")
        print(f"Avg memory reserved: {avg_mem_reserved:.4f}GB ± {std_dev_mem_reserved:.4f}GB")
    else:
        print("No steps timed because total_steps <= warmup_steps.")



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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--layer-size-output", type=str)
    args = parser.parse_args()

    # create tokenizer
    # zaratan compute nodes do not have internet access, so use cached tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # load in dataset
    dataloader = get_wikitext103_dataloader(args.dataset, tokenizer, seq_len=args.max_seq_len, batch_size=args.batch_size)

    # initalize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, dataloader.dataset.vocab_size, device)
    optimizer = build_optimizer(args, model)

    if args.compile:
        model = torch.compile(model)

    # count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_params = f"{num_params/1e9:.1f}B" if num_params >= 1e9 else f"{num_params/1e6:.1f}M"
    print(f"Training {args.model} with {num_params} parameters on {device} device.")

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

    print("Training complete!")

    if args.layer_size_output:
        layer_sizes = model.get_num_params_per_layer()
        layer_sizes.to_csv(args.layer_size_output)
        print(layer_sizes)
        print("Total: ", layer_sizes["No. Params"].sum())


if __name__ == "__main__":
    main()
