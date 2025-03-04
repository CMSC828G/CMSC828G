import argparse
import math
import os
import time
import torch
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark allreduce with varying sizes and report average and stddev timings"
    )
    parser.add_argument("--numel", type=int, nargs="+", default=[10**i for i in range(6, 9)],
                        help="List of tensor sizes to benchmark")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64", "float16"],
                        help="Data type for the tensor")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of repetitions per size")
    parser.add_argument("--backend", type=str, default="nccl",
                        help="Distributed backend to use (e.g., nccl or gloo)")
    return parser.parse_args()

def print0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def get_dtype(dtype_str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "float16":
        return torch.float16
    else:
        raise ValueError("Unsupported dtype")

def main():
    args = parse_args()
    dtype = get_dtype(args.dtype)

    # grab slurm rank and world size
    slurm_rank = int(os.getenv("SLURM_PROCID", default=0))
    slurm_world_size = int(os.getenv("SLURM_NTASKS", default=1))
    slurm_local_rank = int(os.getenv("SLURM_LOCALID", default=0))

    # If using GPUs, select the appropriate device.
    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but backend is nccl")
        device = torch.device("cuda", slurm_local_rank)
        assert slurm_local_rank < torch.cuda.device_count(), f"SLURM_LOCALID={slurm_local_rank} is greater than the number of GPUs"
    else: 
        device = torch.device("cpu")

    # Initialize the distributed process group.
    dist.init_process_group(backend=args.backend, rank=slurm_rank, world_size=slurm_world_size, device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    header = ("backend,world_size,total_elements,total_bytes,average_time,stdev_time")
    print0(header)

    for numel in args.numel:

        # Create a large random tensor on the chosen device.
        message = torch.randn(numel, dtype=dtype, device=device)
        element_size = message.element_size()  # bytes per element
        total_bytes = numel * element_size

        # warmup
        for _ in range(3):
            dist.all_reduce(message, async_op=False)

        # Collect timings for each repetition for this size.
        times = []
        for _ in range(args.repetitions):
            # Synchronize before timing.
            dist.barrier()
            start_time = time.perf_counter()

            dist.all_reduce(message, async_op=False)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            dist.barrier()

        # Compute average and standard deviation across repetitions.
        times = torch.tensor(times, device=device)
        dist.reduce(times, dst=0)
        if rank == 0:
            times = times / world_size
            avg_time = torch.mean(times).item()
            stdev_time = torch.std(times).item()

            csv_line = (
                f"{args.backend},{world_size},"
                f"{numel},{total_bytes},{avg_time:.6f},{stdev_time:.6f}"
            )
            print(csv_line)

        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
