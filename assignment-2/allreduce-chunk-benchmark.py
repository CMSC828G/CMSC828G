import argparse
import math
import os
import time
import statistics
import torch
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark async allreduce with varying chunk sizes and report average and stddev timings"
    )
    parser.add_argument("--numel", type=int, default=10**6,
                        help="Total number of elements in the tensor")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float64", "float16"],
                        help="Data type for the tensor")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of repetitions per chunk size")
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

    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but backend is nccl")
        device = torch.device("cuda", slurm_local_rank)
        assert slurm_local_rank < torch.cuda.device_count(), f"SLURM_LOCALID={slurm_local_rank} is greater than the number of GPUs"
    else: 
        device = torch.device("cpu")

    # initialize the distributed process group
    dist.init_process_group(backend=args.backend, rank=slurm_rank, world_size=slurm_world_size, device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    message = torch.randn(args.numel, dtype=dtype, device=device)
    element_size = message.element_size()  # bytes per element
    total_bytes = args.numel * element_size

    # set the range of chunk sizes (powers of 10)
    max_exp = int(math.log10(args.numel)) if args.numel > 0 else 1

    # only print on rank 0
    if rank == 0:
        header = ("backend,world_size,num_gpus,total_elements,total_bytes,chunk_elements,chunk_bytes,average_time,stdev_time")
        print(header)

    # Loop over different chunk sizes
    for exp in range(3, max_exp + 1):
        chunk_size = 10 ** exp
        # Split the message into chunks. 
        chunks = list(torch.split(message, chunk_size))
        chunk_bytes = chunk_size * element_size

        times = []
        for rep in range(1, args.repetitions + 1):
            # Synchronize before timing.
            dist.barrier()
            start_time = time.perf_counter()

            work_handles = []
            for chunk in chunks:
                work = dist.all_reduce(chunk, async_op=True)
                work_handles.append(work)

            # wait for all async operations to complete.
            for work in work_handles:
                work.wait()

            dist.barrier()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

        if rank == 0:
            avg_time = statistics.mean(times)
            stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

            csv_line = (
                f"{args.backend},{world_size},{torch.cuda.device_count() if torch.cuda.is_available() else 0},"
                f"{args.numel},{total_bytes},{chunk_size},{chunk_bytes},{avg_time:.6f},{stdev_time:.6f}"
            )
            print(csv_line)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
