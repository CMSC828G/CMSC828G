#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gpus=a100:8
#SBATCH -A cmsc828-class
#SBATCH -t 00:05:00


source /scratch/zt1/project/cmsc828/shared/assignment-2/.venv/bin/activate

# CHANGE THIS; numel is the size of the tensor we're allreducing; it will be split into chunks
# and each chunk will be allreduced separately and asynchronously
export NUMEL=10000000
export DTYPE=float32
export BACKEND=nccl

# NCCL ring reduction is slow on Zaratan for some reason
# and letting NCCL choose also leads to some weird performance behavior
# fix to use tree reduction for gathering results
export NCCL_ALGO=TREE

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


srun -u python -u allreduce-chunk-benchmark.py \
    --numel $NUMEL \
    --dtype $DTYPE \
    --backend $BACKEND
    