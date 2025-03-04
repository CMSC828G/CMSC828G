#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gpus=a100:8
#SBATCH -A cmsc828-class
#SBATCH -t 00:15:00


source /scratch/zt1/project/cmsc828/shared/assignment-2/.venv/bin/activate

#CHANGE THIS; numel is a list of sizes to benchmark allreduce with
export NUMEL="1 12792 25584 51169 102338 204677 409355 818710 1637420 3274840 6549680 13099360 26198720 52397440 104794880 209589760 419179520 838359040 1676718080"
export DTYPE=float32
export BACKEND=nccl

# NCCL ring reduction is slow on Zaratan for some reason
# and letting NCCL choose also leads to some weird performance behavior
# fix to use tree reduction for gathering results
export NCCL_ALGO=TREE

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500


srun -u python -u allreduce-benchmark.py \
    --numel $NUMEL \
    --dtype $DTYPE \
    --backend $BACKEND
    