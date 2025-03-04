#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -p gpu
#SBATCH --gpus=a100:8
#SBATCH -A cmsc828-class
#SBATCH -t 01:00:00

source /scratch/zt1/project/cmsc828/shared/assignment-2/.venv/bin/activate
export TIKTOKEN_CACHE_DIR="/scratch/zt1/project/cmsc828/shared/assignment-2/.tiktoken"

# CHANGE THIS; use this parameter to control the bucket size(s) tested (in MB)
BUCKET_SIZES_MB="32 64 128"

DTYPE=float32
MODEL=gpt2-large
BATCH_SIZE=4

# NCCL ring reduction is slow on Zaratan for some reason
# and letting NCCL choose also leads to some weird performance behavior
# fix to use tree reduction for gathering results
export NCCL_ALGO=TREE

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

for BUCKET_SIZE_MB in $BUCKET_SIZES_MB; do

    echo "Running with bucket size $BUCKET_SIZE_MB MB"

    srun -u python -u train.py \
    --dataset "/scratch/zt1/project/cmsc828/shared/assignment-2/datasets/wikitext-103/" \
    --bucket-size-mb $BUCKET_SIZE_MB \
    --dtype $DTYPE \
    --model $MODEL \
    --batch_size $BATCH_SIZE

done
