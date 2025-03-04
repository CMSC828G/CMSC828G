#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16000
#SBATCH -A cmsc828-class
#SBATCH -t 00:10:00

source /scratch/zt1/project/cmsc828/shared/assignment-2/.venv/bin/activate
export TIKTOKEN_CACHE_DIR="/scratch/zt1/project/cmsc828/shared/assignment-2/.tiktoken"
ml cuda/gcc/11.3.0/zen2/12.3.0

MODEL=gpt2-large
BATCH_SIZE=4

PROFILE_OUTPUT="train-profile"
PROJECTED_PROFILE_OUTPUT="train-profile_projected-gpu-times"
FINAL_TIMES_OUTPUT="layer-times-${MODEL}-bs${BATCH_SIZE}.csv"
FINAL_SIZES_OUTPUT="layer-sizes-${MODEL}-bs${BATCH_SIZE}.csv"

CMD="python -u train-benchmark.py --dataset /scratch/zt1/project/cmsc828/shared/assignment-2/datasets/wikitext-103/ --layer-size-output ${FINAL_SIZES_OUTPUT} --dtype float32 --model ${MODEL} --batch_size ${BATCH_SIZE}"

# collect an NSYS profile of the training loop. We'll use this
# to grab the timings of the gradient computation
echo "Profiling a few training iterations."
srun -u nsys profile \
    -w true \
    -t cuda,nvtx \
    -s cpu \
    --capture-range=cudaProfilerApi \
    -x true \
    --force-overwrite true \
    -o $PROFILE_OUTPUT \
    $CMD

if [ ! -f "${PROFILE_OUTPUT}.nsys-rep" ]; then
    echo "Error: ${PROFILE_OUTPUT}.nsys-rep does not exist."
    exit 1
fi

# NSYS NVTX timings are from the CPU; fortunately NSYS provides a tool for projecting
# GPU times based on the CPU times and kernel launch times.
echo "Projecting NVTX timings to GPU times."
nsys recipe nvtx_gpu_proj_trace --input "${PROFILE_OUTPUT}.nsys-rep" --output $PROJECTED_PROFILE_OUTPUT

if [ ! -d "$PROJECTED_PROFILE_OUTPUT" ]; then
    echo "Error: Directory $PROJECTED_PROFILE_OUTPUT does not exist."
    exit 1
fi

# Analyze the output dataframe from the above NSYS recipe
echo "Analyzing profile. Storing gradient computation times into ${FINAL_TIMES_OUTPUT}."
python analyze-nsys-nvtx-times.py $PROJECTED_PROFILE_OUTPUT/trace.parquet $FINAL_TIMES_OUTPUT


# clean up; remove this if you want to inspect the intermediate profiling files
rm "${PROFILE_OUTPUT}.nsys-rep"
rm "${PROFILE_OUTPUT}.sqlite"
rm -r $PROJECTED_PROFILE_OUTPUT
