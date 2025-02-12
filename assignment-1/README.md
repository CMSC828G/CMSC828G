# Assignment 1: Writing Triton Kernels

In this assignment, you will implement custom GPU kernels for a Graph Convolutional Network (GCN) layer and pooling operation.
You will implement these using Triton and run them on A100s on Zaratan. You will be graded for the correctness and
performance of your implementations. This assignment consists of three main tasks:

1. Implement a ReLU forward and backward kernel.
2. Implement an effecient max reduction kernel for graph pooling.
3. Implement kernel fusion and explore performance trade-offs for different performance-related parameters.


## Background

You will implement a subset of the kernels in a graph convolution layer based on the GCN definition 
from [Kipf and Welling](https://openreview.net/pdf?id=SJU4ayYgl). The kernels you will be implementing
are described below, so you do not need to understand this article in detail, but if you would like
a more complete presentation of GNN basics you can check out [this article](https://distill.pub/2021/gnn-intro/).
The GCN layer we will be looking at is defined as below.

$$ H' = \sigma\left( \tilde{A} H W \right) $$

where:

- $\tilde{A}$ is an $N\times N$ adjacency matrix with normalization and self-loops (i.e. $\tilde{A}=\tilde{D}^{1/2}(A+I_N)\tilde{D}^{1/2}$).
- $H$ is a $N\times M$ feature matrix; either the input graph to the model or the output activations of a previous layer.
- $W$ is the trainable weight matrix of this layer with shape $M \times C$.
- $\sigma$ is the activation function (ReLU in our case).

The input to this layer is a graph with $N$ nodes and $M$ features (values) per node.
The output is a graph with $N$ nodes and $C$ features per node.

In addition to the GCN layer, you will also implement graph pooling. Graph pooling is generally employed to use GNNs for graph-level prediction tasks. Since the GCN layer outputs a vector *per node*, we want to reduce this to a vector *per graph*. One way to accomplish this is to compute a reduction along the node dimension of the output matrix. This will map the $N\times C$ output feature matrix to $1 \times C$, which can then be used for tasks like graph classification.

## Part 0: Setting up your Environment

Triton is capable of running on a range of GPU hardware, so feel free to develop on whatever GPUs are available to you. However, we will run and test your code on Zaratan for grading, so we recommend testing your final version there.
Please refer to this [quick primer](https://www.cs.umd.edu/class/spring2025/cmsc828g/zaratan.shtml) and the detailed [Zaratan usage docs](https://hpcc.umd.edu/hpcc/help/usage.html) for getting set up. A Python virtual environment has been created with all the packages required for this assignment. To load this environment, you can run

```bash
source /scratch/zt1/project/cmsc828/shared/assignment-1/.venv/bin/activate
```

With this environment activated, you'll have access to Python, PyTorch, and Triton.
To run your code on the GPUs, you'll need to submit a job using the Slurm scheduler
on Zaratan. There are two ways to do this: an interactive job or a batch job.
You can launch an interactive GPU job on Zaratan as follows.

```bash
# use salloc to request an interactive GPU node for 30 minutes
salloc -A cmsc828-class -n 1 -p gpu --gpus=a100_1g.5gb:1 -t 00:30:00

# once the job is granted you can run any commands; use srun to run on GPUs
source /scratch/zt1/project/cmsc828/shared/assignment-1/.venv/bin/activate
srun --pty python my_triton_python_script.py

# make sure to end the job if you're done before the time limit
exit
```

You can launch a batch job, which will be scheduled at a later time,
by first creating a batch script (shown below) and then submitting it with
`sbatch my_batch_script.job`. This will submit the job to a queue. You can
check the status of your queued jobs with `squeue --me`.

```sbatch
#!/bin/bash
#SBATCH -A cmsc828-class
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH -t 00:30:00

# setup: activate env and cd to your script
source /scratch/zt1/project/cmsc828/shared/assignment-1/.venv/bin/activate
cd /path/to/my/script

# run on the GPUs using srun
srun python my_triton_python_script.py
```


***Note:*** Zaratan is a shared resource used by hundreds of users on campus and compute hours are a limited resource. Please be considerate and do not excessively use resources or leave long unused interactive jobs hanging. The above commands use the A100 MiG partitions on Zaratan to get 1/7th of the GPU (`--gpus=a100_1g.5gb:1`). This should be all you need for debugging/testing correctness, but feel free to switch to the full A100 for performance testing(`--gpus=a100:1`).


## Part 1: ReLU in Triton

For the first part of the assignment you will implement the [ReLU activation function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
This is a simple element-wise function that preserves positive numbers and maps negative numbers to zero.

$$ \mathrm{ReLU}(x) = \max(x, 0) $$

In Python this could be implemented as below.

```python
def relu(x: List[float], out: List[float]):
    for idx in range(len(x)):
        out[idx] = max(x[idx], 0.0)
```

You will also need to implement the backward pass for ReLU. To compute the gradient
for ReLU, we simply backpropagate the gradient values where the input was positive.
This is shown in Python below.

```python
def relu_grad(x: List[float], grad: List[float], output: List[float]):
    for idx in range(len(output)):
        output[idx] = grad[idx] if x[idx] > 0.0 else 0.0
```


The setup, testing, and benchmark code for the ReLU kernel is in the [relu_kernel.py](relu_kernel.py) file. Your task is to implement the `_relu_kernel_forward` and `_relu_kernel_backward` functions using Triton. You can check the correctness of your implementations by running `srun python relu_kernel.py --test-forward` and `srun python relu_kernel.py --test-backward`. You can benchmark them by running `srun python relu_kernel.py --benchmark <results_fpath>`.


## Part 2: Graph Pooling

The next step is to implement the graph pooling operation. In this assignment, you will implement a max pooling operation over the node features. Specifically, you will write a Triton kernel that performs the following operation:

$$ \text{MaxPool}(H) = \max_{i \in \{1, \ldots, N\}} H_{i,:} $$

where $H$ is the $N \times C$ feature matrix, and the result is a $1 \times C$ matrix containing the maximum value of each column. Here is a simple example of how max pooling can be implemented in Python:

```python
def max_pooling(H: List[List[float]]) -> Tuple[List[float], List[int]]:
    n_rows, n_cols = len(H), len(H[0])
    pooled = [float('-inf')] * n_cols
    indices = [-1] * n_cols

    for row in range(n_rows):
        for col in range(n_cols):
            if H[row][col] > pooled[col]:
                pooled[col] = H[row][col]
                indices[col] = row

    return pooled, indices
```

You will also need to implement the gradient kernel for max pooling. For the backwards pass we only backpropagate gradient values where the max occurred in the matrix. An example of this in Python is shown below:

```python
def max_pooling_grad(H: List[List[float]], grad: List[float], indices: List[int]) -> List[List[float]]:
    n_rows, n_cols = len(H), len(H[0])
    grad_output = [[0.0 for _ in range(n_cols)] for _ in range(n_rows)]
    
    for col in range(len(grad)):
        # backpropagate the gradient where the max occurred
        grad_output[indices[col]][col] = grad[col]
    
    return grad_output
```

The setup, testing, and benchmark code for the pooling kernel is in the [max_pool_kernel.py](max_pool_kernel.py) file. To implement the forward kernel you need to write `_max_pool_kernel_forward` and `_max_pool_triton_forward`. To implement the backward kernel you need to write `_max_pool_kernel_backward` and `_max_pool_triton_backward`. You can verify their correctness by running `srun python max_pool_kernel.py --test-forward` and `srun python max_pool_kernel.py --test-backward`. You can benchmark them using `srun python max_pool_kernel.py --benchmark <results_fpath>`.

In addition to correctness, your max pooling forward kernel will also be tested for performance. 
We will test the performance on a full A100 on Zaratan for various matrix sizes (number of rows and cols $\le 2^{15}$). 
To receive full credit your implementation should be near the performance of the PyTorch implementation (`torch.max(x, axis=0)`).
You should not have Triton autotuning on your final submitted kernel.

## Part 3: Kernel fusion, Performance parameters, and Performance analysis

In this part, you will fuse two kernels and try different performance-related parameters to study their impact on performance.

***Kernel Fusion:*** Fuse (combine) two kernels into a single kernel -- the ReLU kernel with the provided [matrix multiplication triton kernel](matmul_kernel.py). How does this change the performance compared to calling the two kernels individually back-to-back? You can also try fusing ReLU with your pooling kernel. Are the performance improvements the same? Feel free to create new files and tests.

***Performance Tuning:*** Try different values of the various performance-related parameters (i.e. block size, number of warps, number of stages) and study the impact these have on performance. Are the performances differences as expected? How does this change across problem sizes?

In your report, write about your findings regarding kernel fusion and the performance tuning of parameters. You should include plots and/or tables to demonstrate your findings.

## What to Turn In and Grading

You will upload a tarball `lastname-firstname-assign1.tar.gz` to [gradescope](https://www.gradescope.com/courses/924314) containing the following:

- Your `relu_kernel.py` file implementing the forward and backward kernels.
- Your `max_pool_kernel.py` file implementing the forward and backward kernels.
- A concise but complete report (1-3 pages) named `report.pdf` addressing your findings in part 3.
- Any other files you created to test or collect results for part 3.

The grading for this assignment is distributed according to the table below.

| Task | Points |
| ---- | ------ |
| Part 1: ReLU Kernel Correctness (forward and backward) | 20 |
| Part 2: Max Pooling Kernel Correctness (forward and backward) | 30 |
| Part 2: Max Pooling Kernel Performance (forward) | 20 |
| Part 3 and Report | 30 |
| **Total** | 100 |


## Part 4 (Optional, For Fun): Training a GNN

If you would like to try training a GNN using your implemented Triton kernels, you can run the graph classification script in [train.py](train.py). Run this script as below to train a 2-layer GCN.

```bash
# run `ls /scratch/zt1/project/cmsc828/shared/assignment-1/datasets` to see available datasets
TRAIN_DATASET="/scratch/zt1/project/cmsc828/shared/assignment-1/datasets/DD_train.pt"
TEST_DATASET="/scratch/zt1/project/cmsc828/shared/assignment-1/datasets/DD_test.pt"

srun python train.py \
    --train_dataset $TRAIN_DATASET \
    --test_dataset $TEST_DATASET \
    --hidden_units 64 \
    --epochs 100 \
    --lr 0.001 \
    --kernel-type triton # or "torch" to compare with PyTorch
```

## Part 5 (Optional, For Fun): More Fusion

For those that would like some more Triton practice, consider if there is more potential for fusion in the GCN layer. We compute two matrix multiplications, an activation function, and, if we are in the last layer, a graph pooling operation. Can these be further fused into single kernels? 

Hints:
- Consider computing two matrix multiplications $D=ABC$. What values of $A$, $B$, $(AB)$, and $C$ do each $D_{ij}$ depend on? Is it possible to replace values in the intermediate matrix $(AB)$ by recomputing and reloading values of $A$ and $B$? For what shapes of $A$, $B$, and $C$ is this beneficial?
- Fusing the matrix multiplication and max pooling is a bit more difficult as Triton cannot synchronize across blocks. Consider splitting the reduction into two kernels: the first part fused with the matrix multiply and the second part in its own kernel.

