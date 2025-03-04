# Assignment 2 

In this assignment, you will explore distributed training through the lens of performance modeling.
Specifically, you will analyze the performance of distributed data parallel (DDP) training in PyTorch, and model the performance with respect to bucket size.


## Main Task

You will develop a performance model for the backward pass and communication in DDP based on the size of the buckets used for AllReduce.
Recall that during the backward pass in DDP, we compute gradients locally on each GPU and then use an AllReduce operation to average the gradients across all participating GPUs.
To improve performance, PyTorch DDP buckets gradients together to send fewer, larger messages and avoid latency overheads.
However, the fewer messages that we send, the less we can overlap communication and computation.
Thus, the size of these buckets can have a significant impact on performance and we would like to know the ideal values for different scenarios.
We could run lots of tests to search for the ideal value, but on a large training run with thousands of GPUs this is impractical.
Instead we can develop a performance model to predict the performance based on bucket size and use that to estimate the optimal bucket size value.

### Part 1

Your task is to design an equation or function that estimates the time to compute the backward pass. 
Your model can take as inputs any or all of the following values.

- $B$ --- total number of bytes of all the gradients in the model
- $b$ --- bucket size in bytes
- $g_l$ --- time, in seconds, to compute the gradient for layer $l$ on the GPU
- $a(x)$ --- time, in seconds, to do an AllReduce operation of $x$ bytes
- $L$ --- number of layers in the model

Your performance model does not need to use all of these, however, and if there are any other values you would like to consider please ask about them on Piazza.
The form of the model is flexible; it can be an equation, set of equations, piecewise function, or short Python function.
The intent is to understand how each component contributes to the final performance of the backward pass.


### Part 2

Now that you have designed a performance model, you can use it to estimate $b^*$, the ideal value of the bucket size $b$. 
You will do this for a GPT-2 model being trained on 8 A100s with a batch size of 4.
The GPT-2 model we are looking at in this part is the gpt2-large model which has 36 transformer block layers.
[gpt2.py](gpt2.py) has more details on the GPT-2 model definition. 

Several scripts are provided in the assignment repo to help you compute/estimate the input values for your performance model.

- [launch-train-benchmark.sbatch](launch-train-benchmark.sbatch) --- A job script that runs several forward passes of the GPT-2 model on one GPU and outputs the parameters and compute time per gradient. You can submit the job to the job queue with `sbatch launch-train-benchmark.sbatch`. It will write out `layer-sizes.csv` and `layer-times.csv` containing the number of parameters per layer gradients and the total time to compute each layer's gradients. Feel free to inspect [launch-train-benchmark.sbatch](launch-train-benchmark.sbatch) and [train-benchmark.py](train-benchmark.py) for more info on how these work.
- [launch-allreduce-benchmark.sbatch](launch-allreduce-benchmark.sbatch) --- A job script that benchmarks PyTorch's AllReduce on 8 GPUs for several input message sizes and reports their runtime. You can change the message sizes benchmarked by changing the `export NUMEL="..."` line.
- [launch-allreduce-chunk-benchmark.sbatch](launch-allreduce-chunk-benchmark.sbatch) --- A job script that reproduces the AllReduce chunk benchmark from the DDP paper (Figures 2a and 2b). For a fixed message size, this script will benchmark doing an allreduce on the data in various chunk sizes on 8 GPUs. Change the `export NUMEL="..."` line to change the fixed number of total elements.

We've provided outputs for these on values of interest if you'd like to use the provided values instead of running it yourself. 
The number of parameters per layer for gpt2-large are in [layer-sizes-gpt2_large-bs4.csv](layer-sizes-gpt2_large-bs4.csv). 
The times to compute the gradient for each layer on a single GPU are in [layer-times-gpt2_large-bs4.csv](layer-times-gpt2_large-bs4.csv). 
Times to compute AllReduce of various message sizes on 8 GPUs are in [allreduce-times.csv](allreduce-times.csv).

The above scripts and pre-collected outputs should be enough to compute and/or estimate the inputs into your performance model (apart from the bucket size $b$ of course), but feel free to write additional benchmark scripts if needed.
Once you have computed the inputs for the performance model, use will them to estimate the ideal bucket size $b^*$ for training the GPT-2 model.


### Part 3

Now you will evaluate your model against real experiments. 
The script [launch-ddp-benchmark.sbatch](launch-ddp-benchmark.sbatch) will launch DDP training for various values of the bucket size (set `BUCKET_SIZE="..."` to change them).
Compare the output of your model with the actual values from DDP. Do the behaviors your model predicts make sense? If your performance model was incorrect, then dive a bit deeper into why you believe it was incorrect. What might you need to account for to improve your model?


### Grading and What to Turn In

What you will do:

- Part 1: Design a performance model for the backward pass of DDP based on the bucket size. Describe your performance model in 1-2 paragraphs.
- Part 2: Use benchmarks to estimate values for your performance model and predict the best bucket size value.
- Part 3: Create a lineplot with bucket size on the x-axis and time on the y-axis comparing your model's predicted times with the actual times from the real experiments. Highlight whether the predicted best bucket size was actually the best bucket size.
- Discussion: 2-3 paragraphs discussion on the accuracy of your model. Discuss how the ideal bucket size will change as the other parameters change (i.e. total bytes $B$ or AllReduce time $a$) in your model. 

To receive full credit for Part 1 you should make a best-effort attempt at designing your performance model and analyzing why it is accurate or inaccurate. There are a lot of ways to model the performance of distributed algorithms and we will not score based on your particular performance model design or its accuracy.

You will upload a tarball `lastname-firstname-assign2.tar.gz` to gradescope containing a report (`report.pdf`) and any other files you created for benchmarking and data collection.

### Hints

Some hints are included below to nudge you in the right direction if you are stuck.
It is recommended that you spend some time thinking about your performance model before consulting any hints as performance modeling is an essential skill in designing efficient ML systems.

<details>
<summary>Hint 1</summary>

Try thinking of the AllReduce communication in terms of latency, $\alpha$, and achieved bandwidth, $\beta$.
Latency is the *overhead* of the AllReduce, a constant time cost to each AllReduce function call.
Achieved bandwidth is the bytes per second that can be transmitted in AllReduce.
Given the latency $\alpha$ and bandwidth $\beta$, the total time to complete an $N$ byte AllReduce would be $T=\alpha + \beta^{-1} N$.

$\beta$ should be straightforward to benchmark.
$\alpha$ is often estimated by sending a 1 byte or 1 element message and using that time to represent the latency.

</details>

<details>
<summary>Hint 2</summary>

One way to think about the modeling problem is minimizing overhead.
We can consider the time to compute a DDP backward pass as below.

$$T_{\mathtt{DDP Backward}} = T_{\mathtt{Sequential Backward}} + T_{\mathtt{DDP Overhead}}$$

To find the ideal bucket size $b$, we want to find a bucket size that minimizes $T_{\mathtt{DDP Overhead}}$.
While this is a useful problem framing, it is not the only one.

</details>
