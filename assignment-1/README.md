# Assignment 1: Implementing Collectives

In this assignment, you will implement collective routines using point-to-point routines to develop an understanding of how the algorithms function in communication libraries. You can choose one collective routine between **ReduceScatter (RS)** or **AllGather (AG)** and implement it using **MPI point-to-point** operations only. You may complete the assignment in **either C++ or Python**. An **optional AllReduce (AR)** stub is included in case you want to explore that (this is not part of the graded assignment).

## What You Need To Implement

You must choose **one** of the following collectives to implement:

- **ReduceScatter**
- **AllGather**

and implement the chosen collective using **two algorithmic variants**:

- **Ring**
- **Recursive** (**doubling** for AG, **halving** for RS)

An optional **AllReduce** stub is provided if you want extra practice. A typical approach is to implement it as a **RS + AG**, but this is not the only possible approach.

## Provided Code

We are providing starter code to make your implementation easier. The C++ stubs live in:

- `collectives.h`
- `collectives.cc`

Python stubs live in:

- `collectives.py`

You must **not** call MPI collective routines in your implementations. Use **MPI point-to-point** routines only.

To keep things simple, we will **not** test with asymmetric message sizes per process, so you can assume **same input message sizes** across processes.

## How To Run

### (C++)

Run the provided harness:

```bash
srun -n 4 ./harness --collective allgather --variant ring --mib 4 
srun -n 4 ./harness --collective reducescatter --variant recursive --mib 4 
```

### Python

A Python environment with all packages required for this assignment is provided, you may load it like so:

```bash
source /scratch/zt1/project/cmsc828/shared/assignment-1/bin/activate
```

Run the provided harness:

```bash
srun -n 4 python3 harness.py --collective allgather --variant ring --mib 4 
srun -n 4 python3 harness.py --collective reducescatter --variant recursive --mib 4 
```

### Testing Scripts

We have also created some bash scripts that can combine several tests into one, you may find them useful.

C++:

```bash
./run_tests.sh <harness> <AG|AR|RS> [num_ranks]
```

Python:

```bash
./run_tests_py.sh <AG|AR|RS> [num_ranks]
```

These scripts run message sizes of **2 and 4 MiB**. For AG and RS, both **ring** and **recursive** variants are tested. For AR, only a single variant is run.

### MiB Interpretation

`--mib` is interpreted as follows:

- **AllGather (AG)**: per-rank input size; per-rank output size is `P × input`
- **AllReduce (AR)**: per-rank input size; per-rank output size is the same
- **ReduceScatter (RS)**: per-rank receive size; per-rank input size is `P × recv`


## Using zaratan or nexusclass

You can run your code on Zaratan or nexusclass. We'll be testing your code on zaratan. While compiling on the login nodes is fine, code should be run only on compute nodes.

*NOTE:* Zaratan and nexusclass are shared resources used by hundreds of users on campus. Compute hours are a limited resource. Please be considerate and do not excessively use resources or leave long unused interactive jobs hanging.


## Timing And Performance Measurements

You should **add timers to either the C++ or Python harness** (whichever you implement) to measure performance. A typical pattern is:

1. Use `MPI_Wtime()` around your collective implementation.
2. Compute timing statistics across all processes using `MPI_Allreduce` (this is the only place you can use an MPI collective):
   - **min** time
   - **max** time
   - **avg** time (sum / P)

Report these numbers for each experiment to characterize performance and variability.

## Evaluation Requirements

For your chosen collective (RS or AG), evaluate:

- **Process counts**: 16, 32, 64, 128, 256
- **Message sizes**: 2 MiB, 4 MiB
- **2 algorithmic variants**: ring and recursive doubling/halving

This yields **20 measurements** per collective implementation.

Use **fixed per-process message sizes** as you scale `P`, so total data volume grows with the number of processes.

## Report

Submit a **~2 page PDF report (3 pages max)** that includes:

- What you implemented and any design decisions
- How you measured performance
- A summary of your results
- Interpretation of performance trends (e.g., scaling, variant comparison)
- Any **unexpected** results you observed
- If you **cannot fully explain** the performance, describe what you would do next to investigate

Focus on explaining what you observed and why you think the performance looks the way it does.

## What To Submit

You will upload a tarball `lastname-firstname-assign1.tar.gz` to gradescope containing the following:

- Your **implementation file(s)**:
  - **C++**: `collectives.cc`
  - **Python**: `collectives.py`
- A concise but complete report (1-3 pages) named `report.pdf`.


## Grading Breakdown

- **40%**: Correctness of the ring implementation for your chosen collective (AG or RS)
- **40%**: Correctness of the recursive implementation for your chosen collective (AG or RS)
- **20%**: Report quality and analysis

For correctness tests, we will run two message sizes and two process counts (10% each).

## Reference

This paper is interesting and may be useful to you in implementing your collectives.

- [Optimization of Collective Communication Operations in MPICH](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf)
