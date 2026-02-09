""" CMSC 828G Assignment 1 Part 2: Implementation of graph max pooling kernel in triton.
"""
import torch

import triton
import triton.language as tl


# MAX POOL
@triton.jit
def _max_pool_kernel_forward(
    # <Pass the required arguments to the kernel>
):
    """ Max pool forward pass. Computes the max value along the columns of the input tensor.
    """
    # <Implement the max pool forward kernel>


def _max_pool_triton_forward(
    x: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty(x.shape[1], device=x.device, dtype=x.dtype)
    indices = torch.empty(x.shape[1], dtype=torch.long, device=x.device)
    num_rows, num_cols = x.shape
    
    # <Add call to the max pool forward kernel here>
    # you can use whatever grid size and BLOCK setup you like
    # you need to write the max values into `out` and the indices of the max values into `indices`

    return out, indices



@triton.jit
def _max_pool_kernel_backward(
    # <Pass the required arguments to the kernel>
):
    """ Max pool backward pass. Computes the gradient of the max pool operation.
        Backpropagates the gradient to the max value in the input tensor.
    """
    # <Implement the max pool backward kernel>


def _max_pool_triton_backward(
    indices: torch.Tensor,
    grad: torch.Tensor,
    shape: tuple[int, int]
) -> torch.Tensor:
    out = torch.empty(shape, device=grad.device)
    num_rows, num_cols = shape
    
    # <Add call to the max pool backward kernel here>

    return out


class _max_pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y, indices = _max_pool_triton_forward(x)
        ctx.save_for_backward(indices)
        ctx.input_shape = x.shape
        return y

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        x_shape = ctx.input_shape
        return _max_pool_triton_backward(indices, grad_output, shape=x_shape)

custom_max_pool = _max_pool.apply


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(6, 16)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],#, 'torch+cpu'],
        line_names=['Triton', 'Torch'],#, 'Torch (CPU)'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='runtime (ms)',
        plot_name='max_pool_forward_performance',
        args={}
    )
)
def benchmark(N, provider):
    x = torch.randn(N, N//4, device='cuda' if not provider.endswith('cpu') else 'cpu', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch' or provider == 'torch+cpu':
        func = lambda: torch.max(x, dim=0).values
    elif provider == 'triton':
        func = lambda: custom_max_pool(x)
    return triton.testing.do_bench(func, quantiles=quantiles)



if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import test_custom_torch_op, time_torch_op

    parser = ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--test-forward', action='store_true')
    action.add_argument('--test-backward', action='store_true')
    action.add_argument('--benchmark', type=str)
    action.add_argument('--time-with-size', type=int, nargs=2, help="Time the max_pool kernel for a tensor of size NxM.")
    args = parser.parse_args()

    if args.test_forward or args.test_backward:
        direction = "forward" if args.test_forward else "backward"
        print(f"Testing max pool {direction} triton kernel")

        for num_rows in [5, 10, 512, 750]:
            for num_cols in [5, 10, 512, 750]:
                test_custom_torch_op(
                    torch_op=lambda x: torch.max(x, dim=0).values,
                    custom_op=custom_max_pool,
                    shape=(num_rows, num_cols),
                    direction=direction,
                )
                print(f"test passed -- N = {num_rows}, M = {num_cols}")

    elif args.benchmark:
        print("Timing max pool triton kernel")
        benchmark.run(print_data=True, save_path=args.benchmark)

    elif args.time_with_size:
        avg_time = time_torch_op(
            custom_max_pool,
            shape=(args.time_with_size[0], args.time_with_size[1]),
        )
        print(f"Average time taken for max pool kernel of size {args.time_with_size[0]}x{args.time_with_size[1]}: {avg_time} ms")

        