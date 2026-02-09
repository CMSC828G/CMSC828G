""" CMSC 828G Assignment 1 Part 1: ReLU Kernel in Triton.
"""
import torch
import triton
import triton.language as tl


# RELU
@triton.jit
def _relu_kernel_forward(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ * Implement the ReLU forward kernel in Triton here *
        Args:
            in_ptr: The input tensor pointer.
            out_ptr: The output tensor pointer.
            n_elements: The number of elements in the input tensor.
            BLOCK_SIZE: The block size to compute per kernel invocation.
    """
    # <Implement the ReLU forward kernel>


def _relu_triton_forward(
    x: torch.Tensor,
) -> torch.Tensor:
    """ Wrapper to call the ReLU forward triton kernel. """
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _relu_kernel_forward[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def _relu_kernel_backward(
    in_ptr,
    out_ptr,
    grad_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ * Implement the ReLU backward kernel in Triton here *
        Args:
            in_ptr: The input tensor pointer.
            out_ptr: The output tensor pointer.
            grad_ptr: The gradient tensor pointer.
            n_elements: The number of elements in the input tensor.
            BLOCK_SIZE: The block size to compute per kernel invocation.
    """
    # <Implement the ReLU backward kernel>


def _relu_triton_backward(
    x: torch.Tensor,
    grad: torch.Tensor,
) -> torch.Tensor:
    """ Wrapper to call the ReLU backward triton kernel. """
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    BLOCK_SIZE = 1024
    _relu_kernel_backward[grid](x, out, grad, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class _relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _relu_triton_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return _relu_triton_backward(x, grad_output)

custom_relu = _relu.apply



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(12, 26)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],# 'torch+cpu'],
        line_names=['Triton', 'Torch'],# 'Torch (CPU)'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='runtime (ms)',
        plot_name='relu_forward_performance',
        args={}
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda' if not provider.endswith('cpu') else 'cpu', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch' or provider == 'torch+cpu':
        func = lambda: torch.nn.functional.relu(x)
    elif provider == 'triton':
        func = lambda: custom_relu(x)
    return triton.testing.do_bench(func, quantiles=quantiles)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from utils import test_custom_torch_op, time_torch_op

    parser = ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--test-forward', action='store_true')
    action.add_argument('--test-backward', action='store_true')
    action.add_argument('--benchmark', type=str)
    action.add_argument('--time-with-size', type=int, help="Time the relu kernel for a tensor of this size.")
    args = parser.parse_args()

    if args.test_forward or args.test_backward:
        direction = "forward" if args.test_forward else "backward"
        print(f"Testing relu {direction} triton kernel")

        for N in [5, 10, 50, 112, 1024, 2048, 2050, 4096]:
            test_custom_torch_op(
                torch_op=torch.nn.functional.relu,
                custom_op=custom_relu,
                shape=(N,),
                direction=direction,
            )
            print(f"N={N} passed")
    elif args.benchmark:
        print("Timing relu triton kernel")
        benchmark.run(print_data=True, save_path=args.benchmark)
    elif args.time_with_size:
        avg_time = time_torch_op(
            custom_relu,
            shape=(args.time_with_size,),
            num_iters=25,
            warm_start=5,
            device='cuda',
        )
        print(f"Average kernel execution time: {avg_time:.5f} ms")

