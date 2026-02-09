""" Implementation of a matmul kernel in Triton.
    Mostly taken from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    with some additions.
"""
import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel_forward(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Compute C = A x B where A is MxK, B is KxN, and C is MxN

    # Each program computes a block of elements of C
    # compute these within groups
    block_id = tl.program_id(axis=0)
    row_block_id = tl.cdiv(M, BLOCK_SIZE_M)
    col_block_id = tl.cdiv(N, BLOCK_SIZE_N)
    proc_in_group = GROUP_SIZE_M * col_block_id
    group_id = block_id // proc_in_group
    group_start = group_id * GROUP_SIZE_M
    group_size = min(row_block_id - group_start, GROUP_SIZE_M)
    row_process_id = group_start + ((block_id % proc_in_group) % group_size)
    col_process_id = (block_id % proc_in_group) // group_size

    # Create pointers to blocks of A and B
    a_offsets = (row_process_id*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    b_offsets = (col_process_id*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    inner_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (a_offsets[:, None] * stride_am + inner_offsets[None, :] * stride_ak)
    b_ptrs = b_ptr + (inner_offsets[:, None] * stride_bk + b_offsets[None, :] * stride_bn)

    # Allocate a block of C to accumulate into
    c_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # Load submatrices of A and B into memory
        a_block = tl.load(a_ptrs, mask=inner_offsets[None, :] < K - k*BLOCK_SIZE_K, other=0.0)
        b_block = tl.load(b_ptrs, mask=inner_offsets[:, None] < K - k*BLOCK_SIZE_K, other=0.0)

        # compute dot product
        c_block = tl.dot(a_block, b_block, c_block)

        # move A and B pointers to next block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store the result in C
    c_row_offsets = row_process_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_col_offsets = col_process_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * c_row_offsets[:, None] + stride_cn * c_col_offsets[None, :]
    c_mask =  (c_row_offsets[:, None] < M) & (c_col_offsets[None, :] < N)
    tl.store(c_ptrs, c_block, mask=c_mask)


def _matmul_triton_forward(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    assert A.shape[1] == B.shape[0]
    assert A.dtype == B.dtype
    #assert A.is_contiguous(), f"Input A must be contiguous: {A}"
    if not A.is_contiguous():
        A = A.contiguous()
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel_forward[grid](
        A, B, C, 
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
    )
    return C


class _matmul(torch.autograd.Function):
    @staticmethod
    def forward(A, B):
        return _matmul_triton_forward(A, B)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B = inputs
        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors

        grad_A = None
        grad_B = None

        if ctx.needs_input_grad[0]:
            grad_A = _matmul_triton_forward(grad_output, B.T)
        if ctx.needs_input_grad[1]:
            grad_B = _matmul_triton_forward(A.T, grad_output)

        return grad_A, grad_B

custom_matmul = _matmul.apply


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(2, 33)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul_forward_performance',
        args={}
    )
)
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        func = lambda: torch.matmul(A, B)
    elif provider == 'triton':
        func = lambda: custom_matmul(A, B)
    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)
    perf = lambda ms: 2*M*N*K*1e-12 / (ms * 1e-3)
    return perf(ms), perf(min_ms), perf(max_ms)



if __name__ == '__main__':
    from argparse import ArgumentParser
    from utils import test_custom_torch_op, time_torch_op
    torch.manual_seed(42)

    parser = ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--test-forward', action='store_true')
    action.add_argument('--test-backward', action='store_true')
    action.add_argument('--benchmark', type=str)
    action.add_argument('--time-with-size', type=int, help="Time the matmul kernel for this size.")
    args = parser.parse_args()

    if args.test_forward or args.test_backward:
        direction = "forward" if args.test_forward else "backward"
        print(f"Testing matmul {direction} triton kernel")

        for M, N, K in [(5,5,5), (32, 64, 128), (128, 32, 64), (64, 128, 32), (128, 128, 128), (517, 85, 119), (2048, 1024, 2048)]:
            A = torch.randn((M, K), device='cuda', dtype=torch.float32)
            B = torch.randn((K, N), device='cuda', dtype=torch.float32)
            A.requires_grad = True
            B.requires_grad = True

            custom_output = custom_matmul(A, B)
            torch_output = torch.matmul(A, B)

            if direction == "forward":
                assert custom_output.shape == torch_output.shape, \
                    f"Output shape mismatch for A.shape={A.shape}, B.shape={B.shape}"
                assert torch.allclose(custom_output, torch_output, atol=0.2, rtol=0), \
                    f"Output value mismatch A=\n{A}\nB=\n{B}\nExpected\n{torch_output}\nbut got\n{custom_output}"

            # test gradients
            if direction == "backward":
                loss = torch.sum(custom_output)
                loss.backward()
                custom_grad_A = A.grad.detach().clone()
                custom_grad_B = B.grad.detach().clone()

                A.grad.zero_()
                B.grad.zero_()
                loss = torch.sum(torch_output)
                loss.backward()
                torch_grad_A = A.grad.detach().clone()
                torch_grad_B = B.grad.detach().clone()

                assert torch.allclose(custom_grad_A, torch_grad_A, atol=0.2, rtol=0), \
                    f"Gradient mismatch for A.shape={A.shape}, B.shape={B.shape}"
                assert torch.allclose(custom_grad_B, torch_grad_B, atol=0.2, rtol=0), \
                    f"Gradient mismatch for A.shape={A.shape}, B.shape={B.shape}"

            
            print(f"Passed for A.shape={A.shape}, B.shape={B.shape}")

    elif args.benchmark:
        print("Benchmarking custom matmul op")
        benchmark.run(print_data=True, save_path='matmul_perf_results')

    elif args.time_with_size:
        print("Timing custom matmul op")
        avg_time = time_torch_op(
            custom_matmul,
            shape=(args.time_with_size, args.time_with_size),
            num_inputs=2,
            num_iters=20,
            warm_start=5,
            device='cuda',
        )
        print(f"Average kernel execution time: {avg_time:.5f} ms")


