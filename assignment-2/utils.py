import torch


def test_custom_torch_op(
    torch_op,
    custom_op,
    shape=(1024,),
    direction = "forward",
    seed=42,
    device='cuda',
):
    """ Test a custom PyTorch operator against a reference operator.

        Args:
            torch_op (callable): Reference PyTorch operator.
            custom_op (callable): Custom PyTorch operator.
            shape (tuple): Input shape for the operator.
            direction (str): Direction to test the operator. Can be 'forward' or 'backward'.
            seed (int): Random seed for reproducibility.
            device (str): Device to run the operator.
    """
    assert direction in ["forward", "backward"], "Direction must be 'forward' or 'backward'"

    torch.manual_seed(seed)

    inp = torch.randn(*shape, dtype=torch.float64).to(device)
    inp.requires_grad = direction != "forward"

    custom_out = custom_op(inp)
    torch_out = torch_op(inp)

    if direction == "forward":
        assert custom_out.shape == torch_out.shape, \
            f"Expected shape {torch_out.shape} but got {custom_out.shape}"
        assert torch.allclose(custom_out, torch_out), \
            f"Expected {torch_out} but got {custom_out} for input {inp}"

    # use the dummy loss to compare gradients
    if direction == "backward":
        torch.autograd.gradcheck(
            custom_op,
            inp,
            raise_exception=True
        )


def time_torch_op(
    op,
    shape=(1024,),
    num_inputs=1,
    num_iters=100,
    warm_start=10,
    device='cuda',
):
    """ Times a PyTorch operator. Returns the average time per iteration.

        Args:
            op (callable): PyTorch operator to time.
            shape (tuple): Input shape for the operator.
            num_iters (int): Number of iterations to run.
            warm_start (int): Number of warm-up iterations.
            device (str): Device to run the operator.
    """
    inputs = [torch.randn(*shape).to(device) for _ in range(num_inputs)]

    # warm up the cache
    for _ in range(warm_start):
        op(*inputs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        op(*inputs)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters