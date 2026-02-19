# NOTE: Do not modify this file except to add timers and time-reporting code

import argparse
import math
from array import array

from mpi4py import MPI

import collectives

def make_input(rank: int, count: int) -> array:
    data = array("d", [0.0] * count)
    for i in range(count):
        data[i] = float(rank) + float(i) * 1e-3
    return data


def check_buffers(expected: array, actual: array, comm: MPI.Comm) -> None:
    local_max = 0.0
    for i in range(len(expected)):
        local_max = max(local_max, abs(expected[i] - actual[i]))

    global_max = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        if global_max <= 1e-6:
            print(f"PASS (max abs diff = {global_max})")
        else:
            print(f"FAIL (max abs diff = {global_max})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MPI collective harness (compares to MPI collectives)."
    )
    parser.add_argument(
        "--collective",
        required=True,
        choices=["allgather", "allreduce", "reducescatter"],
        help="Which collective to run",
    )
    parser.add_argument(
        "--variant",
        required=False,
        default="ring",
        choices=["ring", "recursive"],
        help="Which algorithm variant to run (ring or recursive)",
    )
    parser.add_argument(
        "--mib",
        required=True,
        type=int,
        help="Input size per rank in MiB (for reducescatter, this is the recv size per rank)",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    bytes_per_rank = args.mib * 1024 * 1024
    count = bytes_per_rank // array("d").itemsize
    if count <= 0:
        if comm.rank == 0:
            print("Input size too small for double elements.")
        return 1

    if args.variant not in ("ring", "recursive"):
        if comm.rank == 0:
            print(f"Unknown variant: {args.variant}")
        return 1

    if args.collective == "allgather":
        send = make_input(comm.rank, count)
        expected = array("d", [0.0] * (count * comm.size))
        actual = array("d", [0.0] * (count * comm.size))

        comm.Allgather(send, expected)
        if args.variant == "ring":
            collectives.allgather_ring(send, actual, comm)
        else:
            collectives.allgather_recursive_doubling(send, actual, comm)

        check_buffers(expected, actual, comm)
    elif args.collective == "allreduce":
        send = make_input(comm.rank, count)
        expected = array("d", [0.0] * count)
        actual = array("d", [0.0] * count)

        comm.Allreduce(send, expected, op=MPI.SUM)
        collectives.allreduce(send, actual, comm)

        check_buffers(expected, actual, comm)
    else:
        recvcount = count
        sendcount = recvcount * comm.size

        send = make_input(comm.rank, sendcount)
        expected = array("d", [0.0] * recvcount)
        actual = array("d", [0.0] * recvcount)

        recvcounts = [recvcount] * comm.size
        comm.Reduce_scatter(send, expected, recvcounts, op=MPI.SUM)
        if args.variant == "ring":
            collectives.reducescatter_ring(send, actual, comm)
        else:
            collectives.reducescatter_recursive_halving(send, actual, comm)

        check_buffers(expected, actual, comm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
