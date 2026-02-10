from mpi4py import MPI
from array import array

def allgather_ring(sendbuf, recvbuf, comm: MPI.Comm):
    """AllGather ring using point-to-point only.

    sendbuf: array-like of length sendcount
    recvbuf: preallocated array-like of length sendcount * comm.size
    """
    pass

def allgather_recursive_doubling(sendbuf, recvbuf, comm: MPI.Comm):
    """AllGather recursive doubling using point-to-point only.

    sendbuf: array-like of length sendcount
    recvbuf: preallocated array-like of length sendcount * comm.size
    """
    pass

def allreduce(sendbuf, recvbuf, comm: MPI.Comm):
    """AllReduce using point-to-point only (SUM).

    sendbuf: array-like of length count
    recvbuf: preallocated array-like of length count
    """
    pass


def reducescatter_ring(sendbuf, recvbuf, comm: MPI.Comm):
    """ReduceScatter ring using point-to-point only (SUM).

    sendbuf: array-like of length recvcount * comm.size
    recvbuf: preallocated array-like of length recvcount
    """
    pass

def reducescatter_recursive_halving(sendbuf, recvbuf, comm: MPI.Comm):
    """ReduceScatter recursive halving using point-to-point only (SUM).

    sendbuf: array-like of length recvcount * comm.size
    recvbuf: preallocated array-like of length recvcount
    """
    pass
