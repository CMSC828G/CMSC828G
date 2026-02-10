// NOTE: Do NOT modify this file

#pragma once

#include <mpi.h>

void AllGather_ring(const double *sendbuf, int sendcount, double *recvbuf, MPI_Comm comm);
void AllGather_recursive_doubling(const double *sendbuf, int sendcount, double *recvbuf, MPI_Comm comm);

void AllReduce(const double *sendbuf, double *recvbuf, int count, MPI_Comm comm);

void ReduceScatter_ring(const double *sendbuf, double *recvbuf, int recvcount, MPI_Comm comm);
void ReduceScatter_recursive_halving(const double *sendbuf, double *recvbuf, int recvcount, MPI_Comm comm);
