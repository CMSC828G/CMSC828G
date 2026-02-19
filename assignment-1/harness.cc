// NOTE: Do not modify this file except to add timers and time-reporting code

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "collectives.h"

struct Args {
    std::string collective;
    std::string variant = "ring";
    int mib = 1;
};

bool parse_args(int argc, char **argv, Args *out) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--collective") == 0 && i + 1 < argc) {
            out->collective = argv[++i];
        } else if (std::strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            out->variant = argv[++i];
        } else if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            out->mib = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0) {
            return false;
        } else {
            return false;
        }
    }
    return !out->collective.empty() && out->mib > 0;
}

void usage() {
    std::cerr << "usage: mpirun -n <ranks> ./harness --collective <allgather|allreduce|reducescatter> ";
    std::cerr << "--variant <ring|recursive> --mib <size_per_rank>\n";
}

std::vector<double> make_input(int rank, int count) {
    std::vector<double> v(count);
    for (int i = 0; i < count; ++i) {
        v[i] = static_cast<double>(rank) + static_cast<double>(i) * 1e-3;
    }
    return v;
}

void check_buffers(
    const std::vector<double> &expected,
    const std::vector<double> &actual,
    MPI_Comm comm
) {
    double local_max = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        local_max = std::max(local_max, std::abs(expected[i] - actual[i]));
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        if (global_max <= 1e-6) {
            std::cout << "PASS (max abs diff = " << global_max << ")\n";
        } else {
            std::cout << "FAIL (max abs diff = " << global_max << ")\n";
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    Args args;
    if (!parse_args(argc, argv, &args)) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            usage();
        }
        MPI_Finalize();
        return 1;
    }

    if (args.variant != "ring" && args.variant != "recursive") {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Unknown variant: " << args.variant << "\n";
            usage();
        }
        MPI_Finalize();
        return 1;
    }

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const size_t bytes_per_rank = static_cast<size_t>(args.mib) * 1024 * 1024;
    const int count = static_cast<int>(bytes_per_rank / sizeof(double));
    if (count <= 0) {
        if (rank == 0) {
            std::cerr << "Input size too small for double elements.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (args.collective == "allgather") {
        std::vector<double> send = make_input(rank, count);
        std::vector<double> expected(count * world, 0.0);
        std::vector<double> actual(count * world, 0.0);

        MPI_Allgather(
            send.data(), count, MPI_DOUBLE,
            expected.data(), count, MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        if (args.variant == "ring") {
            AllGather_ring(send.data(), count, actual.data(), MPI_COMM_WORLD);
        } else {
            AllGather_recursive_doubling(send.data(), count, actual.data(), MPI_COMM_WORLD);
        }

        check_buffers(expected, actual, MPI_COMM_WORLD);
    } else if (args.collective == "allreduce") {
        std::vector<double> send = make_input(rank, count);
        std::vector<double> expected(count, 0.0);
        std::vector<double> actual(count, 0.0);

        MPI_Allreduce(send.data(), expected.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        AllReduce(send.data(), actual.data(), count, MPI_COMM_WORLD);

        check_buffers(expected, actual, MPI_COMM_WORLD);
    } else if (args.collective == "reducescatter") {
        const int recvcount = count;
        const int sendcount = recvcount * world;

        std::vector<double> send = make_input(rank, sendcount);
        std::vector<double> expected(recvcount, 0.0);
        std::vector<double> actual(recvcount, 0.0);

        std::vector<int> recvcounts(world, recvcount);
        MPI_Reduce_scatter(send.data(), expected.data(), recvcounts.data(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (args.variant == "ring") {
            ReduceScatter_ring(send.data(), actual.data(), recvcount, MPI_COMM_WORLD);
        } else {
            ReduceScatter_recursive_halving(send.data(), actual.data(), recvcount, MPI_COMM_WORLD);
        }

        check_buffers(expected, actual, MPI_COMM_WORLD);
    } else {
        if (rank == 0) {
            usage();
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}
