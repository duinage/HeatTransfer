#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <mpi.h>

// Heat Equation (2D):
//
// T_t = alpha * (T_xx + T_yy)
//
// T:=T(x,y,t) - temperature function;
// alpha - diffusivity constant;
// x,y - spatial variables;
// t - time variable.

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif

using namespace std;

double T_analytical(double xx, double yy, double t, double alpha) {
    return exp(-2.0 * (M_PI * M_PI) * alpha * t) * sin(M_PI * (xx + 0.5)) * sin(M_PI * (yy + 0.5));
}

int main (int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // CONFIG
    double alpha = 1.0;
    double finish_time = 1.1;
    double xlen = 1.0, ylen = 1.0;
    int N = 100;
    int nx = static_cast<int>(xlen * N);
    int ny = static_cast<int>(ylen * N);
    double dx = xlen / (nx - 1.0);
    double dy = ylen / (ny - 1.0);
    double dt = dx * dx * dy * dy / alpha / (dx * dx + dy * dy) / 2.0;

    // Boundary Conditions
    double T_bot = 0.0, T_top = 0.0, T_left = 0.0, T_right = 0.0;

    // Initial Condition
    int base = nx / size;
    int remainder = nx % size;
    int start = rank * base;
    int stop = start + base + (rank == size - 1 ? remainder : 0);
    int local_nx = stop - start;

    vector<double> x_full(nx);
    for (int i = 0; i < nx; ++i) {
        x_full[i] = -xlen / 2.0 + i * dx;
    }
    vector<double> local_x(local_nx);
    for (int i = 0; i < local_nx; ++i) {
        local_x[i] = x_full[start + i];
    }
    vector<double> y(ny);
    for (int j = 0; j < ny; ++j) {
        y[j] = -ylen / 2.0 + j * dy;
    }

    vector<double> T_local((local_nx + 2) * ny, 0.0);
    for (int i = 0; i < local_nx; ++i) {
        double xx = local_x[i];
        for (int j = 0; j < ny; ++j) {
            double yy = y[j];
            T_local[(i + 1) * ny + j] = T_analytical(xx, yy, 0.0, alpha);
        }
    }
    vector<double> T_new = T_local;

    // Numerical Solution
    double alpha_x = alpha * dt / dx / dx;
    double alpha_y = alpha * dt / dy / dy;
    double t = 0.0;

    MPI_Barrier(comm);
    chrono::high_resolution_clock::time_point t0;
    if (rank == 0) {
        t0 = chrono::high_resolution_clock::now();
    }

    while (t < finish_time) {
        int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        MPI_Request reqs[4];
        MPI_Isend(&T_local[1 * ny], ny, MPI_DOUBLE, left, 11, comm, &reqs[0]);
        MPI_Irecv(&T_local[(local_nx + 1) * ny], ny, MPI_DOUBLE, right, 11, comm, &reqs[1]);
        MPI_Isend(&T_local[local_nx * ny], ny, MPI_DOUBLE, right, 22, comm, &reqs[2]);
        MPI_Irecv(&T_local[0], ny, MPI_DOUBLE, left, 22, comm, &reqs[3]);

        // interior pts
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 2; i < local_nx; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                T_new[i * ny + j] = T_local[i * ny + j] +
                                    alpha_x * (T_local[(i + 1) * ny + j] - 2.0 * T_local[i * ny + j] + T_local[(i - 1) * ny + j]) +
                                    alpha_y * (T_local[i * ny + j + 1] - 2.0 * T_local[i * ny + j] + T_local[i * ny + j - 1]);
            }
        }

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // boundary pts
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < ny - 1; ++j) {
            // left boundary
            T_new[1 * ny + j] = T_local[1 * ny + j] +
                                alpha_x * (T_local[2 * ny + j] - 2.0 * T_local[1 * ny + j] + T_local[0 * ny + j]) +
                                alpha_y * (T_local[1 * ny + j + 1] - 2.0 * T_local[1 * ny + j] + T_local[1 * ny + j - 1]);
            // right boundary
            T_new[local_nx * ny + j] = T_local[local_nx * ny + j] +
                                alpha_x * (T_local[(local_nx + 1) * ny + j] - 2.0 * T_local[local_nx * ny + j] + T_local[(local_nx - 1) * ny + j]) +
                                alpha_y * (T_local[local_nx * ny + j + 1] - 2.0 * T_local[local_nx * ny + j] + T_local[local_nx * ny + j - 1]);
        }

        // BCs
        for (int i = 1; i < local_nx + 1; ++i) {
            T_new[i * ny + 0] = T_bot;
            T_new[i * ny + (ny - 1)] = T_top;
        }

        if (rank == 0) {
            for (int j = 0; j < ny; ++j) {
                T_new[1 * ny + j] = T_left;
            }
        }

        if (rank == size - 1) {
            for (int j = 0; j < ny; ++j) {
                T_new[local_nx * ny + j] = T_right;
            }
        }

        T_local.swap(T_new);
        t += dt;
    }

    // Gather T_global
    int mycount = local_nx * ny;
    vector<int> recvcounts;
    vector<int> displs;
    vector<double> T_global;
    if (rank == 0) {
        recvcounts.resize(size);
        T_global.resize(nx * ny);
    }
    MPI_Gather(&mycount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);
    if (rank == 0) {
        displs.resize(size, 0);
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    MPI_Gatherv(&T_local[1 * ny], mycount, MPI_DOUBLE, T_global.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,0, comm);

    if (rank == 0) {
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = t1 - t0;
        cout << fixed << setprecision(2) << "[MPI version: " << size << " proc.] Got solution in " << duration.count() << " s." << endl;

        cout << "t=" << t << endl;

        // Analytical Solution
        vector<double> T_true(nx * ny);
        for (int i = 0; i < nx; ++i) {
            double xx = x_full[i];
            for (int j = 0; j < ny; ++j) {
                double yy = y[j];
                T_true[i * ny + j] = T_analytical(xx, yy, t, alpha);
            }
        }

        // Errors
        double err_l2 = 0.0;
        for (int k = 0; k < nx * ny; ++k) {
            double err = T_global[k] - T_true[k];
            err_l2 += err * err;
        }
        err_l2 = sqrt(err_l2 * dx * dy);

        double norm_true = 0.0;
        for (int k = 0; k < nx * ny; ++k) {
            norm_true += T_true[k] * T_true[k];
        }
        norm_true = sqrt(norm_true * dx * dy);

        double rel_l2 = err_l2 / norm_true;

        cout << "err_l2=" << setprecision(6) << err_l2 << "; rel_l2=" << rel_l2 << endl;
    }

    MPI_Finalize();
    return 0;
}