import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Heat Equation (2D): 
# 
# T_t = alpha * (T_xx + T_yy)
# 
# T:=T(x,y,t) - temperature function;
# alpha - diffusivity constant; 
# x,y - spatial variables; 
# t - time variable.

### CONFIG ###
alpha = 1
finish_time = 0.1
xlen, ylen = 1, 1
N = 100
nx, ny = xlen*N, ylen*N
dx = xlen/(nx-1)
dy = ylen/(ny-1)
dt = dx*dx*dy*dy/alpha/(dx*dx+dy*dy)/2

### Boundary Conditions ###
T_bot, T_top, T_left, T_right = 0, 0, 0, 0

### Initial Condition ###
base = nx // size
start = rank * base
stop  = start + (base + (nx % size if rank == size-1 else 0))
local_nx = stop - start

x_full = np.linspace(-xlen/2, -xlen/2 + (nx-1)*dx, nx)
local_x = x_full[start:stop]
y = np.linspace(-ylen/2, ylen/2, ny)
local_X, Y = np.meshgrid(local_x, y, indexing='ij')

def T_analytical(X, Y, t, alpha):
    return np.exp(-2*(np.pi**2)*alpha*t) * np.sin(np.pi*(X + 0.5)) * np.sin(np.pi*(Y + 0.5))

T_local = np.zeros((local_nx + 2, ny))
T_local[1:-1, :] = T_analytical(local_X, Y, 0.0, alpha)

### Numerical Solution ###
alpha_x = alpha * dt/dx/dx
alpha_y = alpha * dt/dy/dy
t = 0

comm.Barrier()
if rank == 0:
    t0 = perf_counter()

while t < finish_time:
    left  = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL
    comm.Sendrecv(T_local[1,:],  dest=left,  sendtag=11, recvbuf=T_local[-1,:], source=right, recvtag=11)
    comm.Sendrecv(T_local[-2,:], dest=right, sendtag=22, recvbuf=T_local[0,:],  source=left,  recvtag=22)

    T_new = T_local.copy()

    for i in range(1,local_nx+1):
        for j in range(1,ny-1):
            T_new[i,j] = T_local[i,j] + alpha_x * (T_local[i+1,j] - 2*T_local[i,j] + T_local[i-1,j]) + alpha_y * (T_local[i,j+1] - 2*T_local[i,j] + T_local[i,j-1])

    for i in range(local_nx+2):
        T_new[i, 0] = T_bot
        T_new[i, ny-1] = T_top

    if rank == 0:
        for j in range(ny):
            T_new[1, j] = T_left

    if rank == size - 1:
        for j in range(ny):
            T_new[local_nx, j] = T_right

    T_local = T_new
    t += dt

T_global = comm.gather(T_local[1:-1, :], root=0)

if rank == 0:
    solution_time = perf_counter()-t0
    print(f"[MPI version: {size} proc.] Got solution in {solution_time:.2f} s.")

    T_final = np.concatenate(T_global, axis=0)

    ### Analytical Solution ###
    x_full = np.linspace(-xlen/2, xlen/2, nx)
    X_full, Y_full = np.meshgrid(x_full, y, indexing='ij')
    print(f"{t=}")
    T_true = T_analytical(X_full, Y_full, t, alpha)

    # Errors:
    err = T_final - T_true
    err_l2  = np.sqrt(np.sum(err**2) * dx * dy)
    rel_l2  = err_l2 / np.sqrt(np.sum(T_true**2) * dx * dy)
    print(f"{err_l2=:.6}; {rel_l2=:.6}")


    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot initial condition
    T_init_full = T_analytical(X_full, Y_full, 0.0, alpha)
    im1 = ax1.pcolormesh(X_full, Y_full, T_init_full, vmax=1, cmap="hot")
    fig.colorbar(im1, ax=ax1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title(f"T(x,0):")

    # Plot final numerical solution
    im2 = ax2.pcolormesh(X_full, Y_full, T_final, vmax=1, cmap="hot")
    fig.colorbar(im2, ax=ax2)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f"T(x,{t:.3}):")

    plt.suptitle(f"[MPI: time={solution_time:.2f}s.]")
    plt.tight_layout()
    plt.savefig('heat_solution_mpi.png')
    plt.show()