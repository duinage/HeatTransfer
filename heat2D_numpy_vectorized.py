import numpy as np
import matplotlib.pyplot as plt

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

### Initial Condition ###
x = np.linspace(-xlen/2, xlen/2, nx)
y = np.linspace(-ylen/2, ylen/2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

def T_analytical(X, Y, t, alpha):
    return np.exp(-2*(np.pi**2)*alpha*t) * np.sin(np.pi*(X + 0.5)) * np.sin(np.pi*(Y + 0.5))

T_init = T_analytical(X, Y, 0.0, alpha)

### Boundary Conditions ###
T_bot, T_top, T_left, T_right = 0, 0, 0, 0

T = T_init.copy()

### Numerical Solution ###
alpha_x = alpha * dt/dx/dx
alpha_y = alpha * dt/dy/dy
t = 0

from time import perf_counter
t0 = perf_counter()
while t < finish_time:
    T_new = T.copy()

    T_xx = T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]
    T_yy = T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha_x * T_xx + alpha_y * T_yy

    T_new[:, 0] = T_bot
    T_new[:, -1] = T_top
    T_new[0, :] = T_left
    T_new[-1, :] = T_right

    T = T_new
    t += dt

solution_time = perf_counter()-t0
print(f"[numpy version] Got solution in {solution_time:.2} s.")

### Analytical Solution ###
T_true = T_analytical(X, Y, t, alpha)

# Errors:
err = T - T_true
err_l2  = np.sqrt(np.sum(err**2) * dx * dy)
rel_l2  = err_l2 / np.sqrt(np.sum(T_true**2) * dx * dy)
print(f"{err_l2=:.6}; {rel_l2=:.6}")


# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot initial condition
im1 = ax1.pcolormesh(X, Y, T_init, vmax=1, cmap="hot")
fig.colorbar(im1, ax=ax1)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title(f"T(x,0):")

# Plot final numerical solution
im2 = ax2.pcolormesh(X, Y, T, vmax=1, cmap="hot")
fig.colorbar(im2, ax=ax2)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title(f"T(x,{t:.3}):")

plt.suptitle(f"[Numpy: time={solution_time:.2f}s.]")
plt.tight_layout()
plt.savefig('heat_solution_numpy.png')
plt.show()