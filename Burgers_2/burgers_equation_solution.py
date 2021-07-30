import numpy as np
import matplotlib.pyplot as plt

""" Creating the domains """
nx = 512
nt = 1000
x_grid = np.linspace(-1,1,nx)
dx = x_grid[1] - x_grid[0]
print(dx)

t_grid = np.linspace(0,1,nt)
dt = t_grid[1] - t_grid[0]
print(dt)

print(x_grid.shape)
print(t_grid.shape)

""" Initial condition """
u0 = -np.sin(np.pi*x_grid)

""" Boundary conditions u(t,-1) = u(t,1) = 0 """
ua, ub = 0, 0

""" Finite difference to solve the equation """
u_data = np.zeros((nt, nx))
u_x = np.zeros(nx)
u_xx = np.zeros(nx)

un = u0.copy()
for j in range(nt-1):

    u = un.copy()
    for i in range(1, nx-1):
        u_x[i] = (u[i+1] - u[i-1])/(2*dx)
        u_xx[i] = (u[i-1] - 2*u[i] + u[i+1]) / (dx**2)

    u_t = -u*u_x + (0.01/np.pi)*u_xx

    un = u_t*dt + u

    u_data[j, :] = un


plt.figure()
plt.plot(x_grid, u_data[250])
plt.plot(x_grid, u_data[500])
plt.plot(x_grid, u_data[750])
plt.ylabel('u')
plt.xlabel('x')

X, T = np.meshgrid(x_grid, t_grid)
fig, ax = plt.subplots(figsize=(12, 4))
surf = ax.contourf(T, X, u_data, cmap=plt.get_cmap("seismic"))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel('Space (x)')
plt.xlabel('Time (t)')
plt.tight_layout()
plt.show()


np.save('Burgers_solution.npy', u_data)
np.save('Burgers_space.npy', x_grid)
np.save('Burgers_time.npy', t_grid)




