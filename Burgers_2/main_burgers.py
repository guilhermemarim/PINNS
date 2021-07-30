import scipy.io
from pyDOE import lhs
import numpy as np
from pinn import PhysicsInformedNN
from visualization import Visualization

# Data size on the solution u
N_u = 50
# Collocation points size, where we’ll check for f = 0
N_f = 10000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width, 1-sized output [u]
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# Configuration of the L-BFGS optimizer
lbfgs_config = {
    'maxiter': 50000,
    'maxfun': 50000,
    'm': 50,
    'maxls': 50,
    'factr': 1e5
}


# Getting the data
path = "burgers_shock.mat"

# Reading external data [t is 100x1, usol is 256x100 (solution), x is 256x1]
data = scipy.io.loadmat(path)

# Flatten makes [[]] into [], [:,None] makes it a column vector
t = data['t'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1

# Keeping the 2D data for the solution data (real() is maybe to make it float by default, in case of zeroes)
Exact_u = np.real(data['usol']).T # T x N

# Meshing x and t in 2D (256,100)
X, T = np.meshgrid(x, t)

# Preparing the inputs x and t (meshed as X, T) for predictions in one single array, as X_star
#                   (25600,1)             (25600,1)               = (25600,2)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Preparing the testing u_star  25600,1
u_star = Exact_u.flatten()[:, None]

# Domain bounds (lowerbounds upperbounds) [x, t], which are here ([-1.0, 0.0] and [1.0, 1.0])
lb = X_star.min(axis=0)
ub = X_star.max(axis=0)

# Getting the initial conditions (t=0) shape = (256,2)
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))

# shape = (256,1)
uu1 = Exact_u[0:1,:].T

# Getting the lowest boundary conditions (x=-1) shape = (100,2)
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
# shape = (100,1)
uu2 = Exact_u[:,0:1]

# Getting the highest boundary conditions (x=1) shape = (100,2)
xx3 = np.hstack((X[:,-1:], T[:,-1:]))
# shape = (100,1)
uu3 = Exact_u[:,-1:]

# Stacking them in multidimensional tensors for training (X_u_train is for now the continuous boundaries)
# shape = (456,2)
X_u_train = np.vstack([xx1, xx2, xx3])
# shape = (456,1)
u_train = np.vstack([uu1, uu2, uu3])

# Generating a uniform random sample from ints between 0, and the size of x_u_train, of size N_u (initial data size) and without replacement (unique)
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
# Getting the corresponding X_u_train (which is now scarce boundary/initial coordinates)
# shape = (50,2)
X_u_train = X_u_train[idx,:]
# Getting the corresponding u_train
# shape = (50,1)
u_train = u_train [idx,:]

# Generating the x and t collocation points for f, with each having a N_f size
# We pointwise add and multiply to spread the LHS over the 2D domain
X_f_train = lb + (ub - lb) * lhs(2, N_f)


pinn = PhysicsInformedNN(layers, X_u_train, u_train, X_f_train, X_star, u_star, ub, lb, lbfgs_config, nu=0.01/np.pi, epochs=3000)
pinn.fit()

# Getting the model predictions, from the same (x,t) that the predictions were previously gotten from
u_pred, f_pred = pinn.predict(X_star)

view = Visualization()
view.plot_inf_cont_results(X_star, u_pred, X_u_train, u_train, Exact_u, X, T, x, t)