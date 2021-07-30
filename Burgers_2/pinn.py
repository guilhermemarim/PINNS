import numpy as np
import tensorflow as tf
import scipy.optimize


class PhysicsInformedNN():

    def __init__(self, layers, X_u, u, X_f, X_star, u_star, ub, lb, lbfgs_config, nu, epochs):

        self.layers = layers
        self.X_u = tf.convert_to_tensor(X_u, dtype='float32')
        self.u = tf.convert_to_tensor(u, dtype='float32')

        self.ub = tf.convert_to_tensor(ub, dtype='float32')
        self.lb = tf.convert_to_tensor(lb, dtype='float32')

        self.nu = tf.convert_to_tensor(nu, dtype='float32')
        self.epochs = epochs

        self.X_star = X_star
        self.u_star = u_star

        self.X_f = tf.convert_to_tensor(X_f, dtype='float32')
        # Separating the collocation coordinates
        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype='float32')
        self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype='float32')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.99, epsilon=1e-1)
        self.u_model = self.u_model()

        self.maxiter = lbfgs_config['maxiter']
        self.maxfun = lbfgs_config['maxfun']
        self.m = lbfgs_config['m']
        self.maxls = lbfgs_config['maxls']
        self.factr = lbfgs_config['factr']


        self.it = 0


    def u_model(self):
        u_model = tf.keras.Sequential()
        u_model.add(tf.keras.layers.InputLayer(input_shape=(self.layers[0],)))
        u_model.add(tf.keras.layers.Lambda(lambda X: 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0))
        for width in self.layers[1:]:
            u_model.add(tf.keras.layers.Dense(width, activation=tf.nn.tanh,
                                                   kernel_initializer='glorot_normal'))
        return u_model

    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)

            # Getting the prediction
            u = self.u_model(X_f)
            # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
            du_dx = tape.gradient(u, self.x_f)

        # Getting the other derivatives
        du_dt = tape.gradient(u, self.t_f)
        d2u_dx2 = tape.gradient(du_dx, self.x_f)

        # Letting the tape go
        del tape

        # Buidling the PINNs
        return du_dt + u * du_dx - self.nu * d2u_dx2

    def summary(self):
        return self.u_model.summary()

    def loss_and_grads(self):
        with tf.GradientTape() as tape:
            f_pred = self.f_model()
            u_pred = self.u_model(self.X_u)
            loss = tf.reduce_mean(tf.square(self.u - u_pred)) + tf.reduce_mean(tf.square(f_pred))
        grads = tape.gradient(loss, self.u_model.trainable_variables)
        return loss, grads

    def fit(self):
        print('-------------------------------------')
        print('Starting training with Adam optimizer')
        print('-------------------------------------')


        for epoch in range(self.epochs):
            # Optimization step
            loss, grads = self.loss_and_grads()
            self.optimizer.apply_gradients(zip(grads, self.u_model.trainable_variables))
            error = self.error()
            if epoch % 10 == 0:
                print(f'Epoch {epoch} --- Loss {loss} --- Error {error}')

        print('-------------------------------------')
        print('Starting training with L-BFGS optimizer')
        print('-------------------------------------')
        self.fit_lbfgs()
        self.it = 0


    def set_weights(self, flat_weights):
        """
        Set weights to the model.

        Args:
            flat_weights: flatten weights.
        """

        # get model weights
        shapes = [ w.shape for w in self.u_model.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.u_model.set_weights(weights)

    def evaluate(self, weights):
        """
        Evaluate loss and gradients for weights as ndarray.

        Args:
            weights: flatten weights.

        Returns:
            loss and gradients for weights as ndarray.
        """

        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads = self.loss_and_grads()
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')
        return loss, grads

    def callback(self, weights):
        """
        Callback that prints the progress to stdout.

        Args:
            weights: flatten weights.
        """
        loss, _ = self.loss_and_grads()
        if self.it % 10 == 0:
            print(f'Iteration {self.it} --- Loss {loss} --- Error {self.error()}')
        self.it = self.it + 1

    def fit_lbfgs(self):
        """
        Train the model using L-BFGS-B algorithm.
        """

        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [w.flatten() for w in self.u_model.get_weights()])

        # optimize the weight vector
        print(f'Optimizer: L-BFGS-B (maxiter={self.maxiter})')
        scipy.optimize.fmin_l_bfgs_b(func=self.evaluate, x0=initial_weights, factr=self.factr,
                                     maxfun=self.maxfun, maxiter=self.maxiter,
                                     callback=self.callback, maxls=self.maxls)

    def predict(self, X_star):
        u_star = self.u_model(X_star)
        f_star = self.f_model()
        return u_star, f_star

    def error(self):
        u_pred, _ = self.predict(self.X_star)
        return np.linalg.norm(self.u_star - u_pred, 2) / np.linalg.norm(self.u_star, 2)








