import tensorflow as tf
import cvxpy as cp
import osqp
import scipy.sparse as sparse
import numpy as np

class SolverWrapper:
    def __init__(self, solver_type='OSQP'):
        self.solver_type = solver_type

    def solve_qp(self, P, q, A, l, u):
        if self.solver_type == 'OSQP':
            return self._solve_osqp(P, q, A, l, u)
        elif self.solver_type == 'CVXPY':
            return self._solve_cvxpy(P, q, A, l, u)
        elif self.solver_type == 'SCS':
            return self._solve_scs(P, q, A, l, u)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")

    def _solve_osqp(self, P, q, A, l, u):
        # Convert to sparse matrices
        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace and change alpha parameter
        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

        # Solve problem
        res = prob.solve()

        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        return res.x

    def _solve_cvxpy(self, P, q, A, l, u):
        n = P.shape[0]
        x = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
        constraints = [A @ x <= u, A @ x >= l]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        if prob.status != cp.OPTIMAL:
            raise ValueError('CVXPY did not solve the problem!')
        return x.value

    def _solve_scs(self, P, q, A, l, u):
        n = P.shape[0]
        x = cp.Variable(n)
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
        constraints = [A @ x <= u, A @ x >= l]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL:
            raise ValueError('SCS did not solve the problem!')
        return x.value

class QPLayer(tf.keras.layers.Layer):
    def __init__(self, n_assets, solver_type='OSQP'):
        super(QPLayer, self).__init__()
        self.n_assets = n_assets
        self.solver = SolverWrapper(solver_type)

    def call(self, inputs):
        mu, Sigma = inputs
        P = Sigma
        q = -mu

        # Constraints
        A = np.vstack([np.ones((1, self.n_assets)), np.eye(self.n_assets)])
        l = np.array([1.0] + [0.0] * self.n_assets)
        u = np.array([1.0] + [1.0] * self.n_assets)

        def solve_qp(P, q, A, l, u):
            return self.solver.solve_qp(P.numpy(), q.numpy(), A, l, u)

        optimized_w = tf.py_function(
            func=solve_qp,
            inp=[P, q, A, l, u],
            Tout=tf.float32
        )

        return optimized_w