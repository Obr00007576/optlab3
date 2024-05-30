import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit


def lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef):
    """
    Estimates f(x) - f* via duality gap for 
        f(x) := 0.5 * ||Ax - b||_2^2 + regcoef * ||x||_1.
    """
    primal_obj = 0.5 * Ax_b@Ax_b + regcoef * np.sum(np.abs(x))
    mu = min(1, regcoef / np.linalg.norm(ATAx_b, ord=np.inf)) * Ax_b
    return primal_obj + 0.5 * mu@mu + mu@b


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class BarrierMethodLassoOracle(BaseSmoothOracle):
    def __init__(self, A, b, reg_coef, t):
        self.A = A
        self.b = b
        self.reg_coef = reg_coef
        self.t = t

    def func(self, xu):
        x, u = self._split_xu(xu)
        residual = self.A @ x - self.b
        f_value = 0.5 * np.sum(residual**2) + self.reg_coef * np.sum(u)
        barrier_value = -np.sum(np.log(u + x)) - np.sum(np.log(u - x))
        return self.t * f_value + barrier_value

    def grad(self, xu):
        x, u = self._split_xu(xu)
        residual = self.A @ x - self.b
        grad_f_x = self.A.T @ residual
        grad_f_u = np.full(u.shape, self.reg_coef)

        grad_barrier_x = 1 / (u + x) - 1 / (u - x)
        grad_barrier_u = 1 / (u + x) + 1 / (u - x)

        grad_x = self.t * grad_f_x - grad_barrier_x
        grad_u = self.t * grad_f_u - grad_barrier_u

        return np.concatenate([grad_x, grad_u])

    def hess(self, xu):
        x, u = self._split_xu(xu)
        hessian_f_xx = self.A.T @ self.A
        hessian_barrier_xx = scipy.sparse.diags(
            1 / (u + x)**2 + 1 / (u - x)**2)
        hessian_barrier_uu = scipy.sparse.diags(
            1 / (u + x)**2 + 1 / (u - x)**2)
        hessian_xx = self.t * hessian_f_xx + hessian_barrier_xx
        hessian_xu = np.zeros((x.size, u.size))
        hessian_ux = np.zeros((u.size, x.size))
        hessian_uu = hessian_barrier_uu
        return scipy.sparse.bmat([[hessian_xx, hessian_xu], [hessian_ux, hessian_uu]])

    def _split_xu(self, xu):
        n = self.A.shape[1]
        return xu[:n], xu[n:]
