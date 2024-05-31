from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy
from time import time
from scipy.optimize import line_search
import datetime

import scipy.sparse
import scipy.sparse.linalg
from oracles import lasso_duality_gap, BarrierMethodLassoOracle


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        if self._method == 'Wolfe':
            alpha = self._wolfe_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Armijo':
            alpha = self._armijo_line_search(oracle, x_k, d_k, previous_alpha)
        elif self._method == 'Constant':
            alpha = self.c
        else:
            raise ValueError('Unknown method {}'.format(self._method))
        return alpha

    def _armijo_condition(self, alpha, phi_0, grad_phi_0, phi_alpha):
        if phi_alpha <= phi_0 + self.c1 * alpha * grad_phi_0:
            return True
        return False

    def _armijo_line_search(self, oracle, x_k, d_k, previous_alpha=None):
        alpha = self.alpha_0 if previous_alpha is None else previous_alpha
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        grad_phi_0 = oracle.grad_directional(x_k, d_k, 0)
        while True:
            phi_alpha = oracle.func_directional(x_k, d_k, alpha)
            if self._armijo_condition(alpha, phi_0, grad_phi_0, phi_alpha):
                return alpha
            alpha /= 2.0


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def barrier_method_lasso(A, b, reg_coef, x_0, u_0, tolerance=1e-5,
                         tolerance_inner=1e-8, max_iter=100,
                         max_iter_inner=20, t_0=1, gamma=10,
                         c1=1e-4, lasso_duality_gap=lasso_duality_gap,
                         trace=False, display=False):
    """
    Log-barrier method for solving the problem:
        minimize    f(x, u) := 1/2 * ||Ax - b||_2^2 + reg_coef * \sum_i u_i
        subject to  -u_i <= x_i <= u_i.

    The method constructs the following barrier-approximation of the problem:
        phi_t(x, u) := t * f(x, u) - sum_i( log(u_i + x_i) + log(u_i - x_i) )
    and minimize it as unconstrained problem by Newton's method.

    In the outer loop `t` is increased and we have a sequence of approximations
        { phi_t(x, u) } and solutions { (x_t, u_t)^{*} } which converges in `t`
    to the solution of the original problem.

    Parameters
    ----------
    A : np.array
        Feature matrix for the regression problem.
    b : np.array
        Given vector of responses.
    reg_coef : float
        Regularization coefficient.
    x_0 : np.array
        Starting value for x in optimization algorithm.
    u_0 : np.array
        Starting value for u in optimization algorithm.
    tolerance : float
        Epsilon value for the outer loop stopping criterion:
        Stop the outer loop (which iterates over `k`) when
            `duality_gap(x_k) <= tolerance`
    tolerance_inner : float
        Epsilon value for the inner loop stopping criterion.
        Stop the inner loop (which iterates over `l`) when
            `|| \nabla phi_t(x_k^l) ||_2^2 <= tolerance_inner * \| \nabla \phi_t(x_k) \|_2^2 `
    max_iter : int
        Maximum number of iterations for interior point method.
    max_iter_inner : int
        Maximum number of iterations for inner Newton's method.
    t_0 : float
        Starting value for `t`.
    gamma : float
        Multiplier for changing `t` during the iterations:
        t_{k + 1} = gamma * t_k.
    c1 : float
        Armijo's constant for line search in Newton's method.
    lasso_duality_gap : callable object or None.
        If calable the signature is lasso_duality_gap(x, Ax_b, ATAx_b, b, regcoef)
        Returns duality gap value for esimating the progress of method.
    trace : bool
        If True, the progress information is appended into history dictionary 
        during training. Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    (x_star, u_star) : tuple of np.array
        The point found by the optimization procedure.
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every **outer** iteration of the algorithm
            - history['duality_gap'] : list of duality gaps
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    x, u = x_0, u_0
    t = t_0
    history = {'time': [], 'func': [],
               'duality_gap': [], 'x': []} if trace else None
    start_time = time()
    lst = LineSearchTool(method='Armijo', c1=c1)
    for k in range(max_iter):
        if display:
            print(f"Iteration {k}")

        oracle = BarrierMethodLassoOracle(A, b, reg_coef, t)
        grad_xu0 = None
        for l in range(max_iter_inner):
            xu = np.concatenate((x, u))
            grad_xu = oracle.grad(xu)
            grad_xu0 = grad_xu if grad_xu0 is None else grad_xu0
            hessian_xu = oracle.hess(xu)
            delta_xu = sp.linalg.spsolve(hessian_xu, -grad_xu)
            alpha = lst.line_search(oracle, xu, delta_xu)
            xu += alpha * delta_xu
            x, u = xu[:x.size], xu[x.size:]
            if np.linalg.norm(grad_xu)**2 <= tolerance_inner * np.linalg.norm(grad_xu0)**2:
                break

        gap = lasso_duality_gap(x, A @ x - b, A.T @ (A @ x - b), b, reg_coef)
        if trace:
            current_time = time() - start_time
            history['time'].append(current_time)
            history['func'].append(
                0.5 * np.sum((A @ x - b)**2) + reg_coef * np.sum(u))
            history['duality_gap'].append(gap)
            if x.size <= 2:
                history['x'].append(x.copy())
            if gap <= tolerance:
                return (x, u), "success", history

        t *= gamma
        if gap <= tolerance:
            return (x, u), "success", history
    return (x, u), "iterations_exceeded", history if trace else None
