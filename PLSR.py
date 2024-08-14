#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A number of PLS regression methods, including PLS-R, Sparse PLS-R and most
importantly the regularised PLS-R based on the Krylov formulation.

Created on Thu Jun 24 09:01:04 2021

Copyright (c) 2016-2024, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import abc
import warnings

import numpy as np

import sklearn.base
import sklearn.utils
import sklearn.linear_model
import sklearn.utils.validation
import sklearn.utils.extmath

MACHINE_EPSILON = np.finfo(np.float64).eps


def sigma_max(A):
    """Compute largest singular value of a matrix."""
    sigma = np.linalg.norm(A, ord=2)

    return sigma


class _BasePLSR(sklearn.linear_model._base.LinearModel, metaclass=abc.ABCMeta):
    """Base class for the PLS-R classes."""

    def __init__(self,
                 K=2,
                 *,
                 penalty=None,
                 gamma=0.0,
                 alpha=None,
                 max_iter=None,
                 tol=1e-4,
                 solver="auto",
                 step_size=0.0001,
                 verbose=0,
                 random_state=None,
                 ):

        self.K = max(1, int(K))
        self.penalty = None if penalty is None else str(penalty).lower()
        self.gamma = max(0.0, float(gamma))
        self.alpha = min(max(0.0, float(alpha)), 1.0) \
            if alpha is not None else None
        if (self.penalty is not None) and (self.gamma == 0.0):
            warnings.warn("There is a penalty, but the regularisation "
                          "parameter is zero.")
        if (self.penalty in ["l1l2", "l2l1", "elasticnet"]) \
                and (self.alpha is None):
            raise ValueError(f"Penalty is '{self.penalty}', but the "
                             f"mixing parameter alpha is {self.alpha}.")
        # self.fit_intercept = fit_intercept
        # self.normalize = normalize
        # self.copy_X = copy_X
        if max_iter is not None:
            self.max_iter = max(0, int(max_iter))
        else:
            raise ValueError("Maximum number of iterations (max_iter) must be "
                             "provided.")
        self.tol = max(0.0, float(tol))
        self.solver = str(solver).lower()
        if step_size is not None:
            self.step_size = max(0.0, float(step_size))
        else:
            self.step_size = None
        self.verbose = int(verbose)
        self.random_state = sklearn.utils.validation.check_random_state(
            random_state)

    def _soft_thresholding(self, v, lambda_):
        """The proximal operator of the L1 norm."""
        return np.multiply(np.sign(v), np.maximum(0.0, np.abs(v) - lambda_))

    def _decision_function(self, X):
        sklearn.utils.validation.check_is_fitted(self)

        X = sklearn.utils.check_array(X, accept_sparse=["csr", "csc", "coo"])
        return sklearn.utils.extmath.safe_sparse_dot(X,
                                                     self.coef_,
                                                     dense_output=True)
        # + self.intercept_


class OLS(sklearn.base.RegressorMixin, sklearn.linear_model._base.LinearModel):
    """Ordinary least squares."""

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit PLS regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            multi_output=False,
            y_numeric=True,
        )
        if y.ndim == 1:
            y = y[:, np.newaxis]

        beta_ = np.dot(np.linalg.pinv(X), y)

        self.beta_ = beta_
        self.coef_ = self.beta_
        self.intercept_ = 0.0

        return self

    def _decision_function(self, X):
        sklearn.utils.validation.check_is_fitted(self)

        X = sklearn.utils.check_array(X, accept_sparse=["csr", "csc", "coo"])
        return sklearn.utils.extmath.safe_sparse_dot(X,
                                                     self.coef_,
                                                     dense_output=True)


class Lasso(sklearn.base.RegressorMixin, _BasePLSR):
    """Sparse Ordinary Linear Regression (Lasso).

    Parameters
    ----------
    l1 : float
        The regularisation parameter for the L1 norm.

    max_iter : None or int
        The maximum number of iterations to perform. Defaults (with None) to
        however many iterations are required to reach the tolerance level set.

    tol : float, optional
        The tolerance level to reach (the error in the approximation).

    step_size : float, optional
        The size of the step taken in the ISTA iterations.

    verbose : int, optional
        The verbosity level. Defaults to zero, which means to not print
        any auxiliary information about the fit or predict.

    random_state : None, int or instance of np.random.RandomState
        If seed is None, uses the RandomState singleton used by np.random. If
        seed is an int, uses a new RandomState instance seeded with seed. If
        seed is already a RandomState instance, use it. Otherwise raises
        ValueError.
    """

    def __init__(self,
                 l1=None,
                 *,
                 max_iter=None,
                 tol=1e-4,
                 solver="auto",
                 step_size=0.0001,
                 verbose=0,
                 random_state=None,
                 ):

        super().__init__(
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            step_size=step_size,
            verbose=verbose,
            random_state=random_state,
            )

        self.l1 = max(0.0, float(l1))

    def fit(self, X, y):
        """Fit the Lasso regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            multi_output=False,
            y_numeric=True,
        )
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.solver != "auto":
            raise ValueError("The only available solver is 'auto'.")

        num_samples, num_variables = X.shape

        beta = 0.01 * self.random_state.randn(num_variables, 1)

        it = 0
        while self.max_iter is None or it < self.max_iter:

            grad = np.dot(X.T, np.dot(X, beta) - y) / num_samples

            beta_old = beta
            beta = self._soft_thresholding(beta - self.step_size * grad,
                                           self.l1)

            err_abs = np.linalg.norm(beta_old - beta)
            norm_beta_old = np.linalg.norm(beta_old)
            if norm_beta_old > MACHINE_EPSILON:
                err_rel = err_abs / norm_beta_old
            else:
                err_rel = err_abs

            if self.verbose >= 1:
                print(f"it: {it}, err rel: {err_rel}, err abs: {err_abs}")
            it += 1
            if err_rel < self.tol:
                break

        self.beta_ = beta
        self.coef_ = self.beta_
        self.intercept_ = 0.0

        return self


class PLSR(sklearn.base.RegressorMixin, _BasePLSR):
    """Partial least squares regression.

    Parameters
    ----------
    K : int, optional
        The number of components. Default is 2.

    penalty : None or str, optional
        Either None, for no penalty, or "l1" for the L1 norm penalty, "l2" for
        an L2 (ridge) penalty, or "l1l2", "l2l1", or "elasticnet" for an
        elastic net penalty (L1 plus L2 penalties).

    gamma : float, optional
        The regularisation parameter for L1 and L2 regularistion. The overall
        regularisation strength for elastic net regularisation. Default is 0.0,
        i.e., no penalty.

    alpha: float, optional
        The elastic net mixing parameter, a value in [0, 1]. When zero, the
        penalty is an L2 penalty; when one, the penalty is an L1 penalty; and
        when between zero and one, the penalty is a weighted sum of an L1 and
        an L2 penalty. Default is None, i.e., no penalty.

    max_iter : None or int
        The maximum number of iterations to perform. Defaults (with None) to
        however many iterations are required to reach the tolerance level set.

    tol : float, optional
        The tolerance level to reach (the error in the approximation).

    solver : str, optional
        The solver to use, either "auto", "nipals", or "krylov". The "auto"
        would select the NIPALS algorithm when there are no penalties and the
        Krylov formulation and projected/proximal gradient descent when L1 or
        L2 penalties are present. The option "nipals" would directly choose
        the NIPALS algorithm, but will throw a ValueError if penalty is not
        None. The option "krylov" would directly choose the NIPALS algorithm,
        but will only work if penalty is a valid option. Default is "auto".

    step_size : float, optional
        The size of the step taken in the ISTA iterations. Only used with the
        "krylov" solver.

    reconstruct_components : bool, optional
        Whether to reconstruct the PLS-R components when using the "krylov"
        solver. Only used with the "krylov" solver.

    reconstruct_num_components : None or int, optional
        The number of components to reconstruct. Only used when
        reconstruct_components is True.

    reconstruct_retries : int, optional
        The number of times to attempt to reconstruct the components. The
        solution with the lowest loss will be selected. Only used when
        reconstruct_components is True.

    reconstruct_step_size : float, optional
        The size of the step taken in the projected gradient descent
        iterations for the reconstruction. Only used when
        reconstruct_components is True.

    reconstruct_max_iter : int, optional
        The maximum number of iterations in the reconstruction. Default is
        10000. Only used when reconstruct_components is True.

    reconstruct_tol : float, optional
        The tolerance level to reach (the relative norm change of the weight
        vector). Default is 5e-8.

    reconstruct_max_iter_proj : int, optional
        The maximum number of iterations in the numerically approximated
        projection. Default is 10000. Only used when reconstruct_components is
        True.

    reconstruct_lambda_w_eq_Wk : float, optional
        The regularisation parameter for the penalty on the reconstructed
        weight vector. Default is 1.0. Only used when reconstruct_components is
        True.

    reconstruct_lambda_t_eq_Tk : float, optional
        The regularisation parameter for the penalty on the reconstructed
        score vector. Default is 1.0. Only used when reconstruct_components is
        True.

    reconstruct_lambda_beta : float, optional
        The regularisation parameter for the penalty on the reconstructed
        regression coefficient vector. Default is 1.0. Only used when
        reconstruct_components is True.

    verbose : int, optional
        The verbosity level. Defaults to zero, which means to not print
        any auxiliary information about the fit or predict. Default is 0, print
        nothing.

    random_state : None, int or instance of np.random.RandomState
        If seed is None, uses the RandomState singleton used by np.random. If
        seed is an int, uses a new RandomState instance seeded with seed. If
        seed is already a RandomState instance, use it. Otherwise raises
        ValueError.
    """

    def __init__(self,
                 K=2,
                 *,
                 penalty=None,
                 gamma=0.0,
                 alpha=None,
                 # fit_intercept=True,
                 max_iter=None,
                 tol=1e-4,
                 solver="auto",
                 step_size=0.0001,
                 reconstruct_components=True,
                 reconstruct_num_components=None,
                 reconstruct_retries=3,
                 reconstruct_step_size=0.0001,
                 reconstruct_max_iter=10000,
                 reconstruct_tol=5e-8,
                 reconstruct_max_iter_proj=10000,
                 reconstruct_lambda_w_eq_Wk=1.0,
                 reconstruct_lambda_t_eq_Tk=1.0,
                 reconstruct_lambda_beta=1.0,
                 verbose=0,
                 random_state=None,
                 ):

        super().__init__(
            K=K,
            penalty=penalty,
            gamma=gamma,
            alpha=alpha,
            # fit_intercept=fit_intercept,
            # normalize=normalize,
            # copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            step_size=step_size,
            verbose=verbose,
            random_state=random_state,
            )

        self.reconstruct_components = bool(reconstruct_components)
        if reconstruct_num_components is None:
            self.reconstruct_num_components = None
        else:
            self.reconstruct_num_components = int(reconstruct_num_components)
        self.reconstruct_retries = max(1, int(reconstruct_retries))
        self.reconstruct_step_size = reconstruct_step_size
        self.reconstruct_max_iter = max(0, int(reconstruct_max_iter))
        self.reconstruct_tol = max(0.0, float(reconstruct_tol))
        self.reconstruct_max_iter_proj = max(0, int(reconstruct_max_iter_proj))
        self.reconstruct_lambda_w_eq_Wk = max(
                MACHINE_EPSILON,
                float(reconstruct_lambda_w_eq_Wk))
        self.reconstruct_lambda_t_eq_Tk = max(
                MACHINE_EPSILON,
                float(reconstruct_lambda_t_eq_Tk))
        self.reconstruct_lambda_beta = max(
                MACHINE_EPSILON,
                float(reconstruct_lambda_beta))

    def _nipals(self, X, y):
        """Fit a PLS-R model using the NIPALS algorithm.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.
        """
        num_samples, num_variables = X.shape
        num_components = min(num_samples, num_variables, self.K)

        X_ = X
        W_ = np.zeros((num_variables, num_components))
        T_ = np.zeros((num_samples, num_components))
        P_ = np.zeros((num_variables, num_components))
        C_ = np.zeros((1, num_components))
        D_ = np.zeros((1, num_components))
        for k in range(num_components):
            Xty = np.dot(X_.T, y)
            w = Xty
            it = 0
            while self.max_iter is None or it < self.max_iter:
                w_old = w
                w = np.dot(Xty, np.dot(Xty.T, w))
                w = w / np.linalg.norm(w)
                err_rel = np.linalg.norm(w_old - w) / np.linalg.norm(w_old)
                if self.verbose >= 1:
                    print(f"k: {k}, it: {it}, err: {err_rel}")
                it += 1
                if err_rel < self.tol:
                    break
            t = np.dot(X_, w)
            d = np.dot(t.T, t).item()
            p = np.dot(X_.T, t) / d
            c = np.dot(y.T, t) / d

            # Deflate
            X_ = X_ - np.dot(t, p.T)

            W_[:, [k]] = w
            T_[:, [k]] = t
            P_[:, [k]] = p
            C_[0, k] = c.item()
            D_[0, k] = d

        V_ = np.dot(W_, np.linalg.inv(np.dot(P_.T, W_)))
        beta_ = np.dot(V_, C_.T)

        return beta_, W_, T_, P_, C_, D_, V_

    def _krylov(self, X, y, K):
        """Fit the PLS-R model using the Krylov subspace method.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.

        K : ndarray of shape (n_features, n_components)
            A precomputed Krylov matrix.
        """
        num_samples, num_variables = X.shape
        num_components = min(num_samples, num_variables, self.K)

        XK = np.dot(X, K)

        # TODO: Is 1000 a good threshold for the inverse to be feasible?
        if (self.penalty is None) and (num_components <= 1000):
            # # TODO: Use solve instead!
            # KtXtXK = np.dot(XK.T, XK)
            # alpha_ = np.dot(np.linalg.pinv(KtXtXK), np.dot(XK.T, y))
            self.alpha_, *_ = np.linalg.lstsq(XK, y, rcond=None)
            if self.alpha_.ndim == 1:
                self.alpha_ = self.alpha_[:, np.newaxis]
        else:
            losses = [SqNormAffineLoss(XK, y, gamma=1.0 / (2.0 * num_samples))]

            use_admm = False
            if self.penalty == "l1":
                losses.append(L1Penalty(K, self.gamma))
                use_admm = True

            elif self.penalty == "l2":
                losses.append(L2Penalty(K, self.gamma))
                use_admm = False

            elif self.penalty in ["l1l2", "l2l1", "elasticnet"]:
                if self.alpha < 1.0:
                    losses.append(
                        L2Penalty(K, self.gamma * (1.0 - self.alpha) / 2.0))
                if self.alpha > 0.0:
                    losses.append(L1Penalty(K, self.gamma * self.alpha))
                use_admm = True

            elif self.penalty is not None:
                raise ValueError("Unknown penalty '{self.penalty}'")

            loss = CombinedLoss(losses)

            self.alpha_ = np.random.rand(num_components, 1)

            if not use_admm:
                if self.step_size is None:
                    all_has_L = True
                    for loss_ in losses:
                        # if not hasattr(loss_, "L"):
                        if not isinstance(loss_, LipschitzContinuousGradient):
                            all_has_L = False
                            break
                    if all_has_L:
                        L = loss.L()
                    else:
                        raise ValueError("A step size must be provided.")
                    step_size = 2.0 / L
                else:
                    step_size = self.step_size
                if self.verbose >= 1:
                    print(f"step_size: {step_size}")

                self.func_vals_ = []
                for it in range(self.max_iter):
                    alpha_ = self.alpha_  # Store old vector for stop criterion
                    self.alpha_ = alpha_ - self.step_size * loss.grad(alpha_)

                    f = loss.f(self.alpha_)
                    if len(self.func_vals_) > 0 and f > self.func_vals_[-1]:
                        warnings.warn("The function value increased! Try "
                                      "lowering the step size.")
                    self.func_vals_.append(f)

                    err_rel = np.linalg.norm(self.alpha_ - alpha_) \
                        / (self.step_size * np.linalg.norm(alpha_))

                    if self.verbose >= 1:
                        print(f"it: {it}, err: {err_rel}, "
                              f"f: {loss.f(self.alpha_)}")

                    if err_rel < self.tol:
                        break
            else:
                z = np.dot(K, self.alpha_)

                KtXtXK_n = np.dot(XK.T, XK) / num_samples
                KtK = np.dot(K.T, K)
                KtXty_n = np.dot(XK.T, y) / num_samples

                if self.penalty in ["l2", "l1l2", "l2l1", "elasticnet"]:
                    l2_lm = self.gamma * (1.0 - self.alpha)
                else:
                    l2_lm = 0.0

                if self.penalty in ["l1", "l1l2", "l2l1", "elasticnet"]:
                    l1_lm = self.gamma * self.alpha
                else:
                    l1_lm = 0.0

                # print(f"shape, KtXtXK_n: {KtXtXK_n.shape}")
                # print(f"shape, KtK     : {KtK.shape}")

                # print(f"shape, KtXty_n : {KtXty_n.shape}")

                def argmin_x_L(x, z, y, rho):
                    Ktl = np.dot(K.T, y)
                    Ktz = np.dot(K.T, z)
                    # print(f"shape, Ktl: {Ktl.shape}")
                    # print(f"shape, Ktz: {Ktz.shape}")
                    x = np.linalg.solve(KtXtXK_n + (l2_lm + rho) * KtK,
                                        KtXty_n - Ktl + rho * Ktz)
                    # print(f"shape, x: {x.shape}")
                    return x

                def argmin_z_L(x, z, y, rho):
                    # print(f"shapes, x: {x.shape}, z: {z.shape}, "
                    #       f"y: {y.shape}")
                    Kx = np.dot(K, x)
                    # print(f"shapes, Kx: {Kx.shape}")
                    v = Kx + y / rho
                    # print(f"shapes, v: {v.shape}")
                    th = l1_lm / rho

                    prox = (np.abs(v) > th) * (v - th * np.sign(v - th))
                    # prox = np.multiply(np.sign(v),
                    #                    np.maximum(0.0, np.abs(v) - th))
                    # assert np.linalg.norm(prox - prox_) < 5e-8

                    return prox

                self.alpha_, z, y = ADMM(self.alpha_,
                                         z,
                                         argmin_x_L,
                                         argmin_z_L,
                                         A=K,
                                         B=-1.0,
                                         c=0.0,
                                         max_iter=self.max_iter,
                                         tol=self.tol,
                                         rho=1.0,
                                         verbose=self.verbose)

        beta_ = np.dot(K, self.alpha_)

        return beta_

    def _compute_v(self, V, P, w):
        """Compute the kth w* given components 1, ..., k-1."""
        k = V.shape[1]
        assert (P.shape[1] == k)

        v = w
        for i in range(k):
            v = v - V[:, [i]] * np.dot(P[:, [i]].T, w)

        return v

    def _reconstruct_components(self,
                                beta,
                                X,
                                y,
                                pls_W,
                                pls_T,
                                pls_P,
                                pls_C,
                                K=None,
                                retries=3):
        """Reconstructs PLS-R model components.

        Reconstruct the PLS-R components for the solution found through the
        Krylov subspace formulation.
        """
        num_samples, num_variables = X.shape
        # Num components to reconstruct:
        if self.reconstruct_num_components is not None:
            num_reconstruct_components = self.reconstruct_num_components
        else:
            num_reconstruct_components = min(num_samples,
                                             num_variables,
                                             self.K)

        # Num latent components:
        # num_components = min(num_samples, num_variables, self.K)
        num_components = num_reconstruct_components

        if K is not None:
            K = K[:, :num_components]

        lambda_w_eq_Wk = self.reconstruct_lambda_w_eq_Wk
        lambda_t_eq_Tk = self.reconstruct_lambda_t_eq_Tk
        lambda_p_eq_Pk = 1.0
        lambda_c_eq_Ck = 1.0
        lambda_beta = self.reconstruct_lambda_beta

        factor_increase = 2
        # factor_decrease = 1.1
        factor_decay = 0.01

        Omega = np.zeros((num_components, 0))
        W = np.zeros((num_variables, 0))
        T = np.zeros((num_samples, 0))
        P = np.zeros((num_variables, 0))
        C = np.zeros((1, 0))  # Hard-coded num columns in y
        D = np.zeros((1, 0))  # Hard-coded num columns in y
        V = np.zeros((num_variables, 0))

        # k = 0
        for k in range(num_reconstruct_components):
            if K is None:
                Xty = np.dot(X.T, y)
            else:
                Xty = np.dot(K.T, np.dot(X.T, y))

            loss = CombinedLoss(
                [InnerProductLoss(-Xty),

                 # Similarity to PLS regression vector
                 PLSbetaloss(X,
                             y,
                             beta,
                             T,
                             P,
                             V,
                             lambda_beta,
                             K=K),

                 # Similarity (to PLS components) constraints
                 SqNormAffineLoss(K, pls_W[:, [k]],  # w ≃ W_k
                                  lambda_w_eq_Wk),
                 SqNormAffineLoss(X if K is None else np.dot(X, K),
                                  pls_T[:, [k]],  # t ≃ T_k
                                  lambda_t_eq_Tk),
                 # TODO: XtX can be large! How can we make this more efficient?
                 SqNormAffineLoss(  # p ≃ P_k
                     np.dot(X.T, X) if K is None else np.dot(X.T,
                                                             np.dot(X, K)),
                     pls_P[:, [k]] * np.linalg.norm(pls_T[:, [k]])**2,
                     lambda_p_eq_Pk),
                 SqNormAffineLoss(
                     Xty.T,  # c ≃ C_k
                     pls_C[:, [k]] * np.linalg.norm(pls_T[:, [k]])**2,
                     lambda_c_eq_Ck),
                 # c = np.dot(y.T, Xw) / np.dot(Xw.T Xw)
                 ])

            if k == 0:
                penalties = [
                    ProjSqNorm(1.0, K=K),  # Norm constraint on w.
                    ]
            else:
                penalties = [
                    # Norm constraint on w.
                    ProjSqNorm(1.0, K=K),

                    # Orthogonality constraints
                    ProjNull(W.T if K is None else np.dot(W.T, K)),  # W'w = 0
                    # T't = 0
                    ProjNull(np.dot(T.T, X) if K is None
                             else np.dot(T.T, np.dot(X, K))),
                    # W'p = 0
                    ProjNull(np.dot(np.dot(W.T, X.T), X) if K is None
                             else np.dot(W.T, np.dot(X.T, np.dot(X, K)))),
                    ]

            # Add 10 % noise to the PLS-R weight vector to get a starting point
            w = pls_W[:, [k]] \
                + 0.1 * np.random.randn(num_variables, 1) \
                / np.sqrt(num_variables)
            if K is not None:
                # TODO: Is this starting point any good?
                w = np.dot(np.linalg.pinv(K), w)

            # Make weight vector feasible by projecting it on the feasible set
            w = parallel_dykstra(w,
                                 penalties,
                                 max_iter=self.reconstruct_max_iter_proj,
                                 verbose=self.verbose)

            f_before = f_after = loss.f(w)
            self.func_vals_ = [f_before]
            step_size = self.reconstruct_step_size
            for it in range(self.reconstruct_max_iter):
                f_before = f_after
                # print(f"[{it}] Before      : " +
                #       f"{loss.losses[0].f(w):.4f} and {loss.f(w):.4f}")
                w_old = w  # Store old vector for stop criterion
                # Take a gradient step
                # w = w - step_size * loss.approx_grad(w)
                w = w - step_size * loss.grad(w)
                # print(f"[{it}] Intermediate: "
                #       f"{loss.losses[0].f(w):.4f} and {loss.f(w):.4f}")
                # Take a projection step:
                w = parallel_dykstra(
                    w,
                    penalties,
                    max_iter=self.reconstruct_max_iter_proj,
                    verbose=self.verbose)
                # print(f"NORM OF W: {np.linalg.norm(np.dot(K, w))}, ")
                # print(f"    diff: {np.linalg.norm(np.dot(K, w) - pls_W[:, k])}, ")
                # print(f"    shape w: {np.dot(K, w).shape}")
                # print(f"    shape w_pls: {pls_W[:, k].shape}")
                # print(f"    inner product: {np.dot(np.dot(K, w).ravel(), pls_W[:, k])}")
                # print(f"[{it}] After       : "
                #       f"{loss.f(w):.4f} and "
                #       f"{', '.join([f'{l.f(w):.4f}' for l in loss.losses])}")
                f_after = loss.f(w)

                if len(self.func_vals_) > 0 and f_after > self.func_vals_[-1]:
                    warnings.warn("The function value increased! Try "
                                  "lowering the step size.")

                self.func_vals_.append(f_after)

                w_norm = np.linalg.norm(w_old)
                if w_norm > self.reconstruct_tol:
                    err_rel = np.linalg.norm(w_old - w) \
                        / (self.step_size * w_norm)
                else:
                    err_rel = np.linalg.norm(w_old - w) / self.step_size

                if self.verbose >= 1:
                    print(f"k: {k + 1}, it: {it}, err: {err_rel}, "
                          f"f: {f_after}")

                if err_rel < self.reconstruct_tol:
                    break

                # print()
                if f_after > f_before:
                    step_size /= 1 + max(0, factor_increase - 1) \
                        / (1 + factor_decay * it)
                    if self.verbose >= 2:
                        print(f"[{it}] Loss increased. "
                              f"Step size: {step_size:.6f}")
                        print()
                # else:
                #     step_size *= 1 + max(0, factor_decrease - 1) \
                #         / (1 + factor_decay * it)
                    # d__ = (1 + factor_decay * it)
                    # print(f"[{it}] Loss decreased. "
                    #       f"Step size: {step_size:.4f} "
                    #       f"({1 + max(0, factor_increase - 1) / d__:.4f}, "
                    #       f"{1 + max(0, factor_decrease - 1) / d__:.4f})")
                    # print()

            if K is None:
                w_ = w
            else:
                w_ = np.dot(K, w)
            t = np.dot(X, w_)
            d = np.dot(t.T, t).item()
            p = np.dot(X.T, t) / d
            c = np.dot(y.T, t) / d
            v = self._compute_v(V, P, w_)

            # For all but the last component, do the deflation
            if k < self.K - 1:
                X = X - np.dot(t, p.T)

            if K is not None:
                # print(w.shape)
                Omega = np.c_[Omega, w]
            W = np.c_[W, w_]
            T = np.c_[T, t]
            P = np.c_[P, p]
            C = np.c_[C, c]
            D = np.c_[D, d]
            V = np.c_[V, v]

        # Compute the W* one final time, for numerical stability.
        # TODO: Necessary?
        # V_ = np.dot(W, np.linalg.inv(np.dot(P.T, W)))

        if K is None:
            return W, T, P, C, D, V, None
        else:
            return W, T, P, C, D, V, Omega

    def fit(self, X, y):
        """Fit PLS regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            multi_output=False,
            y_numeric=True,
        )
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.solver == "nipals" and self.penalty is not None:
            raise ValueError("The NIPALS solver does not support penalties.")

        elif self.solver in ["auto", "nipals"] and self.penalty is None:
            beta_, W_, T_, P_, C_, D_, V_ = self._nipals(X, y)
            self.W_ = W_
            self.T_ = T_
            self.P_ = P_
            self.C_ = C_
            self.D_ = D_
            self.V_ = V_

        elif self.solver in ["auto", "krylov"] \
                and (self.penalty in [None, "none", "l1", "l2",
                                      "l1l2", "l2l1", "elasticnet"]):

            # # Build the Krylov matrix, K.
            # K = np.zeros((num_variables, num_components))
            # Kk = None
            # for k in range(num_components):
            #     if Kk is None:
            #         Kk = np.dot(X.T, y)
            #     else:
            #         Kk = np.dot(X.T, np.dot(X, Kk))

            #     K[:, k] = Kk.ravel()

            # # Orthogonalise the columns of K
            # K, _ = np.linalg.qr(K)

            _, pls_W, pls_T, pls_P, pls_C, _, _ = self._nipals(X, y)
            beta_ = self._krylov(X, y, pls_W)

            if self.reconstruct_components:
                W_, T_, P_, C_, D_, V_, Omega_ = self._reconstruct_components(
                    beta_,
                    X,
                    y,
                    pls_W,
                    pls_T,
                    pls_P,
                    pls_C,
                    K=pls_W,
                    retries=self.reconstruct_retries)
                self.W_ = W_
                self.T_ = T_
                self.P_ = P_
                self.C_ = C_
                self.D_ = D_
                self.V_ = V_
                self.Omega_ = Omega_
        else:
            raise ValueError("This combination of solver and penalties is not "
                             "recognised.")

        self.beta_ = beta_
        self.coef_ = self.beta_
        self.intercept_ = 0.0

        return self


class SparsePLSR(sklearn.base.RegressorMixin, _BasePLSR):
    """Sparse Partial Least Squares Regression.

    Parameters
    ----------
    K : int, optional
        The number of components. Default is 2.

    lambda1 : float, required
        The regularisation parameter for the X weight vector, w.

    lambda2 : float, required
        The regularisation parameter for the y weight, c.

    max_iter : None or int
        The maximum number of iterations to perform. Defaults (with None) to
        however many iterations are required to reach the tolerance level set.

    tol : float, optional
        The tolerance level to reach (the error in the approximation).

    verbose : int, optional
        The verbosity level. Defaults to zero, which means to not print
        any auxiliary information about the fit or predict. Default is 0, print
        nothing.

    random_state : None, int or instance of np.random.RandomState
        If seed is None, uses the RandomState singleton used by np.random. If
        seed is an int, uses a new RandomState instance seeded with seed. If
        seed is already a RandomState instance, use it. Otherwise raises
        ValueError.

    """

    def __init__(self,
                 K=2,
                 lambda1=None,
                 lambda2=None,
                 *,
                 max_iter=None,
                 tol=1e-4,
                 solver="auto",
                 verbose=0,
                 random_state=None,
                 ):

        super().__init__(
            K=K,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            verbose=verbose,
            random_state=random_state,
            )

        self.lambda1 = max(0.0, float(lambda1))
        self.lambda2 = max(0.0, float(lambda2))

    # def _soft_thresholding(self, v, lambda_):
    #     return np.multiply(np.sign(v), np.maximum(0.0, np.abs(v) - lambda_))

    def _one_component_svd(self, X, max_iter, tol):
        w = self.random_state.randn(X.shape[0], 1)
        w = w / np.linalg.norm(w)
        for i in range(max_iter):
            c = np.dot(X.T, w)
            c = c / np.linalg.norm(c)
            w_old = w
            w = np.dot(X, c)
            w = w / np.linalg.norm(w)

            err = np.linalg.norm(w - w_old)
            # print(err)
            if err < tol:
                break

        return w, c

    def fit(self, X, y):
        """Fit Sparse PLS regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, 1)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=False,
            dtype=[np.float64, np.float32],
            multi_output=False,
            y_numeric=True,
        )
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.solver != "auto":
            raise ValueError("The only available solver is 'auto'.")

        num_samples, num_variables = X.shape
        num_components = min(num_samples, num_variables, self.K)

        X_ = X
        y_ = y
        self.W_ = np.zeros((num_variables, num_components))
        self.T_ = np.zeros((num_samples, num_components))
        self.P_ = np.zeros((num_variables, num_components))
        self.C_ = np.zeros((1, num_components))
        self.D_ = np.zeros((1, num_components))

        # for k in range(num_components):
        #     M = np.dot(X_.T, y_)
        #     w, c = self._one_component_svd(M, self.max_iter, self.tol)

        #     it = 0
        #     while self.max_iter is None or it < self.max_iter:
        #         w_old = w
        #         c_old = c
        #         w = self._soft_thresholding(np.dot(M, c), self.lambda1)
        #         print(w)
        #         w = w / np.linalg.norm(w)
        #         c = self._soft_thresholding(np.dot(M.T, w), self.lambda2)
        #         c = c / np.linalg.norm(c)

        #         err_rel = np.linalg.norm(w_old - w) + np.linalg.norm(c_old - c)

        #         if self.verbose >= 1:
        #             print(f"k: {k}, it: {it}, err: {err_rel}")

        #         it += 1
        #         if err_rel < self.tol:
        #             break

        #     t = np.dot(X_, w)
        #     u = np.dot(y_, c)  # / np.dot(c.T, c)

        #     d = np.dot(t.T, t).item()
        #     p = np.dot(X_.T, t) / d
        #     # q = np.dot(y_.T, t) / np.dot(t.T, u)  # Is t.u a typo in the paper?
        #     q = np.dot(y_.T, t) / np.dot(t.T, t)  # Is t.u a typo in the paper?

        #     # Deflate
        #     X_ = X_ - np.dot(t, p.T)
        #     y_ = y_ - np.dot(t, q.T)

        #     self.W_[:, [k]] = w
        #     self.T_[:, [k]] = t
        #     self.P_[:, [k]] = p
        #     self.C_[0, k] = c.item()
        #     self.D_[0, k] = d

        for k in range(num_components):
            Xty = np.dot(X_.T, y_)
            # w = Xty
            w, c = self._one_component_svd(Xty, self.max_iter, self.tol)
            it = 0
            while self.max_iter is None or it < self.max_iter:
                w_old = w
                c_old = c
                # w = np.dot(Xty, c)
                w = self._soft_thresholding(np.dot(Xty, c), self.lambda1)

                norm_w = np.linalg.norm(w)
                if norm_w >= MACHINE_EPSILON:
                    w = w / norm_w

                c = self._soft_thresholding(np.dot(Xty.T, w), self.lambda2)
                norm_c = np.linalg.norm(c)
                if norm_c >= MACHINE_EPSILON:
                    c = c / norm_c

                norm_w_old = np.linalg.norm(w_old)
                if norm_w_old < MACHINE_EPSILON:
                    norm_w_old = 1.0
                norm_c_old = np.linalg.norm(c_old)
                if norm_c_old < MACHINE_EPSILON:
                    norm_c_old = 1.0
                err_rel = np.linalg.norm(w_old - w) / norm_w_old + \
                    np.linalg.norm(c_old - c) / norm_c_old

                if self.verbose >= 1:
                    print(f"k: {k}, it: {it}, err: {err_rel}")
                it += 1
                if err_rel < self.tol:
                    break
            t = np.dot(X_, w)
            # u = np.dot(y_, c)
            d = np.dot(t.T, t).item()
            if d >= MACHINE_EPSILON:
                p = np.dot(X_.T, t) / d
                c = np.dot(y_.T, t) / d
            else:
                p = np.dot(X_.T, t)
                c = np.dot(y_.T, t)
            # The paper says to do this, but this must be a typo?
            # c = np.dot(y_.T, t) / np.dot(t.T, u)

            # Deflate
            X_ = X_ - np.dot(t, p.T)
            y_ = y_ - np.dot(t, c.T)

            self.W_[:, [k]] = w
            self.T_[:, [k]] = t
            self.P_[:, [k]] = p
            self.C_[0, k] = c.item()
            self.D_[0, k] = d

        self.V_ = np.dot(self.W_, np.linalg.pinv(np.dot(self.P_.T, self.W_)))
        self.beta_ = np.dot(self.V_, self.C_.T)
        self.coef_ = self.beta_
        self.intercept_ = 0.0

        return self


class BaseLoss(metaclass=abc.ABCMeta):
    """Base class for losses and penalties."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def f(self, x):
        """A function value."""
        raise NotImplementedError("Must be implemented!")

    @abc.abstractmethod
    def grad(self, x):
        """The gradient of the function at x."""
        raise NotImplementedError("Must be implemented!")

    def approx_grad(self, w, eps=1e-4):
        """Numerical approximation of the gradient.

        Parameters
        ----------
        x : numpy.ndarray, shape (p, 1)
            The point at which to evaluate the gradient.

        eps : float, optional
            Positive float. The precision of the numerical solution. Smaller is
            better, but too small may result in floating point precision
            errors. Default is 1e-4.
        """
        p = w.shape[0]
        grad = np.zeros(w.shape)
        for i in range(p):
            w[i, 0] -= eps
            loss1 = self.f(w)
            w[i, 0] += 2.0 * eps
            loss2 = self.f(w)
            w[i, 0] -= eps
            grad[i, 0] = (loss2 - loss1) / (2.0 * eps)

        return grad


class LipschitzContinuousGradient(metaclass=abc.ABCMeta):
    """Base class for losses and penalties with Lipschitz continuous gradient.

    Must implement the Lipscitz constant of the gradient, L.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def L(self):
        """Lipschitz constant of the gradient of the loss."""
        raise NotImplementedError("Must be implemented!")


class CombinedLoss(BaseLoss):
    """Sum of losses or penalties."""

    def __init__(self, losses):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        for loss in losses:
            if not isinstance(loss, BaseLoss):
                raise ValueError("The losses must be a BaseLoss or a list or "
                                 "tuple of BaseLoss.")
        self.losses = losses

    def f(self, x):
        """The sum of the losses or penalties."""
        return np.sum([loss.f(x) for loss in self.losses])

    def grad(self, x):
        """The sum of the gradients of the losses or penalties."""
        return np.sum([loss.grad(x) for loss in self.losses], axis=0)


class InnerProductLoss(BaseLoss,
                       LipschitzContinuousGradient):
    """Inner product loss function.

    The inner product is

        v'.w,

    a function of the weight vector w.
    """

    def __init__(self, v):
        self._v = v

    def f(self, x):
        """The function value of the inner product."""
        val = np.dot(self._v.T, x)

        return val.item()

    def grad(self, x):
        """The gradient of the inner product."""
        grad = self._v

        return grad

    def L(self):
        """Return the Lipschitz constant of the gradient of this function."""
        return 0.0  # It is zero.


class PLSbetaloss(BaseLoss):
    r"""A loss function for the regression coefficients.

    Represents the penalty

        P(w) = \lambda.||\beta^* - \beta(w)||²_2,

    where \beta^* is a given "true" or sought regression vector, and \beta(w)
    is the PLS regression coefficient vector as a function of the weight vector
    w.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(42)
    >>> loss = PLSbetaloss(X, y, beta_pls, T_, P_, V_, 1.0)
    >>> a = np.random.randn(*w.shape)
    >>> vals = []
    >>> xx = np.logspace(-4, 0.5, 1000)
    >>> xx = np.r_[-xx[::-1], xx]
    >>> for i in xx:
    ...     vals.append(loss.f(w + (a - w) * i))
    >>> plt.figure()
    >>> plt.semilogy(xx, vals)
    >>>
    >>> grad = loss.grad(w)
    >>> print(np.linalg.norm(-grad - loss.approx_grad(w, eps=5e-4)))
    >>> print(np.linalg.norm(-grad - loss.approx_grad(w, eps=5e-5)))
    >>> print(np.linalg.norm(-grad - loss.approx_grad(w, eps=5e-6)))
    >>> print(np.linalg.norm(-grad - loss.approx_grad(w, eps=5e-7)))
    >>> print(np.linalg.norm(-grad - loss.approx_grad(w, eps=5e-8)))
    >>> grad_approx = loss.approx_grad(w, eps=5e-7)
    """

    def __init__(self, X, y, beta_, T, P, V, lambda_, K=None):
        self.X = X
        self.y = y
        self.beta_ = beta_
        self.T = T
        self.P = P
        self.V = V
        assert (self.X.shape[0] == self.y.shape[0])
        assert (self.y.shape[1] == 1)
        assert (self.X.shape[1] == self.beta_.shape[0])
        assert (self.beta_.shape[1] == 1)
        assert (self.T.shape[0] == self.X.shape[0])
        assert (self.P.shape[0] == self.X.shape[1])
        assert (self.V.shape[0] == self.X.shape[1])
        assert (self.T.shape[1] == self.P.shape[1])
        assert (self.T.shape[1] == self.V.shape[1])

        self.lambda_ = max(0.0, float(lambda_))

        self.K = K
        if self.K is None:
            self._Xty = np.dot(self.X.T, self.y)
        else:
            assert (self.K.shape[0] == self.V.shape[0])
            self._Xty = np.dot(self.K.T,
                               np.dot(self.X.T, self.y))
        # The regression vector up until the last component
        self._beta_k_1 = np.dot(self.V,
                                np.linalg.solve(np.dot(self.T.T, self.T),
                                                np.dot(self.T.T, y)))

    # def _compute_A(self):
    #     """Compute the correction matrix used to compute the kth w* (v_k).

    #     Multiplying the weigth vector, w, with this matrix gives us the w*,
    #     given all previous components 1, ..., k-1.
    #     """
    #     p = self.V.shape[0]
    #     k = self.V.shape[1]

    #     # TODO: Can this be made more efficiently, without a loop?
    #     #       This matrix can also be very large (p-by-p). Can we make this
    #     #       more efficient?
    #     A = np.eye(p)
    #     for i in range(k):
    #         A -= np.dot(self.V[:, [i]], self.P[:, [i]].T)

    #     return A

    def _compute_Aw(self, w):
        """Compute the kth v_k (a.k.a. w*) given components 1,...,k-1.

        This function computes dot((I - D)K, w), so works for any input vector.
        """
        k = self.V.shape[1]

        # TODO: Can this be made more efficiently, without a loop?
        if self.K is None:
            v = w
            for i in range(k):
                v = v - self.V[:, [i]] * np.dot(self.P[:, [i]].T, w)
        else:
            v = np.dot(self.K, w)
            for i in range(k):
                v = v - self.V[:, [i]] * np.dot(self.P[:, [i]].T,
                                                np.dot(self.K, w))

        return v

    def _compute_Atb(self, b):
        """Compute dot(((I - D)K).T, b) for any input vector, b."""
        k = self.V.shape[1]

        # TODO: Can this be made more efficiently, without a loop?
        if self.K is None:
            v = b
            for i in range(k):
                v = v - self.P[:, [i]] * np.dot(self.V[:, [i]].T, b)
        else:
            v = np.dot(self.K.T, b)
            for i in range(k):
                v = v - np.dot(self.K.T,
                               self.P[:, [i]]) * np.dot(self.V[:, [i]].T, b)

        return v

    def f(self, w):
        """The function value of the PLS beta loss."""
        if self.K is None:
            Xw = np.dot(self.X, w)
        else:
            Xw = np.dot(self.X,
                        np.dot(self.K, w))

        # v_k = self._compute_Aw(w)
        # beta_k = np.dot(v_k, np.dot(w.T, self._Xty)) / np.dot(Xw.T, Xw)
        # beta_diff = self.beta_ - self._beta_k_1  # beta* - beta_{k-1}
        # val = self.lambda_ * np.linalg.norm(beta_diff - beta_k)**2

        v_k = self._compute_Aw(w)  # (I - D)Kw
        beta_k = np.dot(v_k, np.dot(Xw.T, self.y)) / np.dot(Xw.T, Xw)
        val = self.lambda_ * np.linalg.norm(self.beta_
                                            - (self._beta_k_1 + beta_k))**2

        return val

    def grad(self, w):
        """Compute the gradient of the loss."""
        # a = self.beta_ - self._beta_k_1  # beta* - beta_{k-1}
        # b = self._Xty
        # Aw = self._compute_Aw(w)
        # AtAw = self._compute_Ata(Aw)
        # wtAtAw = np.linalg.norm(Aw)**2
        # Ata = self._compute_Ata(a)
        # Xw = np.dot(self.X, w)
        # XtXw = np.dot(self.X.T, Xw)
        # wtb = np.dot(w.T, b)
        # wtXtXw = np.linalg.norm(Xw)**2
        # atAw = np.dot(a.T, Aw)

        # grad = 0
        # grad -= (2 * wtb / wtXtXw) * Ata
        # grad -= (2 * atAw / wtXtXw) * b
        # grad += (4 * wtb * atAw / (wtXtXw**2)) * XtXw
        # grad += (2 * wtb * wtb / wtXtXw**2) * AtAw
        # grad += (2 * wtb * wtAtAw / wtXtXw**2) * b
        # grad -= (4 * wtb * wtAtAw * wtb / wtXtXw**3) * XtXw

        a = self._Xty  # Contains K if provided
        b = self.beta_ - self._beta_k_1  # beta* - beta_{k-1}
        Aw = self._compute_Aw(w)  # Contains K if provided
        Atb = self._compute_Atb(b)
        wta = np.dot(w.T, a)
        if self.K is None:
            Xw = np.dot(self.X, w)
            Bw = np.dot(self.X.T, Xw)
        else:
            Xw = np.dot(self.X, np.dot(self.K, w))
            Bw = np.dot(self.K.T, np.dot(self.X.T, Xw))
        wtBw = np.linalg.norm(Xw)**2
        btAw = np.dot(b.T, Aw)
        AtAw = self._compute_Atb(Aw)
        wtAtAw = np.linalg.norm(Aw)**2

        grad = 0
        grad -= (2 * wta / wtBw) * Atb
        grad -= (2 * btAw / wtBw) * a
        grad += (4 * btAw * wta / (wtBw**2)) * Bw
        grad += (2 * wta * wta / wtBw**2) * AtAw
        grad += (2 * wta * wtAtAw / wtBw**2) * a
        grad -= (4 * wta * wtAtAw * wta / wtBw**3) * Bw

        return self.lambda_ * grad


class SqNormAffineLoss(BaseLoss,
                       LipschitzContinuousGradient):
    r"""A penalty function.

    Represents the penalty

        P(x) = \gamma.||A.x - b||²_2,

    as a function of the weight vector x.
    """

    def __init__(self, A, b, gamma):
        self.A = A
        self.b = b

        if self.A is not None and self.b is not None:
            assert (self.A.shape[0] == self.b.shape[0])

        if self.b.ndim == 1:
            self.b = self.b[:, np.newaxis]

        self.gamma = max(0.0, float(gamma))

        if self.A is not None:
            self._Atb = np.dot(self.A.T, self.b)
        else:
            self._Atb = self.b.copy()

    def f(self, x):
        """Function value of the squared norm of an affine function."""
        if self.A is not None:
            val = self.gamma * np.linalg.norm(np.dot(self.A, x) - self.b)**2
        else:
            val = self.gamma * np.linalg.norm(x - self.b)**2

        return val.item()

    def grad(self, x):
        """Gradient of the loss."""
        if self.A is not None:
            grad = (self.gamma * 2.0) * np.dot(self.A.T,
                                               np.dot(self.A, x) - self.b)
        else:
            grad = (self.gamma * 2.0) * (x - self.b)

        return grad

    def L(self):
        """Lipschitz constant of this loss."""
        A = (self.gamma * 2.0) * np.dot(self.A.T, self.A)
        singval_max = sigma_max(A)

        return singval_max


class L2Penalty(BaseLoss,
                LipschitzContinuousGradient):
    r"""The L2 penalty function.

    Represents the penalty

        P(x) = \gamma.||A.x||²_2,

    as a function of the weight vector x.
    """

    def __init__(self, A, gamma):
        self.A = A
        self.gamma = max(0.0, float(gamma))

    def f(self, x):
        """Function value of the squared L2 norm of a linear function."""
        if self.A is not None:
            val = self.gamma * np.linalg.norm(np.dot(self.A, x))**2
        else:
            val = self.gamma * np.linalg.norm(x)**2

        return val.item()

    def grad(self, x):
        """Gradient of this loss."""
        if self.A is not None:
            grad = (self.gamma * 2.0) * np.dot(self.A.T, np.dot(self.A, x))
        else:
            grad = (self.gamma * 2.0) * x

        return grad

    def L(self):
        """Lipschitz constant of this loss."""
        A = (self.gamma * 2.0) * np.dot(self.A.T, self.A)
        singval_max = sigma_max(A)

        return singval_max


class L1Penalty(BaseLoss):
    r"""The L1 penalty function.

    Represents the penalty

        P(x) = \gamma.||A.x||_1,

    as a function of the weight vector x.
    """

    def __init__(self, A, gamma):
        self.A = A
        self.gamma = max(0.0, float(gamma))

    def f(self, x):
        """Function value of the L1 norm of a linear function."""
        if self.A is not None:
            val = self.gamma * np.linalg.norm(np.dot(self.A, x), ord=1)
        else:
            val = self.gamma * np.linalg.norm(x, ord=1)

        return val.item()

    def grad(self, x):
        """Gradient of this loss."""
        if self.A is not None:
            v = np.dot(self.A, x)
        else:
            v = x.copy()

        ind_nz = np.abs(v) > 0.0
        ind_z = np.logical_not(ind_nz)

        # v[ind_nz] = v[ind_nz] / np.abs(v[ind_nz])
        v[ind_nz] = np.sign(v[ind_nz])
        # v[ind_z] = np.random.choice([-1.0, 1.0],
        #                             size=np.count_nonzero(ind_z))
        v[ind_z] = 0.0  # np.random.uniform(-1.0, 1.0, size=np.count_nonzero(ind_z))

        if self.A is not None:
            grad = self.gamma * np.dot(self.A.T, v)
        else:
            grad = self.gamma * v

        return grad


# TODO: Add base class for constraints.
class ProjSqNorm(object):
    r"""Project onto squared norm ball.

    Project x such that

        ||x||²_2 \leq \varepsilon.

    Assuming that K is an orthonormal matrix.
    """

    def __init__(self, epsilon, K=None):
        self.epsilon = max(0, float(epsilon))
        self.K = K

    def __call__(self, x):  # noqa: D102
        if self.epsilon == 0:
            return np.zeros_like(x)

        if self.K is None:
            norm = np.linalg.norm(x)
        else:
            norm = np.linalg.norm(np.dot(self.K, x))
        if norm**2 <= self.epsilon:
            return x

        lambda_ = norm / (2 * np.sqrt(self.epsilon)) - 0.5

        return (1 / (1 + 2 * lambda_)) * x


class ProjNull(object):
    """Project onto the null space of a matrix A."""

    def __init__(self, A, control=True, lambda_factor=100, maxiter=10):
        self.A = A
        self.control = bool(control)
        self.lambda_factor = max(1, float(lambda_factor))
        self.maxiter = max(1, maxiter)

    def __call__(self, x):  # noqa: D102

        # Already in the null space?
        if np.linalg.norm(np.dot(self.A, x)) < 100 * MACHINE_EPSILON:
            return x

        # In this case \lambda \rightarrow \infty
        if self.A.shape[0] <= self.A.shape[1]:
            AAt = np.dot(self.A, self.A.T)
            invAAt_Ax = np.linalg.solve(AAt, np.dot(self.A, x))
            At_invAAt_Ax = np.dot(self.A.T, invAAt_Ax)

            x_ = x - At_invAAt_Ax

        # In this case \lambda is just "large"
        # TODO: Can we improve this?
        else:
            AtA = np.dot(self.A.T, self.A)
            Ip = np.eye(*AtA.shape)

            if self.control:
                maxiter = 1
            else:
                maxiter = self.maxiter

            lambda_ = self.lambda_factor * max(1,
                                               np.sum(np.abs(np.diag(self.A))))

            x_new = x
            for it in range(maxiter):
                x_old = x_new
                x_new = np.linalg.solve(Ip + (2 * lambda_) * AtA, x)
                print(f"Err: {np.linalg.norm(x_new - x_old)}")
                if it > 0 and \
                        (np.linalg.norm(x_new - x_old) / np.linalg.norm(x_old)
                         < np.sqrt(MACHINE_EPSILON)):
                    break
                lambda_ *= 10

            x_ = x_new

        return x_


def ADMM(x,
         z,
         argmin_x_L,
         argmin_z_L,
         A=None,
         B=None,
         c=None,
         max_iter=None,
         tol=1e-4,
         rho=1.0,
         mu=10.0,
         tau_inc=2.0,
         tau_dec=2.0,
         verbose=0):
    """An ADMM solver for problems in the form f(x) + g(z) with Ax + Bz = c.

    Each step minimises the Lagrangian,

        L(x, z, y) = f(x) + g(z) + y'.(A.x + B.z - c)
            + (rho/2).||A.x + B.z - c||²_2,

    first wrt. x, then wrt. z, and finally wrt. the dual variable, y.

    Parameters
    ----------
    x : Numpy array
        The start values for the x variable.

    z : Numpy array
        The start values for the z variable.

    argmin_x_L : Callable
        Has signature argmin_x_L(x, z, y, rho), returns the minimum of the
        augmented Lagrangian wrt. x.

    argmin_z_L : Callable
        Has signature argmin_z_L(x, z, y, rho), returns the minimum of the
        augmented Lagrangian wrt. z.

    A : A float, a numpy array, or None
        If None, it will be treated as an identity matrix.

    B : A float, a numpy array, or None
        If None, it will be treated as an identity matrix.

    c : A float, a numpy array, or None
        If None, it will be treated as zero.

    max_iter : int or None
        The maximum number of iterations. Default is None, which means to do as
        many iterations as necessary to reach the tolerance stopping criterion.

    tol : int or float
        The tolerance for the stopping criterion. Detault value is 1e-4.
    """
    rho = max(MACHINE_EPSILON, float(rho))
    if c is None:
        c = 0.0
    elif isinstance(c, (int, float)):
        c = float(c)

    if max_iter is not None:
        max_iter = max(1, int(max_iter))
    tol = max(0.0, float(tol))

    if A is None:
        y = x.copy()
    else:
        if isinstance(A, (int, float)):
            y = A * x
        else:
            y = np.dot(A, x)

    it = 0
    while (max_iter is None) or (it < max_iter):
        x_ = x
        x = argmin_x_L(x_, z, y, rho)

        z_ = z
        z = argmin_z_L(x, z_, y, rho)
        # print(f"z_.shape: {z_.shape}")
        # print(f"z.shape: {z.shape}")

        if A is None:
            Ax = x
        else:
            if isinstance(A, (int, float)):
                Ax = A * x
            else:
                Ax = np.dot(A, x)
        if B is None:
            Bz = z
        else:
            if isinstance(B, (int, float)):
                Bz = B * z
            else:
                Bz = np.dot(B, z)

        r = Ax + Bz - c  # The primal residual

        AtB = 1.0
        if A is None:
            if B is None:
                pass
            else:
                AtB = B
        else:
            if B is None:
                if isinstance(A, (int, float)):
                    AtB = A
                else:
                    AtB = A.T
            else:
                if isinstance(B, (int, float)):
                    AtB = A.T * B
                else:
                    AtB = np.dot(A.T, B)

        z_z_ = z - z_
        if isinstance(AtB, (int, float)):  # The dual residual
            s = (rho * AtB) * z_z_
        else:
            s = rho * np.dot(AtB, z_z_)

        norm_r = np.linalg.norm(r)
        norm_s = np.linalg.norm(s)

        if norm_r > mu * norm_s:
            rho = tau_inc * rho
        elif norm_s > mu * norm_r:
            rho = rho / tau_dec

        # Update the Lagrange multipliers
        y_ = y
        y = y_ + rho * r

        # err_rel_x = np.linalg.norm(x - x_) / np.linalg.norm(x_)
        # err_rel_z = np.linalg.norm(z - z_) / np.linalg.norm(z_)

        if verbose >= 1:
            print(f"it: {it}, primal res: {norm_r}, dual res: {norm_s}, "
                  f"rho: {rho} [ADMM]")

        # if err_rel_x < tol and err_rel_z < tol:
        #     break
        if norm_r < tol and norm_s < tol:
            break

        it += 1

    return x, z, y


def parallel_dykstra(x,
                     functions,
                     weights=None,
                     max_iter=100,
                     min_iter=1,
                     eps=MACHINE_EPSILON,
                     verbose=0):
    """Find the projection onto the intersection of multiple sets.

    Parameters
    ----------
    functions : List or tuple with two or more elements.
        Functions defining projection or proximal operators.

    x : Numpy array.
        The point that we wish to project.

    weights : List or tuple with floats.
        Weights for the functions. Default is that they all have the same
        weight. The elements of the list or tuple must sum to 1.

    max_iter : int, optional
        The maximum number of iterations of the parallel Dykstra algorithm.
        Default is 100.

    min_iter : int, optional
        Do at least this number of iterations of the parallel Dykstra
        algorithm. Default is 1.

    eps : float, optional
        The tolerance level to reach, the relative change in the weight vector.
        Default is machine epsilon for 32-bit floating-point numbers, 2.2e-16.

    verbose : int, optional
        The verbosity level. Defaults to zero, which means to not print
        any auxiliary information about the fit or predict. Default is 0, print
        nothing.
    """
    num = len(functions)

    if weights is None:
        weights = [1 / num] * num

    x_new = x_old = x
    p = [0.0] * len(functions)
    z = [0.0] * len(functions)
    for i in range(num):
        z[i] = np.copy(x)

    for it in range(max_iter):

        for i in range(num):
            p[i] = functions[i](z[i])

        # TODO: Does the weights really matter when the function is the
        # indicator function?
        x_old = x_new
        x_new = np.zeros(x_old.shape)
        for i in range(num):
            x_new += weights[i] * p[i]

        for i in range(num):
            z[i] = x_new + z[i] - p[i]

        # print(np.linalg.norm(x_new))

        x_old_norm = np.linalg.norm(x_old)
        if x_old_norm < eps:
            x_old_norm = 1.0
        if np.linalg.norm(x_new - x_old) / x_old_norm < eps \
                and it + 1 >= min_iter:
            if verbose >= 1:
                print("Dykstra converged!")
            break
        # else:
        #     print(np.linalg.norm(x_new - x_old) / np.linalg.norm(x_old))

    return x_new
