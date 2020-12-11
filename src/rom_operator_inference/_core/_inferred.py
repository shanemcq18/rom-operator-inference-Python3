# _core/_inferred.py
"""Nonparametric Operator Inference ROM classes.

Classes
-------
* _InferredMixin
* InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM)
* InferredContinuousROM(_InferredMixin, _NonparametricMixin, _ContinuousROM)
"""
# TODO: _construct_data_matrix() --> _assemble_data_matrix()
# TODO: _construct_f_() --> _init_f_()


__all__ = [
            "InferredDiscreteROM",
            "InferredContinuousROM",
          ]

import numpy as np

from ._base import _DiscreteROM, _ContinuousROM, _NonparametricMixin
from .. import lstsq
from ..utils import kron2c, kron3c


class _InferredMixin:
    """Mixin class for reduced model classes that use Operator Inference."""

    def _check_training_data_shapes(self, datasets, labels):
        """Ensure that each data set has the same number of columns."""
        for data, label in zip(datasets, labels):
            # Ensure each data set is two-dimensional.
            if data.ndim != 2:
                raise ValueError(f"{label} must be two-dimensional")
            # Ensure each data set has the same number of columns.
            if data.shape[1] != datasets[0].shape[1]:
                raise ValueError("training data not aligned "
                                 f"({label}.shape[1] != {labels[0]}.shape[1])")
            # Validate the number of rows.
            if label.startswith("X") and data.shape[0] not in (self.n, self.r):
                raise ValueError(f"invalid training set ({label}.shape[0] "
                                 f"!= n={self.n} or r={self.r})")
            elif label.startswith("U") and data.shape[0] != self.m:
                raise ValueError(f"invalid training input "
                                 f"({label}.shape[0] != m={self.m})")

    def _process_fit_arguments(self, Vr, X, rhs, U):
        """Do sanity checks, extract dimensions, check and fix data sizes, and
        get projected data for the Operator Inference least-squares problem.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        Returns
        -------
        X_ : (r,k) ndarray
            Projected state snapshots.

        rhs_ : (r,k) ndarray
            Projected right-hand-side data.

        U : (m,k) ndarray
            Inputs, potentially reshaped.
        """
        # Check modelform and inputs.
        self._check_modelform(trained=False)
        self._check_inputargs(U, 'U')

        # Store basis and dimensions.
        if Vr is not None:
            self.n, self.r = Vr.shape   # Full dimension, reduced dimension.
        else:
            self.n = None               # No full dimension.
            self.r = X.shape[0]         # Reduced dimension.
        self.Vr = Vr

        # Ensure training data sets have consistent sizes.
        if self.has_inputs:
            if U.ndim == 1:             # Reshape one-dimensional inputs.
                U = U.reshape((1,-1))
            self.m = U.shape[0]         # Input dimension.
            self._check_training_data_shapes([X, rhs, U], ["X", "Xdot", "U"])
        else:
            self.m = None               # No input dimension.
            self._check_training_data_shapes([X, rhs], ["X", "Xdot"])

        # Project states and rhs to the reduced subspace (if not done already).
        X_ = self.project(X, 'X')
        rhs_ = self.project(rhs, 'rhs')

        return X_, rhs_, U

    def _construct_data_matrix(self, X_, U):
        """Construct the Operator Inference data matrix D from projected data.

        If modelform="cAHB", this is D = [1 | X_.T | (X_ ⊗ X_).T | U.T].

        Parameters
        ----------
        X_ : (r,k) ndarray
            Column-wise projected snapshot training data.

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array.

        Returns
        -------
        D : (k,d(r,m)) ndarray
            Operator Inference data matrix (no regularization).
        """
        D = []
        if self.has_constant:           # Constant term.
            D.append(np.ones((X_.shape[1],1)))

        if self.has_linear:             # Linear state term.
            D.append(X_.T)

        if self.has_quadratic:          # (compact) Quadratic state term.
            D.append(kron2c(X_).T)

        if self.has_cubic:              # (compact) Cubic state term.
            D.append(kron3c(X_).T)

        if self.has_inputs:             # Linear input term.
            if self.m == U.ndim == 1:
                U = U.reshape((1,-1))
            D.append(U.T)

        return np.hstack(D)

    def _extract_operators(self, O):
        """Extract and save the inferred operators from the block-matrix
        solution to the least-squares problem.

        Parameters
        ----------
        O : (r,d(r,m)) ndarray
            Block matrix of ROM operator coefficients, the transpose of the
            solution to the Operator Inference linear least-squares problem.
        """
        i = 0
        if self.has_constant:           # Constant term (one-dimensional).
            self.c_ = O[:,i:i+1][:,0]
            i += 1
        else:
            self.c_ = None

        if self.has_linear:             # Linear state matrix.
            self.A_ = O[:,i:i+self.r]
            i += self.r
        else:
            self.A_ = None

        if self.has_quadratic:          # (compact) Qudadratic state matrix.
            _r2 = self.r * (self.r + 1) // 2
            self.H_ = O[:,i:i+_r2]
            i += _r2
        else:
            self.H_ = None

        if self.has_cubic:              # (compact) Cubic state matrix.
            _r3 = self.r * (self.r + 1) * (self.r + 2) // 6
            self.G_ = O[:,i:i+_r3]
            i += _r3
        else:
            self.G_ = None

        if self.has_inputs:             # Linear input matrix.
            self.B_ = O[:,i:i+self.m]
            i += self.m
        else:
            self.B_ = None

        return

    def _construct_solver(self, Vr, X, rhs, U, P, **kwargs):
        """Construct a solver object mapping the regularizer P to solutions
        of the Operator Inference least-squares problem.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB". This parameter is used here
            only to determine the correct type of solver.
        """
        X_, rhs_, U = self._process_fit_arguments(Vr, X, rhs, U)
        D = self._construct_data_matrix(X_, U)
        self.solver_ = lstsq.solver(D, rhs_.T, P, **kwargs)

    def _evaluate_solver(self, P):
        """Evaluate the least-squares solver with regularizer P.

        Parameters
        ----------
        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".
        """
        Otrp = self.solver_.predict(P)
        self._extract_operators(Otrp.T)
        self._construct_f_()

    def fit(self, Vr, X, rhs, U, P, **kwargs):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and rhs are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        rhs : (n,k) or (r,k) ndarray
            Column-wise next-iteration (discrete model) or velocity
            (continuous model) training data. Each column is a snapshot, and
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        **kwargs
            Additional arguments for the least-squares solver.
            See lstsq.solvers().

        Returns
        -------
        self
        """
        self._construct_solver(Vr, X, rhs, U, P, **kwargs)
        self._evaluate_solver(P)
        return self


# Nonparametric Operator Inference models -------------------------------------
class InferredDiscreteROM(_InferredMixin, _NonparametricMixin, _DiscreteROM):
    """Reduced order model for a discrete dynamical system of
    the form

        x_{j+1} = f(x_{j}, u_{j}),              x_{0} = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax.
        'H' : Quadratic state term H(x⊗x).
        'G' : Cubic state term G(x⊗x⊗x).
        'B' : Input term Bu.
        For example, modelform=="AB" means f(x,u) = Ax + Bu.

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax.

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x).

    has_inputs : bool
        Whether or not there is a linear input term Bu.

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    H_ : (r,r(r+1)/2) ndarray or None
        Learned ROM (compact) quadratic state matrix, or None if 'H' is not in
        `modelform`.

    G_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM (compact) cubic state matrix, or None if 'G' is not in
        `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable((r,) ndarray, (m,) ndarray) -> (r,)
        The learned ROM operator, defined by c_, A_, H_, G_, and/or B_.
        The signature is f_(x_) if 'B' is not in `modelform` (no inputs) and
        f_(x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and inputs if appropriate) to reduced state. Calculated in fit().
    """
    def fit(self, Vr, X, U=None, P=0, **kwargs):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X is assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k-1) or (k-1,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        **kwargs
            Additional arguments for the least-squares solver.
            See lstsq.solvers().

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr,
                                  X[:,:-1], X[:,1:],    # x_j's and x_{j+1}'s.
                                  U[...,:X.shape[1]-1] if U is not None else U,
                                  P, **kwargs)


class InferredContinuousROM(_InferredMixin, _NonparametricMixin,
                            _ContinuousROM):
    """Reduced order model for a system of high-dimensional ODEs of the form

        dx / dt = f(t, x(t), u(t)),             x(0) = x0.

    The model form (structure) of the desired reduced model is user specified,
    and the operators of the reduced model are inferred by solving an ordinary
    least-squares problem.

    Parameters
    ----------
    modelform : str containing 'c', 'A', 'H', and/or 'B'
        The structure of the desired reduced-order model. Each character
        indicates the presence of a different term in the model:
        'c' : Constant term c
        'A' : Linear state term Ax(t).
        'H' : Quadratic state term H(x⊗x)(t).
        'G' : Cubic state term G(x⊗x⊗x)(t).
        'B' : Input term Bu(t).
        For example, modelform=="AB" means f(t,x(t),u(t)) = Ax(t) + Bu(t).

    Attributes
    ----------
    has_consant : bool
        Whether or not there is a constant term c.

    has_linear : bool
        Whether or not there is a linear state term Ax(t).

    has_quadratic : bool
        Whether or not there is a quadratic state term H(x⊗x)(t).

    has_cubic : bool
        Whether or not there is a cubic state term G(x⊗x⊗x)(t).

    has_inputs : bool
        Whether or not there is a linear input term Bu(t).

    n : int
        The dimension of the original full-order model (x.size).

    r : int
        The dimension of the learned reduced-order model (x_.size).

    m : int or None
        The dimension of the input u(t), or None if 'B' is not in `modelform`.

    Vr : (n,r) ndarray
        The basis for the linear reduced space (e.g., POD basis matrix).

    datacond_ : float
        Condition number of the raw data matrix for the least-squares problem.

    dataregcond_ : float
        Condition number of the regularized data matrix for the least-squares
        problem. Same as datacond_ if there is no regularization.

    residual_ : float
        The squared Frobenius-norm residual of the regularized least-squares
        problem for computing the reduced-order model operators.

    misfit_ : float
        The squared Frobenius-norm data misfit of the (nonregularized)
        least-squares problem for computing the reduced-order model operators.

    c_ : (r,) ndarray or None
        Learned ROM constant term, or None if 'c' is not in `modelform`.

    A_ : (r,r) ndarray or None
        Learned ROM linear state matrix, or None if 'A' is not in `modelform`.

    H_ : (r,r(r+1)/2) ndarray or None
        Learned ROM (compact) quadratic state matrix, or None if 'H' is not in
        `modelform`.

    G_ : (r,r(r+1)(r+2)/6) ndarray or None
        Learned ROM (compact) cubic state matrix, or None if 'G' is not in
        `modelform`.

    B_ : (r,m) ndarray or None
        Learned ROM input matrix, or None if 'B' is not in `modelform`.

    f_ : callable(float, (r,) ndarray, func?) -> (r,) ndarray
        The learned ROM operator defined by c_, A_, H_, G_, and/or B_.
        The signature is f_(t, x_) if 'B' is not in `modelform` (no inputs) and
        f_(t, x_, u) if 'B' is in `modelform`. That is, f_ maps reduced state
        (and possibly an input function) to reduced state. Calculated in fit().

    sol_ : Bunch object returned by scipy.integrate.solve_ivp(), the result
        of integrating the learned ROM in predict(). For more details, see
        https://docs.scipy.org/doc/scipy/reference/integrate.html.
    """
    def fit(self, Vr, X, Xdot, U=None, P=0, **kwargs):
        """Solve for the reduced model operators via ordinary least squares.

        Parameters
        ----------
        Vr : (n,r) ndarray or None
            The basis for the linear reduced space (e.g., POD basis matrix).
            If None, X and Xdot are assumed to already be projected (r,k).

        X : (n,k) or (r,k) ndarray
            Column-wise snapshot training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        Xdot : (n,k) or (r,k) ndarray
            Column-wise velocity training data (each column is a snapshot),
            either full order (n rows) or projected to reduced order (r rows).

        U : (m,k) or (k,) ndarray or None
            Column-wise inputs corresponding to the snapshots. If m=1 (scalar
            input), then U may be a one-dimensional array. Required if 'B' is
            in `modelform`; must be None if 'B' is not in `modelform`.

        P : float >= 0 or (d,d) ndarray or list of r (floats or (d,d) ndarrays)
            Tikhonov regularization factor(s); see lstsq.solve(). Here, d
            is the number of unknowns in each decoupled least-squares problem,
            e.g., d = r + m when `modelform`="AB".

        **kwargs
            Additional arguments for the least-squares solver.
            See lstsq.solvers().

        Returns
        -------
        self
        """
        return _InferredMixin.fit(self, Vr, X, Xdot, U, P, **kwargs)
