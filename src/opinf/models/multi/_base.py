# models/multi/_base.py
"""Abstract base class for multilithic dynamical systems models."""

__all__ = []


import abc
import warnings
import numpy as np

from .. import errors, lstsq
from ...operators import _utils as oputils


class _ModelMulti(abc.ABC):
    r"""Base class for systems with multivariable block structure.

    This class is for systems where the state that can be partitioned as

    .. math::
       \qhat = \left[\begin{array}{c}
       \qhat_0 \\ \qhat_1 \\ \vdots \\ \qhat_{n_q - 1}
       \end{array}\right]
       \in \RR^r,

    where each :math:`\qhat_i \in \RR^{r_i}`. The total state dimension is the
    sum :math:`r = \sum_{i=0}^{n_q - 1}r_i`.

    Parameters
    ----------
    state_dimensions : tuple of ints
        Number of degrees of freedom in each state variable.
    operators : list
        Block operators
    """

    def __init__(self, state_dimensions, operators):
        """Set the state dimensions and initialize other attributes."""
        # Dimensions.
        self.__rs = tuple([int(r) for r in state_dimensions])
        dimsum = np.cumsum((0,) + self.__rs)
        self.__r = int(dimsum[-1])
        self.__nvars = len(state_dimensions)
        self.__slices = [
            slice(dimsum[i], dimsum[i + 1]) for i in range(self.__nvars)
        ]

        # TODO: modify the checks here because operators is just one list.
        # Operators.
        if len(operators) <= 1:
            warnings.warn(
                "only one set of operators detected, "
                "consider using a monolithic model",
                errors.UsageWarning,
            )
        for oplist in operators:
            for op in oplist:
                op.set_state_dimensions(state_dimensions)
                op.input_dimension = self.input_dimension
        self.__operators = operators

    # Dimensions --------------------------------------------------------------
    @property
    def state_dimensions(self):
        """State dimensions (r0, r1, ...)."""
        return self.__rs

    @property
    def state_dimension(self):
        """Total dimension of the state, r = r0 + r1 + ..."""
        return self.__r

    @property
    def num_variables(self):
        """Number of state variables."""
        return self.__nvars

    @property
    def argslices(self):
        """Slices to access the individual state variables."""
        return self.__slices

    def split(self, arr):
        """Split a state array into the individual variables.

        Parameters
        ----------
        arr : (r,...) ndarray
            State vector or snapshot matrix to split.

        Returns
        -------
        arrs : list ``num_variables`` ndarrays
            Individual state variables.
        """
        return [arr[s] for s in self.argslices]

    @classmethod
    @abc.abstractmethod
    def input_dimension(self):
        """Number of inputs."""
        raise NotImplementedError

    # Operators ---------------------------------------------------------------
    @property
    def operators(self) -> list:
        """Model operators."""
        return self.__operators

    def _extract_operators(self, Ohats):
        """Populate operator entries.

        Parameters
        ----------
        Ohats : list of ndarrays
            Operator matrices, one for each state variable.
        """
        if (l1 := len(Ohats)) != (l2 := len(self.operators)):
            raise ValueError(f"len(Ohats) = {l1} != {l2} = len(operators)")

        for Ohat, oplist in zip(Ohats, self.operators):
            index = 0
            Ohat = np.atleast_2d(Ohat)
            for op in oplist:
                new_index = index + op.operator_dimension()
                op.set_entries(Ohat[:, index:new_index])
                index = new_index

    # Evaluation --------------------------------------------------------------
    def rhs(self, state, input_=None):
        """Right-hand side of the model.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (M,) ndarray or None
            Input vector.

        Returns
        -------
        evaluation : (r,) ndarray
            Right-hand side of the model at the given state / input.
        """
        rhs = np.zeros(self.state_dimension, dtype=state.dtype)
        for op in self.operators:
            rhs[self.argslices[op.indices[0]]] += op.apply(state, input_)

        # # Old version, when operators was a list of lists
        # for ops, rowslice in zip(self.operators, self.argslices):
        #     for op in ops:
        #         rhs[rowslice] += op.apply(state, input_)
        return rhs

    def jacobian(self, state, input_=None):
        """State Jacobian of the right-hand side of the model.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (M,) ndarray or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian of the right-hand side of the model
            at the given state / input.
        """
        jac = np.zeros(
            (self.state_dimension, self.state_dimension),
            dtype=state.dtype,
        )
        for op in self.operators:
            jac[self.argslices[op.indices[0]]] += op.jacobian(state, input_)

        # # Old version, when operatros was a list of lists
        # for ops, rowslice in zip(self.operators, self.argslices):
        #     for op in ops:
        #         jac[rowslice, :] += op.jacobian(state, input_)
        return jac


class _OpInfModel(_ModelMulti):
    """Base class for multilithic models with Operator Inference learning."""

    # Properties: solver ------------------------------------------------------
    @property
    def solvers(self):
        """Solvers for the decoupled least-squares regressions, see
        :mod:`opinf.lstsq`."""
        return self.__solvers

    @solvers.setter
    def solvers(self, solvers):
        """Set the solver, including default options."""
        if self._fully_intrusive:
            if solvers is not None:
                warnings.warn(
                    "all operators initialized explicity, setting solver=None",
                    errors.OpInfWarning,
                )
            self.__solvers = None
            return

        # Defaults and shortcuts.
        if solvers is None:
            # No regularization.
            solvers = [lstsq.PlainSolver() for _ in range(self.num_variables)]
        elif np.isscalar(solvers):
            if solvers == 0:
                # Also no regularization.
                solvers = [
                    lstsq.PlainSolver() for _ in range(self.num_variables)
                ]
            elif solvers > 0:
                # Scalar Tikhonov (L2) regularization.
                solvers = [
                    lstsq.L2Solver(solvers) for _ in range(self.num_variables)
                ]
            else:
                raise ValueError("if a scalar, solver must be nonnegative")

        # Light validation: must be instance w/ fit(), solve().
        for solver in solvers:
            if isinstance(solver, type):
                raise TypeError("solver must be an instance, not a class")
            for mtd in "fit", "solve":
                if not hasattr(solver, mtd) or not callable(
                    getattr(solver, mtd)
                ):
                    warnings.warn(
                        f"solver should have a '{mtd}()' method",
                        errors.OpInfWarning,
                    )

        self.__solvers = solvers

    # Validation methods ------------------------------------------------------
    def _check_inputargs(self, u, argname):
        """Check that the model structure agrees with input arguments."""
        if self._has_inputs and u is None:
            raise ValueError(f"argument '{argname}' required")

        if not self._has_inputs and u is not None:
            warnings.warn(
                f"argument '{argname}' should be None, "
                "argument will be ignored",
                errors.OpInfWarning,
            )

    def _check_is_trained(self):
        """Ensure that the model is trained and ready for prediction."""
        if self.state_dimension is None:
            raise AttributeError("no state_dimension (call fit())")
        if self._has_inputs and (self.input_dimension is None):
            raise AttributeError("no input_dimension (call fit())")
        if any(oputils.is_uncalibrated(op) for op in self.operators):
            raise AttributeError("model not trained (call fit())")

    # Fitting -----------------------------------------------------------------
    @abc.abstractmethod
    def fit(self, *args, **kwargs):  # pragma: no cover
        """Learn model operators from data."""
        pass
