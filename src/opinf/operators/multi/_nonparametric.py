# _nonparametric.py
"""Multilithic OpInf operators with no external parameter dependencies."""

__all__ = [
    "ConstantOperatorMulti",
    "LinearOperatorMulti",
    "BilinearOperatorMulti",
    "QuadraticOperatorMulti",
    "CubicOperatorMulti",
    "InputOperatorMulti",
    "StateInputOperatorMulti",
]

import warnings
import numpy as np
import scipy.linalg as la

from ._base import OpInfOperatorMulti, InputMultiMixin
from .._nonparametric import QuadraticOperator, CubicOperator
from ... import errors, utils


# No dependence on state or input =============================================
class ConstantOperatorMulti(OpInfOperatorMulti):
    r"""Constant operator for multivariable systems,
    :math:`\qhat \mapsto \chat_{i}\in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_index : int
        Index :math:`i` of the substate this operator maps to.
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_i,) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 1

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries):
            entries = np.atleast_1d(entries)
        # self._validate_entries?

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = np.ravel(entries)
            else:
                raise ValueError(
                    "ConstantOperatorMulti entries must be one-dimensional"
                )
        if entries.shape[0] != self.substate_dimensions[self.state_indices[0]]:
            raise ValueError("invalid entries.shape[0]")

        super().set_entries(self, entries)

    @staticmethod
    def operator_dimension():
        """Number of columns in the operator entries matrix."""
        return 1

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        if np.ndim(state) == 2:  # k > 1.
            return np.outer(self.entries, np.ones(state.shape[-1]))
        return self.entries  # k = 1.


# Dependent on state but not on input =========================================
class LinearOperatorMulti(OpInfOperatorMulti):
    r"""Linear state operator for multivariable systems,
    :math:`\qhat \mapsto \Ahat_{i,j}\qhat_{j}\in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_indices : tuple of 2 ints
        Indices of the substates this operator maps to and acts on, i.e.,
        :math:`(i, j).`
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_i, r_j) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 2

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError(
                "LinearOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")
        if (_dim := entries.shape[1]) != (r_j := self.operator_dimension()):
            raise ValueError(f"entries.shape[1] = {_dim} != {r_j} = r_j")

        super().set_entries(self, entries)

    @utils.requires("substate_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix."""
        return self.substate_dimensions[self.state_indices[1]]

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        return self.entries @ state[self.substate_slices[1]]

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        jac : (r_i, r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        jac = np.zeros(
            (self.entries.shape[0], self.state_dimension),
            dtype=state.dtype,
        )
        jac[:, self.substate_slices[1]] = self.entries
        return jac


class BilinearOperatorMulti(OpInfOperatorMulti):
    r"""Bilinear state operator for multivariable systems,
    :math:`\qhat \mapsto
    \Hhat_{i,jk}[\qhat_{j}\otimes\qhat_{k}]\in\mathbb{R}^{r_i}`
    where :math:`j \neq k.`

    See :class:`QuadraticOperatorMulti` for the case where :math:`j = k.`

    Parameters
    ----------
    state_indices : tuple of 3 ints
        Indices of the substates this operator maps to and acts on, i.e.,
        :math:`(i,j,k).`
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_i, r_j * r_k) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 3

    def __init__(
        self,
        state_indices: tuple,
        substate_dimensions=None,
        entries=None,
    ):
        """Set the dimensions and construct access slices."""
        super().__init__(
            state_indices=state_indices,
            substate_dimensions=substate_dimensions,
            entries=entries,
        )
        if self.state_indices[1] == self.state_indices[2]:
            warnings.warn(
                "for quadratic interactions of one substates with itself, "
                "use QuadraticOperatorMulti",
                errors.UsageWarning,
            )

    # Properties --------------------------------------------------------------
    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries?

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError(
                "BilinearOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")
        if (_dim := entries.shape[1]) != (_r2 := self.operator_dimension()):
            raise ValueError(f"entries.shape[1] = {_dim} != {_r2} = r_j r_k")

        super().set_entries(self, entries)

    # Inference ---------------------------------------------------------------
    @utils.requires("substate_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix, based solely on
        the state and/or input dimensions.
        """
        rleft = self.substate_dimensions[self.state_indices[0]]
        rright = self.substate_dimensions[self.state_indices[1]]
        return rleft * rright

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        mult = np.kron if state.ndim == 1 else la.khatri_rao
        return self.entries @ mult(
            self.getarg(state, 0), self.getarg(state, 1)
        )

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        jac : (r_i, r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        jac = np.zeros((self.entries.shape[0], self.state_dimension))
        jac[:, self.substate_slices[1]] = (
            self.entries
            @ np.kron(
                np.eye(self.substate_dimensions[self.state_indices[1]]),
                state[self.substate_slices[2]],
            ).T
        )
        jac[:, self.substate_slices[2]] = (
            self.entries
            @ np.kron(
                state[self.substate_slices[1]],
                np.eye(self.substate_dimensions[self.state_indices[2]]),
            ).T
        )
        return jac


class QuadraticOperatorMulti(OpInfOperatorMulti):
    r"""Quadratic state operator for multivariable systems,
    :math:`\qhat \mapsto
    \Hhat_{i,jj}[\qhat_{j}\otimes\qhat_{j}]\in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_indices : tuple of 2 ints
        Indices of the substates this operator maps to and acts on, i.e.,
        :math:`(i,j).`
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_i, r_j^2) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 2

    def _clear(self):
        """Clear the operator entries and related attributes."""
        super()._clear(self)
        self._submask = None
        self._prejac = None

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError(
                "QuadraticOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")
        r_j = self.substate_dimensions[self.state_indices[1]]
        if entries.shape[1] == r_j**2:
            entries = QuadraticOperator.compress_entries(entries)
        if (_dim := entries.shape[1]) != (_r2 := self.operator_dimension()):
            raise ValueError(
                f"entries.shape[1] = {_dim} != {_r2} = (r_j + 1) * r_j / 2"
            )

        self._submask = QuadraticOperator.ckron_indices(r_j)
        self._prejac = None
        super().set_entries(self, entries)

    # Inference ---------------------------------------------------------------
    @utils.requires("substate_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix, based solely on
        the state and/or input dimensions.
        """
        rj = self.substate_dimensions[self.state_indices[1]]
        return (rj + 1) * rj // 2

    # Evaluation --------------------------------------------------------------
    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        Jt @ q = jacobian(q).
        """
        r_i = self.substate_dimensions[self.state_indices[0]]
        r_j = self.substate_dimensions[self.state_indices[1]]
        Ht = QuadraticOperator.expand_entries(self.entries).reshape(
            (r_i, r_j, r_j)
        )
        self._prejac = Ht + Ht.transpose(0, 2, 1)

    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        return self.entries @ np.prod(
            state[self.substate_slices[1]][self._submask],
            axis=1,
        )

    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        jac : (r_i, r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        if self._prejac is None:
            self._precompute_jacobian_jit()
        jac = np.zeros(
            (self.entries.shape[0], self.state_dimension),
            dtype=state.dtype,
        )
        jac[:, self.substate_slices[1]] = (
            self._prejac @ state[self.substate_slices[1]]
        )
        return jac


class CubicOperatorMulti(OpInfOperatorMulti):
    r"""Cubic state operator for multivariable systems,
    :math:`\qhat \mapsto
    \Ghat_{i,jjj}[\qhat_{j}\otimes\qhat_{j}\otimes\qhat_{j}]
    \in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_indices : tuple of 2 ints
        Indices of the substates this operator maps to and acts on, i.e.,
        :math:`(i,j).`
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_i, r_j^3) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 2

    def _clear(self):
        """Clear the operator entries and related attributes."""
        super()._clear(self)
        self._submask = None
        self._prejac = None

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError(
                "CubicOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")
        r_j = self.substate_dimensions[self.state_indices[1]]
        if entries.shape[1] == r_j**3:
            entries = CubicOperator.compress_entries(entries)
        if (_dim := entries.shape[1]) != (_r3 := self.operator_dimension()):
            raise ValueError(
                f"entries.shape[1] = {_dim} "
                f"!= {_r3} = (r_j + 2) * (r_j + 1) * r_j / 6"
            )

        self._submask = CubicOperator.ckron_indices(r_j)
        self._prejac = None
        super().set_entries(self, entries)

    # Inference ---------------------------------------------------------------
    @utils.requires("substate_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix, based solely on
        the state and/or input dimensions.
        """
        rj = self.substate_dimensions[self.state_indices[1]]
        return (rj + 2) * (rj + 1) * rj // 6

    # Evaluation --------------------------------------------------------------
    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        (Jt @ q) @ q = jacobian(q).
        """
        r_i = self.substate_dimensions[self.indices[0]]
        r_j = self.substate_dimensions[self.indices[1]]
        Gt = CubicOperator.expand_entries(self.entries).reshape(
            (r_i, r_j, r_j, r_j)
        )
        self._prejac = Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        return self.entries @ np.prod(
            state[self.substate_slices[1]][self._submask],
            axis=1,
        )

    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        jac : (r_i, r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        if self._prejac is None:
            self._precompute_jacobian_jit()
        jac = np.zeros(
            (self.entries.shape[0], self.state_dimension),
            dtype=state.dtype,
        )
        q = state[self.substate_slices[1]]
        jac[:, self.substate_slices[1]] = (self._prejac @ q) @ q
        return jac


# Dependent on input but not on state =========================================
class InputOperatorMulti(OpInfOperatorMulti, InputMultiMixin):
    r"""Linear input operator for multivariable systems,
    :math:`(\qhat, \u) \mapsto \Bhat_{i,j}\u_{j}\in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_index : int
        Index :math:`i` of the substate this operator maps to.
    input_index : int or None
        Index :math:`j` of the subindex this operator acts on.
        If ``None``, the input is treated as monolithic.
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    subinput_dimensions : tuple of ints or None
        Dimensions of the subinputs.
        If not provided here, use :math:`set_subinput_dimensions`.
    entries : (r_i, r_j^2) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 1
    _num_input_indices = 1

    def __init__(
        self,
        state_index: int,
        input_index=None,
        substate_dimensions=None,
        subinput_dimensions=None,
        entries=None,
    ):
        InputMultiMixin.__init__(
            self,
            input_indices=input_index,
            subinput_dimensions=subinput_dimensions,
        )
        OpInfOperatorMulti.__init__(
            self,
            indices=(state_index,),
            substate_dimensions=substate_dimensions,
            entries=entries,
        )

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries

        # Ensure that the entries have valid state dimensions.
        if entries.ndim == 1:
            entries = entries.reshape((-1, 1))
        if entries.ndim != 2:
            raise ValueError(
                "InputOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")

        # Ensure that the entries have valid input dimensions.
        # If the subinput_dimensions is None, treat the input monolithically.
        m = entries.shape[1]
        if self.subinput_dimensions is None:
            self.set_subinput_dimensions((m,))
        if (_dim := entries.shape[1]) != (
            m_j := self.subinput_dimensions[self.input_indices[0]]
        ):
            raise ValueError(f"entries.shape[1] = {_dim} != {m_j} = m_j")

        super().set_entries(self, entries)

    @utils.requires("subinput_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix."""
        return self.subinput_dimensions[self.input_indices[0]]

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector (not used).
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        return self.entries @ np.atleast_1d(input_)[self.subinput_slices[0]]


# Dependent on both state and input ===========================================
class StateInputOperatorMulti(OpInfOperatorMulti, InputMultiMixin):
    r"""Bilinear state-input operator for multivariable systems,
    :math:`(\qhat, \u) \mapsto
    \Nhat_{i,jk}[\u_{k}\otimes\qhat_{j}]\in\mathbb{R}^{r_i}.`

    Parameters
    ----------
    state_indices : tuple of 2 ints
        Indices of the substate this operator maps to and acts on, i.e.,
        :math:`(i,j)`.
    input_index : int or None
        Index :math:`k` of the subindex this operator acts on.
        If ``None``, the input is treated as monolithic.
    substate_dimensions : tuple of ints or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    subinput_dimensions : tuple of ints or None
        Dimensions of the subinputs.
        If not provided here, use :math:`set_subinput_dimensions`.
    entries : (r_i, r_j^2) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    _num_state_indices = 2
    _num_input_indices = 1

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        if np.isscalar(entries) or np.shape(entries) == (1,):
            entries = np.atleast_2d(entries)
        # self._validate_entries

        # Ensure that the operator has valid dimensions.
        if entries.ndim != 2:
            raise ValueError(
                "StateInputOperatorMulti entries must be two-dimensional"
            )
        if (_dim := entries.shape[0]) != (
            r_i := self.substate_dimensions[self.state_indices[0]]
        ):
            raise ValueError(f"entries.shape[0] = {_dim} != {r_i} = r_i")

        r_j = self.substate_dimensions[self.state_indices[1]]
        m = entries.shape[1] // r_j
        if self.subinput_dimensions is None:
            self.set_subinput_dimensions((m,))
        if (_dim := entries.shape[1]) != (_d2 := self.operator_dimension()):
            raise ValueError(f"entries.shape[1] = {_dim} != {_d2} = r_j * m_k")

        super().set_entries(self, entries)

    @utils.requires("substate_dimensions")
    @utils.requires("subinput_dimensions")
    def operator_dimension(self):
        """Number of columns in the operator entries matrix, based solely on
        the state and/or input dimensions.
        """
        return (
            self.substate_dimensions[self.state_indices[0]]
            * self.subinput_dimensions[self.input_indices[1]]
        )

    @utils.requires("entries")
    def apply(self, state, input_):
        r"""Apply the operator to the given state / input.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector.

        Returns
        -------
        out : (r_i,) ndarray
            Application of the operator.
        """
        mult = la.khatri_rao
        if state.ndim == 1 or self.input_dimension == 1:
            mult = np.kron
        return self.entries @ mult(
            input_[self.subinput_slices[0]], state[self.substate_slices[1]]
        )

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        """Construct the state Jacobian of the operator.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        jac : (r_i, r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        u = np.atleast_1d(input_)
        jac = np.zeros(
            (self.entries.shape[0], self.state_dimension),
            dtype=state.dtype,
        )
        jac[:, self.substate_slices[1]] = np.sum(
            [
                um * Nm
                for um, Nm in zip(
                    u,
                    np.split(
                        self.entries,
                        self.subinput_dimensions[self.input_indices[0]],
                        axis=1,
                    ),
                )
            ],
            axis=0,
        )
        return jac
