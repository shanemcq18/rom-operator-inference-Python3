# operators.py
r"""OpInf operators for multilithic systems."""

__all__ = [
    "OperatorMulti",
    "OpInfOperatorMulti",
]

import abc
import numpy as np

from ... import utils


# Base classes ================================================================
class OperatorMulti(abc.ABC):
    r"""Base class for operators in multivariable systems.

    These operators operate on states that have a natural partition into
    substates,

    .. math::
        \qhat = \left[\begin{array}{c}
        \qhat_0 \\ \qhat_1 \\ \vdots \\ \qhat_{n_q - 1}
        \end{array}\right]
        \in \RR^r,

    typically arising from using a block-diagonal basis on multivariate data.
    Here, :math:`\qhat_i \in \mathbb{R}^{r_i}` for :math:`i = 0,\ldots,n_q-1`
    and :math:`r = \sum_{i=0}^{n_q - 1} r_i`.

    Parameters
    ----------
    state_indices : tuple(int)
        Indices of the substates this operator is involved with.
        The 0-th index is the substate that this operator maps to, meaning
        the output has the same dimension as the substate with that index.
        The remaining indices are the substates that this operator acts on.
    substate_dimensions : tuple(int) or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    """

    _num_state_indices = NotImplemented  # Required number of state indices.

    def __init__(
        self,
        state_indices: tuple,
        substate_dimensions: tuple = None,
    ):
        # Store the state indices.
        if isinstance(state_indices, int):
            state_indices = (state_indices,)
        if len(state_indices) != self._num_state_indices:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly "
                f"{self._num_state_indices} state_indices"
            )
        self.__state_indices = tuple(state_indices)

        # Store the state dimensions or initialize empty attributes.
        self.set_substate_dimensions(substate_dimensions)

    # Properties --------------------------------------------------------------
    @property
    def state_indices(self) -> tuple:
        """Indices of the substates this operator maps to and acts on."""
        return self.__state_indices

    def set_substate_dimensions(self, substate_dimensions):
        """Set the substate dimensions."""
        if substate_dimensions is None:
            self.__rs = None
            self.__r = None
            self.__nsubstates = None
            self.__stateslices = None
        else:
            self.__rs = tuple([int(r) for r in substate_dimensions])
            dimsum = np.cumsum((0,) + self.__rs)
            self.__r = int(dimsum[-1])
            self.__nsubstates = len(substate_dimensions)
            self.__stateslices = tuple(
                [slice(dimsum[i], dimsum[i + 1]) for i in self.state_indices]
            )

    @property
    def substate_dimensions(self):
        r"""Dimensions of the substates, :math:`r_0,r_1,\ldots,r_{n_q-1}`."""
        return self.__rs

    @property
    def state_dimension(self):
        r"""Total dimension of the state, :math:`r = \sum_{i=0}^{n_q-1}r_i`."""
        return self.__r

    @property
    def num_substates(self):
        """Number of substates :math:`n_q`."""
        return self.__nsubstates

    @property
    def substate_slices(self):
        """Slices to the substates that this operator maps to and acts on."""
        return self.__stateslices

    # Evaluation --------------------------------------------------------------
    @utils.requires("entries")
    @abc.abstractmethod
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input.

        Note that this method receives the full state :math:`\qhat` but returns
        a quantity that matches one of the substates.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray
            Input vector (not used).

        Returns
        -------
        out : (entries.shape[0],) ndarray
            Application of the operator.
        """
        raise NotImplementedError

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
        jac : (entries.shape[0], r) ndarray
            State Jacobian for the block this operator belongs to.
        """
        return 0


class InputMultiMixin:
    r"""Mixin for operators in systems with multivariate inputs.

    These operators operate on inputs that have a natural partition into
    subinputs,

    .. math::
        \u = \left[\begin{array}{c}
        \u_0 \\ \u_1 \\ \vdots \\ \u_{n_u - 1}
        \end{array}\right]
        \in \RR^m,

    for example arising from using a block-diagonal basis on multivariate data
    and having several input sources.
    Here, :math:`\u_i \in \mathbb{R}^{m_i}` for :math:`i = 0,\ldots,n_u-1`
    and :math:`m = \sum_{i=0}^{n_u - 1} m_i`.

    Parameters
    ----------
    input_indices : tuple(int) or None
        Indices of the subinputs this operator acts on.
        If ``None``, treat the input as monolithic.
    subinput_dimensions : tuple(int) or None
        Dimensions of the subinputs.
        If not provided here, use :meth:`set_subinput_dimensions`.
    """

    _num_input_indices = NotImplemented  # Required number of input indices.

    def __init__(self, input_indices=None, subinput_dimensions=None):
        if input_indices is None:
            input_indices = 0
        if isinstance(input_indices, int):
            input_indices = (input_indices,)
        self.__input_indices = tuple(input_indices)
        if len(input_indices) != self._num_input_indices:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly "
                f"{self._num_input_indices} input_indices"
            )

        self.set_subinput_dimensions(subinput_dimensions)

    # Properties --------------------------------------------------------------
    @property
    def input_indices(self) -> tuple:
        """Indices of the subinputs this operator acts on."""
        return self.__input_indices

    def set_subinput_dimensions(self, subinput_dimensions):
        """Set the subinput dimensions."""
        if subinput_dimensions is None:
            self.__ms = None
            self.__m = None
            self.__nsubinputs = None
            self.__inputslices = None
        else:
            self.__ms = tuple([int(m) for m in subinput_dimensions])
            dimsum = np.cumsum((0,) + self.__ms)
            self.__m = int(dimsum[-1])
            self.__nsubinputs = len(subinput_dimensions)
            self.__inputslices = tuple(
                [slice(dimsum[i], dimsum[i + 1]) for i in self.input_indices]
            )

    @property
    def subinput_dimensions(self):
        r"""Dimensions of the subinputs, :math:`m_0,m_1,\ldots,r_{n_u-1}`."""
        return self.__ms

    @property
    def input_dimension(self):
        r"""Total dimension of the input, :math:`m = \sum_{i=0}^{n_u-1}m_i`."""
        return self.__m

    @property
    def num_subinputs(self):
        """Number of subinputs :math:`n_m`."""
        return self.__nsubinputs

    @property
    def subinput_slices(self):
        """Slices to the subinputs that this operator acts on."""
        return self.__inputslices


class OpInfOperatorMulti(OperatorMulti):
    r"""Template for nonparametric multilithic operators that can be calibrated
    through Operator Inference.

    These operators operate on states that have a natural partition into
    substates,

    .. math::
        \qhat = \left[\begin{array}{c}
        \qhat_0 \\ \qhat_1 \\ \vdots \\ \qhat_{n_q - 1}
        \end{array}\right]
        \in \RR^r,

    typically arising from using a block-diagonal basis on multivariate data.
    Here, :math:`\qhat_i \in \mathbb{R}^{r_i}` for :math:`i = 0,\ldots,n_q-1`
    and :math:`r = \sum_{i=0}^{n_q - 1} r_i`.

    These operators can be written as a matrix-vector product
    :math:`\Ophat_{\ell}(\qhat, \u) = \Ohat_{\ell}\d(\qhat, \u)`, where the
    first dimension of the entries matrix :math:`\Ophat_{\ell}` is
    ``substate_dimensions[state_indices[0]]``.

    Parameters
    ----------
    state_indices : tuple(int)
        Indices of the substates this operator maps to and acts on.
    substate_dimensions : tuple(int) or None
        Dimensions of the substates.
        If not provided here, use :meth:`set_substate_dimensions`.
    entries : (r_ell, something) ndarray or None
        Operator entries. If not provided here, use :meth:`set_entries`.
    """

    def __init__(
        self,
        state_indices: tuple,
        substate_dimensions: tuple = None,
        entries=None,
    ):
        super().__init__(
            state_indices=state_indices,
            substate_dimensions=substate_dimensions,
        )

        # Initialize the operator entries and set them if provided.
        self._clear()
        if entries is not None:
            self.set_entries(entries)

    # Entries -----------------------------------------------------------------
    def _clear(self):
        """Clear the operator entries."""
        self.__entries = None

    @property
    def entries(self):
        """Operator entries, an array with
        ``shape[0] == substate_dimensions[state_indices[0]]`` and ``shape[1]``
        depending on the other state dimensions and indices.
        """
        return self.__entries

    @utils.requires("substate_dimensions")
    def set_entries(self, entries):
        """Set the operator entries."""
        self.__entries = entries

    # Inference ---------------------------------------------------------------
    @utils.requires("substate_dimensions")
    @abc.abstractmethod
    def operator_dimension(self) -> int:
        """Number of columns in the operator entries matrix."""
        raise NotImplementedError

    @utils.requires("substate_dimensions")
    @abc.abstractmethod
    def datablock(states: np.ndarray, inputs=None) -> np.ndarray:
        """Construct the data matrix block corresponding to the operator."""
        raise NotImplementedError
