# pre/_shift_scale.py
"""Tools for preprocessing data."""

__all__ = [
            "shift",
            "scale",
            "SnapshotTransformer"
          ]

import numpy as np


# Shifting and MinMax scaling =================================================
def shift(X, shift_by=None):
    """Shift the columns of X by a vector.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots. Each column is a single snapshot.
    shift_by : (n,) or (n,1) ndarray
        A vector that is the same size as a single snapshot. If None,
        set to the mean of the columns of X.

    Returns
    -------
    Xshifted : (n,k) ndarray
        The matrix such that Xshifted[:,j] = X[:,j] - shift_by for j=0,...,k-1.
    xbar : (n,) ndarray
        The shift factor. Since this is a one-dimensional array, it must be
        reshaped to be applied to a matrix: Xshifted + xbar.reshape((-1,1)).
        Only returned if shift_by=None.

    Examples
    --------
    # Shift X by its mean, then shift Y by the same mean.
    >>> Xshifted, xbar = pre.shift(X)
    >>> Yshifted = pre.shift(Y, xbar)

    # Shift X by its mean, then undo the transformation by an inverse shift.
    >>> Xshifted, xbar = pre.shift(X)
    >>> X_again = pre.shift(Xshifted, -xbar)
    """
    # Check dimensions.
    if X.ndim != 2:
        raise ValueError("data X must be two-dimensional")

    # If not shift_by factor is provided, compute the mean column.
    learning = (shift_by is None)
    if learning:
        shift_by = np.mean(X, axis=1)
    elif shift_by.ndim != 1:
        raise ValueError("shift_by must be one-dimensional")

    # Shift the columns by the mean.
    Xshifted = X - shift_by.reshape((-1,1))

    return (Xshifted, shift_by) if learning else Xshifted


def scale(X, scale_to, scale_from=None):
    """Scale the entries of the snapshot matrix X from the interval
    [scale_from[0], scale_from[1]] to [scale_to[0], scale_to[1]].
    Scaling algorithm follows sklearn.preprocessing.MinMaxScaler.

    Parameters
    ----------
    X : (n,k) ndarray
        A matrix of k snapshots to be scaled. Each column is a single snapshot.
    scale_to : (2,) tuple
        The desired minimum and maximum of the scaled data.
    scale_from : (2,) tuple
        The minimum and maximum of the snapshot data. If None, learn the
        scaling from X: scale_from[0] = min(X); scale_from[1] = max(X).

    Returns
    -------
    Xscaled : (n,k) ndarray
        The scaled snapshot matrix.
    scaled_to : (2,) tuple
        The bounds that the snapshot matrix was scaled to, i.e.,
        scaled_to[0] = min(Xscaled); scaled_to[1] = max(Xscaled).
        Only returned if scale_from = None.
    scaled_from : (2,) tuple
        The minimum and maximum of the snapshot data, i.e., the bounds that
        the data was scaled from. Only returned if scale_from = None.

    Examples
    --------
    # Scale X to [-1,1] and then scale Y with the same transformation.
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (-1,1))
    >>> Yscaled = pre.scale(Y, scaled_to, scaled_from)

    # Scale X to [0,1], then undo the transformation by an inverse scaling.
    >>> Xscaled, scaled_to, scaled_from = pre.scale(X, (0,1))
    >>> X_again = pre.scale(Xscaled, scaled_from, scaled_to)
    """
    # If no scale_from bounds are provided, learn them.
    learning = (scale_from is None)
    if learning:
        scale_from = np.min(X), np.max(X)

    # Check scales.
    if len(scale_to) != 2:
        raise ValueError("scale_to must have exactly 2 elements")
    if len(scale_from) != 2:
        raise ValueError("scale_from must have exactly 2 elements")

    # Do the scaling.
    mini, maxi = scale_to
    xmin, xmax = scale_from
    scl = (maxi - mini)/(xmax - xmin)
    Xscaled = X*scl + (mini - xmin*scl)

    return (Xscaled, scale_to, scale_from) if learning else Xscaled


class SnapshotTransformer:
    """Process snapshots by scaling and/or centering."""
    _VALID_SCALINGS = {
        "standard",
        "minmax",
        "symminmax",
        "maxabs",
        "minmaxabs",
    }

    def __init__(self, scaling=None, center=False, verbose=0):
        """Set transformation hyperparameters.

        Parameters
        ----------
        scaling : str or None
            If given, scale (non-dimensionalize) the snapshot data entrywise.
            * 'standard': standardize to zero mean and unit standard deviation.
            * 'minmax': minmax scaling to [0,1].
            * 'symminmax': minmax scaling to [-1,1].
            * 'maxabs': absolute maximum scaling to [-1,1] (no shift).
            * 'minmaxabs': absolute min-max scaling to [-1,1] (mean shift).
        center : bool
            If True, shift the (scaled) snapshots by the mean snapshot.
        verbose : int
            Verbosity level. Options:
            * 0 (default): no printouts.
            * 1: some printouts.
            * 2: lots of printouts.

        Notes
        -----
        Standard scaling:
            X' = (X - mean(X)) / std(X);
            Guarantees mean(X') = 0, std(X') = 1.
        Min-max scaling:
            X' = (X - min(X))/(max(X) - min(X));
            Guarantees min(X') = 0, max(X') = 1.
        Symmetric min-max scaling:
            X' = (X - min(X))*2/(max(X) - min(X)) - 1
            Guarantees min(X') = -1, max(X') = 1.
        Maximum absolute scaling:
            X' = X / max(abs(X));
            Guarantees mean(X') = mean(X) / max(abs(X)), max(abs(X')) = 1.
        Min-max absolute scaling:
            X' = (X - mean(X)) / max(abs(X - mean(X)));
            Guarantees mean(X') = 0, max(abs(X')) = 1.
        Centering:
            X'' = X' - mean(X', axis=1);
            Guarantees mean(X'', axis=1) = [0, ..., 0].
        """
        self.scaling = scaling
        self.center = center
        self.verbose = verbose

    def _clear(self):
        """Delete all learned attributes."""
        for attr in ("scale_", "shift_", "mean_"):
            if hasattr(self, attr):
                delattr(self, attr)

    # Properties --------------------------------------------------------------
    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'symminmax': minmax scaling to [-1,1].
        * 'maxabs': absolute maximum scaling to [-1,1] (no shift).
        * 'minmaxabs': absolute min-max scaling to [-1,1] (mean shift).
        """
        return self.__scaling

    @scaling.setter
    def scaling(self, scl):
        """Set the scaling strategy, checking for validity."""
        if scl is None:
            self._clear()
            self.__scaling = scl
            return
        if not isinstance(scl, str):
            raise TypeError("'scaling' must be of type 'str'")
        if scl not in self._VALID_SCALINGS:
            opts = ", ".join([f"'{v}'" for v in self._VALID_SCALINGS])
            raise ValueError(f"invalid scaling '{scl}'; "
                             f"valid options are {opts}")
        self._clear()
        self.__scaling = scl

    # Reporting ---------------------------------------------------------------
    @staticmethod
    def _statistics_report(X):
        """Return a string of basis statistics about a data set."""
        return " | ".join([f"{f(X):<10.3e}"
                           for f in (np.min, np.mean, np.max, np.std)])

    # Persistence -------------------------------------------------------------
    def save(self, filename, overwrite=False):
        """Save the current transformer to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path of the file to save the transformer in.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        """Load a SnapshotTransformer from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
            SnapshotTransformer
        """
        raise NotImplementedError

    # Main routines -----------------------------------------------------------
    def fit_transform(self, X, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        Y = X if inplace else X.copy()

        # Scale (non-dimensionalize) each variable.
        if self.scaling:
            # Standard: X' = (X - µ)/σ
            if self.scaling == "standard":
                µ = np.mean(X)
                σ = np.std(X)
                self.scale_ = 1/σ
                self.shift_ = -µ*self.scale_

            # Min-max: X' = (X - min(X))/(max(X) - min(X))
            elif self.scaling == "minmax":
                Xmin = np.min(X)
                Xmax = np.max(X)
                self.scale_ = 1/(Xmax - Xmin)
                self.shift_ = -Xmin*self.scale_

            # Symmetric min-max: X' = (X - min(X))*2/(max(X) - min(X)) - 1
            elif self.scaling == "symminmax":
                Xmin = np.min(X)
                Xmax = np.max(X)
                self.scale_ = 2/(Xmax - Xmin)
                self.shift_ = -Xmin*self.scale_ - 1

            # MaxAbs: X' = X / max(abs(X))
            elif self.scaling == "maxabs":
                self.scale_ = 1/np.max(np.abs(X))
                self.shift_ = 0

            # MinMaxAbs: X' = (X - mean(X)) / max(abs(X - mean(X)))
            elif self.scaling == "minmaxabs":
                µ = np.mean(X)
                Y -= µ
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = -µ*self.scale_
                Y += µ

            Y *= self.scale_
            Y += self.shift_

        # Center the scaled snapshots.
        if self.center:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1,1))

        if self.verbose >= 1:
            if self.scaling is None and self.center is False:
                report = ["No scaling learned!"]
            else:
                report = [f"Learned {self.scaling} scaling X -> X'"]
            report.append("   |     min    |    mean    |     max    |    std")
            sep = '-'*12
            report.append(f"---|{sep}|{sep}|{sep}|{sep}")
            report.append(f"X  | {self._statistics_report(X)}")
            report.append(f"X' | {self._statistics_report(Y)}")
            print('\n'.join(report))

        return Y

    def transform(self, X, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        Y = X if inplace else X.copy()

        # Scale (non-dimensionalize) each variable.
        if self.scaling:
            Y *= self.scale_
            Y += self.shift_

        # Shift (center) the scaled snapshots.
        if self.center:
            Y -= self.mean_.reshape((-1,1))

        return Y

    def inverse_transform(self, X, inplace=False):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        X : (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.

        Returns
        -------
        X'': (n,k) ndarray
            Matrix of k untransformed n-dimensional snapshots.
        """
        Y = X if inplace else X.copy()

        # Shift (uncenter) the scaled snapshots.
        if self.center:
            Y += self.mean_.reshape((-1,1))

        # Unscale (re-dimensionalize) the unshifted snapshots.
        if self.scaling:
            Y -= self.shift_
            Y /= self.scale_

        return Y


# Deprecations ================================================================

def mean_shift(X):                              # pragma nocover
    np.warnings.warn("mean_shift() has been renamed shift()",
                     DeprecationWarning, stacklevel=1)
    a,b = shift(X)
    return b,a


mean_shift.__doc__ = "\nDEPRECATED! use shift().\n\n" + shift.__doc__
