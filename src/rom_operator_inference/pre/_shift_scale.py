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
    """Process snapshots by centering and/or scaling (in that order).

    Attributes (transformation hyperparameters)
    -------------------------------------------
    center : bool
        If True, shift the snapshots by the mean of the training snapshots.
    scaling : str or None
        If given, scale (non-dimensionalize) the centered snapshot entries.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
    verbose : bool
        If True, print information about the learned transformation.

    Notes
    -----
    Snapshot centering (center=True):
        X' = X - mean(X, axis=1);
        Guarantees mean(X', axis=1) = [0, ..., 0].
    Standard scaling (scaling='standard'):
        X' = (X - mean(X)) / std(X);
        Guarantees mean(X') = 0, std(X') = 1.
    Min-max scaling (scaling='minmax'):
        X' = (X - min(X))/(max(X) - min(X));
        Guarantees min(X') = 0, max(X') = 1.
    Symmetric min-max scaling  (scaling='minmaxsym'):
        X' = (X - min(X))*2/(max(X) - min(X)) - 1
        Guarantees min(X') = -1, max(X') = 1.
    Maximum absolute scaling:
        X' = X / max(abs(X));
        Guarantees mean(X') = mean(X) / max(abs(X)), max(abs(X')) = 1.
    Min-max absolute scaling:
        X' = (X - mean(X)) / max(abs(X - mean(X)));
        Guarantees mean(X') = 0, max(abs(X')) = 1.
    """
    _VALID_SCALINGS = {
        "standard",
        "minmax",
        "minmaxsym",
        "maxabs",
        "maxabssym",
    }

    _table_header = "    |     min    |    mean    |     max    |    std\n"
    _table_header += "----|------------|------------|------------|------------"

    def __init__(self, center=False, scaling=None, verbose=False):
        """Set transformation hyperparameters."""
        self.center = center
        self.scaling = scaling
        self.verbose = verbose

    def _clear(self):
        """Delete all learned attributes."""
        for attr in ("scale_", "shift_", "mean_"):
            if hasattr(self, attr):
                delattr(self, attr)

    # Properties --------------------------------------------------------------
    @property
    def center(self):
        """Snapshot mean-centering directive (bool)."""
        return self.__center

    @center.setter
    def center(self, ctr):
        """Set the centering directive, resetting the transformation."""
        if ctr not in (True, False):
            raise TypeError("'center' must be True or False")
        self._clear()
        self.__center = ctr

    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0,1].
        * 'minmaxsym': minmax scaling to [-1,1].
        * 'maxabs': maximum absolute scaling to [-1,1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1,1] (mean shift).
        """
        return self.__scaling

    @scaling.setter
    def scaling(self, scl):
        """Set the scaling strategy, resetting the transformation."""
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

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation: scaling type + centering bool."""
        out = ["Snapshot transformer"]
        if self.center:
            out.append("with mean-snapshot centering")
            if self.scaling:
                out.append(f"and '{self.scaling}' scaling")
        elif self.scaling:
            out.append(f"with '{self.scaling}' scaling")
        return ' '.join(out)

    @staticmethod
    def _statistics_report(X):
        """Return a string of basis statistics about a data set."""
        return " | ".join([f"{f(X):>10.3e}"
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

        # Record statistics of the training data.
        if self.verbose:
            report = ["No transformation learned"]
            report.append(self._table_header)
            report.append(f"X   | {self._statistics_report(X)}")

        # Center the snapshots by the mean training snapshot.
        if self.center:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1,1))

            if self.verbose:
                report[0] = "Learned mean centering X -> X'"
                report.append(f"X'  | {self._statistics_report(Y)}")

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling:
            # Standard: X' = (X - µ)/σ
            if self.scaling == "standard":
                µ = np.mean(Y)
                σ = np.std(Y)
                self.scale_ = 1/σ
                self.shift_ = -µ*self.scale_

            # Min-max: X' = (X - min(X))/(max(X) - min(X))
            elif self.scaling == "minmax":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 1/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_

            # Symmetric min-max: X' = (X - min(X))*2/(max(X) - min(X)) - 1
            elif self.scaling == "minmaxsym":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 2/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_ - 1

            # MaxAbs: X' = X / max(abs(X))
            elif self.scaling == "maxabs":
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = 0

            # maxabssym: X' = (X - mean(X)) / max(abs(X - mean(X)))
            elif self.scaling == "maxabssym":
                µ = np.mean(Y)
                Y -= µ
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = -µ*self.scale_
                Y += µ

            else:
                raise RuntimeError(f"invalid scaling '{self.scaling}'")

            Y *= self.scale_
            Y += self.shift_

            if self.verbose:
                if self.center:
                    report[0] += f" and {self.scaling} scaling X' -> X''"
                else:
                    report[0] = f"Learned {self.scaling} scaling X -> X''"
                report.append(f"X'' | {self._statistics_report(Y)}")

        if self.verbose:
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

        # Center the snapshots by the mean training snapshot.
        if self.center is True:
            Y -= self.mean_.reshape((-1,1))

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling is not None:
            Y *= self.scale_
            Y += self.shift_

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

        # Unscale (re-dimensionalize) the data.
        if self.scaling:
            Y -= self.shift_
            Y /= self.scale_

        # Uncenter the unscaled snapshots.
        if self.center:
            Y += self.mean_.reshape((-1,1))

        return Y


# class MultivariableSnapshotTransformer:
#     """Convenience class for processing snapshots by scaling variables
#     (non-dimensionalization) and/or shifting by the mean snapshot.
#     """
#     _VALID_SCALES = ("minmax", "minmaxsym", "minmaxpos", "standard")
#
#     def __init__(self, num_variables=1, variable_names=None,
#                  scale=None, shift=False, verbose=0):
#         """Set transformation hyperparameters.
#
#         Parameters
#         ----------
#         num_variables : int ≥ 1
#             Number of distinct variables represented by the snapshots.
#             If the snapshots have n rows, then n must be evenly divisible by
#             num_variables. Then dof = n / num_variables is assumed to be the
#             number of degrees of freedom for each variables.
#
#         variable_names : list, tuple, or None
#             Labels for the distinct variables represented by the snapshots.
#             Must have length `num_variables`. If None, default to
#             (0, 1, ..., num_variables - 1).
#
#         scale : bool, str, or None
#             If given, scale (non-dimensionalize) each variable entrywise
#             * [a,b] : min-max scaling to [a,b]
#             * 'minmax' or 'minmaxsym' or [-1,1]: minmax scaling to [-1,1]
#             * 'minmaxpos' or [0,1]: min-max scaling to [0,1].
#             * 'standard' or True: normalize variable to have zero mean and
#                 unit standard deviation.
#             * list of length `num_variables`: scaling pattern for each of the
#                 num_variables in the list, e.g., [[0,1], [-1,1], 'standard']
#                scales variable 0 to [0,1], variable 1 to [-1,1], and variable
#                 2 has a standard scaling.
#
#         shift : bool
#             If given, shift the (scaled) snapshots by the mean snapshot.
#
#         verbose : int
#             Verbosity level. Options:
#             * 0 (default): no printouts.
#             * 1: some printouts.
#             * 2: lots of printouts.
#         """
#         self.num_variables = num_variables
#         self.variable_names = variable_names
#         self.scaling = scale
#         self.shift = shift
#         self.verbose = verbose
#
#     def _clear(self):
#         """Delete all learned attributes."""
#         for attr in self.__dict__:
#             if attr.endswith('_'):
#                 delattr(self, attr)
#
#     # Properties ------------------------------------------------------------
#     @property
#     def num_variables(self):
#         """Number of distinct variables represented by the snapshots.
#         If the snapshots have n rows, then n must be evenly divisible by
#         num_variables. Then dof = n / num_variables is assumed to be the
#         number of degrees of freedom for each variables.
#         """
#         return self.__nv
#
#     @num_variables.setter
#     def num_variables(self, nv):
#         """Set the number of variables, clearing the transformation."""
#         if (nv // 1) != nv:
#             raise TypeError("num_variables must be an integer")
#         self.clear()
#         self.__nv = nv
#
#     @property
#     def variable_names(self):
#         """Labels for the distinct variables represented by the snapshots.
#         Must have length `num_variables`.
#         """
#         return self.__names
#
#     @variable_names.setter
#     def variables_names(self, names):
#         if names is None:
#             self.__name = tuple(range(self.num_variables))
#         if not isinstance(names, (int,tuple)):
#             raise TypeError("variables_names must be list or tuple")
#         if len(names) != self.num_variables:
#             raise ValueError(f"{len(variable_names)} = len(variable_names) "
#                              f"!= num_variables = {num_variables}")
#         self.__name = variables_names
#
#     def _check_single_scale(self, s):
#         """Validate an individual scaling directive."""
#         if isinstance(s, str):
#             if s not in self._VALID_SCALES:
#                 raise ValueError(f"invalid scaling '{s}'")
#         elif isinstance(s, (list, tuple)):
#             if len(s) != 2 or s[0] >= s[1]:
#                 raise ValueError(f"invalid scaling '{s}'")
#         elif s is not True:
#             raise TypeError(f"invalid scaling '{s}' of type '{type(s)}'")
#         return s
#
#     @property
#     def scale(self):
#         """Scaling (non-dimensionalize) directives for each variable.
#         * [a,b] : min-max scaling to [a,b]
#         * 'minmax' or 'minmaxsym' or [-1,1]: minmax scaling to [-1,1]
#         * 'minmaxpos' or [0,1]: min-max scaling to [0,1].
#         * 'standard': normalize variable to have zero mean and
#             unit standard deviation.
#         * list of length `num_variables`: scaling pattern for each of the
#             num_variables in the list, e.g., [[0,1], [-1,1], 'standard']
#             scales variable 0 to [0,1], variable 1 to [-1,1], and variable
#             2 has a standard scaling.
#         """
#         return self.__scale
#
#     @scale.setter
#     def scale(self, scale):
#         """Set the scale."""
#         if scale is None:
#             self.__scale = scale
#             return
#
#         if self.num_variables == 1 and (isinstance(scale, (list, tuple))
#                                         and len(scale) == 2):
#             scale = [scale]
#         elif not (isinstance(scale, (list, tuple))
#                   and len(scale) != self.num_variables):
#             scale = [scale]
#
#         scales = [self._check_single_scale(s) for s in scale]
#         if len(scales) != self.num_variables:
#             raise ValueError(f"expected {self.num_variables} scales, "
#                              f"got {len(scales)}")
#         self.__scale = scales
#
#     def fit_transform(self, X, weights=None, inplace=False):
#         """Learn the transformation from a collection of snapshots.
#
#         Parameters
#         ----------
#         X : (n,k) ndarray
#             Matrix of k snapshots. Each column is a snapshot of dimension n.
#
#         weights: (k,) ndarray or None
#             Weights for determining the mean shift.
#
#         inplace : bool
#             If True, overwrite the input data during transformation.
#             If False, create a copy of the data to transform.
#
#         Returns
#         -------
#         self
#         """
#         n, r = X.shape
#
#         # Scale (non-dimensionalize) each variable.
#         if self.scaling:
#             variables = np.split(X, self.num_variables, axis=0)
#         if self.scaling == "standard":
#             self.means_ = np.array([np.mean(v) for v in variables])
#             self.stds_ = np.array([np.std(v) for v in variables])
#         elif self.scaling == "minmax":
#             self.mins_ = np.array([np.min(v) for v in variables])
#             self.maxs_ = np.array([np.max(v) for v in variables])
#
#         # TODO: Shift (center) the scaled snapshots.
#
#         return X
#
#         # Determine whether learning the scaling transformation is needed.
#         learning = (scales is None)
#         if learning:
#             if variables is not None:
#                 raise ValueError("scale=None only valid for variables=None")
#             scales = np.empty((config.NUM_ROMVARS, 2), dtype=np.float)
#         else:
#             # Validate the scales.
#             _shape = (config.NUM_ROMVARS, 2)
#             if scales.shape != _shape:
#                 raise ValueError(f"`scales` must have shape {_shape}")
#
#         # Parse the variables.
#         if variables is None:
#             variables = config.ROM_VARIABLES
#         elif isinstance(variables, str):
#             variables = [variables]
#         varindices = [config.ROM_VARIABLES.index(v) for v in variables]
#
#         # Make sure the data can be split correctly by variable.
#         nchunks = len(variables)
#         chunksize, remainder = divmod(data.shape[0], nchunks)
#         if remainder != 0:
#             raise ValueError("data to scale cannot be split"
#                              f" evenly into {nchunks} chunks")
#
#         # Do the scaling by variable.
#         for i,vidx in enumerate(varindices):
#             s = slice(i*chunksize,(i+1)*chunksize)
#             if learning:
#                 assert i == vidx
#                 if variables[i] in ["p", "T", "xi"]:
#                     scales[vidx,0] = np.mean(data[s])
#                     shifted = data[s] - scales[vidx,0]
#                 else:
#                     scales[vidx,0] = 0
#                     shifted = data[s]
#                 scales[vidx,1] = np.abs(shifted).max()
#                 data[s] = shifted / scales[vidx,1]
#             else:
#                 data[s] = (data[s] - scales[vidx,0]) / scales[vidx,1]
#
#         # Report info on the learned scaling.
#         if verbose >= 1:
#             sep = '|'.join(['-'*12]*2)
#             report = f"""Learned new scaling
#                            Shift    |    Denom
#                         {sep}
#         Pressure        {scales[0,0]:<12.3e}|{scales[0,1]:>12.3e}
#                         {sep}
#         x-velocity      {scales[1,0]:<12.3f}|{scales[1,1]:>12.3f}
#                         {sep}
#         y-velocity      {scales[2,0]:<12.3f}|{scales[2,1]:>12.3f}
#                         {sep}
#         Temperature     {scales[3,0]:<12.3e}|{scales[3,1]:>12.3e}
#                         {sep}
#         Specific Volume {scales[4,0]:<12.3f}|{scales[4,1]:>12.3f}
#                         {sep}
#         CH4 molar       {scales[5,0]:<12.3f}|{scales[5,1]:>12.3f}
#                         {sep}
#         O2  molar       {scales[6,0]:<12.3f}|{scales[6,1]:>12.3f}
#                         {sep}
#         H2O molar       {scales[8,0]:<12.3f}|{scales[8,1]:>12.3f}
#                         {sep}
#         CO2 molar       {scales[7,0]:<12.3f}|{scales[7,1]:>12.3f}
#                         {sep}"""
#             logging.info(report)
#
#         return data, scales
#
#     def transform(self, X):
#         return X
#
#     def untransform(self, X):
#         return X


# Deprecations ================================================================

def mean_shift(X):                              # pragma nocover
    np.warnings.warn("mean_shift() has been renamed shift()",
                     DeprecationWarning, stacklevel=1)
    a,b = shift(X)
    return b,a


mean_shift.__doc__ = "\nDEPRECATED! use shift().\n\n" + shift.__doc__
