# pre/test_shift_scale.py
"""Tests for rom_operator_inference.pre._shift_scale.py."""

import pytest
import itertools
import numpy as np

import rom_operator_inference as opinf


# Data preprocessing: shifting and MinMax scaling / unscaling =================
def test_shift(set_up_basis_data):
    """Test pre._shift_scale.shift()."""
    X = set_up_basis_data

    # Try with bad data shape.
    with pytest.raises(ValueError) as exc:
        opinf.pre.shift(np.random.random((3,3,3)))
    assert exc.value.args[0] == "data X must be two-dimensional"

    # Try with bad shift vector.
    with pytest.raises(ValueError) as exc:
        opinf.pre.shift(X, X)
    assert exc.value.args[0] == "shift_by must be one-dimensional"

    # Correct usage.
    Xshifted, xbar = opinf.pre.shift(X)
    assert xbar.shape == (X.shape[0],)
    assert Xshifted.shape == X.shape
    assert np.allclose(np.mean(Xshifted, axis=1), np.zeros(X.shape[0]))
    for j in range(X.shape[1]):
        assert np.allclose(Xshifted[:,j], X[:,j] - xbar)

    Y = np.random.random(X.shape)
    Yshifted = opinf.pre.shift(Y, xbar)
    for j in range(Y.shape[1]):
        assert np.allclose(Yshifted[:,j], Y[:,j] - xbar)

    # Verify inverse shifting.
    assert np.allclose(X, opinf.pre.shift(Xshifted, -xbar))


def test_scale(set_up_basis_data):
    """Test pre._shift_scale.scale()."""
    X = set_up_basis_data

    # Try with bad scales.
    with pytest.raises(ValueError) as exc:
        opinf.pre.scale(X, (1,2,3), (4,5))
    assert exc.value.args[0] == "scale_to must have exactly 2 elements"

    with pytest.raises(ValueError) as exc:
        opinf.pre.scale(X, (1,2), (3,4,5))
    assert exc.value.args[0] == "scale_from must have exactly 2 elements"

    # Scale X to [-1,1] and then scale Y with the same transformation.
    Xscaled, scaled_to, scaled_from = opinf.pre.scale(X, (-1,1))
    assert Xscaled.shape == X.shape
    assert scaled_to == (-1,1)
    assert isinstance(scaled_from, tuple)
    assert len(scaled_from) == 2
    assert round(scaled_from[0],8) == round(X.min(),8)
    assert round(scaled_from[1],8) == round(X.max(),8)
    assert round(Xscaled.min(),8) == -1
    assert round(Xscaled.max(),8) == 1

    # Verify inverse scaling.
    assert np.allclose(opinf.pre.scale(Xscaled, scaled_from, scaled_to), X)


class TestSnapshotTransformer:
    """Test pre.SnapshotTransformer."""
    def test_init(self):
        """Test pre.SnapshotTransformer.__init__()."""
        st = opinf.pre.SnapshotTransformer()
        for attr in ["scaling", "center", "verbose"]:
            assert hasattr(st, attr)

    def test_properties(self):
        """Test pre.SnapshotTransformer properties (attribute protection)."""
        st = opinf.pre.SnapshotTransformer()

        # Test center.
        with pytest.raises(TypeError) as exc:
            st.center = "nope"
        assert exc.value.args[0] == "'center' must be True or False"
        st.center = True
        st.center = False

        # Test scale.
        with pytest.raises(ValueError) as exc:
            st.scaling = "minimaxii"
        assert exc.value.args[0].startswith("invalid scaling 'minimaxii'")

        with pytest.raises(TypeError) as exc:
            st.scaling = [2, 1]
        assert exc.value.args[0] == "'scaling' must be of type 'str'"

        for s in st._VALID_SCALINGS:
            st.scaling = s
        st.scaling = None

    def test_fit_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.fit_transform()."""

        def fit_transform_copy(st, A):
            """Assert A and B are not the same object but do have the same
            type and shape.
            """
            B = st.fit_transform(A, inplace=False)
            assert B is not A
            assert type(B) is type(A)
            assert B.shape == A.shape
            return B

        st = opinf.pre.SnapshotTransformer(verbose=False)

        # Test null transformation.
        st.center = False
        st.scaling = None
        X = np.random.randint(0, 100, (n,k)).astype(float)
        Y = st.fit_transform(X, inplace=True)
        assert Y is X
        Y = fit_transform_copy(st, X)
        assert np.all(Y == X)

        # Test centering.
        st.center = True
        st.scaling = None
        Y = fit_transform_copy(st, X)
        assert hasattr(st, "mean_")
        assert isinstance(st.mean_, np.ndarray)
        assert st.mean_.shape == (X.shape[0],)
        assert np.allclose(np.mean(Y, axis=1), 0)

        # Test scaling (without and with centering).
        for centering in (False, True):
            st.center = centering

            # Test standard scaling.
            st.scaling = "standard"
            Y = fit_transform_copy(st, X)
            for attr in "scale_", "shift_":
                assert hasattr(st, attr)
                assert isinstance(getattr(st, attr), float)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.std(Y), 1)

            # Test min-max scaling.
            st.scaling = "minmax"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), 0)
            assert np.isclose(np.max(Y), 1)

            # Test symmetric min-max scaling.
            st.scaling = "minmaxsym"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.min(Y), -1)
            assert np.isclose(np.max(Y), 1)

            # Test maximum absolute scaling.
            st.scaling = "maxabs"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.max(np.abs(Y)), 1)

            # Test minimum-maximum absolute scaling.
            st.scaling = "maxabssym"
            Y = fit_transform_copy(st, X)
            assert np.isclose(np.mean(Y), 0)
            assert np.isclose(np.max(np.abs(Y)), 1)

    def test_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.transform()."""
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st = opinf.pre.SnapshotTransformer(verbose=False)

        # Test null transformation.
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st.fit_transform(X)
        Y = np.random.randint(0, 100, (n,k)).astype(float)
        Z = st.transform(Y, inplace=True)
        assert Z is Y
        Z = st.transform(Y, inplace=False)
        assert Z is not Y
        assert Z.shape == Y.shape
        assert np.all(Z == Y)

        # Test mean shift.
        st.center = True
        st.scaling = None
        st.fit_transform(X)
        µ = st.mean_
        Z = st.transform(Y, inplace=False)
        assert np.allclose(Z, Y - µ.reshape(-1,1))

        # Test each scaling.
        st.center = False
        for scl in st._VALID_SCALINGS:
            X = np.random.randint(0, 100, (n,k)).astype(float)
            Y = np.random.randint(0, 100, (n,k)).astype(float)
            st.scaling = scl
            st.fit_transform(X)
            a, b = st.scale_, st.shift_
            Z = st.transform(Y)
            assert np.allclose(Z, a*Y + b)

    def test_inverse_transform(self, n=200, k=50):
        """Test pre.SnapshotTransformer.inverse_transform()."""
        X = np.random.randint(0, 100, (n,k)).astype(float)
        st = opinf.pre.SnapshotTransformer(verbose=False)

        for scaling, center in itertools.product({None, *st._VALID_SCALINGS},
                                                 (True, False)):
            st.scaling = scaling
            st.center = center
            st.fit_transform(X, inplace=False)
            Y = np.random.randint(0, 100, (n,k)).astype(float)
            Z = st.transform(Y, inplace=False)
            st.inverse_transform(Z, inplace=True)
            assert np.allclose(Z, Y)


# class TestMultivariableSnapshotTransformer:
#     """Test pre.MultivariableSnapshotTransformer."""
#     def test_init(self):
#         """Test pre.MultivariableSnapshotTransformer.__init__()."""
#         st = opinf.pre.MultivariableSnapshotTransformer()
#         for attr in ["num_variables", "scale", "shift", "inplace",
#                      "variable_names", "weights", "verbose"]:
#             assert hasattr(st, attr)
#
#     def test_properties(self):
#         """Test pre.SnapshotTransformer properties (attribute protection)."""
#         st = opinf.pre.SnapshotTransformer()
#
#         # Test num_variables.
#         with pytest.raises(TypeError) as exc:
#             st.num_variables = 1.5
#         assert exc.value.args[0] == "num_variables must be an integer"
#
#         st.num_variables = 1
#         st.num_variables = 2
#         st.num_variables = 3.0
#
#         # Test variable_names.
#         st.num_variables = 1
#         with pytest.raises(ValueError) as exc:
#             st.variable_names = [0, 1]
#         assert exc.value.args[0] == \
#             "2 = len(variable_names) != num_variables = 1"
#         st.variable_names = "fred"
#
#         # Test scale.
#         st.num_variables = 1
#         with pytest.raises(ValueError) as exc:
#             st.scaling = "minimaxi"
#         assert exc.value.args[0] == "invalid scaling 'minimaxi'"
#
#         with pytest.raises(ValueError) as exc:
#             st.scaling = [2, 1]
#         assert exc.value.args[0] == "invalid scaling '[2, 1]'"
#
#         for s in list(st._VALID_SCALINGS) + [True, [-3, 7]]:
#             st.scaling = s
#             assert st.scaling == [s]
