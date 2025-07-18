# basis/test_linear.py
"""Tests for basis._linear."""

import os
import pytest
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import opinf

try:
    from .test_base import _TestBasisTemplate
except ImportError:
    from test_base import _TestBasisTemplate


class TestLinearBasis(_TestBasisTemplate):
    """Test basis._linear.LinearBasis."""

    # Setup -------------------------------------------------------------------
    Basis = opinf.basis.LinearBasis

    def _orth(self, n, r):
        return la.svd(np.random.random((n, r)), full_matrices=False)[0]

    def get_bases(self):
        Vr = self._orth(20, 4)
        yield self.Basis(Vr), 20

        w = np.random.random(50)
        w[0] = 0.25
        w[1] = 0.25
        Vr = np.zeros((50, 2))
        Vr[0, 0] = 2
        Vr[1, 1] = 2
        yield self.Basis(Vr, weights=w, check_orthogonality=True), 50

    # Tests -------------------------------------------------------------------
    def test_init(self, n=10, r=3):
        """Test __init__(), fit(), and entries."""
        # Empty basis.
        basis = self.Basis(None, name="something")
        assert basis.entries is None
        assert basis.weights is None
        assert basis.full_state_dimension is None
        assert basis.reduced_state_dimension is None
        assert basis.name == "something"

        # Check that compress() and decompress() are disabled.
        for method in basis.compress, basis.decompress:
            with pytest.raises(AttributeError) as ex:
                method(0)
            assert ex.value.args[0] == "basis entries not initialized"

        # Ensure the state dimensions cannot be set.
        for attr in "full_state_dimension", "reduced_state_dimension":
            with pytest.raises(AttributeError) as ex:
                setattr(basis, attr, 10)
            assert getattr(basis, attr) is None

        # Orthogonal basis.
        Vr = self._orth(n, r)
        basis = self.Basis(Vr)
        assert basis.entries is Vr
        assert basis.shape == (n, r)
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert np.all(basis[:] == Vr)
        assert basis.weights is None
        assert basis.name is None

        # Non-orthogonal basis.
        Vr = np.ones((n, r))
        with pytest.warns(opinf.errors.OpInfWarning) as wn:
            basis = self.Basis(Vr)
        assert wn[0].message.args[0] == "basis not orthogonal"
        assert basis.full_state_dimension == n
        assert basis.reduced_state_dimension == r
        assert np.all(basis.entries == 1)
        assert basis.weights is None
        assert basis.fit() is basis

        # Weighted basis.
        w = 1 + np.random.random(n)
        wrootinv = 1 / np.sqrt(w)
        Vr_euc = la.qr(np.random.random((n, r)), mode="economic")[0]
        Vr = wrootinv.reshape((-1, 1)) * Vr_euc
        basis1 = self.Basis(
            entries=Vr,
            weights=w,
            check_orthogonality=True,
        )
        basis2 = self.Basis(
            entries=Vr,
            weights=np.diag(w),
            check_orthogonality=True,
        )
        assert np.all(basis1.entries == basis2.entries)
        assert np.all(basis2.entries == Vr)

        with pytest.raises(ValueError) as ex:
            self.Basis(
                entries=Vr,
                weights=np.random.random((2, 2, 2)),
                check_orthogonality=False,
            )
        assert ex.value.args[0] == "expected one- or two-dimensional weights"

    def test_plot1D(self, n=20, r=4):
        """Lightly test plot1D()."""
        basis = self.Basis(self._orth(n, r))

        # Turn interactive mode on.
        _pltio = plt.isinteractive()
        plt.ion()

        # Call the plotting routine.
        x = np.linspace(0, 10, n)
        ax = basis.plot1D(x)
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        ax = basis.plot1D(num_vectors=(r - 1))
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

        # Restore interactive mode setting.
        plt.interactive(_pltio)

    def test_eq(self):
        """Test __eq__()."""
        basis1 = self.Basis(self._orth(10, 4))
        assert basis1 != 10
        basis2 = self.Basis(self._orth(10, 3))
        assert basis1 != basis2

        basis1 = self.Basis(basis2.entries)
        assert basis1 == basis2
        basis1 = self.Basis(self._orth(10, 10) @ basis2.entries)
        assert basis1 != basis2

        basis2 = self.Basis(
            basis2.entries,
            weights=np.random.random(basis2.full_state_dimension) + 0.5,
            check_orthogonality=False,
        )
        assert basis1 != basis2
        assert basis2 != basis1
        basis1 = self.Basis(
            basis2.entries,
            weights=basis2.weights,
            check_orthogonality=False,
        )
        assert basis1 == basis2
        basis1 = self.Basis(
            basis2.entries,
            weights=(2 * basis2.weights),
            check_orthogonality=False,
        )
        assert basis1 != basis2

    def test_save(self, n=11, r=2, target="_linearbasissavetest.h5"):
        """Lightly test LinearBasis.save()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)
        w = np.random.random(n)

        basis = self.Basis(Vr, name="mybasis")
        basis.save(target)
        assert os.path.isfile(target)
        os.remove(target)

        basis = self.Basis(Vr, w, check_orthogonality=False)
        basis.save(target)
        assert os.path.isfile(target)
        os.remove(target)

    def test_load(self, n=10, r=5, target="_linearbasisloadtest.h5"):
        """Test LinearBasis.load()."""
        # Clean up after old tests.
        if os.path.isfile(target):  # pragma: no cover
            os.remove(target)

        Vr = self._orth(n, r)
        basis1 = self.Basis(Vr, name="testbasis")
        basis1.save(target)
        basis2 = self.Basis.load(target)
        assert basis2.name == "testbasis"
        assert basis2 == basis1

        w = np.random.random(n) + 0.5
        basis1 = self.Basis(Vr, w, check_orthogonality=False)
        basis1.save(target, overwrite=True)
        with pytest.warns(opinf.errors.OpInfWarning):
            basis2 = self.Basis.load(target)
        assert basis2 == basis1

        # Clean up.
        os.remove(target)


if __name__ == "__main__":
    pytest.main([__file__])
