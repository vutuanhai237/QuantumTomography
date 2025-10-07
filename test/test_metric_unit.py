import numpy as np
import tensorflow as tf
import pytest

from qtomo import generator
from qtomo import metric


# =====================================================
# Fixtures: reusable test data (no special states)
# =====================================================


@pytest.fixture
def single_qubit_states():
    """Generate reproducible full-rank single-qubit density matrices (real only)."""
    np.random.seed(42)

    def random_real_density(d=2):
        A = np.random.randn(d, d)  # real random matrix
        rho = A @ A.T  # symmetric & positive semidefinite
        rho /= np.trace(rho)  # normalize trace = 1
        return rho

    return {
        "rho_1": random_real_density(),
        "rho_2": random_real_density(),
        "rho_3": random_real_density(),
    }


@pytest.fixture
def two_qubit_states():
    """Generate reproducible full-rank two-qubit density matrices (real only)."""
    np.random.seed(123)

    def random_real_density(d=4):
        A = np.random.randn(d, d)
        rho = A @ A.T
        rho /= np.trace(rho)
        return rho

    return {
        "rho_1": random_real_density(),
        "rho_2": random_real_density(),
        "rho_3": random_real_density(),
    }

@pytest.fixture
def orthogonal_density_matrices():
    """Return two full-rank, nearly orthogonal 2x2 density matrices (numerically stable)."""
    eps = 1e-3
    rho_pure = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    sigma_pure = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    rho = (1 - eps) * rho_pure + eps * 0.5 * np.eye(2)
    sigma = (1 - eps) * sigma_pure + eps * 0.5 * np.eye(2)

    rho /= np.trace(rho)
    sigma /= np.trace(sigma)

    return rho, sigma


# ==============================
# MSE and mean infidelity tests
# ==============================

@pytest.mark.parametrize("rho, sigma", [
    (np.eye(2), np.eye(2)),
    (np.eye(2) / 2, np.eye(2)),
])
def test_MSE_values(rho, sigma):
    """Test mean squared error between different states."""
    rho_tensors = [tf.constant(rho, dtype=tf.complex128)]
    sigma_tensors = [tf.constant(sigma, dtype=tf.complex128)]
    result = metric.MSE(sigma_tensors, rho_tensors)

    assert tf.math.real(result) >= 0
    if np.allclose(rho, sigma):
        assert tf.math.abs(result) < 1e-12, "MSE should be 0 for identical states"


def test_mean_infidelity_identical(single_qubit_states):
    """Mean infidelity should be 0 for identical states."""
    rho = [tf.constant(single_qubit_states["rho_1"], dtype=tf.complex128)]
    result = metric.mean_infidelity(rho, rho)
    assert tf.math.abs(tf.math.real(result)) < 1e-12, \
        f"Expected 0, got {result}"


def test_mean_infidelity_different(single_qubit_states):
    """Mean infidelity should be > 0 for different random states."""
    rho = [tf.constant(single_qubit_states["rho_1"], dtype=tf.complex128)]
    sigma = [tf.constant(single_qubit_states["rho_2"], dtype=tf.complex128)]
    result = metric.mean_infidelity(rho, sigma)
    val = tf.math.real(result).numpy()
    assert 0 <= val <= 1, "Infidelity should be in [0,1]"
    assert val > 0.001, f"Expected positive infidelity, got {val}"

def test_mean_infidelity_orthogonal_stable(orthogonal_density_matrices):
    """Numerically safe test — infidelity should be close to 1."""
    rho_np, sigma_np = orthogonal_density_matrices
    rho = tf.constant(rho_np, dtype=tf.complex128)
    sigma = tf.constant(sigma_np, dtype=tf.complex128)

    result = tf.math.real(metric.mean_infidelity([rho], [sigma])).numpy()
    assert not np.isnan(result), "Infidelity result should never be NaN"
    assert 0.9 <= result <= 1.1, f"Expected infidelity ≈ 1, got {result}"


# ============================================
# Frobenius norm and fidelity-related metrics
# ============================================

def test_frobenius_norm_and_infidelity(single_qubit_states):
    """Frobenius norm and infidelity should be well-defined."""
    rho = tf.constant(single_qubit_states["rho_1"], dtype=tf.complex128)
    sigma = tf.constant(single_qubit_states["rho_2"], dtype=tf.complex128)

    f_norm = metric.frobenius_norm(rho, sigma)
    inf = metric.infidelity(rho, sigma)

    assert tf.math.real(f_norm) >= 0, "Frobenius norm must be non-negative"
    assert 0 <= tf.math.real(inf) <= 1, "Infidelity should be in [0,1]"


def test_trace_fidelity_self(two_qubit_states):
    """Trace fidelity of a state with itself should be close to 1."""
    rho = tf.constant(two_qubit_states["rho_1"], dtype=tf.complex128)
    val = metric.compilation_trace_fidelity(rho, rho)
    assert tf.math.real(val) > 0.9, f"Expected high fidelity, got {val}"


# ==============================
# Random and multi-qubit tests
# ==============================

def test_random_density_matrices(two_qubit_states):
    """Random density matrices should yield valid, nonzero Frobenius norm."""
    rho = tf.constant(two_qubit_states["rho_1"], dtype=tf.complex128)
    sigma = tf.constant(two_qubit_states["rho_2"], dtype=tf.complex128)

    result = metric.frobenius_norm(rho, sigma)
    val = tf.math.real(result).numpy()

    assert val > 0, f"Expected positive Frobenius norm, got {val}"


def test_two_qubit_infidelity(two_qubit_states):
    """Infidelity between random 2-qubit states should be > 0."""
    rho = tf.constant(two_qubit_states["rho_1"], dtype=tf.complex128)
    sigma = tf.constant(two_qubit_states["rho_2"], dtype=tf.complex128)

    val = metric.infidelity(rho, sigma)
    real_val = tf.math.real(val).numpy()

    assert 0 <= real_val <= 1, "Infidelity must be within [0,1]"
    assert real_val > 0.001, f"Expected nonzero infidelity, got {real_val}"


# ===========================
# Pauli operator trace tests
# ===========================

def test_trace_Pauli_Z(monkeypatch):
    """Test trace_Pauli with mocked kron_insert."""
    def fake_kron_insert(n, qubit_index, pauli):
        return np.kron(np.eye(2 ** qubit_index),
                       np.kron(pauli, np.eye(2 ** (n - qubit_index - 1))))

    monkeypatch.setattr(generator, "kron_insert", fake_kron_insert)

    np.random.seed(99)
    rho = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = rho @ rho.conj().T
    rho /= np.trace(rho)

    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    result = metric.trace_Pauli(rho, 0, pauli_z)
    assert not np.isnan(result), "Result should not be NaN"
    assert np.isfinite(result), "Result should be finite"
