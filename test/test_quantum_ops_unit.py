import numpy as np
import tensorflow as tf
import pytest
from qtomo import generator
from qtomo import quantum_ops  # assuming your file is qtomo/quantum_ops.py


# =================================
# Helper functions
# =================================

def random_density_matrix(dim=2, seed=42):
    """Generate a random valid density matrix of dimension dim."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    return rho


# ================================================
# Tests for apply_unitary and apply_unitary_dagger
# ================================================

def test_apply_unitary_identity():
    """Applying identity should leave rho unchanged."""
    rho = tf.constant(random_density_matrix(2), dtype=tf.complex128)
    U = tf.eye(2, dtype=tf.complex128)
    result = quantum_ops.apply_unitary(rho, U)
    assert np.allclose(result, rho), "Identity unitary should not change rho"


def test_apply_unitary_and_dagger_consistency():
    """U followed by U† should recover the original rho."""
    rho = tf.constant(random_density_matrix(2), dtype=tf.complex128)
    theta = np.pi / 4
    U = tf.constant([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=tf.complex128)

    transformed = quantum_ops.apply_unitary(rho, U)
    recovered = quantum_ops.apply_unitary_dagger(transformed, U)

    assert np.allclose(recovered, rho, atol=1e-12), "U then U† should recover rho"


def test_apply_unitary_preserves_trace_and_hermiticity():
    """Unitary transformation preserves trace and Hermiticity."""
    rho = tf.constant(random_density_matrix(2), dtype=tf.complex128)
    U = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)  # Pauli X

    result = quantum_ops.apply_unitary(rho, U)
    trace_ok = np.isclose(np.trace(result), 1.0, atol=1e-12)
    hermitian_ok = np.allclose(result, tf.transpose(tf.math.conj(result)))

    assert trace_ok, "Trace should remain 1 under unitary evolution"
    assert hermitian_ok, "Hermiticity should be preserved"


# ================================
# Tests for apply_kraus_operators
# ================================

def test_apply_kraus_operators_trace_preserving():
    """Check Kraus map is trace preserving for amplitude damping channel."""
    gamma = 0.3
    K0 = tf.constant([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=tf.complex128)
    K1 = tf.constant([[0, np.sqrt(gamma)], [0, 0]], dtype=tf.complex128)
    kraus_ops = [K0, K1]

    rho = tf.constant(random_density_matrix(2), dtype=tf.complex128)
    result = quantum_ops.apply_kraus_operators(rho, kraus_ops)
    assert np.isclose(np.trace(result), 1.0, atol=1e-12), "Kraus map should preserve trace"


def test_apply_kraus_operators_no_nan_inf():
    """Output must contain finite values for valid Kraus operators."""
    rho = tf.constant(random_density_matrix(2), dtype=tf.complex128)
    K = tf.constant([[1, 0], [0, 1]], dtype=tf.complex128)
    result = quantum_ops.apply_kraus_operators(rho, [K])
    assert np.all(np.isfinite(result)), "No NaN or Inf in output"


# =================================
# Tests for amplitude damping noise
# =================================

def test_amplitude_damping_gamma_0_no_change():
    """If gamma=0, should return same rho."""
    rho = np.array([[0.6, 0.3], [0.3, 0.4]], dtype=np.complex128)
    result = quantum_ops.apply_amplitude_damping_noise(rho, num_qubits=1, gamma=0.0)
    assert np.allclose(result, rho, atol=1e-12), "No noise when gamma=0"


def test_amplitude_damping_gamma_1_full_decay():
    """If gamma=1, system decays to |0><0|."""
    rho = np.array([[0.4, 0.0], [0.0, 0.6]], dtype=np.complex128)
    result = quantum_ops.apply_amplitude_damping_noise(rho, num_qubits=1, gamma=1.0)
    expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert np.allclose(result, expected, atol=1e-12), "Full damping → ground state"


def test_amplitude_damping_hermitian_trace_positive():
    """Check Hermiticity, trace=1, positive semi-definite."""
    rho = random_density_matrix(2)
    gamma = 0.4
    result = quantum_ops.apply_amplitude_damping_noise(rho, num_qubits=1, gamma=gamma)

    assert np.allclose(result, result.conj().T), "Must remain Hermitian"
    assert np.isclose(np.trace(result), 1.0, atol=1e-12), "Trace must remain 1"
    assert np.all(np.linalg.eigvalsh(result) >= -1e-12), "Density matrix must be PSD"


# =================================
# Tests for dephasing channel
# =================================

@pytest.mark.parametrize("gamma", [0.0, 0.2, 0.8, 1.0])
def test_dephasing_channel_properties(gamma):
    """Dephasing should reduce off-diagonal terms."""
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

    result = quantum_ops.dephasing_channel(rho, gamma)

    # Check Hermitian and trace=1
    assert np.allclose(result, result.conj().T), "Hermitian"
    assert np.isclose(np.trace(result), 1.0, atol=1e-12), "Trace must be preserved"

    # Off-diagonal elements shrink as gamma increases
    assert abs(result[0, 1]) <= abs(rho[0, 1]) + 1e-12, "Dephasing should not increase coherence"


def test_dephasing_channel_no_nan_inf():
    """Ensure numerical stability."""
    rho = np.array([[0.5, 0.3], [0.3, 0.5]], dtype=np.complex128)
    result = quantum_ops.dephasing_channel(rho, gamma=0.5)
    assert np.all(np.isfinite(result)), "Should not contain NaN or Inf"
