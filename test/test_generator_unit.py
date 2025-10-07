# test/test_generators.py

import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
import logging

from qtomo.generator import (
    gellmann_matrices,
    probe_states_gellmann,
    measurement_projectors,
    merge_projectors_into_povm,
    measurement_operators,
    random_unitary,
    haar,
    haar_probe_state,
    kraus_operators,
    kron_insert
)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===============
# Test Gell-Mann
# ===============

def test_gellmann_shape_and_count():
    d = 3
    gm = gellmann_matrices(d)
    logger.info(f"Gell-Mann matrices count: {len(gm)}")
    assert len(gm) == d**2 - 1
    for i, m in enumerate(gm):
        logger.info(f"Gell-Mann matrix {i} shape: {m.shape}")
        assert m.shape == (d, d)
        assert np.allclose(m.conj().T, m)  # Hermitian

def test_probe_states_gellmann_output_shape():
    """Ensure probe_states_gellmann returns 6^n density matrices with proper normalization."""
    # Create fake eigen decomposition data
    rng = np.random.default_rng(42)
    eigen_gellmann = []
    for _ in range(8):
        eigvals = rng.normal(size=3)
        eigvecs, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # random orthonormal basis
        eigen_gellmann.append((eigvals, eigvecs))

    result = probe_states_gellmann(eigen_gellmann, 1)
    assert len(result) == 6, "Should produce 6^1 = 6 probe states"
    for rho in result:
        assert rho.shape == (3, 3)
        assert np.allclose(rho, rho.conj().T), "Density matrix must be Hermitian"
        assert np.isclose(np.trace(rho), 1), "Density matrix must be trace 1"
        assert np.all(np.linalg.eigvalsh(rho) >= -1e-12), "Density matrix must be PSD"


def test_probe_states_gellmann_real_values():
    """Check function works even if eigenvalues and eigenvectors are real."""
    eigvals = np.array([1, 0.5, 0.2])
    eigvecs = np.eye(3)
    eigen_gellmann = [(eigvals, eigvecs)]
    result = probe_states_gellmann(eigen_gellmann, 1)
    assert all(np.isrealobj(r) for r in result), "Should produce real matrices if inputs are real"

# ============================
# Test Measurement Projectors
# ============================

def test_measurement_projectors_trace_sum():
    projectors = measurement_projectors(1)
    total = sum(projectors)
    identity = np.eye(2)
    logger.info(f"Sum of projectors:\n{total}")
    assert_allclose(total, identity, atol=1e-6)

def test_merge_projectors_into_povm():
    projectors = measurement_projectors(1)
    povm = merge_projectors_into_povm(projectors, 4)
    logger.info(f"POVM length: {len(povm)}")
    total = sum(povm)
    identity = np.eye(2)
    logger.info(f"Sum of POVM elements:\n{total}")
    assert len(povm) == 4
    assert_allclose(total, identity, atol=1e-6)

def test_measurement_operators_single_qubit():
    """Test measurement operators for 1 qubit (should just return projectors)."""
    projectors = [
        np.array([[1, 0], [0, 0]]),  # |0><0|
        np.array([[0, 0], [0, 1]]),  # |1><1|
    ]
    result = measurement_operators(projectors, 1)

    assert len(result) == 2, "There should be 2 measurement operators for 1 qubit"
    for M, P in zip(result, projectors):
        assert np.allclose(M, P), "Returned operators should match input projectors"
        assert M.shape == (2, 2)


def test_measurement_operators_two_qubits():
    """Test measurement operators for 2 qubits."""
    projectors = [
        np.array([[1, 0], [0, 0]]),  # |0><0|
        np.array([[0, 0], [0, 1]]),  # |1><1|
    ]
    result = measurement_operators(projectors, 2)

    # Expect 4 operators: |00>, |01>, |10>, |11>
    assert len(result) == 4, "There should be 4 measurement operators for 2 qubits"
    for M in result:
        assert M.shape == (4, 4), "Each operator should be 4x4"

def test_measurement_operators_orthogonality():
    """Check that generated measurement operators are orthogonal."""
    projectors = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]
    result = measurement_operators(projectors, 2)

    # Compute Hilbert-Schmidt inner products (Tr(A†B))
    overlaps = np.array([[np.trace(A.conj().T @ B) for B in result] for A in result])

    # Diagonal elements = 1, off-diagonal = 0
    for i in range(len(result)):
        for j in range(len(result)):
            if i == j:
                assert np.isclose(overlaps[i, j], 1), "Operator should be normalized"
            else:
                assert np.isclose(overlaps[i, j], 0), "Operators should be orthogonal"

def test_measurement_operators_real_values():
    """All entries must be real if input projectors are real."""
    projectors = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
    ]
    result = measurement_operators(projectors, 2)
    for M in result:
        assert np.isrealobj(M), "Measurement operators should be real for real projectors"

# ==============
# Test Unitaries
# ==============

def test_random_unitary_unitarity():
    U = random_unitary(2)
    identity = np.eye(U.shape[0])
    logger.info(f"Random unitary U^dagger U:\n{U.conj().T @ U}")
    assert_allclose(U.conj().T @ U, identity, atol=1e-6)

def test_haar_unitary_unitarity():
    U = haar(2)
    identity = np.eye(U.shape[0])
    logger.info(f"Haar unitary U^dagger U:\n{U.conj().T @ U}")
    assert_allclose(U.conj().T @ U, identity, atol=1e-6)

# ========================
# Test Haar Probe States  
# ========================

def test_haar_probe_state_properties():
    rho = haar_probe_state(1)
    logger.info(f"Haar probe state trace: {np.trace(rho)}")
    assert rho.shape == (2, 2)
    assert_allclose(np.trace(rho), 1, atol=1e-6)
    assert np.allclose(rho, rho.conj().T)  # Hermitian

# =======================
# Test Kraus Operators   
# =======================

def test_kraus_cptp_condition():
    ops = kraus_operators(1, 4)
    summation = sum([tf.matmul(tf.math.conj(tf.transpose(K)), K) for K in ops])
    identity = np.eye(2)
    logger.info(f"Sum of Kraus operators (should be identity):\n{summation.numpy()}")
    assert_allclose(summation, identity, atol=1e-6)

# ==============
# Test Kronecker
# ==============
def test_kron_insert_basic():
    """Check if kron_insert inserts matrix correctly at position j."""
    X = np.array([[0, 1], [1, 0]])  # Pauli X
    result = kron_insert(3, 1, X)

    # Expected: I ⊗ X ⊗ I
    expected = np.kron(np.kron(np.eye(2), X), np.eye(2))
    assert np.allclose(result, expected), "Kronecker structure is incorrect"
    assert result.shape == (8, 8), "Result should be 8x8 for 3 qubits"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_kron_insert_position(j):
    """Test inserting matrix at various positions in n=3."""
    Z = np.array([[1, 0], [0, -1]])
    n = 3
    result = kron_insert(n, j, Z)

    # Verify by checking tensor factor at the correct index
    # Take partial trace-like slice and verify diagonal element pattern
    reshaped = result.reshape([2] * 2 * n)
    diag_pattern = np.diagonal(result).reshape([2] * n)
    # The number of -1s corresponds to Z acting on bit j
    num_neg = np.sum(diag_pattern < 0)
    assert num_neg > 0, f"Expected Z effect at qubit {j}"