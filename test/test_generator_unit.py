# test/test_generators.py

import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
import logging

from qtomo.generator import (
    gellmann_matrices,
    measurement_projectors,
    merge_projectors_into_povm,
    random_unitary,
    haar,
    haar_probe_state,
    kraus_operators,
    choi_matrix
)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============== #
# Test Gell-Mann  #
# =============== #

def test_gellmann_shape_and_count():
    d = 3
    gm = gellmann_matrices(d)
    logger.info(f"Gell-Mann matrices count: {len(gm)}")
    assert len(gm) == d**2 - 1
    for i, m in enumerate(gm):
        logger.info(f"Gell-Mann matrix {i} shape: {m.shape}")
        assert m.shape == (d, d)
        assert np.allclose(m.conj().T, m)  # Hermitian

# ========================= #
# Test Measurement Projectors
# ========================= #

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

# ============== #
# Test Unitaries #
# ============== #

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

# ======================== #
# Test Haar Probe States   #
# ======================== #

def test_haar_probe_state_properties():
    rho = haar_probe_state(1)
    logger.info(f"Haar probe state trace: {np.trace(rho)}")
    assert rho.shape == (2, 2)
    assert_allclose(np.trace(rho), 1, atol=1e-6)
    assert np.allclose(rho, rho.conj().T)  # Hermitian

# ======================= #
# Test Kraus Operators    #
# ======================= #

def test_kraus_cptp_condition():
    ops = kraus_operators(1, 4)
    summation = sum([tf.matmul(tf.math.conj(tf.transpose(K)), K) for K in ops])
    identity = np.eye(2)
    logger.info(f"Sum of Kraus operators (should be identity):\n{summation.numpy()}")
    assert_allclose(summation, identity, atol=1e-6)
