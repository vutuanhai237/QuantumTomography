import pytest
import logging
import numpy as np
import tensorflow as tf
from qtomo.tomography import KrausTomography, ChoiTomography

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# Fixtures: helper functions to create mock data or objects used across tests
# --------------------------

@pytest.fixture
def mock_optimizer():
    class MockOptimizer:
        def step_and_cost(self, param, loss_fn, grad_fn):
            # Simulate update step without changing parameters, just return current loss
            loss = loss_fn(param)
            return param, loss
    return MockOptimizer()

@pytest.fixture
def mock_data_process_fn():
    def _fn(rho_set, param):
        # Simulate data processing step: output = input, no changes
        return rho_set
    return _fn

@pytest.fixture
def mock_gradient_fn():
    def _fn(loss_fn, param):
        # Simulate gradient returning zero vector, loss returning zero
        return tf.zeros_like(param), tf.constant(0.0)
    return _fn

@pytest.fixture
def mock_loss_fn():
    def _fn(pred, target):
        # Simulate loss function returning constant 0.1234
        return tf.constant(0.1234, dtype=tf.float32)
    return _fn


# --------------------------
# Tests for KrausTomography
# --------------------------

def test_kraus_tomography_init():
    # Check that KrausTomography object is correctly initialized with the right number of states and operators
    tomo = KrausTomography(num_qubits=1, num_rho=5, num_kraus=2)
    logger.info(f"KrausTomography initialized with {len(tomo.target_rho_set)} states and {len(tomo.init_kraus)} Kraus operators")
    assert len(tomo.target_rho_set) == 5  # Should have 5 training states
    assert isinstance(tomo.init_kraus, list)  # init_kraus is a list
    assert len(tomo.init_kraus) == 2  # Should have 2 Kraus operators

def test_kraus_fit_returns_results(mock_data_process_fn, mock_gradient_fn, mock_loss_fn, mock_optimizer):
    # Check that the fit function returns correct results using mock functions
    tomo = KrausTomography(num_qubits=1, num_rho=3, num_kraus=2)
    param, loss_history = tomo.fit(
        epochs=3,
        data_process_fn=mock_data_process_fn,
        gradient_fn=mock_gradient_fn,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer
    )
    logger.info(f"Kraus fit completed. Loss history: {[l.numpy() if hasattr(l, 'numpy') else l for l in loss_history]}")
    assert isinstance(param, list)  # Returned parameters should be list of Kraus operators
    assert len(loss_history) == 3  # Loss history should have 3 values (for 3 epochs)

def test_kraus_evaluate_output():
    # Check that evaluate returns correct output, target, and metric score
    tomo = KrausTomography(num_qubits=1, num_rho=1, num_kraus=1)
    tomo.kraus_operators = tomo.init_kraus  # Skip training, use initialized operators

    rho_input = np.eye(2, dtype=np.complex128) / 2  # Input: maximally mixed state

    def target_process_fn(rho): return rho  # target process: identity
    def metric_fn(a, b): return 1.0  # metric mock returns perfect score 1

    out, tgt, score = tomo.evaluate(rho_input, target_process_fn, metric_fn)
    logger.info(f"Kraus evaluate output shape: {out.shape}, score: {score}")
    assert out.shape == (2, 2)  # output shape correct
    assert np.allclose(out, tgt)  # output and target should be equal
    assert score == 1.0  # metric returns correct value


# --------------------------
# Tests for ChoiTomography
# --------------------------

def test_choi_tomography_init():
    # Check initialization of ChoiTomography object
    tomo = ChoiTomography(num_qubits=1, num_rho=5)
    logger.info(f"ChoiTomography initialized with {len(tomo.target_rho_set)} states")
    assert len(tomo.target_rho_set) == 5  # Should have 5 training states
    assert isinstance(tomo.init_unitary, tf.Variable)  # init_unitary should be a tf.Variable

def test_choi_fit_returns_results(mock_data_process_fn, mock_gradient_fn, mock_loss_fn, mock_optimizer):
    # Check fit method returns correct results for ChoiTomography
    tomo = ChoiTomography(num_qubits=1, num_rho=3)
    param, loss_history = tomo.fit(
        epochs=2,
        data_process_fn=mock_data_process_fn,
        gradient_fn=mock_gradient_fn,
        loss_fn=mock_loss_fn,
        optimizer=mock_optimizer
    )
    logger.info(f"Choi fit completed. Loss history: {[l.numpy() if hasattr(l, 'numpy') else l for l in loss_history]}")
    assert isinstance(param, tf.Tensor)  # returned param is a tensor (unitary matrix)
    assert len(loss_history) == 2  # loss history length matches number of epochs

def test_choi_evaluate_output():
    # Check evaluate returns correct output, target, and metric score
    tomo = ChoiTomography(num_qubits=1, num_rho=1)
    tomo.unitary = tomo.init_unitary  # Skip training, use initial unitary

    rho_input = np.eye(2, dtype=np.complex128) / 2  # input state

    def target_process_fn(rho): return rho  # target process: identity
    def metric_fn(a, b): return 0.99  # metric mock returns 0.99 score

    out, tgt, score = tomo.evaluate(rho_input, target_process_fn, metric_fn)
    logger.info(f"Choi evaluate output shape: {out.shape}, score: {score}")
    assert out.shape == (2, 2)  # output shape correct
    assert np.allclose(out, tgt)  # output close to target
    assert score == 0.99  # metric returns expected value
