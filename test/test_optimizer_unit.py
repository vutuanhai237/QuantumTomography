import pytest
import tensorflow as tf
import numpy as np
import logging
from qtomo.optimizer import SGDOptimizer, AdamOptimizer
from qtomo.gradient import riemann

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Define a loss function as a fixture, returns a function that takes x
@pytest.fixture
def simple_loss_fn():
    def loss_fn(x):
        I = tf.eye(x.shape[0], dtype=x.dtype)
        diff = x - 2 * I
        return tf.reduce_sum(tf.math.square(tf.math.abs(diff)))
    return loss_fn


def test_sgd_with_riemann(simple_loss_fn):
    opt = SGDOptimizer(alpha=0.1)
    param = tf.Variable(np.ones((2, 2), dtype=np.complex128))

    updated_param, loss = opt.step_and_cost(param, simple_loss_fn, riemann)

    logger.info(f"SGD Initial Loss: {loss.numpy()}")
    logger.info(f"SGD Updated param:\n{updated_param.numpy()}")

    assert loss.numpy() > 0
    assert not tf.reduce_all(tf.equal(updated_param, param))


def test_adam_with_riemann(simple_loss_fn):
    opt = AdamOptimizer(alpha=0.1)
    param = tf.Variable(np.ones((2, 2), dtype=np.complex128))

    updated_param, loss = opt.step_and_cost(param, simple_loss_fn, riemann)
    logger.info(f"Adam Step 1 Loss: {loss.numpy()}")
    logger.info(f"Adam Step 1 Updated param:\n{updated_param.numpy()}")
    logger.info(f"Adam m state:\n{opt.m.numpy()}")
    logger.info(f"Adam v state:\n{opt.v.numpy()}")
    logger.info(f"Adam step t: {opt.t.numpy()}")

    assert loss.numpy() > 0
    assert updated_param.shape == param.shape
    assert opt.m is not None
    assert opt.v is not None

    updated_param2, loss2 = opt.step_and_cost(updated_param, simple_loss_fn, riemann)
    logger.info(f"Adam Step 2 Loss: {loss2.numpy()}")
    logger.info(f"Adam Step 2 Updated param:\n{updated_param2.numpy()}")
    logger.info(f"Adam step t: {opt.t.numpy()}")

    assert loss2.numpy() <= loss.numpy() + 1e-3


def test_adam_state_increment(simple_loss_fn):
    opt = AdamOptimizer(alpha=0.1)
    param = tf.Variable(np.ones((2, 2), dtype=np.complex128))

    for i in range(5):
        updated_param, loss = opt.step_and_cost(param, simple_loss_fn, riemann)
        param = updated_param
        logger.info(f"Adam Step {i + 1} Loss: {loss.numpy()}")
        logger.info(f"Adam Step {i + 1} t: {opt.t.numpy()}")
        assert opt.t.numpy() == i + 1
