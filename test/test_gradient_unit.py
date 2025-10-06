import tensorflow as tf
import numpy as np
import logging
from qtomo.gradient import riemann

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def simple_loss_fn(x):
    I = tf.eye(x.shape[0], dtype=x.dtype)
    diff = x - 2 * I
    return tf.reduce_sum(tf.math.square(tf.math.abs(diff)))


def test_riemann_gradient_descent():
    param = tf.Variable(np.ones((2, 2), dtype=np.complex128))
    learning_rate = 0.03
    losses = []

    for epoch in range(30):
        grad, loss = riemann(simple_loss_fn, param)
        logger.info(f"Epoch {epoch + 1}, Loss: {loss.numpy():.6f}")
        logger.debug(f"Grad:\n{grad.numpy()}")
        logger.debug(f"Param before:\n{param.numpy()}")

        # Gradient descent step
        param.assign(param - learning_rate * grad)
        losses.append(loss.numpy())

    assert all(x >= y for x, y in zip(losses, losses[1:])), "Loss not decreasing!"
    
