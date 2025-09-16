import tensorflow as tf

def riemann(loss_fn: callable, tensor: tf.Variable) -> tf.Tensor:
    """
    Calculate the projected gradient by computing loss inside GradientTape.

    Args:
        loss_fn (callable): A function that takes `tensor` and returns scalar loss.
        tensor (tf.Variable): Tensor to optimize.

    Returns:
        tf.Tensor: Projected gradient.
    """
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        loss = loss_fn(tensor)

    grad = tape.gradient(loss, tensor)

    if grad is None:
        raise ValueError("Gradient is None. Check if loss depends on tensor.")

    # Project gradient to satisfy Kraus/unitary structure
    adj_grad = tf.linalg.adjoint(grad)
    adj_tensor = tf.linalg.adjoint(tensor)
    proj = grad - tf.matmul(tensor, (tf.matmul(adj_grad, tensor) + tf.matmul(adj_tensor, grad)) / 2)

    return proj, loss # Return both projected gradient and loss
