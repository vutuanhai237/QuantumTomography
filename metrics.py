import tensorflow as tf
def compilation_trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    f = (tf.linalg.sqrtm(rho)) @ sigma @ (tf.linalg.sqrtm(rho))

    '''# Cast to a supported type
    real_part = tf.math.real(rho_2)
    imaginary_part = tf.math.imag(rho_2)

    # Check for NaNs in both real and imaginary parts
    contains_nan_real = tf.reduce_any(tf.math.is_nan(real_part))
    contains_nan_imag = tf.reduce_any(tf.math.is_nan(imaginary_part))

    contains_nan = contains_nan_real or contains_nan_imag

    if contains_nan == True:
        rho_2 = rho'''

    return tf.linalg.trace(f)
def frobenius_norm(rho, sigma):
    """
    Compute the Frobenius norm between two matrices.

    Parameters:
    rho (numpy.ndarray): The first matrix.
    sigma (numpy.ndarray): The second matrix.

    Returns:
    float: The Frobenius norm between rho and rho_prime.
    """
    
    # Ensure the matrices are TensorFlow tensors
    rho = tf.convert_to_tensor(rho, dtype=tf.complex128)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.complex128)

    # Compute the difference between the matrices
    diff = rho - sigma

    # Compute the Frobenius norm
    #norm = tf.linalg.normalize(diff, ord='fro')
    norm = tf.sqrt(tf.reduce_sum(tf.square(diff)))
    return norm