import tensorflow as tf
import init
import numpy as np
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

def trace_Pauli(rho, qubit_index, pauli_matrix):
    """
    Compute the trace of the Pauli-Z operator applied to a specific qubit of the density matrix.

    Parameters:
    rho (numpy.ndarray): The density matrix (2^n x 2^n) where n is the number of qubits.
    qubit_index (int): The index of the qubit to which the Pauli-Z operator is applied.

    Returns:
    float: The trace after applying the Pauli-Z operator.
    """
    
    # Ensure the density matrix is a NumPy array
    rho = np.array(rho, dtype=np.complex128)
    
    
    # Get number of qubits (assuming rho is a 2^n x 2^n matrix)
    n = int(np.log2(rho.shape[0]))

    # Create the full operator that applies the Pauli-Z to the specific qubit
    Z_1 = init.kron_n_identity(n, qubit_index, pauli_matrix)
    
    # Calculate the trace
    trace_result = np.trace(Z_1 @ rho)  # Matrix multiplication
    return trace_result

