import tensorflow as tf
import base.generator as generator
import numpy as np

def mean_fidelity(rho_f_list, rho_list):
    """Compute mean(Fidelity) metric"""

    fidelity_sum = tf.constant(0.0, dtype=tf.complex128)

    for rho, rho_f in zip(rho_list, rho_f_list):
        sqrt_rho = tf.linalg.sqrtm(rho)
        intermediate = sqrt_rho @ rho_f @ sqrt_rho

        fidelity = tf.linalg.trace(tf.linalg.sqrtm(intermediate)) ** 2
        fidelity_sum += fidelity

    fidelity_avg = fidelity_sum / len(rho_list)

    return fidelity_avg

def compilation_trace_fidelity(rho, sigma):
    """Calculating the fidelity metric

    Args:
        - rho (DensityMatrix): first density matrix
        - sigma (DensityMatrix): second density matrix

    Returns:
        - float: trace metric has value from 0 to 1
    """
    f = (tf.linalg.sqrtm(rho)) @ sigma @ (tf.linalg.sqrtm(rho))
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
    Z_1 = generator.kron_n_identity(n, qubit_index, pauli_matrix)
    
    # Calculate the trace
    trace_result = np.trace(Z_1 @ rho)  # Matrix multiplication
    return trace_result

