
import numpy as np
import tensorflow as tf
from qtomo import generator
from typing import List, Union

# ==============================
# Type aliases for clarity
# ==============================
# rho_f_list, rho_list: list of density matrices, each of shape (d, d)
# Usually dtype=tf.complex128 tensor or np.ndarray with shape (d, d)
# List length = N (number of states)

TensorOrArray = Union[tf.Tensor, np.ndarray]
DensityMatrix = TensorOrArray
DensityMatrixList = List[DensityMatrix]

# ----------------------------------------
# 1. Diff between density matrices
# ----------------------------------------

def MSE(rho_f_list: DensityMatrixList, rho_list: DensityMatrixList) -> tf.Tensor:
    """
    Compute Mean Squared Error (MSE) between lists of density matrices.

    Args:
        rho_f_list (List[tf.Tensor or np.ndarray]): List of predicted density matrices (N x d x d).
        rho_list (List[tf.Tensor or np.ndarray]): List of target density matrices (N x d x d).

    Returns:
        tf.Tensor: Scalar tensor representing average MSE loss.
    """
    mse_sum = 0.0

    for rho, rho_f in zip(rho_list, rho_f_list):
        difference = rho - rho_f  # element-wise difference (d x d)
        mse = tf.reduce_sum(tf.abs(difference)**2)
        mse_sum += mse

    mse_avg = mse_sum / len(rho_list)
    return mse_avg  # Lower value indicates better matching

def mean_infidelity(rho_f_list: DensityMatrixList, rho_list: DensityMatrixList) -> tf.Tensor:
    """
    Compute mean fidelity metric between two lists of density matrices.

    Args:
        rho_f_list (List[tf.Tensor or np.ndarray]): Predicted density matrices, shape N x d x d.
        rho_list (List[tf.Tensor or np.ndarray]): Target density matrices, shape N x d x d.

    Returns:
        tf.Tensor: Scalar tensor for average fidelity.
    """
    fidelity_sum = tf.constant(0.0, dtype=tf.complex128)

    for rho, rho_f in zip(rho_list, rho_f_list):
        overlap = tf.matmul(rho_f, rho)
        fidelity = tf.math.square(tf.linalg.trace(overlap))
        fidelity_sum += fidelity

    fidelity_avg = fidelity_sum / len(rho_list)
    return 1 - fidelity_avg

# ----------------------------------------
# 2. Diff between two matrices
# ----------------------------------------

def compilation_trace_fidelity(rho: DensityMatrix, sigma: DensityMatrix) -> tf.Tensor:
    """
    Calculate trace fidelity metric between two density matrices.

    Args:
        rho (tf.Tensor or np.ndarray): Density matrix of shape (d, d).
        sigma (tf.Tensor or np.ndarray): Density matrix of shape (d, d).

    Returns:
        tf.Tensor: Scalar tensor fidelity value (0 to 1).
    """
    f = (tf.linalg.sqrtm(rho)) @ sigma @ (tf.linalg.sqrtm(rho))
    return tf.linalg.trace(f)

def frobenius_norm(rho: DensityMatrix, sigma: DensityMatrix) -> tf.Tensor:
    """
    Compute Frobenius norm ||rho - sigma||_F between two matrices.

    Args:
        rho (tf.Tensor or np.ndarray): First matrix (d x d).
        sigma (tf.Tensor or np.ndarray): Second matrix (d x d).

    Returns:
        tf.Tensor: Scalar tensor norm value.
    """
    rho = tf.convert_to_tensor(rho, dtype=tf.complex128)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.complex128)
    diff = rho - sigma
    norm = tf.sqrt(tf.reduce_sum(tf.square(diff)))
    return norm

def infidelity(rho: DensityMatrix, sigma: DensityMatrix) -> tf.Tensor:
    """
    Compute loss using trace fidelity metric between two density matrices.

    Args:
        rho (tf.Tensor or np.ndarray): First density matrix.
        sigma (tf.Tensor or np.ndarray): Second density matrix.

    Returns:
        tf.Tensor: Scalar tensor loss.
    """
    return 1 - compilation_trace_fidelity(rho, sigma)


# ----------------------------------------
# 3. Trace of Pauli operator on specific qubit
# ----------------------------------------
def trace_Pauli(rho: np.ndarray, qubit_index: int, pauli_matrix: np.ndarray) -> complex:
    """
    Compute trace of Pauli operator applied to a specific qubit of density matrix.

    Args:
        rho (np.ndarray): Density matrix (2^n x 2^n), n = number of qubits.
        qubit_index (int): Index of qubit to apply Pauli operator.
        pauli_matrix (np.ndarray): 2x2 Pauli matrix.

    Returns:
        complex: Trace result as a complex number.
    """
    rho = np.array(rho, dtype=np.complex128)
    n = int(np.log2(rho.shape[0]))
    Z_1 = generator.kron_insert(n, qubit_index, pauli_matrix)
    trace_result = np.trace(Z_1 @ rho)
    return trace_result

