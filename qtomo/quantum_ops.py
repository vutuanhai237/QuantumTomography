
import numpy as np
import tensorflow as tf
from qtomo import generator


def dephasing_channel(input_rho, num_qubits: int, gamma: float) -> np.ndarray:
    """
    Apply dephasing noise channel on the input density matrix.

    Args:
        input_rho: Density matrix object with `.data` attribute (numpy array).
        num_qubits: Number of qubits in the system.
        gamma: Dephasing parameter in [0,1].

    Returns:
        Noisy density matrix as a numpy array after dephasing.
    """
    rho = input_rho.data
    sigma_z = np.array([[1, 0], [0, -1]])
    alpha = 0.5 * (1 + np.sqrt(1 - gamma))
    beta = 0.5 * (1 - np.sqrt(1 - gamma))

    for i in range(num_qubits):
        sigma_z_i = np.eye(1)
        for j in range(num_qubits):
            if j == i:
                sigma_z_i = np.kron(sigma_z_i, sigma_z)
            else:
                sigma_z_i = np.kron(sigma_z_i, np.eye(2))
        rho = alpha * rho + beta * (sigma_z_i @ rho @ sigma_z_i)

    return rho


def apply_amplitude_damping_noise(input_rho, num_qubits: int, gamma: float) -> np.ndarray:
    """
    Apply amplitude damping noise channel on the input density matrix.

    Args:
        input_rho: Density matrix as a cupy array.
        num_qubits: Number of qubits.
        gamma: Amplitude damping parameter in [0,1].

    Returns:
        Noisy density matrix as a cupy array after amplitude damping.
    """
    rho = input_rho.copy()
    for k in range(num_qubits):
        K0_k = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1_k = np.array([[0, np.sqrt(gamma)], [0, 0]])
        K0 = generator.kron_insert(num_qubits, k, K0_k)
        K1 = generator.kron_insert(num_qubits, k, K1_k)
        rho = K0 @ rho @ np.transpose(np.conjugate(K0)) + K1 @ rho @ np.transpose(np.conjugate(K1))
    return rho


def apply_unitary(rho: tf.Tensor, unitary: tf.Tensor) -> tf.Tensor:
    """
    Apply a unitary transformation U * rho * U^dagger on the density matrix.

    Args:
        rho: Density matrix as a TensorFlow tensor.
        unitary: Unitary operator as TensorFlow tensor.

    Returns:
        Transformed density matrix as TensorFlow tensor.
    """
    return tf.matmul(tf.matmul(unitary, rho), tf.transpose(tf.math.conj(unitary)))


def apply_unitary_dagger(rho: tf.Tensor, unitary: tf.Tensor) -> tf.Tensor:
    """
    Apply the adjoint unitary transformation U^dagger * rho * U.

    Args:
        rho: Density matrix as a TensorFlow tensor.
        unitary: Unitary operator as TensorFlow tensor.

    Returns:
        Transformed density matrix as TensorFlow tensor.
    """
    return tf.matmul(tf.matmul(tf.transpose(tf.math.conj(unitary)), rho), unitary)


def apply_kraus_operators(rho: tf.Tensor, kraus_ops: list[tf.Tensor]) -> tf.Tensor:
    """
    Apply a set of Kraus operators on a density matrix.

    Args:
        rho: Density matrix as TensorFlow tensor.
        kraus_ops: List of Kraus operators (TensorFlow tensors).

    Returns:
        The density matrix after Kraus operators applied (sum over K*rho*K^dagger).
    """
    return sum(tf.matmul(tf.matmul(K, rho), tf.transpose(tf.math.conj(K))) for K in kraus_ops)