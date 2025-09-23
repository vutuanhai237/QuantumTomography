# Những hàm có chức năng khác như nhập xuất file để ở đây, hoặc chèn thêm các hàm linh tinh khác
import tensorflow as tf
import numpy as np

def normalize_kraus(kraus_operators):
    """
    Normalize Kraus operators so that Σ K_i† K_i = I.
    Uses TensorFlow backend.
    """
    summation = sum(tf.matmul(tf.linalg.adjoint(K), K) for K in kraus_operators)
    inv_sqrt = tf.linalg.inv(tf.linalg.sqrtm(summation))
    return [K @ inv_sqrt for K in kraus_operators]

def normalize_unitary(matrix):  
    # Perform QR decomposition to get the unitary matrix Q
    Q, _ = tf.linalg.qr(matrix)

    return Q

def diagonalize_matrices(matrices):
    """
    Diagonalize each matrix.
    Returns a list of (eigenvalues, eigenvectors).
    """
    eigen_decomps = []
    for mat in matrices:
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigen_decomps.append((eigvals, eigvecs))
    return eigen_decomps

class FileManager:
    pass