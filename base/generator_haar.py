import numpy as np
from scipy.linalg import qr, kron
from numpy import diagonal, absolute, multiply
import tensorflow as tf

def random_unitary(d):
    """
    Generate a random unitary matrix of size d x d.
    """
    # Generate a random complex matrix
    random_matrix = np.random.normal(size=(d, d)) + 0j
    # Perform QR decomposition
    q, _ = qr(random_matrix)
    return q

def haar(d):
    """
    Generates a Haar-random unitary matrix of size d x d.
    The function creates a random matrix, performs a QR decomposition,
    and scales it to ensure that the matrix is unitary.
    
    Reference: https://arxiv.org/pdf/math-ph/0609050.pdf
    """
    # Generate a random complex matrix using numpy's randn
    array = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2.0)
    
    # Perform QR decomposition to obtain a unitary matrix
    ortho, upper = qr(array)
    
    # Adjust the phases to ensure unitary
    diag = diagonal(upper)
    temp = diag / absolute(diag)
    
    # Multiply to get the final unitary matrix
    result = multiply(ortho, temp, ortho)
    
    return result

def generate_rho_haar(n):
    """Generate probe state for an n-qubit system."""
    size = 2**n
    # Generate a random state vector (pure state) |psi> in Hilbert space
    psi = np.random.randn(size) + 1j * np.random.randn(size)
    psi /= np.linalg.norm(psi)  # Normalize the vector
        
    # Generate a Haar-random unitary matrix U
    U = haar(size)
        
    # Construct the density matrix rho = U |psi><psi| U†
    rho = np.outer(psi, np.conj(psi))
    rho = np.dot(np.dot(U, rho), U.conj().T)
    return rho
    
def generate_n_qubits_rho_haar(n):
    """Generate 6^n probe states for an n-qubit system."""
    density_matrices = []

    for _ in range(6**n):
        # Construct the density matrix rho
        rho = generate_rho_haar(n)
        
        # Append the density matrix to the list
        density_matrices.append(rho)
    
    return density_matrices

def generate_kraus_operators(unitary):
    '''Create a set of Kraus Operators from the input unitary matrix, using QR decomposition'''
    kraus_ops = []
    Q, R = qr(unitary)

    #Q: a 2^N x 2^N matrix, N is the number of qubits
    for q in Q:
        q = np.expand_dims(q, 1)
        kraus_ops.append(q @ np.transpose(np.conjugate(q)))
    return tf.convert_to_tensor(kraus_ops)

def normalize_kraus(kraus_operators):
    """Ensure Kraus operators satisfy Σ K_i^† K_i = I"""
    summation = sum(tf.linalg.adjoint(K) @ K for K in kraus_operators)
    sqrt_inv = tf.linalg.inv(tf.linalg.sqrtm(summation))  # (Σ K_i† K_i)^(-1/2)
    return [K @ sqrt_inv for K in kraus_operators]

def generate_choi_matrix(U, d):
    """Construct the Choi matrix from a Haar random unitary U."""
    choi = np.zeros((d**2,d**2), dtype=complex)

    for i in range(d):
        for j in range(d):
            # Create the outer product |i><j|
            outer_ij = np.outer(np.eye(d)[i], np.eye(d)[j].conj())
            # Apply the unitary operation U on |i><j|
            Phi_ij = U @ outer_ij @ U.conj().T
            # Update Choi matrix
            choi += kron(np.eye(d)[i], np.eye(d)[j].conj()) @ Phi_ij
    
    return choi