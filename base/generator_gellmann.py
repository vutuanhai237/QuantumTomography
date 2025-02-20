import numpy as np
import itertools
import random
def generate_gellmann_matrices(d):
    """Generate generalized Gell-Mann matrices for SU(d)."""
    matrices = []
    
    # Off-diagonal symmetric and anti-symmetric
    for i in range(d):
        for j in range(i + 1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = mat[j, i] = 1
            matrices.append(mat)

            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = -1j
            mat[j, i] = 1j
            matrices.append(mat)

    # Diagonal traceless matrices
    for k in range(1, d):
        diag_matrix = np.zeros((d, d), dtype=complex)
        for i in range(k):
            diag_matrix[i, i] = 1
        diag_matrix[k, k] = -k
        matrices.append(diag_matrix / np.sqrt(k * (k + 1)))

    return matrices

def diagonalize_gellmann(gellmann_matrices):
    """Diagonalize the Gell-Mann matrices for SU(d)."""
    lambdas = gellmann_matrices
    eigen_gellmann = []

    for l in lambdas:
        eigvals, eigvecs = np.linalg.eigh(l) 
        eigen_gellmann.append((eigvals, eigvecs)) 
    
    return eigen_gellmann


def generate_measurement_projector_gellmann():
    projectors = []
    eigen_gellmann = diagonalize_gellmann(generate_gellmann_matrices(2)) # ??????
    for _, eigenvectors in eigen_gellmann:
        for i in range(eigenvectors.shape[1]):  # Each eigenstate
            vi = eigenvectors[:, i].reshape(-1, 1)  # |v_i⟩
            projector = vi @ vi.conj().T  # |v_i⟩⟨v_i|
            projectors.append(projector)
    # Ensure they sum to identity
    sum_projectors = sum(projectors)
    scale_factor = np.trace(sum_projectors) / np.trace(np.eye(2))  # Normalize to identity
    projectors = [M / scale_factor for M in projectors]

    return projectors

def generate_measurement_operators_gellmann(projectors, n):
    single_qubit_M = projectors
    M_n_qubit = []

    for combination in itertools.product(single_qubit_M, repeat=n):
        M = combination[0]
        for i in range(1, n):
            M = np.kron(M, combination[i])  # Extend to n-qubit
        M_n_qubit.append(M)
        

    return M_n_qubit

def generate_n_qubits_rho_gellmann(eigen_gellmann, n):
    """Generate 6^n probe states for an n-qubit system."""

    # Extract eigenvalues and eigenvectors
    eigvals, eigvecs = zip(*eigen_gellmann)
    eigvals = np.abs(eigvals)
    eigvals /= np.sum(eigvals)
    rho_list = []

    for _ in range(6**n):
        eig_set = random.choice(eigvals)  # Sample eigenvalues
        eig_diag = np.diag(eig_set) 

        eig_vec = eigvecs[0]
        rho = eig_vec @ eig_diag @ eig_vec.conj().T 
        rho /= np.trace(rho)  # Normalize

        rho_list.append(rho)

    return rho_list
