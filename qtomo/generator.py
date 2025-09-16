# ========================== #
#    Quantum Data Generators #
# ========================== #
import numpy as np
import itertools
import random
from scipy.linalg import qr
from qtomo import utils

# =============================== #
# 1. Gell-Mann Matrix Generators  #
# =============================== #

def gellmann_matrices(d: int) -> list[np.ndarray]:
    """
    Generate the generalized Gell-Mann matrices for SU(d).
    Returns a list of (d x d) numpy arrays.
    """
    matrices = []
    
    # Off-diagonal symmetric and anti-symmetric matrices
    for i in range(d):
        for j in range(i + 1, d):
            sym = np.zeros((d, d), dtype=complex)
            sym[i, j] = sym[j, i] = 1
            matrices.append(sym)

            antisym = np.zeros((d, d), dtype=complex)
            antisym[i, j] = -1j
            antisym[j, i] = 1j
            matrices.append(antisym)

    # Diagonal traceless matrices
    for k in range(1, d):
        diag = np.zeros((d, d), dtype=complex)
        for i in range(k):
            diag[i, i] = 1
        diag[k, k] = -k
        matrices.append(diag / np.sqrt(k * (k + 1)))

    return matrices

# ==================================== #
# 2. Measurement Projector Generators  #
# ==================================== #

def measurement_projectors(num_qubits: int = 1) -> list[np.ndarray]:
    """
    Generate measurement projectors from diagonalized Gell-Mann matrices
    for a d=2^n dimensional system.
    """
    d = 2 ** num_qubits
    eigen_gellmann = utils.diagonalize_matrices(gellmann_matrices(d))
    projectors = []

    for _, eigvecs in eigen_gellmann:
        for i in range(eigvecs.shape[1]):
            v = eigvecs[:, i].reshape(-1, 1)
            projector = v @ v.conj().T
            projectors.append(projector)

    # Normalize to ensure sum(projectors) = I
    sum_proj = sum(projectors)
    scale = np.trace(sum_proj) / d
    projectors = [P / scale for P in projectors]

    return projectors

def merge_projectors_into_povm(projectors: list[np.ndarray], num_elements: int) -> list[np.ndarray]:
    """
    Merge a list of projectors into `num_elements` POVM elements.
    Ensures the final POVM elements sum to identity.
    """
    total = len(projectors)
    assert num_elements <= total, "Cannot request more POVM elements than available projectors"

    # Round-robin grouping
    groups = [[] for _ in range(num_elements)]
    for i, P in enumerate(projectors):
        groups[i % num_elements].append(P)

    povms = [sum(group) for group in groups]

    # Optional: check sum
    print(f"Sum of POVMs:\n{sum(povms)}")

    return povms

def measurement_operators(projectors: list[np.ndarray], n: int) -> list[np.ndarray]:
    """
    Generate n-qubit measurement operators via Kronecker product
    of single-qubit projectors.
    """
    M_n = []
    for combo in itertools.product(projectors, repeat=n):
        M = combo[0]
        for i in range(1, n):
            M = np.kron(M, combo[i])
        M_n.append(M)
    return M_n

# ============================ #
# 3. Probe State Generators    #
# ============================ #

def probe_states_gellmann(eigen_gellmann: list[tuple[np.ndarray, np.ndarray]], n: int) -> list[np.ndarray]:
    """
    Generate 6^n probe states using eigen-decomposition from Gell-Mann basis.
    """
    eigvals_list, eigvecs_list = zip(*eigen_gellmann)
    eigvals_list = [np.abs(ev) / np.sum(np.abs(ev)) for ev in eigvals_list]

    rho_list = []
    for _ in range(6 ** n):
        eigvals = random.choice(eigvals_list)
        eigvecs = eigvecs_list[0]  # Use first eigvec basis
        diag = np.diag(eigvals)
        rho = eigvecs @ diag @ eigvecs.conj().T
        rho /= np.trace(rho)
        rho_list.append(rho)

    return rho_list

# ============================== #
# 4. Haar / Random Unitary Gen   #
# ============================== #

def random_unitary(n: int) -> np.ndarray:
    """
    Generate a random unitary matrix of for n qubits using QR decomposition.
    """
    d = 2 ** n
    rand_mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    Q, _ = qr(rand_mat)
    return Q

def haar(n: int) -> np.ndarray:
    """
    Generates a Haar-random unitary matrix for n qubits.
    The function creates a random matrix, performs a QR decomposition,
    and scales it to ensure that the matrix is unitary.
    
    Reference: https://arxiv.org/pdf/math-ph/0609050.pdf
    """
    d = 2 ** n
    z = (np.random.randn(d, d) + 1j * np.random.randn(d, d)) / np.sqrt(2)
    Q, R = qr(z)
    D = np.diag(R)
    ph = D / np.abs(D)
    return Q * ph

# =================================== #
# 5. Probe States via Haar Unitaries  #
# =================================== #

def haar_probe_state(n: int) -> np.ndarray:
    """
    Generate a random pure state |psi> and rotate by Haar unitary to form rho.
    """
    d = 2 ** n
    psi = np.random.randn(d) + 1j * np.random.randn(d)
    psi /= np.linalg.norm(psi)
    U = haar(n)
    rho = U @ np.outer(psi, psi.conj()) @ U.conj().T
    return rho

def haar_probe_states(n: int, num_rho: int = -1) -> list[np.ndarray]:
    """
    Generate `num_rho` Haar-random probe states for n qubits.
    If num_rho < 0, default to 6^n states.
    """
    if num_rho < 0:
        num_rho = 6 ** n
    return [haar_probe_state(n) for _ in range(num_rho)]

# ========================== #
# 6. Kraus Operator Generator #
# ========================== #

def kraus_operators(n: int, num_operators: int = -1) -> list[np.ndarray]:
    """
    Generate a list of random Kraus operators (unitary matrices).
    Automatically normalize them to satisfy CPTP condition.
    """
    dim = 2 ** n
    max_ops = dim ** 2
    if num_operators < 0 or num_operators > max_ops:
        num_operators = max_ops

    kraus_ops = [random_unitary(n) for _ in range(num_operators)]
    return utils.normalize_kraus(kraus_ops)

# ========================== #
# 7. Choi Matrix Generator   #
# ========================== #

def choi_matrix(U: np.ndarray, d: int) -> np.ndarray:
    """
    Construct the Choi matrix from a unitary U acting on d-dimensional space.
    """
    choi = np.zeros((d ** 2, d ** 2), dtype=complex)
    for i in range(d):
        for j in range(d):
            ketbra = np.outer(np.eye(d)[i], np.eye(d)[j].conj())
            Phi = U @ ketbra @ U.conj().T
            kron_ij = np.kron(np.eye(d)[i], np.eye(d)[j].conj())
            choi += kron_ij @ Phi
    return choi

# ========================== #
# 8. Quantum Circuit Gen     #
# ========================== #

from qoop.core import ansatz

def asigned_circuit(num_qubits: int):
    """
    Create a parameterized quantum circuit (ansatz) for given number of qubits.
    Parameters are randomly initialized.
    """
    circuit = ansatz.graph(num_qubits=num_qubits)
    x0 = 2 * np.pi * np.random.rand(circuit.num_parameters)
    return circuit.assign_parameters(dict(zip(circuit.parameters, x0)))

def kron_insert(n: int, j: int, matrix: np.ndarray) -> np.ndarray:
    """
    Return the Kronecker product of n identity matrices with `matrix` at position j.
    """
    identity = np.eye(2)
    matrices = [identity] * n
    matrices[j] = matrix
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result
