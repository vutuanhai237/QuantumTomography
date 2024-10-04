from scipy.linalg import qr
import numpy as np
import tensorflow as tf
from qoop.core import ansatz
from qiskit.quantum_info import DensityMatrix

def create_kraus_operators(num_qubits: int):
    unitary = create_unitary_matrix(num_qubits=num_qubits)
    kraus_operators = create_kraus_operators(unitary)
    return kraus_operators
    

def create_kraus_operators_from_unitary(unitary_matrix):
    '''
        Create a set of Kraus Operators from the input unitary matrix, using QR decomposition
    '''
    kraus_ops = []
    Q, R = qr(unitary_matrix)

    #Q: a 2^N x 2^N matrix, N is the number of qubits
    for q in Q:
        q = np.expand_dims(q, 1)
        kraus_ops.append(q @ np.transpose(np.conjugate(q)))
    return tf.convert_to_tensor(kraus_ops)

def create_unitary_matrix(num_qubits: int):
    """
    Generate a random unitary matrix of size 2^n x 2^n.
    """
    dimension = 2 ** num_qubits
    # Generate a random complex matrix
    random_matrix = np.random.normal(size=(dimension, dimension)) + 0j
    # Perform QR decomposition
    q, _ = qr(random_matrix)
    return q

def create_circuit(num_qubits: int):
    """
        Create a ansatz V with n qubits and assign its parameters
        Args:
        - num_qubits (int): number of qubits

        Returns:
        - qiskit.QuantumCircuit: parameter assigned Quantum circuit
    """
    #Assign random parameter
    circuit = ansatz.graph(num_qubits=num_qubits)
    num_params = circuit.num_parameters
    x0 = 2 * np.pi * np.random.random(num_params)
    circuit = circuit.assign_parameters(dict(zip(circuit.parameters, x0)))
    return circuit

def create_plus_state(num_qubits:int):
    """Initialize a |+⟩^{⊗n} state as rho and a random first set of Kraus operators:
        Args:
        - num_qubits (int): number of qubits

        Returns:
        - DensityMatrix: |+⟩^{⊗n} state
    """
    # Create |+⟩^{⊗n} state
    plus_state = (1/np.sqrt(2)) * np.array([1, 1])
    initial_state_vector = plus_state
    for _ in range(num_qubits - 1):
        initial_state_vector = np.kron(initial_state_vector, plus_state)
    rho_matrix = np.outer(initial_state_vector, initial_state_vector.conj())
    rho = DensityMatrix(rho_matrix)
    return rho
def create_random_state(num_qubits:int):
    """Initialize a |+⟩^{⊗n} state as rho and a random first set of Kraus operators:
        Args:
        - num_qubits (int): number of qubits

        Returns:
        - DensityMatrix: |+⟩^{⊗n} state
    """
    initial_state_vector = np.random.rand(2**num_qubits)
    # Normalize the state vector
    initial_state_vector /= np.linalg.norm(initial_state_vector)

    # Construct the density matrix
    rho_matrix = np.outer(initial_state_vector, np.transpose(np.conjugate(initial_state_vector)))
    rho = DensityMatrix(rho_matrix)
    return rho

def kron_n_identity(n, j, matrix):
    """
    Kronecker product of n identity matrices, except at position j where we place the matrix.
    """
    identity = np.eye(2)
    matrices = [identity] * n
    matrices[j] = matrix
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result