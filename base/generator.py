import numpy as np
import tensorflow as tf
from qoop.core import ansatz

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