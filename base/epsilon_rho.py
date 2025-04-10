import numpy as np
import cupy as cp
import tensorflow as tf
import base.generator as generator
def calculate_dephasing(input_rho, num_qubits: int, gamma: float): #for verification
    '''
    Calculate rho2 by dephasing
    '''
    
    # Convert DensityMatrix to numpy array
    rho = input_rho.data
    # Define the Pauli-Z matrix
    sigma_z = cp.array([[1, 0], [0, -1]])

    # Calculate the factors
    alpha = (1 + cp.sqrt(1 - gamma)) * 1/2
    beta = (1 - cp.sqrt(1 - gamma)) * 1/2
    
    # n qubits => qubit thứ n => I @ I @ .... sigma_z (n) @....I
    # Loop for multiple qubits
    for i in range(num_qubits):
        # Create the tensor product of Pauli-Z matrices for all qubits
        sigma_z_i = cp.eye(1)
        for j in range(num_qubits):
            if j == i:
                sigma_z_i = cp.kron(sigma_z_i, sigma_z)
            else:
                sigma_z_i = cp.kron(sigma_z_i, cp.eye(2))
        
        # Apply the dephasing formula
        rho = alpha * rho + beta * (sigma_z_i @ rho @ sigma_z_i)
        # Normalize the density matrix (optional, depending on the context)
        # rho /= cp.trace(rho)

    return rho

def calculate_from_unitary(rho, unitary_matrix):
    '''
    Calculate rho' by applying U @ rho @ U(dagger)
    '''

    rho_2 = tf.matmul(tf.matmul(unitary_matrix, rho), tf.transpose(tf.math.conj(unitary_matrix)))
    #rho_2 = unitary_matrix  @ rho @ cp.transpose(cp.conjugate(unitary_matrix))

    return rho_2

def calculate_from_unitary_dagger(rho, unitary_matrix):
    '''
    Calculate rho' by applying U @ rho @ U(dagger)
    '''

    rho_2 = tf.matmul(tf.matmul(tf.transpose(tf.math.conj(unitary_matrix)), rho), unitary_matrix)
    
    #rho_2 =  cp.transpose(cp.conjugate(unitary_matrix)) @ rho @ unitary_matrix 

    return rho_2

def apply_amplitude_noise(input_rho, num_qubits, gamma):
    rho = input_rho.copy()
    for k in range(num_qubits):
        K0_k= cp.array([[1, 0], [0, cp.sqrt(1 - gamma)]])
        K1_k= cp.array([[0, cp.sqrt(gamma)], [0, 0]])
        K0 = generator.kron_n_identity(num_qubits, k, K0_k)
        K1 = generator.kron_n_identity(num_qubits, k, K1_k)
        rho = K0 @ rho @ cp.transpose(cp.conjugate(K0)) + K1 @ rho @ cp.transpose(cp.conjugate(K1))
    return rho

def calculate_from_kraus_operators(rho, kraus_operators):
    '''
    Calculate rho' by applying K @ rho @ K(dagger)
    '''

    rho_2 = sum(tf.matmul(tf.matmul(K, rho), tf.transpose(tf.math.conj(K))) for K in kraus_operators)

    return rho_2

def calculate_set_from_kraus_operators(kraus_operators, rho_list, epsilon):
    """Compute rho_f_i = E_rand(sum(K@rho_i@K_dagger))"""

    data = []
    
    for i, rho in enumerate(rho_list):

        rho2 = calculate_from_kraus_operators(rho, kraus_operators)
        data.append(calculate_from_unitary_dagger(rho2, epsilon))
        
    return data 

def calculate_set_from_unitary_dagger(unitary, rho2_list):
    """Compute rho_f_i = E_rand^-1(rho2)"""

    data = []
    
    for i, rho2 in enumerate(rho2_list):

        rho3 = calculate_from_unitary_dagger(rho2, unitary)
        data.append(rho3)
        
    return data 