import numpy as np

def calculate_dephasing(input_rho, num_qubits: int, gamma: float): #for verification
    '''
    Calculate rho2 by dephasing
    '''
    
    # Convert DensityMatrix to numpy array
    rho = input_rho.data
    # Define the Pauli-Z matrix
    sigma_z = np.array([[1, 0], [0, -1]])

    # Calculate the factors
    alpha = (1 + np.sqrt(1 - gamma)) * 1/2
    beta = (1 - np.sqrt(1 - gamma)) * 1/2
    
    # n qubits => qubit thứ n => I @ I @ .... sigma_z (n) @....I
    # Loop for multiple qubits
    for i in range(num_qubits):
        # Create the tensor product of Pauli-Z matrices for all qubits
        sigma_z_i = np.eye(1)
        for j in range(num_qubits):
            if j == i:
                sigma_z_i = np.kron(sigma_z_i, sigma_z)
            else:
                sigma_z_i = np.kron(sigma_z_i, np.eye(2))
        
        # Apply the dephasing formula
        rho = alpha * rho + beta * (sigma_z_i @ rho @ sigma_z_i)
        # Normalize the density matrix (optional, depending on the context)
        # rho /= np.trace(rho)

    return rho

def calculate_from_unitary(rho, unitary_matrix):
    '''
    Calculate rho' by applying U @ rho @ U(dagger)
    '''
    rho_2 = unitary_matrix  @ rho.data @ np.transpose(np.conjugate(unitary_matrix))

    return rho_2

def calculate_from_kraus_operators(rho, kraus_operators):
    '''
    Calculate rho' by applying K @ rho @ K(dagger)
    '''
    rho_2 = sum(K @ rho @ np.transpose(np.conjugate(K)) for K in kraus_operators)

    return rho_2