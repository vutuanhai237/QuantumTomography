from . import epsilon_rho
from . import generator_haar
import numpy as np
import tensorflow as tf

def compute_simulated_data(M_list, rho_list, epsilon):
    """Compute d_{i,j} = Tr(M_j E_rand(rho_i))"""

    data = np.zeros((len(rho_list), len(M_list)), dtype=complex)
    
    for i, rho in enumerate(rho_list):
        rho_transformed = epsilon_rho.calculate_from_unitary(rho=rho, unitary_matrix=epsilon)  # Apply channel
        for j, M in enumerate(M_list):
            data[i, j] = np.trace(M @ rho_transformed)  # Compute expectation value
    
    return data

def diff_measure_rho(d_ij, M_list, rho_list, kraus_operators):
    """Compute loss = Σ (d_{i,j} - Tr(M_j * E(rho_i)))^2"""
    
    loss = tf.constant(0.0, dtype=tf.complex128)  # Initialize loss
    
    for i, rho in enumerate(rho_list):
        rho_transformed = epsilon_rho.calculate_from_kraus_operators(
            rho=rho, kraus_operators=kraus_operators
        )  # Apply channel
        
        for j, M in enumerate(M_list): 
            predicted = tf.linalg.trace(tf.linalg.matmul(M, rho_transformed))  # Tr( M_j * rho_i' )
            diff = d_ij[i, j] - predicted
            loss += diff ** 2  # Squared absolute error
    
    return tf.math.real(loss)

def calculate_adam_kraus(M_list, rho_list, unitary, kraus_operators, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'fidelity'):
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)

    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        data = compute_simulated_data(M_list, rho_list, unitary)
        f = diff_measure_rho(data, M_list, rho_list, tensorKraus)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Calculate projection
    proj = c - tensorKraus @ (np.transpose(np.conjugate(c)) @ tensorKraus + np.transpose(np.conjugate(tensorKraus)) @ c) / 2

    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_kraus_operators = tensorKraus - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    return updated_kraus_operators, m, v, f


def optimize_adam_kraus(M_list, rho_list, unitary, kraus_operators, num_qubits, alpha=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_loop=1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    
    # Initialize m, v to zero matrices
    m = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    v = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):
        kraus_operators_copy, m, v, cost = calculate_adam_kraus(M_list, rho_list, unitary, kraus_operators_copy, m = m, v = v, t = i, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
        print(cost)
        
        
        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)
        
        # Reshape the matrices
        kraus_operators_copy = generator_haar.normalize_kraus(kraus_operators_copy)
        m = tf.reshape(m, (2**num_qubits, 2**num_qubits, 2**num_qubits))
        v = tf.reshape(v, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    return kraus_operators_copy, cost_dict