import tensorflow as tf
import numpy as np
import base.lost_func as lost_func
from . import epsilon_rho as epsilon_rho
import cupy as cp

def calculate_adam_kraus_set(rho_list, unitary, kraus_operators, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'fidelity'):
    # tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex64)
    
    # beta1 = tf.constant(beta1, dtype=tf.complex64)
    # beta2 = tf.constant(beta2, dtype=tf.complex64)
    # t = tf.constant(t, dtype=tf.complex64)
    
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)
        
    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)


    with tf.GradientTape() as tape:
        data = epsilon_rho.calculate_set_from_kraus_operators(tensorKraus, rho_list, unitary)
        f = lost_func.diff_infidelity(data, rho_list)

    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    c_adj = tf.linalg.adjoint(c)  # conjugate transpose
    tensorKraus_adj = tf.linalg.adjoint(tensorKraus)

    term1 = tf.matmul(c_adj, tensorKraus)         # (batch, d, d)
    term2 = tf.matmul(tensorKraus_adj, c)         # (batch, d, d)
    sum_terms = term1 + term2

    proj = c - 0.5 * tf.matmul(tensorKraus, sum_terms)
    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_kraus_operators = tensorKraus - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    return updated_kraus_operators, m, v, f

def calculate_adam_unitary_dagger_set(rho_list, rho2_list, unitary, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'fidelity'):
    tensorUnitary = tf.Variable(unitary, dtype=tf.complex128)
    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        data = epsilon_rho.calculate_set_from_unitary_dagger(tensorUnitary, rho2_list)
        f = lost_func.diff_infidelity(data, rho_list)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorUnitary)

    # Calculate projection
    #proj = c - tensorUnitary @ (cp.transpose(cp.conjugate(c)) @ tensorUnitary + cp.transpose(cp.conjugate(tensorUnitary)) @ c) / 2
    proj = c - tf.matmul(tensorUnitary, tf.matmul(tf.transpose(tf.math.conj(c)), tensorUnitary) + tf.transpose(tf.math.conj(tensorUnitary)) @ c) / 2
    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_unitary = tensorUnitary - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    return updated_unitary, m, v, f

def calculate_adam_unitary_dagger(rho, rho2, unitary, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    tensorUnitary = tf.Variable(unitary, dtype=tf.complex128)
    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, tensorUnitary)
        f = lost_func.diff_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorUnitary)

    # Calculate projection
    proj = c - tensorUnitary @ (cp.transpose(cp.conjugate(c)) @ tensorUnitary + cp.transpose(cp.conjugate(tensorUnitary)) @ c) / 2

    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_unitary = tensorUnitary - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    return updated_unitary, m, v, f

# Adam optimizer function for updating Kraus operators
def calculate_adam_kraus(rho, kraus_operators, unitary, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)

    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho2 = epsilon_rho.calculate_from_kraus_operators(rho, tensorKraus)
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, unitary)
        f = lost_func.diff_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)

    # Calculate projection
    proj = c - tensorKraus @ (cp.transpose(cp.conjugate(c)) @ tensorKraus + cp.transpose(cp.conjugate(tensorKraus)) @ c) / 2

    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_kraus_operators = tensorKraus - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    return updated_kraus_operators, m, v, f
    
def calculate_derivative_kraus(rho, kraus_operators, unitary, num_qubits, alpha=0.001):
    tensorKraus = tf.Variable(kraus_operators)
    with tf.GradientTape() as tape:
        rho2 = epsilon_rho.calculate_from_kraus_operators(rho, tensorKraus)
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, unitary)
        f = lost_func.diff_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    # Calculate projection
    proj = c - tensorKraus @ (cp.transpose(cp.conjugate(c)) @ tensorKraus + cp.transpose(cp.conjugate(tensorKraus)) @ c) / 2

    # Update the Kraus operators
    updated_kraus_operators = tensorKraus - alpha * proj
    return updated_kraus_operators, f

def calculate_derivative_unitary_dagger(rho, rho2, unitary_matrix, alpha=0.001):
    tensorUnitary = tf.Variable(unitary_matrix, dtype=tf.complex128)
    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, tensorUnitary)
        f = lost_func.diff_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorUnitary)

    # Calculate projection
    proj = c - tensorUnitary @ (cp.transpose(cp.conjugate(c)) @ tensorUnitary + cp.transpose(cp.conjugate(tensorUnitary)) @ c) / 2

    # Update the Kraus operators
    updated_unitary = tensorUnitary - alpha * proj
    return updated_unitary, f
