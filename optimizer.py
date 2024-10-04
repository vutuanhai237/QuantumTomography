import tensorflow as tf
import numpy as np
import metrics as mtr
import epsilon_rho
#cost func to compare 2 given rhos
def cost(rho, rho_3):
    return tf.square(mtr.frobenius_norm(rho, rho_3))
    #return 1 - compilation_trace_fidelity(rho, rho_3)

# Adam optimizer function for updating Kraus operators
def calculate_derivative_adam(rho, rho2, kraus_operators, m, v, num_qubits, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)

    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, tensorKraus)
        f = cost(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits, 2**num_qubits, 2**num_qubits))

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

    return updated_kraus_operators, m, v

'''
    rho: The initial rho
    unitary_matrix: randomly generated circle in initialization step
    kraus_operators: set of kraus operators
    n: number of qubits
    alpha: learning rate (?)'''
def calculate_derivative(rho, rho2, kraus_operators, num_qubits, alpha=0.001):
    tensorKraus = tf.Variable(kraus_operators)
    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, tensorKraus)
        f = cost(rho, rho3) 
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    # Calculate projection
    proj = c - tensorKraus @ (np.transpose(np.conjugate(c)) @ tensorKraus + np.transpose(np.conjugate(tensorKraus)) @ c) / 2

    # Update the Kraus operators
    updated_kraus_operators = tensorKraus - alpha * proj
    
    # Return the updated Kraus operators
    return updated_kraus_operators
