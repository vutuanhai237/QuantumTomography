import tensorflow as tf
import numpy as np
from . import metrics as mtr
from . import epsilon_rho as epsilon_rho
#cost func to compare 2 given rhos
def cost_frobe(rho, rho_3):
    return tf.square(mtr.frobenius_norm(rho, rho_3))

#cost func to compare 2 given rhos
def cost_fidelity(rho, rho_3):
    return 1 - mtr.compilation_trace_fidelity(rho, rho_3)

# Adam optimizer function for updating Kraus operators
def calculate_adam_unitary_dagger(rho, rho2, unitary, m, v, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'frobe'):
    tensorUnitary = tf.Variable(unitary, dtype=tf.complex128)
    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, tensorUnitary)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorUnitary)

    # Calculate projection
    proj = c - tensorUnitary @ (np.transpose(np.conjugate(c)) @ tensorUnitary + np.transpose(np.conjugate(tensorUnitary)) @ c) / 2

    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_unitary = tensorUnitary - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)

    rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, updated_unitary)
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_unitary, m, v, cost

# Adam optimizer function for updating Kraus operators
def calculate_adam_kraus(rho, kraus_operators, unitary, m, v, num_qubits, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'fidelity'):
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)

    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho2 = epsilon_rho.calculate_from_kraus_operators(rho, tensorKraus)
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, unitary)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits * 2**num_qubits, 2**num_qubits))
    tensorKraus = tf.reshape(tensorKraus, (2**num_qubits * 2**num_qubits, 2**num_qubits))
    # Calculate projection
    proj = c - tensorKraus @ (np.transpose(np.conjugate(c)) @ tensorKraus + np.transpose(np.conjugate(tensorKraus)) @ c) / 2
    proj = tf.reshape(proj, (2**num_qubits, 2**num_qubits, 2**num_qubits))
    tensorKraus = tf.reshape(tensorKraus, (2**num_qubits, 2**num_qubits, 2**num_qubits))
    # Update Adam variables
    m = beta1 * m + (1 - beta1) * proj
    v = beta2 * v + (1 - beta2) * tf.math.square(proj)

    # Bias correction
    m_hat = m / (1 - tf.pow(beta1, t + 1))
    v_hat = v / (1 - tf.pow(beta2, t + 1))

    # Update the Kraus operators using Adam update rule
    updated_kraus_operators = tensorKraus - alpha * m_hat / (tf.math.sqrt(v_hat) + epsilon)
    rho2 = epsilon_rho.calculate_from_kraus_operators(rho, tensorKraus)
    rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, unitary)
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_kraus_operators, m, v, cost
    
def calculate_derivative_kraus(rho, kraus_operators, unitary, num_qubits, alpha=0.001, mode = 'frobe'):
    tensorKraus = tf.Variable(kraus_operators)
    with tf.GradientTape() as tape:
        rho2 = epsilon_rho.calculate_from_kraus_operators(rho, tensorKraus)
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, unitary)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    c = tf.reshape(c, (2**num_qubits * 2**num_qubits, 2**num_qubits))
    tensorKraus = tf.reshape(tensorKraus, (2**num_qubits * 2**num_qubits, 2**num_qubits))
    # Calculate projection
    proj = c - tensorKraus @ (np.transpose(np.conjugate(c)) @ tensorKraus + np.transpose(np.conjugate(tensorKraus)) @ c) / 2
    proj = tf.reshape(proj, (2**num_qubits, 2**num_qubits, 2**num_qubits))
    tensorKraus = tf.reshape(tensorKraus, (2**num_qubits, 2**num_qubits, 2**num_qubits))
    # Update the Kraus operators
    updated_kraus_operators = tensorKraus - alpha * proj
    
    rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, updated_kraus_operators)
    # Return the updated Kraus operators
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_kraus_operators, cost

def calculate_derivative_unitary_dagger(rho, rho2, unitary_matrix, alpha=0.001, mode = 'frobe'):
    tensorUnitary = tf.Variable(unitary_matrix, dtype=tf.complex128)
    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, tensorUnitary)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorUnitary)

    # Calculate projection
    proj = c - tensorUnitary @ (np.transpose(np.conjugate(c)) @ tensorUnitary + np.transpose(np.conjugate(tensorUnitary)) @ c) / 2

    # Update the Kraus operators
    updated_unitary = tensorUnitary - alpha * proj
    
    rho3 = epsilon_rho.calculate_from_unitary_dagger(rho2, updated_unitary)
    # Return the updated Kraus operators
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_unitary, cost
