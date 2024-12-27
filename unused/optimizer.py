''' OLD FUNC
#Adam optimizer function for updating Kraus operators
def calculate_derivative_adam(rho, rho2, kraus_operators, m, v, num_qubits, t, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, mode = 'frobe'):
    tensorKraus = tf.Variable(kraus_operators, dtype=tf.complex128)

    beta1 = tf.constant(beta1, dtype=tf.complex128)
    beta2 = tf.constant(beta2, dtype=tf.complex128)
    t = tf.constant(t, dtype=tf.complex128)

    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, tensorKraus)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
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

    rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, updated_kraus_operators)
    
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_kraus_operators, m, v, cost

def calculate_derivative_kraus(rho, rho2, kraus_operators, num_qubits, alpha=0.001, mode = 'frobe'):
    tensorKraus = tf.Variable(kraus_operators)
    with tf.GradientTape() as tape:
        rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, tensorKraus)
        if (mode == 'frobe'):
            f = cost_frobe(rho, rho3)
        elif (mode == 'fidelity'):
            f = cost_fidelity(rho, rho3)
    
    # Calculate the gradient
    c = tape.gradient(f, tensorKraus)
    
    # Ensure proper reshaping (ensure this matches the dimensions of your system)
    c = tf.reshape(c, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    # Calculate projection
    proj = c - tensorKraus @ (np.transpose(np.conjugate(c)) @ tensorKraus + np.transpose(np.conjugate(tensorKraus)) @ c) / 2

    # Update the Kraus operators
    updated_kraus_operators = tensorKraus - alpha * proj
    
    rho3 = epsilon_rho.calculate_from_kraus_operators(rho2, updated_kraus_operators)
    # Return the updated Kraus operators
    if (mode == 'frobe'):
        cost = cost_frobe(rho, rho3)
    elif (mode == 'fidelity'):
        cost = cost_fidelity(rho, rho3)
    return updated_kraus_operators, cost
'''