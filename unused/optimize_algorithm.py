''' OLD FUNC
def optimize_adam(rho, rho2, kraus_operators, num_qubits, alpha=0.001, num_loop=1000, mode = 'frobe'):
    if (isModeValid(mode) == False):
        raise Exception("Invalid cost mode, should be 'frobe', 'fidelity'")
    kraus_operators_copy = tf.identity(kraus_operators)
    
    # Initialize m, v to zero matrices
    m = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    v = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):
        # Update Kraus Operators
        kraus_operators_copy, m, v, cost = op.calculate_derivative_adam(rho, rho2, kraus_operators_copy, m, v, num_qubits, i, alpha)
        
        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)
        
        # Reshape the matrices
        kraus_operators_copy = tf.reshape(kraus_operators_copy, (2**num_qubits, 2**num_qubits, 2**num_qubits))
        m = tf.reshape(m, (2**num_qubits, 2**num_qubits, 2**num_qubits))
        v = tf.reshape(v, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    return kraus_operators_copy, cost_dict
'''