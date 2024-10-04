import optimizer as op
import tensorflow as tf

def optimize_adam(rho, rho2, kraus_operators, num_qubits, alpha=0.001, num_loop = 1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    # Initialize m, v to zero matrices
    m = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    v = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    # try looping manually
    for i in range (0, num_loop):
        #Update Kraus Operators
        kraus_operators_copy, m, v =op.calculate_derivative_adam(rho, rho2, kraus_operators_copy, m, v, num_qubits, i, alpha)

        #Reshape
        kraus_operators_copy = tf.reshape(kraus_operators_copy, (2**num_qubits,2**num_qubits,2**num_qubits))
        m = tf.reshape(m, (2**num_qubits,2**num_qubits,2**num_qubits))
        v = tf.reshape(v, (2**num_qubits,2**num_qubits,2**num_qubits))

    return kraus_operators_copy


def optimize_derivative(rho, rho2, kraus_operators, num_qubits, alpha=0.001, num_loop = 1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    # try looping manually
    for i in range (0, num_loop):
        #Update Kraus Operators
        kraus_operators_copy=op.calculate_derivative(rho, rho2, kraus_operators_copy, num_qubits, alpha)

        #Reshape
        kraus_operators_copy = tf.reshape(kraus_operators_copy, (2**num_qubits,2**num_qubits,2**num_qubits))
        
    return kraus_operators_copy
