from . import optimizer as op
from . import generator_haar
import tensorflow as tf
import base.metrics as metrics
import base.epsilon_rho as epsilon_rho

def optimize_adam_kraus_set(rho_list, unitary, kraus_operators, num_qubits, alpha=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_loop=1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    
    # Initialize m, v to zero matrices
    m = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    v = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):

        kraus_operators_copy, m, v, cost = op.calculate_adam_kraus_set(rho_list, unitary, kraus_operators_copy, m = m, v = v, t = i, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
        print(cost)
        
        kraus_operators_copy = generator_haar.normalize_kraus(kraus_operators_copy)
        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)
        # Reshape the matrices
        m = tf.reshape(m, (2**num_qubits, 2**num_qubits, 2**num_qubits))
        v = tf.reshape(v, (2**num_qubits, 2**num_qubits, 2**num_qubits))

    return kraus_operators_copy, cost_dict

def optimize_adam_unitary_dagger(rho, rho2, unitary, alpha=0.001, beta1=0.9, beta2=0.999, epsilon = 1e-8, num_loop=1000):
    unitary_copy = tf.identity(unitary)
    
    # Initialize m, v to zero matrices
    m = tf.zeros_like(unitary_copy, dtype=tf.complex128)
    v = tf.zeros_like(unitary_copy, dtype=tf.complex128)
    
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):
        # Update Kraus Operators
        unitary_copy, m, v, cost = op.calculate_adam_unitary_dagger(rho=rho, rho2=rho2, unitary=unitary_copy, m = m, v = v, t = i, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)

        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)

    return unitary_copy, cost_dict


def optimize_adam_kraus(rho, kraus_operators, unitary, num_qubits, alpha=0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_loop=1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    
    # Initialize m, v to zero matrices
    m = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    v = tf.zeros_like(kraus_operators_copy, dtype=tf.complex128)
    
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):
        # Update Kraus Operators
        kraus_operators_copy, m, v, cost = op.calculate_adam_kraus(rho=rho, kraus_operators=kraus_operators_copy, unitary=unitary, m = m, v = v, t = i, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
        
        # Reshape the matrices
        kraus_operators_copy = generator_haar.normalize_kraus(kraus_operators_copy)
        m = tf.reshape(m, (2**num_qubits, 2**num_qubits, 2**num_qubits))
        v = tf.reshape(v, (2**num_qubits, 2**num_qubits, 2**num_qubits))

        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)

    return kraus_operators_copy, cost_dict



def optimize_derivative_kraus(rho, kraus_operators, unitary, num_qubits, alpha=0.001, num_loop = 1000):
    kraus_operators_copy = tf.identity(kraus_operators)
    # try looping manually
    cost_dict = []
    for i in range (0, num_loop):
        #Update Kraus Operators
        kraus_operators_copy, cost= op.calculate_derivative_kraus(rho=rho, kraus_operators=kraus_operators_copy, unitary=unitary, num_qubits=num_qubits, alpha=alpha)
        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)

        #Reshape
        kraus_operators_copy = generator_haar.normalize_kraus(kraus_operators_copy)

        
    return kraus_operators_copy, cost_dict

def optimize_derivative_unitary_dagger(rho, rho2, unitary, alpha=0.001, num_loop=1000):
    unitary_copy = tf.identity(unitary)
    # Initialize a dictionary to track cost at each iteration
    cost_dict = []

    # Try looping manually
    for i in range(num_loop):
        # Update Kraus Operators
        unitary_copy, cost = op.calculate_derivative_unitary_dagger(rho=rho, rho2=rho2, unitary_matrix=unitary_copy, alpha=alpha)
    
        # Store the cost for this iteration
        cost_dict.append(cost.numpy().real)


    return unitary_copy, cost_dict




