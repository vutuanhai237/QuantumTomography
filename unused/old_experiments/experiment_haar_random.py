from base import generator_gellmann
from base import generator_haar

def init_experiment(n, num_rho, num_kraus):
    d = 2**n

    # Generate epsilon
    epsilon = generator_haar.random_unitary(d)
    #epsilon = epsilon.astype(np.complex64)
    print(f"Generated {epsilon.shape} epsilon.")

    # Generate K list
    kraus_operators = generator_haar.generate_kraus_operators(d, num_kraus)
    #kraus_operators = [tf.cast(K, tf.complex64) for K in kraus_operators]
    print(f"Generated {len(kraus_operators)} of {kraus_operators[0].shape} kraus operators.")

    # Generate 6^n density matrices
    rho_list = generator_haar.generate_n_qubits_rho_haar(n, num_rho)
    print(f"Generated {len(rho_list)} of {rho_list[0].shape} rho.")

    
    return rho_list, epsilon, kraus_operators

from base import epsilon_rho
def calculate_rho2_lists(rho_list, epsilon, kraus_operators):
    rho2 = []
    rho2_kraus = []
    for rho in rho_list:
        rho2.append(epsilon_rho.calculate_from_unitary(rho, epsilon))
        rho2_kraus.append(epsilon_rho.calculate_from_kraus_operators(rho, kraus_operators))
    return rho2, rho2_kraus

import numpy as np
import tensorflow as tf
def write_to_file(filename, data):
    """Write TensorFlow tensor data to a text file without truncation."""
    tensor_data = data.numpy() if isinstance(data, tf.Tensor) else data

    # Open the file and write the tensor data
    with open(filename, 'w') as f:
        if isinstance(data, np.ndarray):
            np.savetxt(f, data, fmt="%.6f")
        elif isinstance(data, list):
            for item in data:
                f.write(f"{item}\n")
        else:
            f.write(str(data))


import re
def parse_tensor_from_file(file_path, shape):
    with open(file_path, 'r') as file:
        # Read the content of the file
        tensor_str = file.read()

    components = re.findall(r"([+-]?\d+\.?\d*[eE]?[+-]?\d*(?:\s*[+-]?\d*\.?\d*[eE]?[+-]?\d*j)?)", tensor_str)
    # Convert the components into a numpy array of complex numbers
    complex_numbers = [complex(c.replace(' ','')) for c in components]

    # Convert the list to a NumPy array and reshape it
    numpy_tensor = np.array(complex_numbers, dtype=np.complex128)

    if (shape == 3):
        numpy_tensor = numpy_tensor.reshape(round(numpy_tensor.size ** (1/shape)),round(numpy_tensor.size ** (1/shape)),round(numpy_tensor.size ** (1/shape)))
    if (shape == 2):
        size = numpy_tensor.size
        new_shape = (round(size ** (1/shape)), round(size ** (1/shape))) if shape == 2 else (size,)
        numpy_tensor = numpy_tensor.reshape(new_shape)
    # Convert NumPy array to TensorFlow tensor
    tf_tensor = tf.convert_to_tensor(numpy_tensor)

    return tf_tensor

import os
from base import optimize_algorithm
from base import metrics
#1 - 0.05, 2 - 0.1, 3 - 0.01
experiment_folder = 'results/experiment_new/haar_random_temp'

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"TensorFlow is using the following GPU(s): {physical_devices}")
else:
    print("No GPU found. TensorFlow will use CPU.")

num_rhos = [-1, 3**1, 3**2, 3**3, 3**4, 3**5, 3**6, 800, 800, 800, 800]   
alphas = [-1, 0.01, 0.01, 0.008, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.0013, 0.001]   
for num_qubits in range(9, 11):
    if (experiment_folder == ''):
        break
    else:
        write_folder = os.path.join(experiment_folder, str(num_qubits) + "_qubits")
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
    print(f"N={num_qubits}")

    #-----Init experiment-----
    rho_list, epsilon, kraus_operators = init_experiment(num_qubits, num_rhos[num_qubits], 1)

    #-----Learn kraus operators-----
    kraus_operators_res, cost_dict = optimize_algorithm.optimize_adam_kraus_set(rho_list, epsilon, kraus_operators, num_qubits, alphas[num_qubits], num_loop=200)
    
    #-----Calculate result data-----
    rho3_list = epsilon_rho.calculate_set_from_kraus_operators(kraus_operators_res, rho_list, epsilon)
    rho2_list, rho2_kraus_list = calculate_rho2_lists(rho_list, epsilon, kraus_operators_res)

    mean_fidelity_rho_rho3 = metrics.mean_fidelity(rho3_list, rho_list)
    mean_fidelity_rho2_rho2 = metrics.mean_fidelity(rho2_kraus_list, rho2_list)

    print(kraus_operators_res)
    print(epsilon)
    print(mean_fidelity_rho_rho3)
    # #-----Write to folder-----
    write_to_file(os.path.join(write_folder, "rho_list.txt"), rho_list)
    write_to_file(os.path.join(write_folder,"epsilon.txt"), epsilon)
    
    write_to_file(os.path.join(write_folder,"cost_dict.txt"), cost_dict)

    write_to_file(os.path.join(write_folder,"rho3_list.txt"), rho3_list)
    write_to_file(os.path.join(write_folder,"rho2_list.txt"), rho2_list)
    write_to_file(os.path.join(write_folder,"rho2_kraus_list.txt"), rho2_kraus_list)

    write_to_file(os.path.join(write_folder,"mean_fidelity_rho_rho3.txt"), mean_fidelity_rho_rho3.numpy())
    write_to_file(os.path.join(write_folder,"mean_fidelity_rho2_rho2.txt"), mean_fidelity_rho2_rho2.numpy())

    np.set_printoptions(threshold=np.inf)
    write_to_file(os.path.join(write_folder,"kraus_operators.txt"), kraus_operators_res)
    np.set_printoptions(threshold=1000)


# 6^n rho

    
    