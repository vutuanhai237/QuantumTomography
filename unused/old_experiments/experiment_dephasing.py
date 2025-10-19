from base import generator_gellmann
from base import generator_haar

def init_experiment(n, num_rho):
    d = 2**n

    # Generate epsilon
    unitary = generator_haar.random_unitary(d)
    print(f"Generated {unitary.shape} unitary.")

    # Generate 6^n density matrices
    rho_list = generator_haar.generate_n_qubits_rho_haar(n, num_rho)
    print(f"Generated {len(rho_list)} of {rho_list[0].shape} rho.")

    return rho_list, unitary

import numpy as np
import tensorflow as tf

from base import epsilon_rho
def calculate_rho2_unitary(rho_list, unitary):
    rho2_unitary = []
    for rho in rho_list:
        rho2_unitary.append(epsilon_rho.calculate_from_unitary(rho, unitary))
    return rho2_unitary

def calculate_rho2_dephasing(rho_list, n, gamma):
    rho2 = []
    for rho in rho_list:
        rho2.append(epsilon_rho.calculate_dephasing(rho, n, gamma))
    return rho2

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

import os
from base import optimize_algorithm
from base import metrics
experiment_folder = 'results/experiment_new/dephasing_temp'

num_rhos = [-1, 3**1, 3**2, 3**3, 3**4, 3**5, 3**6, 800, 800, 800, 800]   
alphas = [-1, 0.01, 0.01, 0.008, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.0013, 0.001]  
limits = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  
for num_qubits in range(7, 10):
    if (experiment_folder == ''):
        break
    else:
        write_folder = os.path.join(experiment_folder, str(num_qubits) + "_qubits")
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
    print(f"N={num_qubits}")

    #-----Init experiment-----
    rho_list, unitary = init_experiment(num_qubits, num_rhos[num_qubits])
    write_to_file(os.path.join(write_folder, "rho_list.txt"), rho_list)

    g_s = np.linspace(1, 10e-3, 20)
    g_index = 0

    while g_index < len(g_s):
        g = g_s[g_index]
        folder_path = os.path.join(write_folder, "_{:.2f}".format(g))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        rho2_list = calculate_rho2_dephasing(rho_list, num_qubits, g)
    
        #-----Learn kraus operators-----
        unitary_res, cost_dict = optimize_algorithm.optimize_adam_unitary_dagger_set(rho_list, rho2_list, unitary, alphas[num_qubits], num_loop=200)
    
        #-----Calculate result data-----
        rho3_list = epsilon_rho.calculate_set_from_unitary_dagger(unitary_res, rho2_list)
        rho2_unitary_list = calculate_rho2_unitary(rho_list, unitary_res)
    
        mean_fidelity_rho_rho3 = metrics.mean_fidelity(rho3_list, rho_list)
        mean_fidelity_rho2_rho2 = metrics.mean_fidelity(rho2_unitary_list, rho2_list)

        g_index = g_index + 1

        #-----Write to folder-----    
        write_to_file(os.path.join(folder_path,"unitary.txt"), unitary)
        write_to_file(os.path.join(folder_path,"unitary_res.txt"), unitary_res)
        write_to_file(os.path.join(folder_path,"cost_dict.txt"), cost_dict)

        write_to_file(os.path.join(folder_path,"rho2_list.txt"), rho2_list)
        write_to_file(os.path.join(folder_path,"rho2_unitary_list.txt"), rho2_unitary_list)

        write_to_file(os.path.join(folder_path,"mean_fidelity_rho_rho3.txt"), mean_fidelity_rho_rho3.numpy())
        write_to_file(os.path.join(folder_path,"mean_fidelity_rho2_rho2.txt"), mean_fidelity_rho2_rho2.numpy())

        print(g, num_qubits)
        print(cost_dict[-1])

    
    

