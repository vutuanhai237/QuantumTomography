from base import generator_haar

def init_experiment(n):
    d = 2**n

    # Generate 6^n density matrices
    rho_list = generator_haar.generate_n_qubits_rho_haar(n)
    print(f"Generated {len(rho_list)} of {rho_list[0].shape} rho.")

    # Generate unitary
    unitary = generator_haar.random_unitary(d)
    print(f"Generated {unitary.shape} unitary operators.")
    return rho_list, unitary

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

import numpy as np

def custom_steps():
    t1 = np.arange(0, 1, 0.05)        # very fine for 0–1 (initial fast changes)
    t2 = np.arange(1, 5, 0.2)         # still fine-grained
    t3 = np.arange(5, 15, 0.5)        # mid-range
    t4 = np.arange(15, 30, 1)         # coarser
    t5 = np.arange(30, 60, 2.5)       # even coarser
    t6 = np.arange(60, 101, 5)        # late regime — saturation zone

    t_all = np.concatenate([t1, t2, t3, t4, t5, t6])
    t_unique_sorted = np.unique(np.round(t_all, decimals=8))  # clean up precision & duplicates

    return t_unique_sorted


import os
from base import optimize_algorithm
from base import metrics
experiment_folder_expected = 'results/experiment_new/traceX_expected'
experiment_folder_result = 'results/experiment_new/traceX_result'

rho_list = [
    np.array([[0.14364795+2.57800736e-17j, 0.34609293-5.68586052e-02j],
              [0.34609293+5.68586052e-02j, 0.85635205+2.62258235e-17j]]),

    np.array([[0.18453881-2.09650606e-17j, 0.32984386-2.04174601e-01j],
              [0.32984386+2.04174601e-01j, 0.81546119-3.62109073e-17j]]),

    np.array([[ 0.97969528+2.26771768e-17j, -0.11034934+8.78376829e-02j],
              [-0.11034934-8.78376829e-02j,  0.02030472+1.48518504e-18j]]),

    np.array([[ 0.96087281+6.89577879e-17j, -0.02912983-1.91696906e-01j],
              [-0.02912983+1.91696906e-01j,  0.03912719-3.66353739e-18j]]),
   
    np.array([[ 0.03131311+2.38095825e-17j, -0.15769847-7.39174724e-02j],
              [-0.15769847+7.39174724e-02j,  0.96868689-2.09847285e-17j]]),

    np.array([[ 0.05876332+7.52428375e-18j, -0.2180843 +8.80308368e-02j],
              [-0.2180843 -8.80308368e-02j,  0.94123668-3.55421136e-17j]])
    ]

unitary = np.array([
        [(-0.658611+0.000000j),  (-0.752483+0.000000j)],
        [(-0.752483-0.000000j),  (0.658611+0.000000j)]
    ])
beta = 0.08

for num_qubits in range(1, 2):
    if (experiment_folder_expected == ''):
        break
    # if (experiment_folder_result == ''):
    #     break
    print(f"N={num_qubits}")
    
    # Create 50 t from 0 to 48
    t_s_expect = np.linspace(100, 0, 400)
    # t_s = [0, 5, 10]

    t_s_result = custom_steps()
    print(t_s_expect)
    print(t_s_result)
    for t in t_s_expect:
        t=round(t, 1)
        
        folder_path_a = os.path.join(experiment_folder_expected, str(num_qubits)+"_qubits_"+"{:06.2f}".format(t)+'_a')
        folder_path_b = os.path.join(experiment_folder_expected, str(num_qubits)+"_qubits_"+"{:06.2f}".format(t)+'_b')
        if not os.path.exists(folder_path_a):
            os.makedirs(folder_path_a)
        if not os.path.exists(folder_path_b):
            os.makedirs(folder_path_b)
        g_a = 1 - np.exp(-2 * beta * t)
        g_b = 1 - np.exp(-2 * beta * t**2)

        print(num_qubits, t)
        rho2_list_a = calculate_rho2_dephasing(rho_list, num_qubits, g_a)
        rho2_list_b = calculate_rho2_dephasing(rho_list, num_qubits, g_b)
        
        rho2_a = rho2_list_a[1]
        rho2_b = rho2_list_b[1]

        pauli_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        # Calculate tr(X1, rho)
        trace_rho2_a = metrics.trace_Pauli(rho2_a, 0, pauli_X)
        trace_rho2_b = metrics.trace_Pauli(rho2_b, 0, pauli_X)

        #-----Write to folder-----    
        write_to_file(os.path.join(folder_path_a,"rho2.txt"), rho2_a)
        write_to_file(os.path.join(folder_path_b,"rho2.txt"), rho2_b)

        write_to_file(os.path.join(folder_path_a,"trace_rho2.txt"), trace_rho2_a)
        write_to_file(os.path.join(folder_path_b,"trace_rho2.txt"), trace_rho2_b)


    # for t in t_s_result:
    #     t=round(t, 1)
        
    #     folder_path_a = os.path.join(experiment_folder_result, str(num_qubits)+"_qubits_"+"{:06.2f}".format(t)+'_a')
    #     folder_path_b = os.path.join(experiment_folder_result, str(num_qubits)+"_qubits_"+"{:06.2f}".format(t)+'_b')
    #     if not os.path.exists(folder_path_a):
    #         os.makedirs(folder_path_a)
    #     if not os.path.exists(folder_path_b):
    #         os.makedirs(folder_path_b)
    #     g_a = 1 - np.exp(-2 * beta * t)
    #     g_b = 1 - np.exp(-2 * beta * t**2)

    #     # Adjust learning rate (alpha) and number of loops based on noise
    #     alpha_a = max(0.001, 0.01 * (1 - g_a))  # smaller learning rate with more noise
    #     alpha_b = max(0.001, 0.01 * (1 - g_b))
        
    #     num_loop_a = int(min(1000, max(100, 500 * (1 + g_a))))  # more loops for higher noise
    #     num_loop_b = int(min(1000, max(100, 500 * (1 + g_b))))

    #     print(num_qubits, t)
    #     rho2_list_a = calculate_rho2_dephasing(rho_list, num_qubits, g_a)
    #     rho2_list_b = calculate_rho2_dephasing(rho_list, num_qubits, g_b)
    #     #-----Learn kraus operators-----
    #     unitary_res_a, costdict_a = optimize_algorithm.optimize_adam_unitary_dagger_set(rho_list, rho2_list_a, unitary, alpha_a, num_loop=num_loop_a)
    #     unitary_res_b, costdict_b = optimize_algorithm.optimize_adam_unitary_dagger_set(rho_list, rho2_list_b, unitary, alpha_b, num_loop=num_loop_b)
        
    #     #-----Calculate result data-----
    #     rho3_list_a = epsilon_rho.calculate_set_from_unitary_dagger(unitary_res_a, rho2_list_a)
    #     rho3_list_b = epsilon_rho.calculate_set_from_unitary_dagger(unitary_res_b, rho2_list_b)
    #     mean_fidelity_a = metrics.mean_fidelity(rho3_list_a, rho_list)
    #     mean_fidelity_b = metrics.mean_fidelity(rho3_list_b, rho_list)

    #     print(g_a, mean_fidelity_a)
    #     print(g_b, mean_fidelity_b)
    #     rho2_a = rho2_list_a[1]
    #     rho2_out_a = epsilon_rho.calculate_from_unitary(rho2_a, unitary_res_a)
    #     rho2_b = rho2_list_b[1]
    #     rho2_out_b = epsilon_rho.calculate_from_unitary(rho2_b, unitary_res_b)
    #     # Define Pauli-X matrix
    #     pauli_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    #     # Calculate tr(X1, rho)
    #     trace_rho2_a = metrics.trace_Pauli(rho2_a, 0, pauli_X)
    #     trace_out_rho2_a = metrics.trace_Pauli(rho2_out_a, 0, pauli_X)
    #     trace_rho2_b = metrics.trace_Pauli(rho2_b, 0, pauli_X)
    #     trace_out_rho2_b = metrics.trace_Pauli(rho2_out_b, 0, pauli_X)

    #     #-----Write to folder-----    
    #     write_to_file(os.path.join(folder_path_a,"costdict.txt"), costdict_a)
    #     write_to_file(os.path.join(folder_path_b,"costdict.txt"), costdict_b)
        
    #     write_to_file(os.path.join(folder_path_a,"unitary_res.txt"), unitary_res_a)
    #     write_to_file(os.path.join(folder_path_b,"unitary_res.txt"), unitary_res_b)

    #     write_to_file(os.path.join(folder_path_a,"rho2.txt"), rho2_a)
    #     write_to_file(os.path.join(folder_path_b,"rho2.txt"), rho2_b)
    #     write_to_file(os.path.join(folder_path_a,"rho2_out.txt"), rho2_out_a)
    #     write_to_file(os.path.join(folder_path_b,"rho2_out.txt"), rho2_out_b)

    #     write_to_file(os.path.join(folder_path_a,"trace_rho2.txt"), trace_rho2_a)
    #     write_to_file(os.path.join(folder_path_b,"trace_rho2.txt"), trace_rho2_b)

    #     write_to_file(os.path.join(folder_path_a,"trace_rho2_out.txt"), trace_out_rho2_a)
    #     write_to_file(os.path.join(folder_path_b,"trace_rho2_out.txt"), trace_out_rho2_b)

    #     write_to_file(os.path.join(folder_path_a,"fidelity_a.txt"), mean_fidelity_a)
    #     write_to_file(os.path.join(folder_path_b,"fidelity_b.txt"), mean_fidelity_b)

        


    
    