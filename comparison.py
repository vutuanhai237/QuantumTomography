import base.epsilon_rho as epsilon_rho
import numpy as np
from base import metrics
import tensorflow as tf
from base import generator_gellmann
from base import generator_haar
import time
import os
from base import fast_QPT_recreate
from base import optimize_algorithm

def initExperiment(n, num_rho, num_kraus):
    d = 2**n
    rho_list = generator_haar.generate_n_qubits_rho_haar(n, num_rho)

    print(f"Generated {len(rho_list)} of {rho_list[0].shape} rho.")
    # ----------------------------
    single_qubits_projectors = generator_gellmann.generate_measurement_projector_gellmann(num_qubits = 1)
    full_measurement_operators = generator_gellmann.generate_measurement_operators_gellmann(single_qubits_projectors, n)
    measurement_operators = generator_gellmann.merge_projectors_into_povm(full_measurement_operators, num_rho)
    print(f"Generated {len(measurement_operators)} of {measurement_operators[0].shape} M.")
    # ----------------------------
    # Generate epsilon
    epsilon = generator_haar.random_unitary(d)
    print(f"Generated {epsilon.shape} epsilon.")
    # Generate K list
    kraus_operators = generator_haar.generate_kraus_operators(d, num_kraus)
    print(f"Generated {len(kraus_operators)} of {kraus_operators[0].shape} kraus operators.")
    return rho_list, measurement_operators, epsilon, kraus_operators
def calculate_rho2_lists(rho_list, epsilon, kraus_operators):
    rho2 = []
    rho2_kraus = []
    for rho in rho_list:
        rho2.append(epsilon_rho.calculate_from_unitary(rho, epsilon))
        rho2_kraus.append(epsilon_rho.calculate_from_kraus_operators(rho, kraus_operators))
    return rho2, rho2_kraus

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


experiment_folder = 'results/experiment_new/comparison_temp'

for n in range (3, 6):
    if (experiment_folder == ''):
        break
    else:
        write_folder = os.path.join(experiment_folder, str(n) + "_qubits")
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)
    print(f"N={n}")

    rho_list, measurement_operators, epsilon, kraus_operators = initExperiment(n, 3**n, 1)
    
    # ----------------------------
    # Fast QPT based on multi-shot measurement
    start_time = time.time()
    kraus_operators_res, cost_dict = fast_QPT_recreate.optimize_adam_kraus(measurement_operators, rho_list, epsilon, kraus_operators, n, 0.05, num_loop=200)
    end_time = time.time()
    multi_shot_time = end_time - start_time
    
    rho3_list = epsilon_rho.calculate_set_from_kraus_operators(kraus_operators_res, rho_list, epsilon)
    rho2_list, rho2_kraus_list = calculate_rho2_lists(rho_list, epsilon, kraus_operators_res)
    
    mean_fidelity_rho_rho3 = metrics.mean_fidelity(rho3_list, rho_list)
    mean_fidelity_rho2_rho2 = metrics.mean_fidelity(rho2_kraus_list, rho2_list)

    # ----------------------------
    # Fast QPT based on single-shot measurement
    start_time = time.time()
    kraus_operators_res_our_method, cost_dict_our_method = optimize_algorithm.optimize_adam_kraus_set(rho_list, epsilon, kraus_operators, n, alpha=0.05, num_loop=200)
    end_time = time.time()
    single_shot_time = end_time - start_time

    rho3_list = epsilon_rho.calculate_set_from_kraus_operators(kraus_operators_res_our_method, rho_list, epsilon)
    rho2_list, rho2_kraus_list = calculate_rho2_lists(rho_list, epsilon, kraus_operators_res_our_method)
    
    mean_fidelity_rho_rho3_our_method = metrics.mean_fidelity(rho3_list, rho_list)
    mean_fidelity_rho2_rho2_our_method = metrics.mean_fidelity(rho2_kraus_list, rho2_list)

    write_to_file(os.path.join(write_folder, "rho_list.txt"), rho_list)
    write_to_file(os.path.join(write_folder,"epsilon.txt"), epsilon)
    write_to_file(os.path.join(write_folder,"measurement_operators.txt"), measurement_operators)
    write_to_file(os.path.join(write_folder,"init_kraus_operators.txt"), kraus_operators)

    write_to_file(os.path.join(write_folder,"kraus_operators.txt"), kraus_operators_res)
    write_to_file(os.path.join(write_folder,"cost_dict.txt"), cost_dict)
    write_to_file(os.path.join(write_folder,"time.txt"), multi_shot_time)

    write_to_file(os.path.join(write_folder,"kraus_operators_ours.txt"), kraus_operators_res_our_method)
    write_to_file(os.path.join(write_folder,"cost_dict_ours.txt"), cost_dict_our_method)
    write_to_file(os.path.join(write_folder,"time_ours.txt"), single_shot_time)

    write_to_file(os.path.join(write_folder,"mean_fidelity_rho_rho3.txt"), mean_fidelity_rho_rho3.numpy())
    write_to_file(os.path.join(write_folder,"mean_fidelity_rho2_rho2.txt"), mean_fidelity_rho2_rho2.numpy())
    write_to_file(os.path.join(write_folder,"mean_fidelity_rho_rho3_ours.txt"), mean_fidelity_rho_rho3_our_method.numpy())
    write_to_file(os.path.join(write_folder,"mean_fidelity_rho2_rho2_ours.txt"), mean_fidelity_rho2_rho2_our_method.numpy())