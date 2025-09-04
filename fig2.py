import tensorflow as tf
import numpy as np
import os
import re
import matplotlib.pyplot as plt

# Function to read and parse the tensor data from a text file
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

def retrieve_dict(base_dir, filename, reverse = False):
    # Dictionary to hold costs organized by prefix (i)
    fidelity_dict = {}
    
    # Iterate through each subfolder in the base directory
    folder_names = os.listdir(base_dir)
    
    # Sort folders numerically or lexicographically
    folder_names.sort(key=lambda x: int(x.split('_')[0]))

    # Iterate through each subfolder in the base directory
    for folder_name in folder_names:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # Extract the prefix for identifying the 'i' value
            prefix = folder_name.split('_')[0]
            # Path to the cost.txt file
            cost_file_path = os.path.join(folder_path, filename)
            
            # Read the cost value from the cost.txt file
            if os.path.exists(cost_file_path):
                    # Extract the complex number part using regular expression
                    tensor = parse_tensor_from_file(cost_file_path, 1)
                    # Convert the extracted string to a complex number
                    for t in tensor.numpy():
                         # Get the real part and convert to float
                         float_value = t.real
                         if (reverse == True):
                             float_value = 1 - t.real
                         # if (float_value < 10e-4): continue
                         if prefix not in fidelity_dict:
                              fidelity_dict[prefix] = []
                         fidelity_dict[prefix].append(abs(float_value))
    
    return fidelity_dict


def merge_dicts(dict_list, labels):
    """
    Merges multiple dictionaries into one, with new keys labeled by the given labels.

    Parameters:
        dict_list (list of dict): List of dictionaries to merge.
        labels (list of str): Labels for each dictionary.

    Returns:
        dict: Merged dictionary with new keys.
    """
    merged_dict = {}
    for i, (d, label) in enumerate(zip(dict_list, labels)):
        for key, values in d.items():
            new_key = f"{label}"
            if new_key not in merged_dict:
               merged_dict[new_key] = []
            for v in values:
                merged_dict[new_key].append(v)
    return merged_dict

folder_path = 'results/experiment_new/haar_random_9_qubits'

#Get final fidelity dict
dict = retrieve_dict(folder_path, 'mean_fidelity_rho_rho3.txt', True)
dict2 = retrieve_dict(folder_path, 'mean_fidelity_rho2_rho2.txt', True)
qubits = [1, 2, 3, 4, 5, 6,7,8,9]
merged = merge_dicts([dict, dict2], ["03", "22"])

inf_rho_i_f = merged["03"]
inf_rho_e_k = merged["22"]
#Get final cost dict
cost_dict = retrieve_dict(folder_path, 'cost_dict.txt')

x_cost = list(range(1, 201))
x_infidelity = list(range(1, len(inf_rho_i_f) + 1))

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(7.5, 4))

# INFIDELITY plot
axs[1].plot(x_infidelity, inf_rho_i_f, marker='o', linestyle='-', color='blue', label=r'$\rho_i$ vs $\rho_f$')
axs[1].plot(x_infidelity, inf_rho_e_k, marker='s', linestyle='--', color='orange', label=r'$\rho_e$ vs $\rho_k$')
#axs[1].set_title('INFIDELITY Comparison')
axs[1].set_xlabel('N')
axs[1].set_ylabel('Infidelity')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Cost evolution plot
for qubit, values in cost_dict.items():
    axs[0].plot(x_cost, values, label=f'{qubit} qubits')

#axs[0].set_title('Cost Evolution Across Qubit Numbers')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Cost')
axs[0].legend()
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

axs[0].tick_params(axis='both', labelsize=12)  # for subplot 0
axs[1].tick_params(axis='both', labelsize=12)  # for subplot 1

plt.tight_layout()
plt.savefig('fig2.eps')
plt.savefig('fig2.png')

