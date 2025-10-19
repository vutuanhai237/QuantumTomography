import tensorflow as tf
import numpy as np
import re
import os

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
def retrieve_dict(base_dir, filename):
    # Dictionary to hold costs organized by prefix (i)
    fidelity_dict = {}
    # Iterate through each subfolder in the base directory
    folder_names = os.listdir(base_dir)
    
    # Sort folders numerically or lexicographically
    folder_names.sort(key=lambda x: int(x.split('_')[0]))

    # Iterate through each subfolder in the base directory
    for folder_name_1 in folder_names:
        folder_path = os.path.join(base_dir, folder_name_1)
        if os.path.isdir(folder_path):
            # Extract the prefix for identifying the 'i' value
            prefix = folder_name_1.split('_')[0]
            folder_paths = os.listdir(folder_path)
            directories = [item for item in folder_paths if os.path.isdir(os.path.join(folder_path, item))]
            directories.sort(key=lambda x: float(x.split('_')[-1]))
            for folder_name_2 in directories:
                 
                 folder_path_2 = os.path.join(folder_path, folder_name_2)
                 cost_file_path = os.path.join(folder_path_2, filename)
            
                # Read the cost value from the cost.txt file
                 if os.path.exists(cost_file_path):
                        # Extract the complex number part using regular expression
                        tensor = parse_tensor_from_file(cost_file_path, 1)
                     
                        # Convert the extracted string to a complex number
                        for t in tensor.numpy():
                             # Get the real part and convert to float
                            float_value = 1-t.real
                            # if (float_value < 10e-4): continue
                            if prefix not in fidelity_dict:
                                fidelity_dict[prefix] = []
                            fidelity_dict[prefix].append(abs(float_value))
    
    return fidelity_dict

import matplotlib.pyplot as plt

infidelity_rho_if = retrieve_dict('results/experiment_new/dephasing_9_qubits', 'mean_fidelity_rho_rho3.txt')
print(infidelity_rho_if)
infidelity_rho_eu = retrieve_dict('results/experiment_new/dephasing_9_qubits', 'mean_fidelity_rho2_rho2.txt')
print(infidelity_rho_eu)
x_points = np.linspace(0.01, 1, 20)

fig, axs = plt.subplots(1, 2, figsize=(7.5, 4), sharey=True)

# Plot INFIDELITY rho_i, rho_f
for num_qubits, values in infidelity_rho_if.items():
    axs[0].plot(x_points, values, marker='o', label=f'{num_qubits} qubit(s)')
#axs[0].set_title('INFIDELITY rho_i, rho_f')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Infidelity')
axs[0].legend()
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot INFIDELITY rho_e, rho_U
for num_qubits, values in infidelity_rho_eu.items():
    axs[1].plot(x_points, values, marker='o', label=f'{num_qubits} qubit(s)')
#axs[1].set_title('INFIDELITY rho_e, rho_U')
axs[1].set_xlabel('Index')
axs[1].legend()
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[0].tick_params(axis='both', labelsize=12)  # for subplot 0
axs[1].tick_params(axis='both', labelsize=12)  # for subplot 1

plt.tight_layout()
plt.savefig('fig4.eps')
plt.savefig('fig4.png')
