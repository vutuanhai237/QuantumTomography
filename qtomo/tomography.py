# This is the main class that manages data and training for quantum process tomography.
# It stores training data, the model (Kraus or Choi), optimizer, and evaluation metrics.
# Combines functionality from multiple files.

from qtomo import generator
from qtomo import gradient
from qtomo import metric
from qtomo.optimizer import Optimizer
from qtomo import utils
from qtomo import quantum_ops
import tensorflow as tf
import numpy as np

class Tomography:
    def __init__(self):
        self.target_rho_set = None  # list[np.ndarray]: Training quantum states (ρ_i)

    def init_training_set(self, num_qubits: int, num_rho: int):
        """
        Initialize the training dataset with random Haar-distributed density matrices.

        Args:
            num_qubits (int): Number of qubits
            num_rho (int): Number of training quantum states
        """
        self.target_rho_set = generator.haar_probe_states(num_qubits, num_rho)

    def fit(self, *args, **kwargs):
        """To be implemented in subclasses."""
        pass


class KrausTomography(Tomography):
    def __init__(self, num_qubits: int, num_rho: int, num_kraus: int):
        """
        Initialize the Kraus operators and training set.

        Args:
            num_qubits (int): Number of qubits
            num_rho (int): Number of training quantum states
            num_kraus (int): Number of Kraus operators
        """
        self.kraus_operators = generator.kraus_operators(n=num_qubits, num_operators=num_kraus)
        self.init_training_set(num_qubits, num_rho)

    def fit(self, epochs: int = 100, 
            data_process_fn: callable = None, 
            gradient_fn: callable = None, 
            loss_fn: callable = None, 
            optimizer: Optimizer = None) -> list[tf.Tensor]:
        """
        Train the model using the chosen optimizer.

        Args:
            epochs (int): Number of optimization steps
            optimizer_name (str): Name of the optimizer ('adam' or 'sgd')
            optimizer_params (dict): Optional hyperparameters for the optimizer

        Returns:
            list[tf.Tensor]: The optimized Kraus operators
        """
        param = self.kraus_operators

        def calculate_loss(kraus_operators: list) -> tf.Tensor:
            """
            Compute the loss between predicted and target quantum states.

            Args:
                kraus_operators (list[tf.Tensor]): Current set of Kraus operators

            Returns:
                tf.Tensor: Scalar loss value
            """
            rho_final_set = data_process_fn(self.target_rho_set, kraus_operators)
            loss = loss_fn(rho_final_set, self.target_rho_set)
            
            return loss
        
        for epoch in range(epochs):
            param, loss = optimizer.step_and_cost(param, calculate_loss, gradient_fn)
            param = utils.normalize_kraus(param)
            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.numpy()}")

        self.kraus_operators = param
        return param

    def evaluate(self, rho_input: np.ndarray, target_process_fn: callable, evaluate_fn: callable) -> tuple:
        """
        Evaluate the learned process on a new input state and compare with ideal.

        Args:
            rho_input (np.ndarray): Input quantum state
            target_process_fn (callable): Ideal target process (e.g., a unitary)

        Returns:
            tuple:
                - rho_output (np.ndarray): Output from learned Kraus process
                - rho_target (np.ndarray): Target output from ideal process
                - comparison_result (float): Evaluation metric (e.g., fidelity)
        """
        rho_output = quantum_ops.apply_kraus_operators(rho_input, self.kraus_operators)
        rho_target = target_process_fn(rho_input)
        comparison_result = evaluate_fn(rho_output, rho_target)
        return rho_output, rho_target, comparison_result


class ChoiTomography(Tomography):
    def __init__(self):
        super().__init__()
        self.unitary = None  # tf.Tensor: The current unitary operator to optimize

    def init(self, num_qubits: int, num_rho: int):
        """
        Initialize the unitary matrix for Choi representation and the training states.

        Args:
            num_qubits (int): Number of qubits
            num_rho (int): Number of training quantum states
        """
        self.unitary = generator.choi_matrix(num_qubits)
        self.init_training_set(num_qubits, num_rho)
