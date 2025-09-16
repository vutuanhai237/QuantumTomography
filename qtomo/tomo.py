# This is the main class that manages data and training for quantum process tomography.
# It stores training data, the model (Kraus or Choi), optimizer, and evaluation metrics.
# Combines functionality from multiple files.

from qtomo import generator
from qtomo import gradient
from qtomo import metric
from qtomo.optimizer import AdamOptimizer, SGDOptimizer
from qtomo import utils
from qtomo import quantum_ops
import tensorflow as tf
import numpy as np

class Tomography:
    def __init__(self):
        self.target_rho_set = None  # list[np.ndarray]: Training quantum states (ρ_i)
        self.gradient_fn = gradient.calculate_gradient  # callable: Gradient computation function
        self.loss_fn = metric.mean_infidelity            # callable: Loss function (between target and output)
        self.evaluate_fn = metric.compilation_trace_fidelity  # callable: Evaluation metric (e.g., fidelity)
        self.data_process_fn = None  # callable: Function to process states through the quantum process

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
    def __init__(self):
        super().__init__()
        self.kraus_operators = None  # list[tf.Tensor]: The trainable Kraus operators

    def init(self, num_qubits: int, num_rho: int, num_kraus: int):
        """
        Initialize the Kraus operators and training set.

        Args:
            num_qubits (int): Number of qubits
            num_rho (int): Number of training quantum states
            num_kraus (int): Number of Kraus operators
        """
        dim = 2 ** num_qubits
        self.kraus_operators = generator.kraus_operators(dim=dim, num_operators=num_kraus)
        self.init_training_set(num_qubits, num_rho)

    def calculate_loss(self, kraus_operators: list) -> tf.Tensor:
        """
        Compute the loss (infidelity) between predicted and target quantum states.

        Args:
            kraus_operators (list[tf.Tensor]): Current set of Kraus operators

        Returns:
            tf.Tensor: Scalar loss value
        """
        rho_final_set = self.data_process_fn(self.target_rho_set, kraus_operators)
        loss = self.loss_fn(rho_final_set, self.target_rho_set)
        return loss

    def fit(self, epochs: int = 100, optimizer_name: str = 'adam', optimizer_params: dict = None):
        """
        Train the model using the chosen optimizer.

        Args:
            epochs (int): Number of optimization steps
            optimizer_name (str): Name of the optimizer ('adam' or 'sgd')
            optimizer_params (dict): Optional hyperparameters for the optimizer

        Returns:
            list[tf.Tensor]: The optimized Kraus operators
        """
        if optimizer_params is None:
            optimizer_params = {}

        if optimizer_name.lower() == 'adam':
            defaults = {'alpha': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
            defaults.update(optimizer_params)
            self.optimizer = AdamOptimizer(**defaults)

        elif optimizer_name.lower() == 'sgd':
            defaults = {'alpha': 0.01}
            defaults.update(optimizer_params)
            self.optimizer = SGDOptimizer(**defaults)

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        param = self.kraus_operators

        for epoch in range(epochs):
            param, loss = self.optimizer.step_and_cost(param, self.calculate_loss, self.gradient_fn)
            param = utils.normalize_kraus(param)
            print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.numpy()}")

        self.kraus_operators = param
        return param

    def evaluate(self, rho_input: np.ndarray, target_process_fn: callable) -> tuple:
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
        if self.data_process_fn is None:
            raise ValueError("Data process function is not set.")
        
        rho_output = quantum_ops.apply_kraus_operators(rho_input, self.kraus_operators)
        rho_target = target_process_fn(rho_input)
        comparison_result = self.evaluate_fn(rho_output, rho_target)
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
