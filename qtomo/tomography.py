# This module defines the main classes for quantum process tomography (QPT).
# It manages training data, the process model (Kraus or Choi), optimization,
# and evaluation against ideal target processes.

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
        self.target_rho_set = None  # list[np.ndarray]: Training states ρ_i
        self.is_logging = False  # bool: Whether to log training progress
    def set_logging(self, is_logging: bool):
        """
        Enable or disable logging of training progress.

        Args:
            is_logging (bool): If True, logs training progress.
        """
        self.is_logging = is_logging
    def init_training_set(self, num_qubits: int, num_rho: int):
        """
        Generate a set of random Haar-distributed density matrices for training.

        Args:
            num_qubits (int): Number of qubits in each state.
            num_rho (int): Number of training states to generate.
        """
        self.target_rho_set = generator.haar_probe_states(num_qubits, num_rho)

    def fit(self, *args, **kwargs):
        """Abstract method — implemented in subclasses."""
        pass


class KrausTomography(Tomography):
    def __init__(self, num_qubits: int, num_rho: int, num_kraus: int):
        super().__init__()
        """
        Initialize a tomography model with Kraus operators.

        Args:
            num_qubits (int): Number of qubits in the system.
            num_rho (int): Number of training states.
            num_kraus (int): Number of Kraus operators to represent the process.
        """
        self.init_kraus = generator.kraus_operators(
            n=num_qubits, num_operators=num_kraus
        )
        self.kraus_operators = None
        self.init_training_set(num_qubits, num_rho)

    def fit(self, epochs: int = 100, 
            data_process_fn: callable = None, 
            gradient_fn: callable = None, 
            loss_fn: callable = None, 
            optimizer: Optimizer = None) -> list[tf.Tensor]:
        """
        Optimize Kraus operators by minimizing a loss between model output and targets.

        Args:
            epochs (int): Number of training iterations.

            data_process_fn (callable): Maps training states through an pre-processing step.
                Signature:
                    data_process_fn(rho_set: list[np.ndarray], 
                                    kraus_operators: list[tf.Tensor]) 
                    -> list[np.ndarray]

            gradient_fn (callable): Gradient computation rule for optimization.
                Example: `gradient.riemann`

            loss_fn (callable): Compares predicted vs. target states.
                Signature:
                    loss_fn(rho_final_set: list[np.ndarray], 
                            target_rho_set: list[np.ndarray]) 
                    -> tf.Tensor
                Example: `metric.mean_infidelity`

            optimizer (Optimizer): Optimizer instance (e.g. Adam, SGD).
                Example: `optimizer.AdamOptimizer(alpha=0.01)`

        Returns:
        (list[tf.Tensor], list[complex]):
            - Optimized loss operators.
            - Training loss history over epochs.
        """
        param = self.init_kraus
        lost_dict = []
        def calculate_loss(kraus_operators: list) -> tf.Tensor:
            """Helper to compute loss for the current Kraus operators."""
            rho_final_set = data_process_fn(self.target_rho_set, kraus_operators)
            return loss_fn(rho_final_set, self.target_rho_set)
        
        for epoch in range(epochs):
            param, loss = optimizer.step_and_cost(param, calculate_loss, gradient_fn)
            param = utils.normalize_kraus(param)
            lost_dict.append(loss.numpy())
            if (self.is_logging):
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.numpy()}")

        self.kraus_operators = param
        return param, lost_dict

    def evaluate(self, rho_input: np.ndarray, target_process_fn: callable, metric_fn: callable) -> tuple:
        """
        Compare the learned Kraus process with an ideal reference process.

        Args:
            rho_input (np.ndarray): Input density matrix to evaluate.

            target_process_fn (callable): Ideal process to apply to rho_input.
                Signature:
                    target_process_fn(rho_input: np.ndarray) -> np.ndarray
                Example:
                    lambda rho: quantum_ops.apply_unitary(rho, unitary=epsilon)

            metric_fn (callable): Metric comparing learned vs. target outputs.
                Signature:
                    metric_fn(rho_output: np.ndarray, rho_target: np.ndarray) -> float
                Example:
                    metric.compilation_trace_fidelity

        Returns:
            tuple:
                - rho_output (np.ndarray): Output from the learned Kraus process.
                - rho_target (np.ndarray): Output from the ideal process.
                - comparison_result (float): Metric value (e.g. fidelity).
        """
        rho_output = quantum_ops.apply_kraus_operators(rho_input, self.kraus_operators)
        rho_target = target_process_fn(rho_input)
        comparison_result = metric_fn(rho_output, rho_target)
        return rho_output, rho_target, comparison_result


class ChoiTomography(Tomography):
    def __init__(self, num_qubits: int, num_rho: int):
        super().__init__()
        """
        Initialize with a Haar-random unitary and Haar-random training states.

        Args:
            num_qubits (int): Number of qubits.
            num_rho (int): Number of training states.
        """
        self.init_unitary = tf.Variable(generator.haar(num_qubits), dtype=tf.complex128)
        self.unitary = None
        self.init_training_set(num_qubits, num_rho)
    
    def fit(self, epochs: int = 100, 
            data_process_fn: callable = None, 
            gradient_fn: callable = None, 
            loss_fn: callable = None, 
            optimizer: Optimizer = None) -> tf.Tensor:
        """
        Optimize the unitary operator by minimizing a loss between model output and targets.

        Args:
            epochs (int): Number of training iterations.

            data_process_fn (callable): Maps training states through an pre-processing step.
                Signature:
                    data_process_fn(rho_set: list[np.ndarray], 
                                    unitary: tf.Tensor) 
                    -> list[np.ndarray]

            gradient_fn (callable): Gradient computation rule for optimization.
                Example: `gradient.riemann`

            loss_fn (callable): Compares predicted vs. target states.
                Signature:
                    loss_fn(rho_final_set: list[np.ndarray], 
                            target_rho_set: list[np.ndarray]) 
                    -> tf.Tensor
                Example: `metric.mean_infidelity`

            optimizer (Optimizer): Optimizer instance (e.g. Adam, SGD).
                Example: `optimizer.AdamOptimizer(alpha=0.01)`

        Returns:
        (tf.Tensor, list[complex]):
            - Optimized unitary matrix.
            - Training loss history over epochs.
        """
        param = self.init_unitary
        loss_dict = []
        def calculate_loss(unitary: list) -> tf.Tensor:
            """Helper to compute loss for the current unitary."""
            rho_final_set = data_process_fn(self.target_rho_set, unitary)
            return loss_fn(rho_final_set, self.target_rho_set)
        
        for epoch in range(epochs):
            param, loss = optimizer.step_and_cost(param, calculate_loss, gradient_fn)
            param = utils.normalize_unitary(param)
            loss_dict.append(loss.numpy())
            if (self.is_logging):
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss.numpy()}")

        self.unitary = param
        return param, loss_dict

    def evaluate(self, rho_input: np.ndarray, target_process_fn: callable, metric_fn: callable) -> tuple:
        """
        Compare the learned unitary process with an ideal reference process.

        Args:
            rho_input (np.ndarray): Input density matrix to evaluate.

            target_process_fn (callable): Ideal process to apply to rho_input.
                Signature:
                    target_process_fn(rho_input: np.ndarray) -> np.ndarray
                Example:
                    lambda rho: quantum_ops.apply_unitary(rho, unitary=epsilon)

            metric_fn (callable): Metric comparing learned vs. target outputs.
                Signature:
                    metric_fn(rho_output: np.ndarray, rho_target: np.ndarray) -> float
                Example:
                    metric.compilation_trace_fidelity

        Returns:
            tuple:
                - rho_output (np.ndarray): Output from the learned unitary process.
                - rho_target (np.ndarray): Output from the ideal process.
                - comparison_result (float): Metric value (e.g. fidelity).
        """
        rho_output = quantum_ops.apply_unitary(rho_input, self.unitary)
        rho_target = target_process_fn(rho_input)
        comparison_result = metric_fn(rho_output, rho_target)
        return rho_output, rho_target, comparison_result
