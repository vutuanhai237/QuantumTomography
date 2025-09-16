
# Đây là class chính lưu dữ liệu trong quá trình tomograhpy
# Lưu trữ data, model, optimizer, metric, generator
# Gom lại từ các fuke

from qtomo import generator
from qtomo import gradient
from qtomo import metric
from qtomo.optimizer import AdamOptimizer, SGDOptimizer
from qtomo import utils
from qtomo import quantum_ops
class Tomography:
    def __init__(self):
        self.target_rho_set = None  # Training resistivity for projection
        self.gradient_fn = gradient.calculate_gradient  # Gradient function
        self.loss_fn = metric.mean_infidelity  # Loss function
        self.evaluate_fn = metric.compilation_trace_fidelity  # Loss function
        self.data_process_fn = None # Target process as function

    def init_training_set(self, num_qubits, num_rho):
        self.target_rho_set = generator.haar_probe_states(num_qubits, num_rho)

    def fit(self, *args, **kwargs):
        pass

class KrausTomography(Tomography):
    def __init__(self):
        super().__init__()
        self.kraus_operators = None  # Current Kraus operators

    def init(self, num_qubits, num_rho, num_kraus):
        dim = 2 ** num_qubits
        self.kraus_operators = generator.kraus_operators(dim=dim, num_operators=num_kraus)
        self.init_training_set(num_qubits, num_rho)

    def calculate_loss(self, kraus_operators):
        rho_final_set = self.data_process_fn(self.target_rho_set, kraus_operators)
        loss = self.loss_fn(rho_final_set, self.target_rho_set)
        return loss
    
    def fit(self, epochs=100, optimizer_name='adam', optimizer_params=None):
        if optimizer_params is None:
            optimizer_params = {}

        if optimizer_name.lower() == 'adam':
            # Gán default nếu người dùng không truyền
            defaults = {'alpha': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
            # Cập nhật defaults với params do người dùng truyền
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
            print(f"Epoch {epoch+1}/{epochs}, loss: {loss.numpy()}")

        self.kraus_operators = param
        return param
    
    def evaluate(self, rho_input, target_process_fn):
        """
        Apply the learned process to a new input state.

        Args:
            rho_input (DensityMatrix): Input density matrix.

        Returns:
            list: Processed output states from the learned model.
        """
        if self.data_process_fn is None:
            raise ValueError("Could not find a data process method.")
        rho_output = quantum_ops.apply_kraus_operators(rho_input, self.kraus_operators)
        rho_target = target_process_fn(rho_input)
        comparison_result = self.evaluate_fn(rho_output, rho_target)
        return rho_output, rho_target, comparison_result
    
class ChoiTomography(Tomography):
    def __init__(self):
        super().__init__()
        self.unitary = None  # Current unitary operator

    def init(self, num_qubits, num_rho):
        self.unitary = generator.choi_matrix(num_qubits)
        self.init_training_set(num_qubits, num_rho)

# Chủ yếu bao gồm input (unitary, epsilon, ...) và tập Kraus operator
# Qúa trình tính toắn update theo kiểu sau
# Tomography.init() => khởi tạo Kraus operator, gọi từ generator.py
# Tomography.fit() => optimize Karus operator, có sử dụng optimizer.py
# Tomography.evaluate() => tính toán metric, có sử dụng metric.py
# Thống nhất tên biến (tham số) xuyên suốt là gì, ví dụ bên optimize_adam_unitary_dagger_set gọi là 