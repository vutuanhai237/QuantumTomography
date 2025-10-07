import numpy as np
import tensorflow as tf
import logging
from qtomo.tomography import ChoiTomography
import qtomo.quantum_ops as quantum_ops
import qtomo.gradient as gradient
import qtomo.optimizer as optimizer
import qtomo.metric as metric

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def fixed_probe_states():
    # Return 3 deterministic 1-qubit states
    return [
        tf.constant([[1.0, 0.0], [0.0, 0.0]], dtype=tf.complex128),  # |0⟩⟨0|
        tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.complex128),  # |+⟩⟨+|
        tf.constant([[0.5, -0.5j], [0.5j, 0.5]], dtype=tf.complex128)  # |+i⟩⟨+i|
    ]


def test_choi_tomography_fixed_dephasing():
    gamma = 0.05
    max_epochs = 300

    logger.info("Initializing ChoiTomography...")
    tomog = ChoiTomography(num_qubits=1, num_rho=3)
    tomog.probe_rhos = fixed_probe_states()

    logger.info(f"Using gamma = {gamma}, epochs = {max_epochs}")
    optimizer_obj = optimizer.AdamOptimizer(alpha=0.03)

    def data_process(rho_set, unitary):
        return [
            quantum_ops.apply_unitary_dagger(
                quantum_ops.dephasing_channel(rho, gamma), unitary
            )
            for rho in rho_set
        ]

    logger.info("Starting training...")
    _, loss_dict = tomog.fit(
        epochs=max_epochs,
        data_process_fn=data_process,
        gradient_fn=gradient.riemann,
        loss_fn=metric.mean_infidelity,
        optimizer=optimizer_obj
    )

    final_loss = loss_dict[-1]
    logger.info(f"Final training loss: {final_loss:.6f}")
    assert final_loss < 0.1, f"Final loss too high: {final_loss}"

    # Evaluation
    rho_test = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    _, _, fidelity = tomog.evaluate(
        rho_input=rho_test,
        target_process_fn=lambda rho: quantum_ops.dephasing_channel(rho, gamma),
        metric_fn=metric.compilation_trace_fidelity,
    )

    fidelity_val = tf.math.real(fidelity).numpy()
    logger.info(f"Fidelity on test state: {fidelity_val:.6f}")
    assert fidelity_val > 0.95, f"Fidelity too low: {fidelity}"

    # Check unitarity of learned unitary
    U = tomog.unitary
    identity = tf.eye(2, dtype=tf.complex128)
    residual = tf.linalg.norm(tf.matmul(tf.linalg.adjoint(U), U) - identity)
    logger.info(f"Unitarity residual ||U†U - I|| = {residual.numpy():.2e}")
    assert residual.numpy() < 1e-6, f"Learned matrix is not unitary! ||U†U - I|| = {residual.numpy()}"
