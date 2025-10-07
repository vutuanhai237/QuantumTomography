import numpy as np
import tensorflow as tf
import pytest

from qtomo.tomography import KrausTomography
import qtomo.quantum_ops as quantum_ops
import qtomo.generator as generator
import qtomo.gradient as gradient
import qtomo.optimizer as optimizer
import qtomo.metric as metric

def fixed_probe_states():
    # Fixed 3 deterministic 1-qubit states for reproducible tests
    return [
        tf.constant([[1.0, 0.0], [0.0, 0.0]], dtype=tf.complex128),      # |0><0|
        tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.complex128),      # |+><+|
        tf.constant([[0.5, -0.5j], [0.5j, 0.5]], dtype=tf.complex128)    # |+i><+i|
    ]

@pytest.mark.parametrize("epsilon", [
    np.array([[0, 1], [1, 0]], dtype=np.complex128),          # Pauli-X
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),       # Pauli-Y
    np.array([[1, 0], [0, -1]], dtype=np.complex128),         # Pauli-Z
    (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)  # Hadamard
])
def test_kraus_tomography_learns_multiple_epsilons(epsilon):
    np.random.seed(42)
    tf.random.set_seed(42)

    # Initialize KrausTomography with fixed probe states for deterministic test
    tomog = KrausTomography(num_qubits=1, num_rho=3, num_kraus=1)
    tomog.probe_rhos = fixed_probe_states()
    tomog.set_logging(False)  # Optional: turn off for test speed

    # Data process: apply learned Kraus, then epsilon† (inverse)
    def data_process(rho_set, kraus_ops):
        return [
            quantum_ops.apply_unitary_dagger(
                quantum_ops.apply_kraus_operators(rho, kraus_ops),
                epsilon
            )
            for rho in rho_set
        ]

    # Adam optimizer
    optimizer_obj = optimizer.AdamOptimizer(alpha=0.05)

    # Train
    final_kraus, loss_dict = tomog.fit(
        epochs=200,
        data_process_fn=data_process,
        gradient_fn=gradient.riemann,
        loss_fn=metric.mean_infidelity,
        optimizer=optimizer_obj,
    )

    final_loss = loss_dict[-1]
    print(f"\nTesting epsilon:\n{epsilon}")
    print("Final loss:", final_loss)
    assert final_loss < 0.1, f"Final loss too high: {final_loss}"

    # Check Kraus operators shape
    assert len(final_kraus) == 1
    assert final_kraus[0].shape == (2, 2)

    # Evaluate on fixed + random test states
    test_states = fixed_probe_states() + generator.haar_probe_states(1, 3)
    for i, rho in enumerate(test_states):
        rho_output, rho_target, fidelity = tomog.evaluate(
            rho_input=rho,
            target_process_fn=lambda r: quantum_ops.apply_unitary(r, epsilon),
            metric_fn=metric.compilation_trace_fidelity,
        )
        print(f"\nTest state {i}:")
        print(f"Input state:\n{rho}")
        print(f"Output after learned Kraus:\n{rho_output}")
        print(f"Output after epsilon (ideal):\n{rho_target}")
        print(f"Fidelity: {fidelity:.6f}")
        assert tf.math.real(fidelity).numpy() > 0.90, f"Fidelity too low on test state {i}: {fidelity}"

    # Check if learned Kraus operator approximates a unitary (loosely)
    U = final_kraus[0]
    identity = tf.eye(2, dtype=tf.complex128)
    residual = tf.linalg.norm(tf.matmul(tf.linalg.adjoint(U), U) - identity)
    print(f"\nUnitarity residual ||K†K - I||: {residual.numpy()}")
    assert residual.numpy() < 0.2, "Learned Kraus operator is far from unitary"
