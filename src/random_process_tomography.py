import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qtomo.tomography import KrausTomography
from functools import partial
import qtomo.quantum_ops as quantum_ops
import qtomo.generator as generator
import qtomo.gradient as gradient
import qtomo.optimizer as optimizer
import qtomo.metric as metric

# ============================================================
# 1. Initialize tomography model
# ------------------------------------------------------------
# We create a KrausTomography object with:
#   - 1 qubit
#   - 3 random probe states (for training)
#   - 1 Kraus operator (to model the process)
# ============================================================
tomog = KrausTomography(num_qubits=1, num_rho=3, num_kraus=1)
tomog.set_logging(True)  # Enable logging of training progress

# ============================================================
# 2. Define target inverse unitary (epsilon)
# ------------------------------------------------------------
# This unitary represents the *ideal process* we want the
# Kraus operators to learn to approximate. 
# Here we generate a random Haar-distributed 2x2 unitary.
# ============================================================
epsilon = generator.haar(n=1)  # 1-qubit → 2x2 matrix

# ============================================================
# 3. Define data processing function
# ------------------------------------------------------------
# This function describes how training states are transformed:
#   1. Apply the current Kraus operators (learned process).
#   2. Apply the *inverse* of epsilon (ideal correction).
#
# During training, this ensures the learned process is driven
# toward approximating the effect of epsilon.
# ============================================================
def data_process(rho_set, kraus_operators, epsilon):
    rho_final_set = []
    for rho in rho_set:
        rho2 = quantum_ops.apply_kraus_operators(rho, kraus_operators)
        rho_final = quantum_ops.apply_unitary_dagger(rho2, epsilon)
        rho_final_set.append(rho_final)
    return rho_final_set

# ============================================================
# 4. Set up optimizer
# ------------------------------------------------------------
# We use Adam with learning rate 0.01. 
# The optimizer updates Kraus operators to minimize the loss.
# ============================================================
optimizer_obj = optimizer.AdamOptimizer(alpha=0.01)

# ============================================================
# 5. Run training loop
# ------------------------------------------------------------
# - fit() minimizes the difference between outputs of
#   data_process() and the target states.
# - Loss is measured by mean infidelity.
# - Gradients are computed with the Riemannian method.
# ============================================================
final_kraus, loss_dict = tomog.fit(
    epochs=400,
    data_process_fn=lambda rho_set, kraus_operators: data_process(
        rho_set, kraus_operators, epsilon
    ),
    gradient_fn=gradient.riemann,
    loss_fn=metric.mean_infidelity,
    optimizer=optimizer_obj,
)

# ============================================================
# 6. Print results
# ------------------------------------------------------------
# - Loss history shows optimization progress.
# - Print training probe states.
# - Print the final learned Kraus operators.
# - Print the target epsilon for comparison.
# ============================================================
print("Loss history:")
print(loss_dict)

print("\nTarget Rho Set:")
for rho in tomog.target_rho_set:
    print(rho)

print("\nFinal Kraus Operators (Learned):")
for op in final_kraus:
    print(op.numpy())

print("\nEpsilon (Ideal Inverse Unitary):")
print(epsilon)

# ============================================================
# 7. Evaluate on test probe states
# ------------------------------------------------------------
# - Generate 6 new Haar-random test states.
# - Compare the learned Kraus process against epsilon.
# - Report input, outputs, and fidelity for each test case.
# ============================================================
print("\n--- Comparison of Final States ---")

rho_set = generator.haar_probe_states(1, 6)  # fresh probe states
for rho in rho_set:
    rho_output, rho_target, fide = tomog.evaluate(
        rho_input=rho,
        target_process_fn=lambda rho_input: quantum_ops.apply_unitary(
            rho_input, unitary=epsilon
        ),
        metric_fn=metric.compilation_trace_fidelity,
    )
    print(f"\nInput ρ:\n{rho}")
    print(f"ρ after learned Kraus process:\n{rho_output}")
    print(f"ρ after epsilon (ideal):\n{rho_target}")
    print(f"Fidelity(Kraus vs. Epsilon): {fide:.6f}\n")
