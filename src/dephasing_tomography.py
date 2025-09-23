import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qtomo.tomography import ChoiTomography
from functools import partial
import qtomo.quantum_ops as quantum_ops
import qtomo.generator as generator
import qtomo.gradient as gradient
import qtomo.optimizer as optimizer
import qtomo.metric as metric
import numpy as np

# ============================================================
# 1. Initialize Choi tomography model
# ------------------------------------------------------------
# - We use ChoiTomography (unitary parameterization of process).
# - System: 1 qubit
# - Training: 3 Haar-random probe states
# ============================================================
tomog = ChoiTomography(num_qubits=1, num_rho=3)

# ============================================================
# 2. Define data processing function
# ------------------------------------------------------------
# This function describes how training states are transformed:
#   1. Apply a *dephasing channel* (with noise strength γ).
#   2. Apply the inverse of the current unitary (learned process).
#
# This way, training drives the learned unitary toward inverting
# the effect of dephasing.
# ============================================================
def data_process(rho_set, unitary, gamma):
    rho_final_set = []
    for rho in rho_set:
        rho2 = quantum_ops.dephasing_channel(rho, gamma)
        rho_final = quantum_ops.apply_unitary_dagger(rho2, unitary)
        rho_final_set.append(rho_final)
    return rho_final_set

# ============================================================
# 3. Optimizer setup
# ------------------------------------------------------------
# Adam optimizer with learning rate 0.01 is used to update
# the Choi-tomography unitary parameters.
# ============================================================
optimizer_obj = optimizer.AdamOptimizer(alpha=0.01)

# ============================================================
# 4. Define noise sweep
# ------------------------------------------------------------
# We vary the dephasing noise parameter γ over a range of values.
# For each γ:
#   - Train tomography
#   - Report final loss
#   - Inspect learned unitary vs. true dephasing channel
# ============================================================
g_s = np.linspace(1, 10e-3, 20)  # γ values to sweep
g_index = 0

# Fixed probe state (to test learned process during evaluation)
rho = generator.haar_probe_state(1)

# ============================================================
# 5. Training loop over noise parameters
# ============================================================
while g_index < len(g_s):
    g = g_s[g_index]

    print(f"\n--- Training for noise parameter (gamma) = {g:.6f} ---")

    # --------------------------------------------------------
    # Train Choi tomography
    # - fit() minimizes mean infidelity between
    #   processed probe states and their targets
    # - Gradient computed with Riemannian method
    # --------------------------------------------------------
    _, loss_dict = tomog.fit(
        epochs=200,
        data_process_fn=lambda rho_set, unitary: data_process(
            rho_set, unitary, gamma=g
        ),
        gradient_fn=gradient.riemann,
        loss_fn=metric.mean_infidelity,
        optimizer=optimizer_obj,
    )
    g_index += 1

    # --------------------------------------------------------
    # 6. Output results of training
    # --------------------------------------------------------
    print("Final loss:", loss_dict[-1])

    print("\nFinal Unitary (Learned):")
    print(tomog.unitary)

    # --------------------------------------------------------
    # 7. Evaluate on a test state
    # --------------------------------------------------------
    # - Compare learned unitary vs. true dephasing channel
    # - Report fidelity between outputs
    # --------------------------------------------------------
    print("\n--- Comparison of Final State ---")
    rho_output, rho_target, fide = tomog.evaluate(
        rho_input=rho,
        target_process_fn=lambda rho_input: quantum_ops.dephasing_channel(
            rho_input, gamma=g
        ),
        metric_fn=metric.compilation_trace_fidelity,
    )
    print(f"ρ after learned Unitary:\n{rho_output}")
    print(f"ρ after true Dephasing:\n{rho_target}")
    print(f"Fidelity(Unitary vs. Dephasing): {fide:.6f}\n")
