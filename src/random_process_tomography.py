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

# ----------------------------- #
# 1. Create a tomography object
# ----------------------------- #
tomog = KrausTomography(num_qubits=1, num_rho=3, num_kraus=1)

# -------------------------------------------------------- #
# 3. Define epsilon (unitary used for inverse transformation)
# -------------------------------------------------------- #
epsilon = generator.haar(1)  # 1 qubit => 2x2 unitary

# ------------------------------------------------------------------------ #
# 4. Define the process function that applies the Kraus op and inverse unitary
# ------------------------------------------------------------------------ #
def data_process(rho_set, kraus_operators, epsilon):
    """
    Apply a quantum process to a list of states using Kraus operators,
    then apply the inverse unitary (epsilon).
    """
    rho_final_set = []
    for rho in rho_set:
        rho2 = quantum_ops.apply_kraus_operators(rho, kraus_operators)
        rho_final = quantum_ops.apply_unitary_dagger(rho2, epsilon)
        rho_final_set.append(rho_final)
    return rho_final_set

optimizer_obj = optimizer.AdamOptimizer(alpha=0.01)  # Optimizer instance

# -------------------------------------------------------- #
# 6. Run optimization (training)
# -------------------------------------------------------- #
tomog.fit(
    epochs=400,
    data_process_fn=partial(data_process, epsilon=epsilon),
    gradient_fn=gradient.riemann,
    loss_fn = metric.mean_infidelity,
    optimizer=optimizer_obj
)

# -------------------------------------------------------- #
# 7. Output results
# -------------------------------------------------------- #
final_kraus = tomog.kraus_operators

print("Target Rho Set:")
for rho in tomog.target_rho_set:
    print(rho)

print("\nFinal Kraus Operators (Learned):")
for op in final_kraus:
    print(op.numpy())

print("\nEpsilon (Ideal Inverse Unitary):")
print(epsilon)

print("\n--- Comparison of Final States ---")

rho_set = generator.haar_probe_states(1, 6)  # Test with a random probe state
rho_output_set = []
rho_target_set = []
for rho in rho_set:
    rho_output, rho_target, fide = tomog.evaluate(
        rho_input=rho, 
        target_process_fn=partial(quantum_ops.apply_unitary, unitary=epsilon),
        evaluate_fn=metric.compilation_trace_fidelity)
    
    print(f"\nInput ρ]:\n{rho}")
    print(f"ρ after Kraus:\n{rho_output}")
    print(f"ρ after epsilon:\n{rho_target}")
    print(f"Fidelity(Kraus, Epsilon): {fide:.6f} \n")



