import base.epsilon_rho as epsilon_rho
import tensorflow as tf
import base.metrics as mtr

def diff_MSE(rho_f_list, rho_list):
    """Compute loss using Mean Squared Error (MSE) between the density matrices."""
    
    mse_sum = 0.0

    for rho, rho_f in zip(rho_list, rho_f_list):
        # Compute the squared difference between the density matrices
        difference = rho - rho_f
        mse = tf.reduce_sum(tf.abs(difference)**2)
        mse_sum += mse

    mse_avg = mse_sum / len(rho_list)

    return mse_avg  # Lower value indicates better matching

def diff_infidelity(rho_f_list, rho_list):
    """Compute a fidelity-based loss between predicted and target density matrices.

    The loss is defined as 1 - average squared fidelity (Tr[ρ_pred ρ_target])^2.
    """

    total_fidelity = 0.0

    for rho, rho_f in zip(rho_list, rho_f_list):
        overlap = tf.matmul(rho_f, rho)
        fidelity = tf.math.square(tf.linalg.trace(overlap))
        total_fidelity += fidelity

    average_fidelity = total_fidelity / len(rho_list)
    loss = 1.0 - average_fidelity
    return loss

def diff_fidelity(rho, rho_3):
    """Compute loss using Trace Fidelity between the density matrices."""
    return 1 - mtr.compilation_trace_fidelity(rho, rho_3)