import base.epsilon_rho as epsilon_rho
import tensorflow as tf
import base.metrics as mtr

def diff_MSE(rho_f_list, rho_list):
    """Compute loss using Mean Squared Error (MSE) between the density matrices."""
    
    mse_sum = 0.0

    for rho, rho_f in zip(rho_list, rho_f_list):
        # Compute the squared difference between the density matrices
        difference = rho - rho_f
        mse = tf.reduce_sum(tf.abs(difference)**2)  # MSE on matrix elements
        mse_sum += mse

    mse_avg = mse_sum / len(rho_list)

    return mse_avg  # Lower value indicates better matching

def diff_fidelity(rho, rho_3):
    """Compute loss using Trace Fidelity between the density matrices."""
    return 1 - mtr.compilation_trace_fidelity(rho, rho_3)