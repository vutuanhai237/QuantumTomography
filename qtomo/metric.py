
# Gọp các function từ 2 file metrics.py và lost_function.py vào 1 file metric.py
# Thêm kiểu dữ liệu cho in-out, ví dụ:
# Chỉ rõ kiểu dữ liệu đang thuộc miền nào, ví dụ rho_f_list đang là N X d X K tensor?
import numpy as np


def MSE(rho_f_list: np.ndarray, rho_list: np.ndarray) -> float:
    """Compute loss using Mean Squared Error (MSE) between the density matrices."""
    
    mse_sum = 0.0

    for rho, rho_f in zip(rho_list, rho_f_list):
        # Compute the squared difference between the density matrices
        difference = rho - rho_f
        mse = tf.reduce_sum(tf.abs(difference)**2)
        mse_sum += mse

    mse_avg = mse_sum / len(rho_list)

    return mse_avg  # Lower value indicates better matching