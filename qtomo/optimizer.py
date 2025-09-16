
# Xem lại file optimize_algorithm.py và optimize.py
# Adam và SGD đang bị duplicate và inject nhiều chỗ
# Tạo 1 class optimizer, adam / sgd kế thừa từ class này
# phương thức trừu tượng là hàm step_and_cost, hoặc đơn giản sử dụng optimizer từ Pennylane cho lẹ
# https://github.com/vutuanhai237/qoop/blob/master/core/optimizer_pennylane.py
# Optimizer thì ko nhận bát kì tham số nào khác ngoài hyperparameter của chính nào và gradient/cost_fn, theta

from qtomo import gradient
from qtomo import utils
import tensorflow as tf

class Optimizer:
    def __init__(self, alpha=0.001):
        """
        Initialize the optimizer.

        Args:
            alpha (float): Learning rate.
        """
        self.alpha = alpha
    def step_and_cost(self, 
                      param: tf.Variable, 
                      loss_fn: callable, 
                      grad_fn: callable,
                     ):
        pass
    
class SGDOptimizer(Optimizer):
    def step_and_cost(self, 
                      param: tf.Variable, 
                      loss_fn: callable, 
                      grad_fn: callable,
                     ):
        pass
    
class AdamOptimizer(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = tf.constant(beta1, dtype=tf.complex128)
        self.beta2 = tf.constant(beta2, dtype=tf.complex128)
        self.epsilon = epsilon

        self.m = None
        self.v = None

        self.t = tf.constant(0, dtype=tf.complex128)
    
    def step_and_cost(self, 
                      param: tf.Variable, 
                      loss_fn: callable, 
                      grad_fn: callable,
                     ) -> tuple[tf.Variable, tf.Tensor]:
        """
        Perform one Adam optimization step with projection on the parameter.
        
        Args:
          param (tf.Variable): Complex tensor parameter to optimize.
          loss_fn (callable): Function that takes `param` and returns scalar loss tensor.
          grad_fn (callable): Function that calculates the projected gradient, default is gradient.calculate_gradient.
        
        Returns:
          param (tf.Variable): Updated parameter after one optimization step.
          loss (tf.Tensor): Scalar loss value *before* the parameter update.
        """
        if self.m is None:
            self.m = tf.zeros_like(param, dtype=tf.complex128)
            self.v = tf.zeros_like(param, dtype=tf.complex128)
        
        self.t += 1
        
        # Compute loss before update (for gradient calculation)
        proj_grad, loss = grad_fn(loss_fn, param)

        self.m = self.beta1 * self.m + (1 - self.beta1) * proj_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * tf.math.square(proj_grad)

        m_hat = self.m / (1 - tf.pow(self.beta1, self.t))
        v_hat = self.v / (1 - tf.pow(self.beta2, self.t))

        update = param - self.alpha * m_hat / (tf.math.sqrt(v_hat) + self.epsilon)
        
        return update, loss
