
# Xem lại file optimize_algorithm.py và optimize.py
# Adam và SGD đang bị duplicate và inject nhiều chỗ
# Tạo 1 class optimizer, adam / sgd kế thừa từ class này
# phương thức trừu tượng là hàm step_and_cost, hoặc đơn giản sử dụng optimizer từ Pennylane cho lẹ
# https://github.com/vutuanhai237/qoop/blob/master/core/optimizer_pennylane.py
# Optimizer thì ko nhận bát kì tham số nào khác ngoài hyperparameter của chính nào và gradient/cost_fn, theta


class Optimizer:
    pass

class AdamOptimizer(Optimizer):
    pass