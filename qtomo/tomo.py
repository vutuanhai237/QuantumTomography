
# Đây là class chính lưu dữ liệu trong quá trình tomograhpy
# Lưu trữ data, model, optimizer, metric, generator
# Gom lại từ các fuke






class Tomography:
    pass
# Chủ yếu bao gồm input (unitary, epsilon, ...) và tập Kraus operator
# Qúa trình tính toắn update theo kiểu sau
# Tomography.init() => khởi tạo Kraus operator, gọi từ generator.py
# Tomography.fit() => optimize Karus operator, có sử dụng optimizer.py
# Tomography.evaluate() => tính toán metric, có sử dụng metric.py
# Thống nhất tên biến (tham số) xuyên suốt là gì, ví dụ bên optimize_adam_unitary_dagger_set gọi là 