import tensorflow as tf
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===================== CONFIG =====================

T_MAX = 30
REGIONS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
NUM_POINTS_EACH = [6, 8, 6, 6, 4, 2]

# ===================== PARSE =====================

def parse_tensor_from_file(file_path, shape):
    with open(file_path, 'r') as file:
        tensor_str = file.read()
    components = re.findall(r"([+-]?\d+\.?\d*[eE]?[+-]?\d*(?:\s*[+-]?\d*\.?\d*[eE]?[+-]?\d*j)?)", tensor_str)
    complex_numbers = [complex(c.replace(' ','')) for c in components]
    numpy_tensor = np.array(complex_numbers, dtype=np.complex128)

    if shape == 3:
        n = round(numpy_tensor.size ** (1/shape))
        numpy_tensor = numpy_tensor.reshape(n, n, n)
    elif shape == 2:
        n = round(numpy_tensor.size ** (1/shape))
        numpy_tensor = numpy_tensor.reshape(n, n)
    return tf.convert_to_tensor(numpy_tensor)

# ===================== LOAD DATA =====================

def retrieve_X_dict_dual(expected_dir, result_dir):
    X_dict = {}

    # --- Load Expected ---
    exp_folders = sorted(
        [f for f in os.listdir(expected_dir) if os.path.isdir(os.path.join(expected_dir, f))],
        key=lambda x: float(x.split('_')[2])
    )

    for folder_name in exp_folders:
        time_val = float(folder_name.split('_')[2])
        if time_val > T_MAX:
            continue

        label = folder_name.split('_')[-1]
        key = f"$\\rho'$ ({label})"
        file_path = os.path.join(expected_dir, folder_name, 'trace_rho2.txt')

        if not os.path.exists(file_path):
            continue

        value = parse_tensor_from_file(file_path, 1).numpy().real
        X_dict.setdefault(key, []).append({'t': time_val, 'value': value[0]})

    # --- Load Result (Downsampled) ---
    res_folders = sorted(
        [f for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))],
        key=lambda x: float(x.split('_')[2])
    )

    # Extract all (folder_name, t_val)
    folder_tuples = []
    for f in res_folders:
        try:
            t_val = float(f.split('_')[2])
            folder_tuples.append((f, t_val))
        except:
            continue

    from collections import defaultdict
    grouped_by_t = defaultdict(list)
    for f, t in folder_tuples:
        grouped_by_t[t].append(f)

    selected_folders = []

    for i, ((start, end), n_points) in enumerate(zip(REGIONS, NUM_POINTS_EACH)):
        if i == len(REGIONS) - 1:
            times_in_region = [t for t in grouped_by_t if start <= t <= end]
        else:
            times_in_region = [t for t in grouped_by_t if start <= t < end]

        times_in_region.sort()

        # Danh sách nhóm folder theo thời điểm t (cặp folder)
        pairs = [grouped_by_t[t] for t in times_in_region]

        if len(pairs) <= n_points:
            selected_pairs = pairs
        else:
            indices = np.linspace(0, len(pairs) - 1, n_points, dtype=int)
            selected_pairs = [pairs[i] for i in indices]

        # Thêm tất cả folder trong nhóm được chọn vào selected_folders
        for pair_folders in selected_pairs:
            for folder_name in pair_folders:
                time_val = float(folder_name.split('_')[2])
                selected_folders.append((folder_name, time_val))

        print(f"Region {start}-{end}: selected {len(selected_pairs)} pairs (time points)")
        print("Times selected:", [float(f.split('_')[2]) for pair in selected_pairs for f in pair])
        print("----")

    # Load selected result folders
    for folder_name, time_val in selected_folders:
        if time_val > T_MAX:
            continue

        label = folder_name.split('_')[-1]
        key = f"$\\overline{{\\rho'}}$ ({label})"
        file_path = os.path.join(result_dir, folder_name, 'trace_rho2_out.txt')

        if not os.path.exists(file_path):
            continue

        value = parse_tensor_from_file(file_path, 1).numpy().real
        X_dict.setdefault(key, []).append({'t': time_val, 'value': value[0]})

    return X_dict

# ===================== PLOT =====================

def plot_dict_with_inset(data_dict, xlabel, ylabel, file_name, inset_range=(0, 1.5)):
    plt.figure(figsize=(7.5, 5))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Vẽ từng đường
    for key, data in data_dict.items():
        t = np.array([d['t'] for d in data])
        y = np.array([d['value'] for d in data])

        if "overline" in key.lower():
            marker = 'o'
            linestyle = 'dotted'
            linewidth = 1.5
        else:
            marker = None
            linestyle = 'solid'
            linewidth = 2

        ax.plot(t, y, label=key, linestyle=linestyle, linewidth=linewidth, marker=marker)

    # Trục chính
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    

    # Inset plot
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
    for key, data in data_dict.items():
        t = np.array([d['t'] for d in data])
        y = np.array([d['value'] for d in data])
        mask = (t >= inset_range[0]) & (t <= inset_range[1])

        if "overline" in key.lower():
            marker = 'o'
            linestyle = 'dotted'
            linewidth = 1.5
        else:
            marker = None
            linestyle = 'solid'
            linewidth = 2

        ax_inset.plot(t[mask], y[mask], linestyle=linestyle, linewidth=linewidth, marker=marker)

    ax_inset.set_xlim(inset_range)
    ax_inset.grid(True, linestyle="--", alpha=0.5)
    ax_inset.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=14, loc='lower right')

    # Lưu hình
    plt.savefig(f"./docs/new_fig/{file_name}.png", format='png', bbox_inches='tight')
    plt.savefig(f"./docs/new_fig/{file_name}.eps", format='eps', bbox_inches='tight')
    plt.show()


# ===================== RUN =====================

trace_dict = retrieve_X_dict_dual(
    'results/experiment_new/traceX_expected',
    'results/experiment_new/traceX_result'
)

# Kiểm tra số lượng điểm
for key, entries in trace_dict.items():
    print(f"{key}: {len(entries)} điểm, t trong [{min(e['t'] for e in entries):.2f}, {max(e['t'] for e in entries):.2f}]")

print(trace_dict)
plot_dict_with_inset(
    trace_dict,
    xlabel='t',
    ylabel="Expectation values",
    file_name="trace_X_inset",
    inset_range=(0, 2)
)
