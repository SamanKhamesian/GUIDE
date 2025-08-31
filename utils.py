import os
import random
import re
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from scipy.stats import pearsonr

from config import Action, Threshold


def set_seed(seed=42):
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)  # Torch CPU
    torch.cuda.manual_seed(seed)  # Torch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Force deterministic algorithm
    torch.backends.cudnn.benchmark = False  # Turn off auto-tuner that may introduce randomness

    # TensorFlow
    tf.random.set_seed(seed)

    # Ensure deterministic operations (for TF 2.x)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Optional: force deterministic TF ops


def numerical_sort_key(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(numbers[0])
    return s


def cal_time_in_range(cgm_array):
    cgm_array = np.array(cgm_array)
    count = ((Threshold.HYPOGLYCEMIA <= cgm_array) & (cgm_array <= Threshold.HYPERGLYCEMIA)).sum()
    return (count / len(cgm_array)) * 100


def cal_time_below_range(cgm_array):
    cgm_array = np.array(cgm_array)
    count = (Threshold.HYPOGLYCEMIA > cgm_array).sum()
    return (count / len(cgm_array)) * 100


def cal_time_above_range(cgm_array):
    cgm_array = np.array(cgm_array)
    count = (cgm_array > Threshold.HYPERGLYCEMIA).sum()
    return (count / len(cgm_array)) * 100


def select_main_meal_hours():
    breakfast_hour = np.random.choice([7, 8, 9])
    lunch_hour     = np.random.choice([12, 13, 14])
    dinner_hour    = np.random.choice([19, 20, 21, 22])
    return [breakfast_hour, lunch_hour, dinner_hour]


def select_main_meal_portion(low, high, mean, sd):
    while True:
        x = np.random.normal(mean, sd)
        if low <= x <= high:
            return x


def evaluate_action_success(test_cgms, test_actions, threshold, direction, target_action_type):
    """
    Parameters:
        - threshold: value to compare CGM against
        - direction: '>' for hyper, '<' for hypo
        - target_action_type: Action.EAT or Action.INJECT
    """
    tp = fp = fn = tn = 0

    for cgm_seq, actions in zip(test_cgms, test_actions):
        cgm_seq = np.asarray(cgm_seq, dtype=float)
        n_hours = len(actions)
        if n_hours == 0:
            continue

        slots_per_hour = len(cgm_seq) // n_hours
        usable_len = n_hours * slots_per_hour
        hourly_means = cgm_seq[:usable_len].reshape(n_hours, slots_per_hour).mean(axis=1)
        action_types = np.array([t for (t, _, _) in actions], dtype=int)

        for mean_cgm, act_type in zip(hourly_means, action_types):
            is_event = (mean_cgm > threshold) if direction == '>' else (mean_cgm < threshold)
            is_action = act_type == target_action_type

            if is_event and is_action:
                tp += 1
            elif is_event and not is_action:
                fn += 1
            elif not is_event and is_action:
                fp += 1
            else:
                tn += 1

    # Use np.nan instead of 0.0 to indicate undefined precision/recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": precision,
        "Recall": recall
    }


def print_and_save_action_eval(title, result_dict, log_path):
    """
    Parameters:
        title: str → Section title like "Injection on Hyperglycemia"
        result_dict: dict → Output from evaluate_action_success()
        log_path: str → Path to the log file
    """
    tp = result_dict["TP"]
    fp = result_dict["FP"]
    fn = result_dict["FN"]
    tn = result_dict["TN"]
    precision = result_dict["Precision"]
    recall = result_dict["Recall"]

    total = tp + fp + fn + tn
    pct = lambda x: (x / total) * 100 if total > 0 else np.nan
    fmt = lambda x: f"{x:.2f}" if not np.isnan(x) else "NaN"

    # Console output
    print(f"\n------------------ {title} ------------------")
    print(f"True Positives : {tp} ({fmt(pct(tp))}%)")
    print(f"False Positives: {fp} ({fmt(pct(fp))}%)")
    print(f"False Negatives: {fn} ({fmt(pct(fn))}%)")
    print(f"True Negatives : {tn} ({fmt(pct(tn))}%)")
    print(f"Precision      : {fmt(precision)}")
    print(f"Recall         : {fmt(recall)}")

    # File output
    with open(log_path, "a") as f:
        f.write(f"\n------------------ {title} ------------------\n")
        f.write(f"True Positives : {tp} ({fmt(pct(tp))}%)\n")
        f.write(f"False Positives: {fp} ({fmt(pct(fp))}%)\n")
        f.write(f"False Negatives: {fn} ({fmt(pct(fn))}%)\n")
        f.write(f"True Negatives : {tn} ({fmt(pct(tn))}%)\n")
        f.write(f"Precision      : {fmt(precision)}\n")
        f.write(f"Recall         : {fmt(recall)}\n")



def __safe_corr(x, y):
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    r, _ = pearsonr(x, y)

    if np.isnan(r):
        return 0.0
    return r

def compute_event_corr(test_cgms, test_actions):
    """
    Returns a 2x2 matrix with rows [InsulinDesired(>185), MealDesired(<110)],
    cols [Eat, Inject], computed across ALL tests for this patient.
    Uses HOURLY mean CGM values aligned to hourly actions (no expansion).
    """
    hourly_means_all = []
    hourly_action_types_all = []

    for cgm_seq, actions in zip(test_cgms, test_actions):
        cgm_seq = np.asarray(cgm_seq, dtype=float)
        n_hours = len(actions)
        if n_hours == 0:
            continue

        slots_per_hour = len(cgm_seq) // n_hours
        if slots_per_hour <= 0:
            continue

        usable_len = n_hours * slots_per_hour
        hourly_means = cgm_seq[:usable_len].reshape(n_hours, slots_per_hour).mean(axis=1)

        action_types = np.array([t for (t, _, _) in actions], dtype=int)

        m = min(len(hourly_means), len(action_types))
        hourly_means_all.append(hourly_means[:m])
        hourly_action_types_all.append(action_types[:m])

    if not hourly_means_all:
        return np.zeros((2, 2), dtype=float)

    mean_cgm_hourly = np.concatenate(hourly_means_all)
    action_types_hourly = np.concatenate(hourly_action_types_all)

    # Event masks from reward logic
    insulin_desired = (mean_cgm_hourly > 165).astype(int)
    meal_desired    = (mean_cgm_hourly < 120).astype(int)

    # Action masks
    eat_actions    = (action_types_hourly == Action.EAT).astype(int)
    inject_actions = (action_types_hourly == Action.INJECT).astype(int)

    # Correlations
    corr_insulin_eat    = __safe_corr(insulin_desired, eat_actions)
    corr_insulin_inject = __safe_corr(insulin_desired, inject_actions)
    corr_meal_eat       = __safe_corr(meal_desired,    eat_actions)
    corr_meal_inject    = __safe_corr(meal_desired,    inject_actions)

    mat = np.array([
        [corr_insulin_eat,   corr_insulin_inject],
        [corr_meal_eat,      corr_meal_inject ],
    ], dtype=float)

    return np.nan_to_num(mat, nan=0.0)


def plot_event_corr(corr_matrix, save_path):
    """
    Plots a single 2x2 heatmap (rows: Hyper/Hypo, cols: Eat/Inject).
    If any entry was undefined originally, we expect it was set to 0 by nan_policy="zero".
    """
    rows = ["Hyperglycemia", "Hypoglycemia"]
    cols = ["Eat", "Inject"]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(cols)));  ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)));  ax.set_yticklabels(rows)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)

    ax.set_title("Correlation (Events × Actions)")
    fig.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    fig.savefig(f"{save_path}/event_corr.png", dpi=300)
    plt.show()


def plot_rewards(reward_list, title="Reward per Step", save_path="test.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_list, label='Reward')
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cgm_levels(cgm_sequence, hour_series, title="Predicted CGM Levels", save_path="test.png"):
    cgm_sequence = np.array(cgm_sequence)
    plt.figure(figsize=(12, 6))
    plt.plot(cgm_sequence, label="CGM", color="black")

    hypo_mask = cgm_sequence < Threshold.HYPOGLYCEMIA
    hyper_mask = cgm_sequence > Threshold.HYPERGLYCEMIA
    in_range_mask = ~hypo_mask & ~hyper_mask

    plt.plot(np.where(hypo_mask)[0], cgm_sequence[hypo_mask], 'r-', label='Hypoglycemia')
    plt.plot(np.where(hyper_mask)[0], cgm_sequence[hyper_mask], 'orange', label='Hyperglycemia')
    plt.plot(np.where(in_range_mask)[0], cgm_sequence[in_range_mask], 'g-', label='In Range')

    plt.axhline(y=Threshold.HYPOGLYCEMIA, color='red', linestyle='--', linewidth=0.5)
    plt.axhline(y=Threshold.HYPERGLYCEMIA, color='orange', linestyle='--', linewidth=0.5)

    tick_indices = np.arange(0, len(hour_series), 12 * 6)
    tick_labels = [str(int(hour_series[i])) for i in tick_indices]

    plt.xticks(tick_indices, tick_labels, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("CGM Value", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_cgm_reward_action_with_legend(cgm_sequence, hour_series, reward_list, action_list, test_index, main_meal_actions=None, save_path_prefix=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    start = 0
    end = 24
    day_length = 24 * 12

    # CGM values
    cgm_day = np.array(cgm_sequence[start * day_length:end * day_length])
    x_cgm = np.arange(day_length)

    # Highlight ranges
    hypo_mask = cgm_day < Threshold.HYPOGLYCEMIA
    hyper_mask = cgm_day > Threshold.HYPERGLYCEMIA
    in_range_mask = ~hypo_mask & ~hyper_mask

    ax1.plot(x_cgm, cgm_day, color="gray", linestyle='--')

    def plot_continuous_segments(x, y, mask, color, label):
        segments = []
        current = []
        for i in range(len(mask)):
            if mask[i]:
                current.append(i)
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)

        if segments:
            for i, segment in enumerate(segments):
                ax1.plot(x[segment], y[segment], color=color, label=label if i == 0 else None)
        else:
            ax1.plot([], [], color=color, label=label)

    plot_continuous_segments(x_cgm, cgm_day, hypo_mask, 'red', 'Hypoglycemia')
    plot_continuous_segments(x_cgm, cgm_day, hyper_mask, 'orange', 'Hyperglycemia')
    plot_continuous_segments(x_cgm, cgm_day, in_range_mask, 'green', 'In Range')

    ax1.axhline(y=Threshold.HYPOGLYCEMIA, color='red', linestyle='--', linewidth=0.5)
    ax1.axhline(y=Threshold.HYPERGLYCEMIA, color='orange', linestyle='--', linewidth=0.5)

    ax1.set_title(f"Results for Test {test_index}", fontsize=16)
    ax1.set_ylabel("CGM Level (mg/dL)", fontsize=14)
    ax1.set_xlabel("Time of the day (Hour)", fontsize=14)
    ax1.set_ylim(0, 300)
    ax1.grid(True)

    # Reward plot
    reward_day = reward_list[start:end]
    x_reward = np.arange(0, day_length, 12)
    ax2.plot(x_reward, reward_day, label='Reward', color='orange')
    ax2.set_ylabel("Reward", fontsize=14)
    ax2.set_xlabel("Time of the day (Hour)", fontsize=14)
    ax2.grid(True)

    # Action markers
    for i in range(start, end):
        action_type, action_value, action_time = action_list[i]
        x = (i - start) * 12 + int(action_time)

        if action_type == 0:
            color = 'green'
            label = None
        elif action_type == 1:
            color = 'blue'
            label = f"{action_value:.1f}"
        elif action_type == 2:
            color = 'red'
            label = f"{action_value:.1f}"
        else:
            continue

        for ax in [ax1, ax2]:
            ax.axvline(x=x, ymin=0, ymax=0.1, color=color, linewidth=1.5)
            if label and ax == ax1:
                ylim = ax.get_ylim()
                y_pos = ylim[0] + 0.12 * (ylim[1] - ylim[0])
                ax.text(x, y_pos, label, ha='center', va='bottom', fontsize=10, color=color)

    # Plot main meal actions (if any)
    if main_meal_actions:
        for step_index, action_type, action_value, action_time in main_meal_actions:
            x = step_index * 12 + int(action_time)

            for ax in [ax1, ax2]:
                ax.axvline(x=x, ymin=0, ymax=0.1, color='m', linewidth=1.5)
                if ax == ax1:
                    ylim = ax.get_ylim()
                    y_pos = ylim[0] + 0.12 * (ylim[1] - ylim[0])
                    ax.text(x, y_pos, f"{action_value:.1f}", ha='center', va='bottom', fontsize=10, color='m')

    # Hour ticks and labels
    hour_day = hour_series[start * day_length:end * day_length]
    tick_indices = np.arange(0, day_length, 12)
    tick_labels = [str(int(hour_day[i])) for i in tick_indices]

    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, fontsize=14)

    ax1.tick_params(axis='x', labelbottom=True, labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    # Legends (split)
    cgm_legend = ax1.legend(loc='upper right', fontsize=12)
    ax1.add_artist(cgm_legend)

    action_patches = [mpatches.Patch(color='green', label='Nothing'), mpatches.Patch(color='blue', label='Eat (g)'),
        mpatches.Patch(color='red', label='Inject (U)'), mpatches.Patch(color='m', label='Meal (g)')]

    ax1.legend(handles=action_patches, loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}/test_{test_index}_results.png", dpi=300)
    plt.show()


def plot_tir_tbr_tar(tir_list, tar_list, tbr_list, save_path, title='Glucose Range Metrics Across Tests'):
    days = np.arange(1, len(tir_list) + 1)
    labels = [f'Test {i}' for i in days]

    tir_list = np.array(tir_list)
    tar_list = np.array(tar_list)
    tbr_list = np.array(tbr_list)

    plt.figure(figsize=(10, 5))

    plt.plot(days, tir_list, label='TIR', color='maroon', marker='o', linestyle='-')
    plt.plot(days, tar_list, label='TAR', color='orange', marker='o', linestyle='--')
    plt.plot(days, tbr_list, label='TBR', color='dodgerblue', marker='o', linestyle=':')

    plt.xticks(days, labels, rotation=45)
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_eat_action_distribution(test_actions, test_time_window, save_path):
    hour_eat_counts = defaultdict(int)
    hour_eat_amounts = defaultdict(list)
    num_tests = len(test_actions)

    for actions, time_window in zip(test_actions, test_time_window):
        for i, (action_type, value, time_index) in enumerate(actions):
            if action_type == 1:  # EAT
                index_in_time = 12 * i + time_index
                if index_in_time >= len(time_window):
                    continue  # safety check
                hour = int(time_window[index_in_time])
                hour_eat_counts[hour] += 1
                hour_eat_amounts[hour].append(value)

    hours = list(range(24))
    counts = [hour_eat_counts[h] for h in hours]
    avg_amounts = [np.mean(hour_eat_amounts[h]) if hour_eat_amounts[h] else 0 for h in hours]
    std_amounts = [np.std(hour_eat_amounts[h]) if hour_eat_amounts[h] else 0 for h in hours]

    # Plot 1: Raw count of EAT events
    plt.figure(figsize=(10, 5))
    plt.bar(hours, counts, color='goldenrod')
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Meal Events")
    plt.title(f"Distribution of Eat Actions by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_carb_distribution.png', dpi=300)
    plt.show()

    # Plot 2: Average carb amount with std dev
    plt.figure(figsize=(10, 5))
    plt.bar(hours, avg_amounts, yerr=std_amounts, capsize=4, color='dodgerblue', alpha=0.9,
            error_kw=dict(ecolor='red', linewidth=1.5))
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Carb Amount (g)")
    plt.title(f"Average Carb Intake by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_carb_amount.png', dpi=300)
    plt.show()


def plot_insulin_action_distribution(test_actions, test_time_window, save_path):
    hour_inject_counts = defaultdict(int)
    hour_inject_amounts = defaultdict(list)
    num_tests = len(test_actions)

    for actions, time_window in zip(test_actions, test_time_window):
        for i, (action_type, value, time_index) in enumerate(actions):
            if action_type == 2:  # INJECT
                index_in_time = 12 * i + time_index
                if index_in_time >= len(time_window):
                    continue
                hour = int(time_window[index_in_time])
                hour_inject_counts[hour] += 1
                hour_inject_amounts[hour].append(value)

    hours = list(range(24))
    counts = [hour_inject_counts[h] for h in hours]
    avg_amounts = [np.mean(hour_inject_amounts[h]) if hour_inject_amounts[h] else 0 for h in hours]
    std_amounts = [np.std(hour_inject_amounts[h]) if hour_inject_amounts[h] else 0 for h in hours]

    # Plot 1: Raw count of INJECT events
    plt.figure(figsize=(10, 5))
    plt.bar(hours, counts, color='darkred')
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Bolus Insulin Injections")
    plt.title(f"Distribution of Bolus Insulin Injections by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.9)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_insulin_distribution.png', dpi=300)
    plt.show()

    # Plot 2: Average insulin amount with std dev
    plt.figure(figsize=(10, 5))
    plt.bar(hours, avg_amounts, yerr=std_amounts, capsize=4, color='mediumseagreen', alpha=0.9,
            error_kw=dict(ecolor='black', linewidth=1.5))
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Bolus Insulin Amount (units)")
    plt.title(f"Average Bolus Insulin Injection by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_insulin_amount.png', dpi=300)
    plt.show()
