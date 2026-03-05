import os
import random
import re
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

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


def cal_coefficient_of_variation(cgm_array):
    cgm_array = np.array(cgm_array)

    mean_cgm = np.mean(cgm_array)
    std_cgm = np.std(cgm_array)

    cv_percent = (std_cgm / mean_cgm) * 100
    return cv_percent


def cal_mean_relative_deviation(x_patient, x_agent, eps=1e-6):
    rel_dev = np.abs(x_agent - x_patient) / (np.abs(x_patient) + eps)
    return rel_dev.mean(), rel_dev


def cal_normalized_l2_distance(x_patient, x_agent, eps=1e-6):
    x_p = x_patient / (x_patient + eps)
    x_a = x_agent / (x_patient + eps)
    return np.linalg.norm(x_a - x_p)


def cal_pnd(x_patient, x_agent, x_avg_patient, eps=1e-6):
    x_patient = np.asarray(x_patient, dtype=float)
    x_agent = np.asarray(x_agent, dtype=float)
    x_avg_patient = np.asarray(x_avg_patient, dtype=float)

    return np.mean(np.abs(x_agent - x_patient) / (np.abs(x_avg_patient) + eps))


def cal_cosine_similarity(x_patient, x_agent, eps=1e-6):
    num = np.dot(x_patient, x_agent)
    den = (np.linalg.norm(x_patient) * np.linalg.norm(x_agent)) + eps
    return num / den


def count_glycemic_events(data, threshold, mode='hyper'):
    if mode == 'hyper':
        condition = data[0] > threshold
    elif mode == 'hypo':
        condition = data[0] < threshold
    else:
        raise ValueError("Invalid mode. Use 'hyper' for hyperglycemia or 'hypo' for hypoglycemia.")

    count = int(condition)

    for value in data:
        if mode == 'hyper':
            if value > threshold and not condition:
                count += 1
                condition = True
            elif value <= threshold and condition:
                condition = False
        elif mode == 'hypo':
            if value < threshold and not condition:
                count += 1
                condition = True
            elif value >= threshold and condition:
                condition = False

    return count


def cal_wilcoxon_pair(df, algo1, algo2, metric):
    """
    Wilcoxon signed-rank test between two algorithms for one metric.

    Returns:
        statistic, p_value
    """
    x = np.array(df[algo1][metric])
    y = np.array(df[algo2][metric])

    assert len(x) == len(y), "Paired samples must have same length"

    res = wilcoxon(x, y)
    return res.pvalue, res.statistic


def cal_wilcoxon_matrix(df, algorithms, metric, correction=None):
    """
    Compute pairwise Wilcoxon signed-rank p-value matrix.

    correction:
        None       -> return raw p-values
        "holm"     -> Holm–Bonferroni correction
        "fdr_bh"   -> Benjamini–Hochberg correction
    """

    n = len(algorithms)
    raw_pvals = np.full((n, n), np.nan)

    # Step 1: compute raw p-values (upper triangle only)
    pval_list = []
    pair_indices = []

    for i in range(n):
        for j in range(i + 1, n):
            p, _ = cal_wilcoxon_pair(df, algorithms[i], algorithms[j], metric)
            raw_pvals[i, j] = p
            raw_pvals[j, i] = p

            pval_list.append(p)
            pair_indices.append((i, j))

    raw_df = pd.DataFrame(raw_pvals, index=algorithms, columns=algorithms)

    # Step 2: apply correction if requested
    if correction is not None:
        rejected, adj_pvals, _, _ = multipletests(pval_list, method=correction)

        adj_matrix = np.full((n, n), np.nan)

        for (i, j), adj_p in zip(pair_indices, adj_pvals):
            adj_matrix[i, j] = adj_p
            adj_matrix[j, i] = adj_p

        adj_df = pd.DataFrame(adj_matrix, index=algorithms, columns=algorithms)

        print("\nRaw p-values:\n")
        print(raw_df.to_string())

        print(f"\nAdjusted p-values ({correction}):\n")
        print(adj_df.to_string())

        return raw_df, adj_df

    else:
        print("\nRaw p-values:\n")
        print(raw_df.to_string())
        return raw_df


def plot_tir_comparison(df, algorithms):
    subjects = np.arange(1, 26)
    num_algos = len(algorithms)

    bar_width = 0.125
    x = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(16, 6))

    for i, algo in enumerate(algorithms):
        tir_values = df[algo]["tir"]
        offset = i * bar_width
        plt.bar(
            x + offset,
            tir_values,
            width=bar_width,
            label=algo,
        )

    plt.xlabel("Subject ID", fontsize=14)
    plt.ylabel("Time in Range (TIR %)", fontsize=14)
    plt.xticks(x + bar_width * (num_algos - 1) / 2, subjects, fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_facecolor("whitesmoke")
    plt.legend(fontsize=14, loc="lower right")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("tir_comparison.png", dpi=300)
    plt.show()


def extract_behavior_features_from_actions(actions, main_meal_actions):
    bolus_events = []
    snack_meal_events = []

    # Process hourly actions (snacks + boluses)
    for hour in range(len(actions)):
        a, v, idx = actions[hour]
        abs_time = hour + idx / 12.0

        if a == Action.INJECT:
            bolus_events.append((abs_time, v))
        elif a == Action.EAT:
            snack_meal_events.append((abs_time, v))

    # Process main meals separately
    main_meal_events = [(hour + idx / 12.0, v) for hour, _, v, idx in main_meal_actions]

    # Combine all meals for counts and carb totals
    all_meals = snack_meal_events + main_meal_events

    meals_per_day = len(all_meals)
    boluses_per_day = len(bolus_events)
    total_bolus = sum(v for _, v in bolus_events)
    total_carb = sum(v for _, v in all_meals)
    total_insulin_per_day = total_bolus
    bolus_to_carb_ratio = total_bolus / total_carb if total_carb > 0 else 0.0

    meal_times = sorted(t for t, _ in all_meals)
    if len(meal_times) >= 2:
        meal_diffs = np.diff(meal_times) * 60
        avg_meal_gap = np.mean(meal_diffs)
    else:
        avg_meal_gap = 0.0

    bolus_times = sorted(t for t, _ in bolus_events)
    if len(bolus_times) >= 2:
        bolus_diffs = np.diff(bolus_times) * 60
        avg_bolus_gap = np.mean(bolus_diffs)
    else:
        avg_bolus_gap = 0.0

    return {"Number of Injections": round(boluses_per_day, 2),
            "Total Bolus Units": round(total_insulin_per_day, 2),
            "Number of Meals": round(meals_per_day, 2),
            "Total Carb Size": round(total_carb, 2),
            "Bolus to Carb Ratio": round(bolus_to_carb_ratio, 2),
            "Meal Gap \n(min)": round(avg_meal_gap, 2),
            "Bolus Gap \n(min)": round(avg_bolus_gap, 2)}


def extract_patient_behavior_features(x_patient):
    df = pd.DataFrame(x_patient,
                      columns=['timestamp', 'hour', 'sleep', 'time_since_last_meal', 'time_since_last_insulin', 'carb', 'bolus', 'basal', 'glucose',
                          'ga_200', 'cgm_class'])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["carb"] = df["carb"].astype(float)
    df["bolus"] = df["bolus"].astype(float)

    meal_events = df[df["carb"] > 0].copy().sort_values("timestamp")
    bolus_events = df[df["bolus"] > 0].copy().sort_values("timestamp")

    # Per-day stats
    meals_per_day = meal_events.groupby("date").size().mean()
    boluses_per_day = bolus_events.groupby("date").size().mean()
    total_carb_per_day = meal_events.groupby("date")["carb"].sum().mean()
    total_bolus_per_day = bolus_events.groupby("date")["bolus"].sum().mean()
    bolus_to_carb_ratio = total_bolus_per_day / total_carb_per_day if total_carb_per_day > 0 else 0.0

    # NEW: Avg time between meals (entire dataset)
    meal_times = meal_events["timestamp"].values
    if len(meal_times) >= 2:
        meal_diffs = np.diff(meal_times).astype('timedelta64[m]').astype(int)
        avg_meal_gap = np.mean(meal_diffs)
    else:
        avg_meal_gap = 0.0

    # NEW: Avg time between injections (entire dataset)
    bolus_times = bolus_events["timestamp"].values
    if len(bolus_times) >= 2:
        bolus_diffs = np.diff(bolus_times).astype('timedelta64[m]').astype(int)
        avg_bolus_gap = np.mean(bolus_diffs)
    else:
        avg_bolus_gap = 0.0

    return {"Number of Injections": round(boluses_per_day, 2),
            "Total Bolus Units": round(total_bolus_per_day, 2),
            "Number of Meals": round(meals_per_day, 2),
            "Total Carb Size": round(total_carb_per_day, 2),
            "Bolus to Carb Ratio": round(bolus_to_carb_ratio, 2),
            "Meal Gap \n(min)": round(avg_meal_gap, 2),
            "Bolus Gap \n(min)": round(avg_bolus_gap, 2)}


def compare_behavioral_feature_vector():
    root = "model/cql_bc/tests/final/azt1d"
    filename = "behavioral_features.csv"

    all_x_patient = []
    all_records = []

    for patient_dir in sorted(os.listdir(root)):
        if not patient_dir.startswith("azt1d_patient_"):
            continue

        if patient_dir in ['azt1d_patient_23']:
            continue

        patient_path = os.path.join(root, patient_dir)

        agent_values = []
        patient_df = None

        for seed_dir in sorted(os.listdir(patient_path)):
            if not seed_dir.startswith("seed_"):
                continue

            csv_path = os.path.join(patient_path, seed_dir, filename)
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            # patient values are identical across seeds → take once
            if patient_df is None:
                patient_df = df[["feature", "patient"]]

            agent_values.append(df["agent"].values)

        # average agent over seeds
        agent_mean = np.mean(agent_values, axis=0)

        behavioral_df = patient_df.copy()
        behavioral_df["agent"] = agent_mean

        agent_dic = {
            row["feature"]: round(row["agent"], 2)
            for _, row in behavioral_df.iterrows()
        }

        patient_dic = {
            row["feature"]: round(row["patient"], 2)
            for _, row in behavioral_df.iterrows()
        }

        plot_behavior_radar(patient_dic, agent_dic, patient_path)

        x_patient = np.array(behavioral_df["patient"].values)
        x_agent = np.array(behavioral_df["agent"].values)

        all_x_patient.append(x_patient)
        all_records.append((patient_dir, x_patient, x_agent))

    x_avg_patient = np.mean(np.stack(all_x_patient), axis=0)

    for patient_dir, x_patient, x_agent in all_records:
        pnd = cal_pnd( x_patient=x_patient, x_agent=x_agent, x_avg_patient=x_avg_patient)
        cosin = cal_cosine_similarity(x_patient, x_agent)
        mrd = cal_mean_relative_deviation(x_patient, x_agent)


def plot_cgm_reward_action(cgm_sequence,
                           hour_series,
                           reward_list,
                           action_list,
                           test_index,
                           main_meal_actions=None,
                           save_path_prefix=None,
                           show=False):

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

    if show:
        plt.show()


def plot_tir_tbr_tar(tir_list, tar_list, tbr_list, save_path, title='Glucose Range Metrics Across Tests', show=False):
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

    if show:
        plt.show()


def plot_eat_action_distribution(test_actions, test_time_window, save_path, show=False):
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

    if show:
        plt.show()

    # Plot 2: Average carb amount with std dev
    plt.figure(figsize=(10, 5))
    plt.bar(hours, avg_amounts, yerr=std_amounts, capsize=4, color='dodgerblue', alpha=0.9, error_kw=dict(ecolor='red', linewidth=1.5))
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Carb Amount (g)")
    plt.title(f"Average Carb Intake by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_carb_amount.png', dpi=300)

    if show:
        plt.show()


def plot_insulin_action_distribution(test_actions, test_time_window, save_path, show=False):
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

    if show:
        plt.show()

    # Plot 2: Average insulin amount with std dev
    plt.figure(figsize=(10, 5))
    plt.bar(hours, avg_amounts, yerr=std_amounts, capsize=4, color='mediumseagreen', alpha=0.9, error_kw=dict(ecolor='black', linewidth=1.5))
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Bolus Insulin Amount (units)")
    plt.title(f"Average Bolus Insulin Injection by Hour Across {num_tests} Tests")
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{save_path}/all_insulin_amount.png', dpi=300)

    if show:
        plt.show()


def plot_behavior_radar(patient, agent, save_path, show=False):
    labels = list(patient.keys())
    num_vars = len(labels)

    df = pd.DataFrame({"feature": labels, "patient": [patient[feat] for feat in labels], "agent": [agent[feat] for feat in labels], })
    df.to_csv(f"{save_path}/behavioral_features.csv", index=False)

    max_vals_dict = {}
    for feat in labels:
        max_val = max(patient.get(feat, 0), agent.get(feat, 0))
        max_vals_dict[feat] = max_val * 1.1
        if max_vals_dict[feat] == 0:
            max_vals_dict[feat] = 1.0

    patient_vals_norm = [patient[feat] / max_vals_dict[feat] for feat in labels]
    agent_vals_norm = [agent[feat] / max_vals_dict[feat] for feat in labels]

    patient_vals_norm += patient_vals_norm[:1]
    agent_vals_norm += agent_vals_norm[:1]

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # --- Custom Styling ---
    ax.set_title(f'Average Daily Behavioral Feature Comparison', fontsize=16, pad=20)
    ax.title.set_y(1.1)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10, pad=25)
    ax.xaxis.grid(True, color='#AAAAAA')
    ax.yaxis.grid(True, color='#AAAAAA')

    # Plot the normalized data
    ax.plot(angles, patient_vals_norm, label='Patient', color='#FF69B4', linewidth=2)
    ax.fill(angles, patient_vals_norm, color='#FF69B4', alpha=0.3)

    ax.plot(angles, agent_vals_norm, label='RL Agent', color='deepskyblue', linewidth=2)
    ax.fill(angles, agent_vals_norm, color='deepskyblue', alpha=0.3)

    r_grids_norm = [0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_ylim(0, 1.0)
    ax.set_yticks(r_grids_norm)
    ax.set_yticklabels([])

    for i in range(num_vars):
        feat = labels[i]
        angle = angles[i]

        max_val_real = max_vals_dict[feat] * 1.0

        if 0.0 < angle < np.pi:
            text_offset = 0.03
            alignment = 'left'
        elif angle > np.pi:
            text_offset = -0.03
            alignment = 'right'
        else:
            text_offset = 0.0
            alignment = 'center'

        for r_norm in r_grids_norm:
            r_real = max_val_real * r_norm

            if r_real >= 10:
                format_str = f'{r_real:.0f}'
            elif r_real >= 1:
                format_str = f'{r_real:.1f}'
            elif r_real >= 0.1:
                format_str = f'{r_real:.2f}'
            else:
                format_str = f'{r_real:.3f}'

            ax.text(angle + text_offset, r_norm, format_str, ha=alignment, va='center', fontsize=10, color='dimgray', clip_on=True)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=12, ncol=2)

    plt.tight_layout()
    plt.savefig(f'{save_path}/behavioral_comparison_radar.png', dpi=300)

    if show:
        plt.show()
