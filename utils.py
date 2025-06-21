import os
import random

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from config import Threshold


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


def plot_cgm_reward_action_with_legend(cgm_sequence, hour_series, reward_list, action_list, days, save_path_prefix=None):
    for day in range(days):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        start = day * 24
        end = (day + 1) * 24

        # Plot CGM
        day_length = 24 * 12
        cgm_day = np.array(cgm_sequence[day * day_length:(day + 1) * day_length])
        x_cgm = np.arange(day_length)

        ax1.plot(x_cgm, cgm_day, label="CGM", color="black")

        # CGM Ranges
        hypo_mask = cgm_day < Threshold.HYPOGLYCEMIA
        hyper_mask = cgm_day > Threshold.HYPERGLYCEMIA
        in_range_mask = ~hypo_mask & ~hyper_mask
        ax1.plot(x_cgm[hypo_mask], cgm_day[hypo_mask], 'r-', label='Hypoglycemia')
        ax1.plot(x_cgm[hyper_mask], cgm_day[hyper_mask], 'orange', label='Hyperglycemia')
        ax1.plot(x_cgm[in_range_mask], cgm_day[in_range_mask], 'g-', label='In Range')

        ax1.axhline(y=Threshold.HYPOGLYCEMIA, color='red', linestyle='--', linewidth=0.5)
        ax1.axhline(y=Threshold.HYPERGLYCEMIA, color='orange', linestyle='--', linewidth=0.5)

        ax1.set_title(f"Day {day + 1} - CGM with Actions")
        ax1.set_ylabel("CGM Value")
        ax1.grid(True)

        # Plot Reward
        reward_day = reward_list[start:end]
        x_reward = np.arange(0, day_length, 12)

        ax2.plot(x_reward, reward_day, label='Reward', color='orange')

        # Use day-specific slice of hour_series
        hour_day = hour_series[day * day_length:(day + 1) * day_length]

        # For tick labels, show every hour (every 12 points = 1 hour)
        tick_indices = np.arange(0, day_length, 12)
        tick_labels = [str(int(hour_day[i])) for i in tick_indices]

        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels)

        ax2.set_title(f"Day {day + 1} - Reward with Actions")
        ax2.set_ylabel("Reward")
        ax2.set_xlabel("Time")
        ax2.grid(True)

        # Plot vertical lines for actions
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
                ax.axvline(x=x, ymin=0, ymax=0.1, color=color, linewidth=1)
                if label and ax == ax1:
                    ylim = ax.get_ylim()
                    y_pos = ylim[0] + 0.12 * (ylim[1] - ylim[0])
                    ax.text(x, y_pos, label, ha='center', va='bottom', fontsize=10, color=color)

        # Legends
        ax1.legend(loc='lower right')
        ax2.legend(loc='upper right')

        action_patches = [mpatches.Patch(color='green', label='Nothing'), mpatches.Patch(color='blue', label='Eat'),
                          mpatches.Patch(color='red', label='Inject')]
        ax2.legend(handles=action_patches, loc='lower right')

        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_day_0{day + 1}.png", dpi=300)
        plt.show()
