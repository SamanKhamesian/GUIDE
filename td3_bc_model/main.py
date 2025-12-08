import os
import sys

import numpy as np
import torch

from config import Action, Threshold, TD3Config, TD3RewardShaping, DataConfig
from simulator import Simulator
from td3_bc_model.replay_buffer import ReplayBuffer
from td3_bc_model.td3_bc_agent import TD3_BC
from utils import (plot_cgm_reward_action, set_seed, cal_time_in_range, cal_time_below_range, cal_time_above_range, cal_coefficient_of_variation,
                   plot_tir_tbr_tar, plot_eat_action_distribution, plot_insulin_action_distribution, count_glycemic_events,
                   extract_behavior_features_from_actions, extract_patient_behavior_features, plot_behavior_radar)


class EnvironmentAdvanced:
    def __init__(self, dataset_name, patient_id):
        self.simulator = Simulator(dataset_name=dataset_name, patient_id=patient_id)
        self.simulator.train()

        # 7 features, and 6-hour data for each
        # features: [hour, sleep, time_since_last_meal, time_since_last_insulin, carbs, bolus, cgm]
        self.state_size = TD3Config.STATE_SIZE
        self.action_space = [Action.NOTHING, Action.EAT, Action.INJECT]

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)  # (type, value, time_index)
        self.main_meal_hours = self.set_main_meal_hours()
        self.main_meal_action_log = []
        self.repeat_counter = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()
        self.episode_cgm_history = []

    def get_window(self):
        return self.simulator.get_full_current_window()

    def get_state(self):
        window = self.get_window()
        # remove basal from state (it's only been in the glimmer input)
        window = np.delete(window, 6, axis=1)
        return window[:, :7].flatten()

    def get_time_series(self):
        return self.simulator.get_time_window()

    def get_current_hour(self):
        return int(self.get_time_series()[5])

    def get_sleep_signal(self):
        return self.simulator.get_sleep_mode()

    def set_main_meal_hours(self):
        return self.simulator.select_main_meal_hours()

    def set_main_mean_portion(self):
        return self.simulator.select_main_meal_portion(20, 100, mean=65, sd=15)

    def reset(self, state_index, is_testing):
        self.simulator.reset(state_index, is_testing)

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)
        self.main_meal_hours = self.set_main_meal_hours()
        self.main_meal_action_log = []
        self.repeat_counter = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()
        self.episode_cgm_history = []

        return self.current_state

    def step(self, step_idx, action):
        # Step 1: Predict next CGM values
        predicted_cgm = self.simulator.predict_next_cgm()
        main_meal_action = None

        # Step 2: Track predicted CGM values per step
        self.episode_cgm_history.extend(predicted_cgm)

        # Step 2.5: Add possible main meal action
        if (int(self.current_hour + 1) % 24) in self.main_meal_hours:
            main_meal_size = self.set_main_mean_portion()
            t_index = np.random.randint(0, 12)
            main_meal_action = (Action.EAT, main_meal_size, t_index)
            self.main_meal_action_log.append((step_idx, Action.EAT, main_meal_size, t_index))

        basal_array = [self.simulator.heuristic_basal_controller(predicted_cgm)] * 12

        # Step 3: Apply action to simulator
        bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array = (
            self.simulator.apply_action_to_inputs(self.full_current_window, action, main_meal_action))

        # Step 4: Commit inputs and predicted CGM
        self.simulator.commit_next_input(predicted_cgm,
                                         basal_array,
                                         bolus_array,
                                         time_since_last_injection_array[1:],
                                         carb_array,
                                         time_since_last_meal_array[1:])

        # Step 5: Update state
        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Step 6: Compute reward
        reward = self.compute_reward(time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action)

        self.prev_action = action

        return self.current_state, predicted_cgm, self.time_series, reward, False, {}

    def compute_reward(self, time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action):
        action_type, action_value, time_index = action

        mean_cgm = predicted_cgm.mean()

        x = np.arange(len(predicted_cgm))
        y = predicted_cgm
        slope, _ = np.polyfit(x, y, 1)

        # Identify glycemic ranges
        hypo = predicted_cgm < Threshold.HYPOGLYCEMIA
        hyper = predicted_cgm > Threshold.HYPERGLYCEMIA
        normal = ~hypo & ~hyper

        # Extract reward weights for different glycemic zones
        w_normal = TD3RewardShaping.WEIGHTS[0]
        w_hypo = TD3RewardShaping.WEIGHTS[1]
        w_hyper = TD3RewardShaping.WEIGHTS[2]
        ideal_cgm = TD3RewardShaping.IDEAL_CGM

        # Compute reward for each CGM point based on zone
        reward = np.zeros_like(predicted_cgm)
        normal_vals = predicted_cgm[normal]

        # Penalize hypoglycemia proportionally
        reward[hypo] = -w_hypo * (Threshold.HYPOGLYCEMIA - predicted_cgm[hypo])

        # Penalize hyperglycemia proportionally
        reward[hyper] = -w_hyper * (predicted_cgm[hyper] - Threshold.HYPERGLYCEMIA)

        reward[normal] = np.where(normal_vals <= ideal_cgm,
                                  (normal_vals - Threshold.HYPOGLYCEMIA) / (ideal_cgm - Threshold.HYPOGLYCEMIA) * w_normal,
                                  (Threshold.HYPERGLYCEMIA - normal_vals) / (Threshold.HYPERGLYCEMIA - ideal_cgm) * w_normal)

        # Aggregate reward across all time steps
        total_reward = np.sum(reward)

        # Update time since last meal and insulin for this time slot
        time_since_last_meal = time_since_last_meal_array[time_index] + 1
        time_since_last_insulin = time_since_last_injection_array[time_index] + 1

        # During sleep, reward doing nothing and penalize any action
        if self.sleep_mode:
            if action_type != Action.NOTHING:
                total_reward -= 100
            else:
                total_reward += 100

        # Bonus for maintaining CGM in stable, healthy range without action
        if action_type == Action.NOTHING and 120 < mean_cgm < 140:
            total_reward += 50  # increased from 25

        # Penalize not injecting insulin when hyperglycemia is severe
        if action_type != Action.INJECT and mean_cgm > 185:
            total_reward -= min(2000, (mean_cgm - 185) * 20)

        # Penalize not eating when CGM is low
        if action_type != Action.EAT and mean_cgm < 90:
            total_reward -= min(200, (90 - mean_cgm) * 5)

        # Evaluate insulin action
        if action_type == Action.INJECT:
            # Penalize injecting when CGM is already low
            if mean_cgm < 150:
                total_reward -= min(500, (150 - mean_cgm) * 10)

            # Reward insulin if CGM is high
            if mean_cgm > 185:
                total_reward += min(2000, (mean_cgm - 185) * 20)

            if time_since_last_insulin < 2 * 12:  # increased from 18
                total_reward -= 100

            if time_since_last_meal <= 18 and mean_cgm > 140 and slope > 0:
                total_reward += 75

        # Evaluate meal action
        if action_type == Action.EAT:
            # Reward eating when CGM is low
            if mean_cgm < 90:
                total_reward += min(200, (90 - mean_cgm) * 5)

            if mean_cgm > 150 and time_since_last_meal < 2 * 12:
                total_reward -= 100

            # Reward eating after recent insulin injection (may prevent hypo)
            if time_since_last_insulin < 12:
                total_reward += 25  # reduced from 50
            else:
                total_reward += 5  # reduced from 10

        # Update repeat counter based on current action
        if self.prev_action[0] == action_type and action_type != Action.NOTHING:
            self.repeat_counter += 1

            # Penalize repeated eating or injecting behavior
            if self.repeat_counter >= 2:
                total_reward -= (self.repeat_counter - 1) * 950

        else:
            self.repeat_counter = 0

        total_reward = total_reward / 1000.0
        return total_reward

    def compute_episode_reward(self):
        cgm_array = np.array(self.episode_cgm_history)

        # Compute daily metrics clearly:
        tir_ratio = cal_time_in_range(cgm_array)

        hypo_events = count_glycemic_events(data=list(cgm_array), threshold=Threshold.HYPERGLYCEMIA, mode='hypo')
        hyper_events = count_glycemic_events(data=list(cgm_array), threshold=Threshold.HYPERGLYCEMIA, mode='hyper')

        # Calculate event duration (in 5-minute intervals)
        hypo_duration = np.sum(cgm_array < Threshold.HYPOGLYCEMIA) * 5
        hyper_duration = np.sum(cgm_array > Threshold.HYPERGLYCEMIA) * 5

        # Apply clear penalty thresholds
        episode_reward = 0

        # Reward/Penalty for TIR (daily)
        episode_reward += (20 * (tir_ratio - 70))

        # Penalty for hypo events
        if hypo_events == 0:
            episode_reward += TD3RewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hypo_events <= 3:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hypo_events <= 5:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_3_PENALTY

        # Penalty for hyper events
        if hyper_events == 0:
            episode_reward += TD3RewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hyper_events <= 3:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hyper_events <= 5:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= TD3RewardShaping.HYPO_HYPER_3_PENALTY

        # Duration penalty (additional fine-tuning, optional but recommended)
        episode_reward -= ((hypo_duration + hyper_duration) * 0.5)  # penalty per minute outside range

        episode_reward = episode_reward / 1000.0
        return episode_reward


def fill_replay_buffer(env, buffer):
    print("Filling initial replay buffer...")
    for i in range(TD3Config.NUM_TRAIN_INIT_STATE):
        for episode in range(TD3Config.MAX_EPISODES):
            state = env.reset(state_index=i, is_testing=False)
            episode_states, episode_actions, episode_rewards, episode_next_states = [], [], [], []

            for step in range(TD3Config.MAX_STEPS_PER_EPISODE):
                probs = np.random.dirichlet(np.ones(3))
                action_type = np.argmax(probs)
                carb_amount = np.random.uniform(*TD3Config.CARB_RANGE)
                insulin_amount = np.random.uniform(*TD3Config.INSULIN_RANGE)
                time_index = np.random.randint(0, 12)

                action_vector = np.array([probs[0], probs[1], probs[2], carb_amount, insulin_amount, time_index], dtype=np.float32)
                action = (action_type, carb_amount if action_type == 1 else insulin_amount, time_index)

                next_state, _, _, reward, _, _ = env.step(step, action)

                episode_states.append(state)
                episode_actions.append(action_vector)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)

                state = next_state

            episode_end_reward = env.compute_episode_reward()

            total_steps = len(episode_rewards)
            adjusted_rewards = [r + (episode_end_reward / total_steps) for r in episode_rewards]

            for s, a, r_adj, s_next in zip(episode_states, episode_actions, adjusted_rewards, episode_next_states):
                buffer.add(s, a, r_adj, s_next, False)

    print("Replay buffer filled with episode-end rewards considered.")


def train_td3_bc(agent, buffer):
    print("Starting TD3-BC training...")
    for step in range(TD3Config.TRAINING_STEPS):
        agent.train(buffer, batch_size=TD3Config.BATCH_SIZE)


def test_td3_bc(env, agent, max_action, folder_path):
    print("Evaluating policy...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    log_path = os.path.join(folder_path, "eval_results.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)

    test_rewards, test_cgms, test_actions, test_time_window, test_behavioral_features = [], [], [], [], []
    test_tir, test_tar, test_tbr, test_cv = [], [], [], []

    y_history = env.simulator.data.y_history
    x_history = env.simulator.data.X_history
    patient_behavioral_features = extract_patient_behavior_features(x_history)

    for i in range(TD3Config.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)

        total_reward = 0
        rewards, predicted_cgms, actions, time_window = [], [], [], []

        print(f"\n--------------------- Test {i + 1} ---------------------", )
        for step in range(TD3Config.TESTING_STEPS):
            raw_action = agent.select_action(state)
            raw_action = np.clip(raw_action, [0, 0, 0, 0, 0, 0], max_action)

            probs = raw_action[:3]
            mapped_type = int(np.argmax(probs))
            carb_amt = raw_action[3]
            insulin_amt = raw_action[4]

            mapped_time = int(np.clip(np.round(raw_action[5]), 0, 11))
            mapped_value = carb_amt if mapped_type == 1 else insulin_amt if mapped_type == 2 else 0.0

            action = (mapped_type, mapped_value, mapped_time)
            state, predicted_cgm, time_series, reward, _, _ = env.step(step, action)

            total_reward += reward
            rewards.append(reward)
            actions.append(action)
            predicted_cgms.extend(predicted_cgm)
            time_window.extend(time_series)

        main_meal_actions = env.main_meal_action_log.copy()
        episode_end_reward = env.compute_episode_reward()
        total_reward += episode_end_reward

        agent_features = extract_behavior_features_from_actions(actions, main_meal_actions)
        test_behavioral_features.append(agent_features)

        test_rewards.append(total_reward)
        test_cgms.append(predicted_cgms)
        test_actions.append(actions)
        test_time_window.append(time_window)

        print(f"\n-------------- Results for Test {i + 1} ---------------\n")
        print(f"Episode-end reward      : {episode_end_reward:.2f}")
        print(f"Evaluation reward       : {total_reward - episode_end_reward:.2f}")
        print(f"Total Evaluation reward : {total_reward:.2f}")

        tir = cal_time_in_range(predicted_cgms)
        tar = cal_time_above_range(predicted_cgms)
        tbr = cal_time_below_range(predicted_cgms)
        cv  = cal_coefficient_of_variation(predicted_cgms)

        test_tir.append(tir)
        test_tar.append(tar)
        test_tbr.append(tbr)
        test_cv.append(cv)

        print(f"\nTime-in-Range           : {tir:.2f}%")
        print(f"Time-above-Range        : {tar:.2f}%")
        print(f"Time-below-Range        : {tbr:.2f}%")
        print(f"Coefficient of Variation: {cv:.2f}%")

        plot_cgm_reward_action(cgm_sequence=predicted_cgms,
                               hour_series=time_window,
                               reward_list=rewards,
                               action_list=actions,
                               test_index=i + 1,
                               main_meal_actions=main_meal_actions,
                               save_path_prefix=folder_path)

        with open(log_path, "a") as f:
            f.write(f"\n--------------------- Results for Test {i + 1} ----------------------\n")
            f.write(f"\nEpisode-end reward      : {episode_end_reward:.2f}\n")
            f.write(f"Evaluation reward       : {total_reward - episode_end_reward:.2f}\n")
            f.write(f"Total Evaluation reward : {total_reward:.2f}\n")
            f.write(f"\nTime-in-Range           : {tir:.2f}%\n")
            f.write(f"Time-above-Range        : {tar:.2f}%\n")
            f.write(f"Time-below-Range        : {tbr:.2f}%\n")
            f.write(f"Coefficient of Variation: {cv:.2f}%\n")

    evaluate_performance(test_actions, test_time_window, y_history, test_tir, test_tar, test_tbr, test_cv, log_path, folder_path)

    if test_behavioral_features:
        avg_features = {}
        keys = test_behavioral_features[0].keys()
        for key in keys:
            values = [f[key] for f in test_behavioral_features]
            avg_features[key] = round(np.mean(values), 2)

        plot_behavior_radar(patient_behavioral_features, avg_features, save_path=folder_path)


def evaluate_performance(test_actions, test_time_window, y_history, test_tir, test_tar, test_tbr, test_cv, log_path, folder_path):
    print(f"\n----------------- Final Results ------------------")

    # TIR
    avg_tir = np.mean(test_tir)
    std_tir = np.std(test_tir)
    med_tir = np.median(test_tir)

    # TAR
    avg_tar = np.mean(test_tar)
    std_tar = np.std(test_tar)
    med_tar = np.median(test_tar)

    # TBR
    avg_tbr = np.mean(test_tbr)
    std_tbr = np.std(test_tbr)
    med_tbr = np.median(test_tbr)

    # CV
    avg_cv = np.mean(test_cv)
    std_cv = np.std(test_cv)
    med_cv = np.median(test_cv)

    plot_tir_tbr_tar(test_tir, test_tar, test_tbr, save_path=f'{folder_path}/all_tests.png')
    plot_eat_action_distribution(test_actions, test_time_window, save_path=folder_path)
    plot_insulin_action_distribution(test_actions, test_time_window, save_path=folder_path)

    print(f"\nAverage Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%")
    print(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%")
    print(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%")
    print(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%")

    print(f"\nHistory Time-in-Range (TIR)          : {cal_time_in_range(y_history):.2f}%")
    print(f"History Time-above-Range (TAR)       : {cal_time_above_range(y_history):.2f}%")
    print(f"History Time-below-Range (TBR)       : {cal_time_below_range(y_history):.2f}%")
    print(f"History Coefficient of Variation (CV): {cal_coefficient_of_variation(y_history):.2f}%")

    with open(log_path, "a") as f:
        # Summary statistics block
        f.write("\n----------- Summary Time-in-Range Stats Across Tests -----------\n")
        f.write(f"Average Time-in-Range (TIR)          : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%\n")
        f.write(f"Average Time-above-Range (TAR)       : {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%\n")
        f.write(f"Average Time-below-Range (TBR)       : {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%\n")
        f.write(f"Average Coefficient of Variation (CV): {avg_cv:.2f}% (± {std_cv:.2f}%), Median: {med_cv:.2f}%\n")

        # History block
        f.write("\n---------------- Historical Time-in-Range Stats ----------------\n")
        f.write(f"History Time-in-Range (TIR)          : {cal_time_in_range(y_history):.2f}%\n")
        f.write(f"History Time-above-Range (TAR)       : {cal_time_above_range(y_history):.2f}%\n")
        f.write(f"History Time-below-Range (TBR)       : {cal_time_below_range(y_history):.2f}%\n")
        f.write(f"History Coefficient of Variation (CV): {cal_coefficient_of_variation(y_history):.2f}%")


def main(dataset_name, patient_id):
    env = EnvironmentAdvanced(dataset_name=dataset_name, patient_id=patient_id)
    device = torch.device("cpu")
    folder_path = f'./td3_bc_model/tests/final/azt1d_extra/{dataset_name}_patient_{patient_id}'

    max_action = np.array([1.0, 1.0, 1.0, TD3Config.CARB_RANGE[1], TD3Config.INSULIN_RANGE[1], 11.0])

    agent = TD3_BC(state_dim=TD3Config.STATE_SIZE, action_dim=len(max_action), max_action=max_action, device=device)
    buffer = ReplayBuffer()

    fill_replay_buffer(env, buffer)
    train_td3_bc(agent, buffer)
    test_td3_bc(env, agent, max_action, folder_path)


if __name__ == "__main__":
    # patient_id = sys.argv[1]
    set_seed(42)
    main(dataset_name=DataConfig.DATASET, patient_id=str(DataConfig.PATIENT_ID))
