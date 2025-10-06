import os

import numpy as np
import gc, torch

from buffer import PPOBuffer
from config import Action, Threshold, PPORewardShaping, PPOConfig, DataConfig
from ppo_model.ppo_agent import PPOAgent
from simulator import Simulator
from utils import (plot_cgm_reward_action, set_seed, cal_time_in_range, cal_time_above_range, cal_time_below_range, plot_tir_tbr_tar,
                   plot_eat_action_distribution, plot_insulin_action_distribution)


def generate_discrete_values(start, stop, step):
    count = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(count)]


def magni_scaled_array(cgm):
    risk = 10 * (1.509 * (np.log(cgm) ** 1.084 - 5.381)) ** 2
    risk_clipped = np.clip(risk, 0, 15.5)
    return 100 * (1 - risk_clipped / 7.75)


class EnvironmentAdvanced:
    def __init__(self, dataset_name, patient_id):
        self.simulator = Simulator(dataset_name=dataset_name, patient_id=patient_id)
        self.simulator.train()

        # 8 features, and 6-hour data for each
        # [hour, time_since_last_meal, time_since_last_insulin, sleep, basal, carbs, bolus, cgm]
        self.state_size = 72 * 8
        self.action_space = [Action.NOTHING, Action.EAT, Action.INJECT]

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)  # (type, value, time_index)
        self.repeat_counter = 0
        self.meal_counter = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Track CGM history per episode
        self.episode_cgm_history = []

    def get_window(self):
        return self.simulator.get_full_current_window()

    def get_state(self):
        window = self.get_window()
        return window[:, :8].flatten()

    def get_time_series(self):
        return self.simulator.get_time_window()

    def get_current_hour(self):
        return int(self.get_time_series()[5])

    def get_sleep_signal(self):
        return self.simulator.get_sleep_mode()

    def reset(self, state_index, is_testing=False):
        self.simulator.reset(state_index, is_testing)

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)
        self.repeat_counter = 0
        self.meal_counter = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Reset CGM history at the beginning of each episode
        self.episode_cgm_history = []

        return self.current_state

    def step(self, action):
        # Step 1: Predict next CGM values
        predicted_cgm = self.simulator.predict_next_cgm()

        # Step 2: Track predicted CGM values per step
        self.episode_cgm_history.extend(predicted_cgm)

        # Step 3: Apply action to simulator
        bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array = (
            self.simulator.apply_action_to_inputs(self.full_current_window, action))

        # Step 4: Commit inputs and predicted CGM
        self.simulator.commit_next_input(predicted_cgm, bolus_array, time_since_last_injection_array[1:], carb_array, time_since_last_meal_array[1:])

        # Step 5: Update state
        self.current_state = self.get_state()
        self.time_series = self.get_time_series()
        self.current_hour = self.get_current_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Step 6: Compute reward
        reward = self.compute_reward(time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action)

        self.prev_action = action

        return self.current_state, predicted_cgm, self.time_series, reward, False, {}

    # --> TODO: Add reward logging (hypo/hyper breakdown) per step for debugging and performance visualization
    def compute_reward(self, time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action):
        action_type, action_value, time_index = action

        mean_cgm = predicted_cgm.mean()

        hypo = predicted_cgm < Threshold.HYPOGLYCEMIA
        hyper = predicted_cgm > Threshold.HYPERGLYCEMIA
        normal = ~hypo & ~hyper

        # Extract weights
        w_normal = PPORewardShaping.WEIGHTS[0]
        w_hypo = PPORewardShaping.WEIGHTS[1]
        w_hyper = PPORewardShaping.WEIGHTS[2]

        # Piecewise reward components
        reward = np.zeros_like(predicted_cgm)
        normal_vals = predicted_cgm[normal]

        reward[hypo] = -w_hypo * (Threshold.HYPOGLYCEMIA - predicted_cgm[hypo])
        reward[hyper] = -w_hyper * (predicted_cgm[hyper] - Threshold.HYPERGLYCEMIA)
        reward[normal] = np.where(normal_vals <= PPORewardShaping.IDEAL_CGM,
                                  (normal_vals - Threshold.HYPOGLYCEMIA) / (PPORewardShaping.IDEAL_CGM - Threshold.HYPOGLYCEMIA) * w_normal,
                                  (Threshold.HYPERGLYCEMIA - normal_vals) / (Threshold.HYPERGLYCEMIA - PPORewardShaping.IDEAL_CGM) * w_normal)

        total_reward = np.sum(reward)

        time_since_last_meal = time_since_last_meal_array[time_index] + 1
        time_since_last_insulin = time_since_last_injection_array[time_index] + 1

        if self.sleep_mode:
            if action_type != Action.NOTHING:
                total_reward -= PPORewardShaping.DO_NOTHING_IN_SLEEP
            else:
                total_reward += PPORewardShaping.DO_NOTHING_IN_SLEEP

        if action_type == Action.NOTHING and np.all(predicted_cgm > 120) and np.all(predicted_cgm < 140):
            total_reward += PPORewardShaping.DO_NOTHING_BONUS  # encourages stability

        if action_type == Action.INJECT:
            if mean_cgm < 150:
                total_reward -= (150 - mean_cgm) * 10

            elif mean_cgm > 190:
                total_reward += (mean_cgm - 190) * 10

            if time_since_last_insulin < 2 * 12:
                total_reward -= PPORewardShaping.EARLY_INJECTION_PENALTY

            if time_since_last_meal <= 12:
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS

        if action_type == Action.EAT:
            if not self.sleep_mode:
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS
                self.meal_counter += 1

            if mean_cgm < 110:
                total_reward += (110 - mean_cgm) * 10

            if time_since_last_meal < 12 or time_since_last_meal > 6 * 12:
                total_reward -= PPORewardShaping.EARLY_MEAL_PENALTY

            elif 12 <= time_since_last_meal <= 6 * 12:
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS

            if time_since_last_insulin < 2 * 12:
                # good timing
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS
            else:
                # insulin was too long ago, then maybe it missed the chance (still some reward, but less)
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS * 0.2

        if self.repeat_counter >= 1:
            total_reward -= PPORewardShaping.REPEATED_ACTION_PENALTY

        if self.prev_action[0] == action_type and action_type != Action.NOTHING:
            self.repeat_counter += 1
        else:
            self.repeat_counter = 0

        return total_reward

    # New method to calculate episode-end reward
    def compute_episode_reward(self):
        cgm_array = np.array(self.episode_cgm_history)

        # Compute daily metrics clearly:
        tir_ratio = cal_time_in_range(cgm_array)

        hypo_events = np.sum(cgm_array < Threshold.HYPOGLYCEMIA)
        hyper_events = np.sum(cgm_array > Threshold.HYPERGLYCEMIA)

        # Calculate event duration (in 5-minute intervals)
        hypo_duration = np.sum(cgm_array < Threshold.HYPOGLYCEMIA) * 5
        hyper_duration = np.sum(cgm_array > Threshold.HYPERGLYCEMIA) * 5

        # Apply clear penalty thresholds
        episode_reward = 0

        # Reward/Penalty for TIR (daily)
        episode_reward += (10 * (tir_ratio - 70))

        # Penalty for hypo events
        if hypo_events == 0:
            episode_reward += PPORewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hypo_events <= 3:
            episode_reward -= PPORewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hypo_events <= 5:
            episode_reward -= PPORewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= PPORewardShaping.HYPO_HYPER_3_PENALTY

        # Penalty for hyper events
        if hyper_events == 0:
            episode_reward += PPORewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hyper_events <= 3:
            episode_reward -= PPORewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hyper_events <= 5:
            episode_reward -= PPORewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= PPORewardShaping.HYPO_HYPER_3_PENALTY

        # Duration penalty (additional fine-tuning, optional but recommended)
        episode_reward -= (hypo_duration + hyper_duration) * 0.5  # penalty per minute outside range

        if 3 <= self.meal_counter <= 6:
            episode_reward += PPORewardShaping.EAT_COUNT_REWARD
        else:
            episode_reward -= PPORewardShaping.EAT_COUNT_REWARD

        return episode_reward


def train_ppo(env, agent, buffer, carb_values, insulin_values):
    print("Starting PPO training...")

    for epoch in range(PPOConfig.TRAINING_EPOCHS):
        for i in range(PPOConfig.NUM_TRAIN_INIT_STATE):
            state = env.reset(state_index=i, is_testing=False)
            print("\n-------------------- Epoch {}, Step {} --------------------".format(epoch + 1, i + 1))
            start_idx = len(buffer.rewards)

            for _ in range(PPOConfig.MAX_STEPS_PER_EPISODE):

                # Disable gradient tracking during rollout
                with torch.no_grad():
                    action_type, carb_idx, insulin_idx, time_index, log_prob = agent.select_action(state)
                    value_estimate = agent.evaluate(state)

                # Map to actual values
                carb_amount = carb_values[carb_idx]
                insulin_amount = insulin_values[insulin_idx]
                value = carb_amount if action_type == Action.EAT else (
                    insulin_amount if action_type == Action.INJECT else 0.0
                )

                action = (action_type, value, time_index)

                # Step the environment
                next_state, _, _, reward, done, _ = env.step(action)

                # Convert everything to plain CPU/numpy values before storing
                buf_state = np.asarray(state, dtype=np.float32)
                buf_action = [int(action_type), int(carb_idx), int(insulin_idx), int(time_index)]
                buf_reward = float(reward)
                buf_value = float(value_estimate)
                buf_logp = float(log_prob)

                buffer.store(buf_state, buf_action, buf_reward, buf_value, buf_logp, bool(done))

                state = next_state

            # Add episode-end reward to only this episode’s steps
            ep_end_reward = env.compute_episode_reward()
            end_idx = len(buffer.rewards)
            steps_this_ep = end_idx - start_idx
            if steps_this_ep > 0:
                add_per_step = ep_end_reward / steps_this_ep
                for k in range(start_idx, end_idx):
                    buffer.rewards[k] += add_per_step

        # Train after all init states, then reset buffer and free memory
        agent.train(buffer)
        buffer.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_ppo(env, agent, carb_values, insulin_values, folder_path):
    print("Evaluating PPO policy...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    test_rewards, test_cgms, test_actions, test_time_window = [], [], [], []
    test_tir, test_tar, test_tbr = [], [], []
    true_tir, true_tar, true_tbr = [], [], []

    y_history = env.simulator.data.y_history

    for i in range(PPOConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=i, is_testing=True)

        total_reward = 0
        rewards, predicted_cgms, actions, time_window = [], [], [], []
        y_true = env.simulator.data.y_rl_test[i]

        print(f"\n--------------------- Test {i + 1} ---------------------")
        for _ in range(PPOConfig.TESTING_STEPS):
            action_type, carb_idx, insulin_idx, time_index, _ = agent.select_action(state)

            # Map to actual values
            carb_amount = carb_values[carb_idx]
            insulin_amount = insulin_values[insulin_idx]

            mapped_type = action_type
            mapped_value = carb_amount if mapped_type == Action.EAT else (insulin_amount if mapped_type == Action.INJECT else 0.0)
            mapped_time = time_index

            action = (mapped_type, mapped_value, mapped_time)
            state, predicted_cgm, ts, reward, done, _ = env.step(action)

            total_reward += reward
            rewards.append(reward)
            predicted_cgms.extend(predicted_cgm)
            actions.append(action)
            time_window.extend(ts)

            print(f"Generated action: Type={mapped_type}, Value={mapped_value:.2f}, Time Index={mapped_time}")
            print(f"Calculated reward: {reward:.2f}")

        episode_end_reward = env.compute_episode_reward()
        total_reward += episode_end_reward

        test_rewards.append(total_reward)
        test_cgms.append(predicted_cgms)
        test_actions.append(actions)
        test_time_window.append(time_window)

        print(f"\n-------------- Results for Test {i + 1} ---------------\n")
        print(f"Episode-end reward: {episode_end_reward:.2f}")
        print(f"Total Evaluation reward: {total_reward - episode_end_reward:.2f}")
        print(f"Total Evaluation reward (including episode-end): {total_reward:.2f}")

        tir = cal_time_in_range(predicted_cgms)
        tar = cal_time_above_range(predicted_cgms)
        tbr = cal_time_below_range(predicted_cgms)

        test_tir.append(tir)
        test_tar.append(tar)
        test_tbr.append(tbr)

        print(f"\nTime-in-Range : {tir:.2f}%")
        print(f"Time-above-Range: {tar:.2f}%")
        print(f"Time-below-Range: {tbr:.2f}%")

        _true_tir = cal_time_in_range(y_true)
        _true_tar = cal_time_above_range(y_true)
        _true_tbr = cal_time_below_range(y_true)

        true_tir.append(_true_tir)
        true_tar.append(_true_tar)
        true_tbr.append(_true_tbr)

        plot_cgm_reward_action(
            cgm_sequence=predicted_cgms,
            true_cgm_sequence=None,
            hour_series=time_window,
            reward_list=rewards,
            action_list=actions,
            days=1,
            save_path_prefix=f"{folder_path}/test_{i + 1}_results"
        )

    evaluate_performance_ppo(test_rewards,
                             test_cgms,
                             test_actions,
                             test_time_window,
                             test_tir,
                             test_tar,
                             test_tbr,
                             true_tir,
                             true_tar,
                             true_tbr,
                             folder_path,
                             y_history)


def evaluate_performance_ppo(test_rewards,
                             test_cgms,
                             test_actions,
                             test_time_window,
                             test_tir,
                             test_tar,
                             test_tbr,
                             true_tir,
                             true_tar,
                             true_tbr,
                             folder_path,
                             y_history):
    print(f"\n----------------- Final Results ------------------\n")

    avg_tir = np.mean(test_tir)
    std_tir = np.std(test_tir)
    med_tir = np.median(test_tir)

    avg_true_tir = np.mean(true_tir)
    std_true_tir = np.std(true_tir)
    med_true_tir = np.median(true_tir)

    avg_tar = np.mean(test_tar)
    std_tar = np.std(test_tar)
    med_tar = np.median(test_tar)

    avg_true_tar = np.mean(true_tar)
    std_true_tar = np.std(true_tar)
    med_true_tar = np.median(true_tar)

    avg_tbr = np.mean(test_tbr)
    std_tbr = np.std(test_tbr)
    med_tbr = np.median(test_tbr)

    avg_true_tbr = np.mean(true_tbr)
    std_true_tbr = np.std(true_tbr)
    med_true_tbr = np.median(true_tbr)

    plot_tir_tbr_tar(test_tir, test_tar, test_tbr, save_path=f'{folder_path}/all_tir.png')
    plot_eat_action_distribution(test_actions, test_time_window, save_path=folder_path)
    plot_insulin_action_distribution(test_actions, test_time_window, save_path=folder_path)

    print(f"Time-in-Range (TIR)   : {avg_tir:.2f}% (± {std_tir:.2f}%), Median: {med_tir:.2f}%")
    print(f"Time-above-Range (TAR): {avg_tar:.2f}% (± {std_tar:.2f}%), Median: {med_tar:.2f}%")
    print(f"Time-below-Range (TBR): {avg_tbr:.2f}% (± {std_tbr:.2f}%), Median: {med_tbr:.2f}%")

    print(f"\nTrue Time-in-Rage (TTIR): {avg_true_tir:.2f}% (± {std_true_tir:.2f}%), Median: {med_true_tir:.2f}%)")
    print(f"True Time-above_Range (TTAR): {avg_true_tar:.2f}% (± {std_true_tar:.2f}%), Median: {med_true_tar:.2f}%)")
    print(f"True Time-below_Range (TTBR): {avg_true_tbr:.2f}% (± {std_true_tbr:.2f}%), Median: {med_true_tbr:.2f}%)")

    print(f"\nHistory Time-in-Range: {cal_time_in_range(y_history):.2f}%")
    print(f"History Time-above-Range: {cal_time_above_range(y_history):.2f}%")
    print(f"History Time-below-Range: {cal_time_below_range(y_history):.2f}%")


def main(dataset_name, patient_id):
    env = EnvironmentAdvanced(dataset_name=dataset_name, patient_id=patient_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = "./ppo_model/tests/test_1"

    carb_values = list(range(PPOConfig.CARB_RANGE[0], PPOConfig.CARB_RANGE[1] + 1, PPOConfig.CARB_STEP))
    insulin_values = generate_discrete_values(PPOConfig.INSULIN_RANGE[0], PPOConfig.INSULIN_RANGE[1], PPOConfig.INSULIN_STEP)
    time_indices = list(range(PPOConfig.TIME_INDEX_RANGE[0], PPOConfig.TIME_INDEX_RANGE[1] + 1, PPOConfig.TIME_STEP))

    agent = PPOAgent(state_dim=PPOConfig.STATE_SIZE, device=device, n_carb=len(carb_values), n_insulin=len(insulin_values), n_time=len(time_indices))
    buffer = PPOBuffer(state_dim=PPOConfig.STATE_SIZE, device=device)

    train_ppo(env, agent, buffer, carb_values, insulin_values)
    test_ppo(env, agent, carb_values, insulin_values, folder_path)


if __name__ == "__main__":
    set_seed(42)
    main(dataset_name=DataConfig.DATASET, patient_id=DataConfig.PATIENT_ID)
