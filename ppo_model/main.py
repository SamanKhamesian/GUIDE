import numpy as np
import torch

from buffer import PPOBuffer
from config import Action, Threshold, PPORewardShaping, PPOConfig, DataConfig
from ppo_agent import PPOAgent
from simulator import Simulator
from utils import plot_rewards, plot_cgm_levels, plot_cgm_reward_action_with_legend, set_seed


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

        # [hour, time_since_last_meal, time_since_last_insulin, sleep, basal, carbs, bolus, cgm]
        self.state_size = 72 * 8
        self.action_space = [Action.NOTHING, Action.EAT, Action.INJECT]

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)  # (type, value, time_index)
        self.repeat_counter = 0
        self.eat_count = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Track CGM history per episode
        self.episode_cgm_history = []

    def get_time_series(self):
        return self.simulator.get_time_window()

    def get_hour(self):
        return int(self.get_time_series()[5])

    def get_sleep_signal(self):
        return self.simulator.get_sleep_mode()

    def get_window(self):
        return self.simulator.get_full_current_window()

    def get_state(self):
        window = self.get_window()
        return window[:, :8].flatten()

    def reset(self, state_index, is_testing=False):
        self.simulator.reset(state_index, is_testing)

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)
        self.repeat_counter = 0
        self.eat_count = 0
        self.time_series = self.get_time_series()
        self.current_hour = self.get_hour()
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
        bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array = self.simulator.apply_action_to_inputs(self.full_current_window,
                                                                                                                                     action)

        self.time_series = self.get_time_series()
        self.current_hour = self.get_hour()
        self.sleep_mode = self.get_sleep_signal()

        # Step 4: Compute reward
        reward = self.compute_reward(time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action)

        # Step 5: Commit inputs and predicted CGM
        self.simulator.commit_next_input(predicted_cgm, bolus_array, time_since_last_injection_array[1:], carb_array, time_since_last_meal_array[1:])

        # Step 6: Update state
        self.current_state = self.get_state()
        self.prev_action = action

        return self.current_state, predicted_cgm, self.time_series, reward, False, {}

    # --> TODO: Add reward logging (hypo/hyper breakdown) per step for debugging and performance visualization
    def compute_reward(self, time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action):
        action_type, action_value, time_index = action

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

        # TEST 2
        reward[hypo] = -w_hypo * (Threshold.HYPOGLYCEMIA - predicted_cgm[hypo])
        reward[hyper] = -w_hyper * (predicted_cgm[hyper] - Threshold.HYPERGLYCEMIA)
        reward[normal] = np.where(normal_vals <= PPORewardShaping.IDEAL_CGM,
                                  (normal_vals - Threshold.HYPOGLYCEMIA) / (PPORewardShaping.IDEAL_CGM - Threshold.HYPOGLYCEMIA) * w_normal,
                                  (Threshold.HYPERGLYCEMIA - normal_vals) / (Threshold.HYPERGLYCEMIA - PPORewardShaping.IDEAL_CGM) * w_normal)

        total_reward = np.sum(reward)

        mean_cgm = predicted_cgm.mean()
        time_since_last_meal = time_since_last_meal_array[time_index] + 1
        time_since_last_insulin = time_since_last_injection_array[time_index] + 1

        if self.sleep_mode:
            if action_type != Action.NOTHING:
                total_reward -= PPORewardShaping.DO_NOTHING_IN_SLEEP
            else:
                total_reward += PPORewardShaping.DO_NOTHING_IN_SLEEP

        if action_type == Action.EAT:
            self.eat_count += 1

        # TEST 3 - Reward for skipping unnecessary actions
        if action_type == Action.NOTHING and np.all(predicted_cgm > 120) and np.all(predicted_cgm < 140):
            total_reward += PPORewardShaping.DO_NOTHING_BONUS  # encourages stability

        # TEST 3 - Penalize unnecessary insulin
        if action_type == Action.INJECT:
            if mean_cgm < 150:
                total_reward -= (150 - mean_cgm) * 10

            elif mean_cgm > 200:
                total_reward += (mean_cgm - 200) * 5

            else:
                total_reward -= (200 - mean_cgm)

            if time_since_last_insulin < 2 * 12:
                total_reward -= PPORewardShaping.EARLY_INJECTION_PENALTY

        # Test 6
        if action_type == Action.EAT:
            if 6 <= self.current_hour <= 9 or 13 <= self.current_hour <= 15 or 19 <= self.current_hour <= 22:
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS

            if mean_cgm < 110:
                total_reward += (110 - mean_cgm) * 10

            if time_since_last_meal < 12 or time_since_last_meal > 6 * 12:
                total_reward -= PPORewardShaping.EARLY_MEAL_PENALTY

            elif 12 <= time_since_last_meal <= 6 * 12:
                total_reward += PPORewardShaping.GOOD_MEAL_TIMING_BONUS

            if time_since_last_insulin < 24:
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
        tir_ratio = np.mean((cgm_array >= Threshold.HYPOGLYCEMIA) & (cgm_array <= Threshold.HYPERGLYCEMIA))
        hypo_events = np.sum(cgm_array < Threshold.HYPOGLYCEMIA)
        hyper_events = np.sum(cgm_array > Threshold.HYPERGLYCEMIA)

        # Calculate event duration (in 5-minute intervals)
        hypo_duration = np.sum(cgm_array < Threshold.HYPOGLYCEMIA) * 5
        hyper_duration = np.sum(cgm_array > Threshold.HYPERGLYCEMIA) * 5

        # Apply clear penalty thresholds
        episode_reward = 0

        # Reward/Penalty for TIR (daily)
        episode_reward += 100 if tir_ratio >= 0.7 else -100 * (0.7 - tir_ratio)

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

        if 3 <= self.eat_count <= 6:
            episode_reward += PPORewardShaping.EAT_COUNT_REWARD
        else:
            episode_reward -= PPORewardShaping.EAT_COUNT_REWARD

        return episode_reward


def main(dataset_name, patient_id):
    env = EnvironmentAdvanced(dataset_name=dataset_name, patient_id=patient_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = PPOConfig.STATE_SIZE
    # Prepare discrete action values
    carb_values = list(range(PPOConfig.CARB_RANGE[0], PPOConfig.CARB_RANGE[1] + 1, PPOConfig.CARB_STEP))
    insulin_values = generate_discrete_values(PPOConfig.INSULIN_RANGE[0], PPOConfig.INSULIN_RANGE[1], PPOConfig.INSULIN_STEP)
    time_indices = list(range(PPOConfig.TIME_INDEX_RANGE[0], PPOConfig.TIME_INDEX_RANGE[1] + 1, PPOConfig.TIME_STEP))

    n_carb = len(carb_values)
    n_insulin = len(insulin_values)
    n_time = len(time_indices)

    agent = PPOAgent(state_dim=state_dim, device=device, n_carb=n_carb, n_insulin=n_insulin, n_time=n_time)
    buffer = PPOBuffer(state_dim=state_dim, device=device)

    for i in range(PPOConfig.NUM_TRAIN_INIT_STATE):
        for episode in range(PPOConfig.MAX_EPISODES):
            state = env.reset(i, False)

            for step in range(PPOConfig.MAX_STEPS_PER_EPISODE):
                action_type, carb_idx, insulin_idx, time_index, log_prob = agent.select_action(state)

                carb_amount = carb_values[carb_idx]
                insulin_amount = insulin_values[insulin_idx]

                value = (carb_amount if action_type == Action.EAT else insulin_amount if action_type == Action.INJECT else 0.0)

                action = (action_type, value, time_index)
                next_state, predicted_cgm, ts, reward, done, _ = env.step(action)

                value_estimate = agent.evaluate(state)
                buffer.store(state, [action_type, carb_idx, insulin_idx, time_index], reward, value_estimate, log_prob, done)

                state = next_state

            ep_end_reward = env.compute_episode_reward()

            for i in range(PPOConfig.MAX_STEPS_PER_EPISODE):
                buffer.rewards[i] += ep_end_reward / PPOConfig.MAX_STEPS_PER_EPISODE

            agent.train(buffer)
            buffer.reset()

    print("Evaluating PPO policy...")
    state = env.reset(0, True)
    total_reward = 0
    rewards = []
    predicted_cgms = []
    actions = []
    time_window = []

    for _ in range(PPOConfig.TESTING_STEPS):
        action_type, carb_idx, insulin_idx, time_index, _ = agent.select_action(state)

        carb_amount = carb_values[carb_idx]
        insulin_amount = insulin_values[insulin_idx]

        value = carb_amount if action_type == Action.EAT else insulin_amount if action_type == Action.INJECT else 0.0
        action = (action_type, value, time_index)

        state, predicted_cgm, ts, reward, done, _ = env.step(action)
        total_reward += reward

        rewards.append(reward)
        predicted_cgms.extend(predicted_cgm)
        actions.append(action)
        time_window.extend(ts)

    # add this back here
    episode_end_reward = env.compute_episode_reward()
    total_reward += episode_end_reward

    print(f"Episode-end reward: {episode_end_reward:.2f}")
    print(f"Total Evaluation reward (including episode-end): {total_reward:.2f}")

    print(f"Final evaluation reward: {total_reward:.2f}")
    plot_rewards(rewards, title="PPO Reward per Step", save_path="./ppo_model/tests/reward_plot.png")
    plot_cgm_levels(predicted_cgms, time_window, title="Predicted CGM Levels", save_path="./ppo_model/tests/cgm_plot.png")
    plot_cgm_reward_action_with_legend(cgm_sequence=predicted_cgms,
                                       hour_series=time_window,
                                       reward_list=rewards,
                                       action_list=actions,
                                       days=1,
                                       save_path_prefix="./ppo_model/tests/ultimate_test")


if __name__ == "__main__":
    set_seed(42)
    main(dataset_name=DataConfig.DATASET, patient_id=DataConfig.PATIENT_ID)
