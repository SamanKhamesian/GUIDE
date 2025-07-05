import numpy as np
import torch
import os
from config import Action, Threshold, KnowledgeBase, RLConfig, RewardFunction
from replay_buffer import ReplayBuffer
from advanced.simulator import Simulator
from advanced.td3_bc_agent import TD3_BC
from utils import plot_rewards, plot_cgm_levels, plot_cgm_reward_action_with_legend, set_seed, numerical_sort_key


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

    def reset(self):
        self.simulator.reset()

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
        bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array = self.simulator.apply_action_to_inputs(self.full_current_window, action)

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
        w_normal = RewardFunction.WEIGHTS[0]
        w_hypo = RewardFunction.WEIGHTS[1]
        w_hyper = RewardFunction.WEIGHTS[2]

        # Piecewise reward components
        reward = np.zeros_like(predicted_cgm)
        normal_vals = predicted_cgm[normal]

        # TEST 2
        reward[hypo] = -w_hypo * (Threshold.HYPOGLYCEMIA - predicted_cgm[hypo])
        reward[hyper] = -w_hyper * (predicted_cgm[hyper] - Threshold.HYPERGLYCEMIA)
        reward[normal] = np.where(normal_vals <= RewardFunction.IDEAL_CGM,
                                  (normal_vals - Threshold.HYPOGLYCEMIA) / (RewardFunction.IDEAL_CGM - Threshold.HYPOGLYCEMIA) * w_normal,
                                  (Threshold.HYPERGLYCEMIA - normal_vals) / (Threshold.HYPERGLYCEMIA - RewardFunction.IDEAL_CGM) * w_normal)

        total_reward = np.sum(reward)

        mean_cgm = predicted_cgm.mean()
        time_since_last_meal = time_since_last_meal_array[time_index] + 1
        time_since_last_insulin = time_since_last_injection_array[time_index] + 1

        if self.sleep_mode:
            if action_type != Action.NOTHING:
                total_reward -= RewardFunction.DO_NOTHING_IN_SLEEP
            else:
                total_reward += RewardFunction.DO_NOTHING_IN_SLEEP

        if action_type == Action.EAT:
            self.eat_count += 1

        # TEST 3 - Reward for skipping unnecessary actions
        if action_type == Action.NOTHING and np.all(predicted_cgm > 120) and np.all(predicted_cgm < 140):
            total_reward += RewardFunction.DO_NOTHING_BONUS  # encourages stability

        # TEST 3 - Penalize unnecessary insulin
        if action_type == Action.INJECT:
            if mean_cgm < 150:
                total_reward -= (150 - mean_cgm) * 10

            elif mean_cgm > 200:
                total_reward += (mean_cgm - 200) * 5

            if time_since_last_insulin < 2 * 12:
                total_reward -= RewardFunction.EARLY_INJECTION_PENALTY

        # Test 6
        if action_type == Action.EAT:
            if 6 <= self.current_hour <= 9 or 13 <= self.current_hour <= 15 or 19 <= self.current_hour <= 22:
                total_reward += RewardFunction.GOOD_MEAL_TIMING_BONUS

            if mean_cgm < 110:
                total_reward += (110 - mean_cgm) * 10

            if time_since_last_meal < 12 or time_since_last_meal > 6 * 12:
                total_reward -= RewardFunction.EARLY_MEAL_PENALTY

            elif 12 <= time_since_last_meal <= 6 * 12:
                total_reward += RewardFunction.GOOD_MEAL_TIMING_BONUS

            if time_since_last_insulin < 24:
                # good timing
                total_reward += RewardFunction.GOOD_MEAL_TIMING_BONUS
            else:
                # insulin was too long ago, then maybe it missed the chance (still some reward, but less)
                total_reward += RewardFunction.GOOD_MEAL_TIMING_BONUS * 0.2

        if self.prev_action[0] == action_type and action_type != Action.NOTHING:
            self.repeat_counter += 1
        else:
            self.repeat_counter = 0

        if self.repeat_counter >= 2:
            total_reward -= RewardFunction.REPEATED_ACTION_PENALTY

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
            episode_reward += RewardFunction.HYPO_HYPER_1_PENALTY
        elif 1 <= hypo_events <= 3:
            episode_reward -= RewardFunction.HYPO_HYPER_1_PENALTY
        elif 4 <= hypo_events <= 5:
            episode_reward -= RewardFunction.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= RewardFunction.HYPO_HYPER_3_PENALTY

        # Penalty for hyper events
        if hyper_events == 0:
            episode_reward += RewardFunction.HYPO_HYPER_1_PENALTY
        elif 1 <= hyper_events <= 3:
            episode_reward -= RewardFunction.HYPO_HYPER_1_PENALTY
        elif 4 <= hyper_events <= 5:
            episode_reward -= RewardFunction.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= RewardFunction.HYPO_HYPER_3_PENALTY

        # Duration penalty (additional fine-tuning, optional but recommended)
        episode_reward -= (hypo_duration + hyper_duration) * 0.5  # penalty per minute outside range

        if 3 <= self.eat_count <= 6:
            episode_reward += RewardFunction.EAT_COUNT_REWARD
        else:
            episode_reward -= RewardFunction.EAT_COUNT_REWARD

        return episode_reward

def main(dataset_name, patient_id):
    env = EnvironmentAdvanced(dataset_name=dataset_name, patient_id=patient_id)
    device = torch.device("cpu")

    # Define max_action in real-world space
    max_action = np.array([1.0, 1.0, 1.0,  # softmax logits (prob_type_*)
        KnowledgeBase.CARB_RANGE[1],  # max carb in grams
        KnowledgeBase.INSULIN_RANGE[1],  # max insulin in units
        11.0  # time index (12 slots)
    ])

    agent = TD3_BC(state_dim=72 * 8, action_dim=6, max_action=max_action, device=device)
    buffer = ReplayBuffer()

    print("Filling initial replay buffer...")
    for episode in range(RLConfig.MAX_EPISODES):
        state = env.reset()
        episode_states, episode_actions, episode_rewards, episode_next_states = [], [], [], []

        for step in range(RLConfig.MAX_STEPS_PER_EPISODE):
            probs = np.random.dirichlet(np.ones(3))
            action_type = np.argmax(probs)
            carb_amount = np.random.uniform(*KnowledgeBase.CARB_RANGE)
            insulin_amount = np.random.uniform(*KnowledgeBase.INSULIN_RANGE)
            time_index = np.random.randint(0, 12)

            action_vector = np.array([probs[0], probs[1], probs[2], carb_amount, insulin_amount, time_index], dtype=np.float32)
            action = (action_type, carb_amount if action_type == 1 else insulin_amount, time_index)

            next_state, _, _, reward, _, _ = env.step(action)

            # Temporarily store states, actions, rewards per step
            episode_states.append(state)
            episode_actions.append(action_vector)
            episode_rewards.append(reward)
            episode_next_states.append(next_state)

            state = next_state

        # At episode-end, compute the final daily metrics reward
        episode_end_reward = env.compute_episode_reward()

        # Distribute the episode-end reward equally across all steps, or just add clearly at the end:
        # Option 1 (equal distribution, recommended):
        total_episode_steps = len(episode_rewards)
        adjusted_episode_rewards = [r + (episode_end_reward / total_episode_steps) for r in episode_rewards]

        # Store all adjusted transitions in the replay buffer
        for s, a, r_adj, s_next in zip(episode_states, episode_actions, adjusted_episode_rewards, episode_next_states):
            buffer.add(s, a, r_adj, s_next, False)

    print("Replay buffer filled with episode-end rewards considered.")

    print("Starting TD3-BC training...")
    for step in range(RLConfig.TRAINING_STEPS):
        agent.train(buffer, batch_size=256)

    print("Evaluating policy...")
    state = env.reset()
    total_reward = 0

    rewards = []
    predicted_cgms = []
    actions = []
    time_window = []

    for _ in range(RLConfig.TESTING_STEPS):
        raw_action = agent.select_action(state)
        raw_action = np.clip(raw_action, [0, 0, 0, 0, 0, 0], max_action)

        # Map softmax logits to discrete action type
        probs = raw_action[:3]
        mapped_type = int(np.argmax(probs))

        carb_amt = raw_action[3]
        insulin_amt = raw_action[4]
        mapped_time = int(np.clip(np.round(raw_action[5]), 0, 11))

        mapped_value = carb_amt if mapped_type == 1 else insulin_amt if mapped_type == 2 else 0.0
        action = (mapped_type, mapped_value, mapped_time)
        actions.append(action)
        print(f"Generated action: Type={mapped_type}, Value={mapped_value:.2f}, Time Index={mapped_time}")

        state, predicted_cgm, time_series, reward, done, _ = env.step(action)
        total_reward += reward

        rewards.append(reward)
        predicted_cgms.extend(predicted_cgm)
        time_window.extend(time_series)

    # Compute and add the episode-end reward
    episode_end_reward = env.compute_episode_reward()
    total_reward += episode_end_reward

    print(f"Episode-end reward: {episode_end_reward:.2f}")
    print(f"Total Evaluation reward (including episode-end): {total_reward:.2f}")

    print(f"Evaluation reward: {total_reward:.2f}")
    plot_rewards(rewards, title="Evaluation Reward per Step", save_path="./advanced/tests/reward_levels.png")
    plot_cgm_levels(predicted_cgms, time_window, title="Predicted CGM with Ranges", save_path="./advanced/tests/cgm_levels.png")
    plot_cgm_reward_action_with_legend(cgm_sequence=predicted_cgms,
                                       hour_series=time_window,
                                       reward_list=rewards,
                                       action_list=actions,
                                       days=3,
                                       save_path_prefix="./advanced/tests/advanced_test")


if __name__ == "__main__":
    set_seed(42)
    main(dataset_name=RLConfig.DATASET, patient_id=RLConfig.PATIENT_ID)
