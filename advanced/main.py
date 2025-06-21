import numpy as np
import torch

from config import Action, Threshold, KnowledgeBase, RLConfig, RewardFunction
from replay_buffer import ReplayBuffer
from advanced.simulator import Simulator
from advanced.td3_bc_agent import TD3_BC
from utils import plot_rewards, plot_cgm_levels, plot_cgm_reward_action_with_legend, set_seed


class EnvironmentAdvanced:
    def __init__(self, patient_id):
        self.simulator = Simulator(patient_id=patient_id)
        self.simulator.train()

        self.state_size = 72 * 8  # [hour, time_since_last_meal, time_since_last_insulin, sleep, basal, carbs, bolus, cgm]
        self.action_space = [Action.NOTHING, Action.EAT, Action.INJECT]

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)  # (type, value, time_index)
        self.repeat_counter = 0
        self.sleep_mode = self.get_sleep_signal()

    def get_sleep_signal(self):
        return self.simulator.get_sleep_mode()

    def get_window(self):
        return self.simulator.get_full_current_window()

    def get_state(self):
        window = self.get_window()
        return window[:, :8].flatten()  # [hour, time_since_last_meal, time_since_last_insulin, sleep, basal, carbs, bolus, cgm]

    def reset(self):
        self.simulator.reset()

        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = (Action.NOTHING, 0.0, 0)
        self.repeat_counter = 0
        self.sleep_mode = self.get_sleep_signal()

        return self.current_state

    def step(self, action):
        # Step 1: Predict next CGM values
        predicted_cgm = self.simulator.predict_next_cgm()

        # Step 2: Apply action to simulator
        bolus_array, time_since_last_injection_array, carb_array, time_since_last_meal_array = self.simulator.apply_action_to_inputs(self.full_current_window, action)

        self.sleep_mode = self.get_sleep_signal()

        # Step 3: Compute reward
        reward = self.compute_reward(time_since_last_injection_array, time_since_last_meal_array, predicted_cgm, action)

        # Step 4: Commit inputs and predicted CGM
        self.simulator.commit_next_input(predicted_cgm, bolus_array, time_since_last_injection_array[1:], carb_array, time_since_last_meal_array[1:])

        # Step 5: Update state
        self.current_state = self.get_state()
        self.prev_action = action

        time_series = self.simulator.get_time_window()
        return self.current_state, predicted_cgm, time_series, reward, False, {}

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

        # TEST 4 - Do nothing when users are in sleep
        # if self.sleep_mode:
        #     if action_type != Action.NOTHING:
        #         total_reward -= RewardFunction.DO_NOTHING_IN_SLEEP
        #     else:
        #         total_reward += RewardFunction.DO_NOTHING_IN_SLEEP

        # TEST 3 - Reward for skipping unnecessary actions
        if action_type == Action.NOTHING and np.all(predicted_cgm > 90) and np.all(predicted_cgm < 150):
            total_reward += RewardFunction.DO_NOTHING_BONUS  # encourages stability

        # TEST 3 - Penalize unnecessary insulin
        if action_type == Action.INJECT:
            mean_cgm = predicted_cgm.mean()
            if mean_cgm < 150:
                total_reward -= (150 - mean_cgm) * 10
            elif mean_cgm > 200:
                total_reward += (mean_cgm - 200) * 5

        # Test 6
        if action_type == Action.EAT:
            time_since_last_meal = time_since_last_meal_array[time_index] + 1

            if time_since_last_meal < 12:
                total_reward -= RewardFunction.EARLY_MEAL_PENALTY

            elif 3 * 12 <= time_since_last_meal <= 6 * 12:
                total_reward += RewardFunction.GOOD_MEAL_TIMING_BONUS

        if action_type == Action.INJECT:
            time_since_last_injection = time_since_last_injection_array[time_index] + 1

            if time_since_last_injection < 2 * 12:
                total_reward -= RewardFunction.EARLY_INJECTION_PENALTY

        if self.prev_action[0] == action_type and action_type != Action.NOTHING:
            self.repeat_counter += 1
        else:
            self.repeat_counter = 0

        if self.repeat_counter >= 2:
            total_reward -= RewardFunction.REPEATED_ACTION_PENALTY

        return total_reward



def main():
    patient_id = "540"
    env = EnvironmentAdvanced(patient_id)
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
    for _ in range(RLConfig.MAX_EPISODES):
        state = env.reset()
        for _ in range(RLConfig.MAX_STEPS_PER_EPISODE):
            # Random but valid 6D action
            probs = np.random.dirichlet(np.ones(3))
            action_type = np.argmax(probs)
            carb_amount = np.random.uniform(*KnowledgeBase.CARB_RANGE)
            insulin_amount = np.random.uniform(*KnowledgeBase.INSULIN_RANGE)
            time_index = np.random.randint(0, 12)

            action_vector = np.array([probs[0], probs[1], probs[2], carb_amount, insulin_amount, time_index], dtype=np.float32)

            # Encode the chosen action only
            action = (action_type, carb_amount if action_type == 1 else insulin_amount, time_index)

            next_state, predicted_cgm, time_series, reward, done, _ = env.step(action)
            buffer.add(state, action_vector, reward, next_state, done)
            state = next_state

    print("Replay buffer filled.")

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
    main()
