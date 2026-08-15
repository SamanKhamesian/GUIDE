import numpy as np

from config import RewardShaping, Action, Threshold, EnvConfig
from simulator import Simulator
from utils import cal_time_in_range, count_glycemic_events


class Environment:
    def __init__(self, dataset_name, patient_id):
        self.simulator = Simulator(dataset_name=dataset_name, patient_id=patient_id)
        self.simulator.train()

        # 8 features, and 6-hour data for each
        # features: [hour, sleep, time_since_last_meal, time_since_last_insulin, carbs, bolus, basal, cgm]
        self.state_size = EnvConfig.STATE_DIM
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
        # Old: remove basal from state (it's only been in the glimmer input)
        # Old: window = np.delete(window, 6, axis=1)
        return window[:, :8].flatten()

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
        w_normal = RewardShaping.WEIGHTS[0]
        w_hypo = RewardShaping.WEIGHTS[1]
        w_hyper = RewardShaping.WEIGHTS[2]
        ideal_cgm = RewardShaping.IDEAL_CGM

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
            episode_reward += RewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hypo_events <= 3:
            episode_reward -= RewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hypo_events <= 5:
            episode_reward -= RewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= RewardShaping.HYPO_HYPER_3_PENALTY

        # Penalty for hyper events
        if hyper_events == 0:
            episode_reward += RewardShaping.HYPO_HYPER_1_PENALTY
        elif 1 <= hyper_events <= 3:
            episode_reward -= RewardShaping.HYPO_HYPER_1_PENALTY
        elif 4 <= hyper_events <= 5:
            episode_reward -= RewardShaping.HYPO_HYPER_2_PENALTY
        else:
            episode_reward -= RewardShaping.HYPO_HYPER_3_PENALTY

        # Duration penalty (additional fine-tuning, optional but recommended)
        episode_reward -= ((hypo_duration + hyper_duration) * 0.5)  # penalty per minute outside range

        episode_reward = episode_reward / 1000.0
        return episode_reward