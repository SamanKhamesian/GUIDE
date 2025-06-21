import numpy as np
import torch

from basic.simulator import Simulator
from basic.td3_bc_agent import TD3_BC
from config import Action, Threshold, PredictorConfig, RLConfig
from replay_buffer import ReplayBuffer
from utils import plot_rewards, plot_cgm_levels


class Environment:
    def __init__(self, patient_id):
        # --> TODO: Consider extending the action space to include action magnitude (e.g., dose of insulin or carbs) for finer control
        self.simulator = Simulator(patient_id=patient_id)
        self.simulator.train()

        self.state_size = 72 * 5  # 72 time steps × 5 features: [hour, basal, carbs, bolus, CGM]
        self.action_space = [Action.NOTHING, Action.EAT, Action.INJECT]
        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = Action.NOTHING

    def get_window(self):
        return self.simulator.get_full_current_window()

    def get_state(self):
        window = self.get_window()
        return window[:, :5].flatten()  # Flatten [hour, basal, carb, bolus, CGM] inputs

    def step(self, action):
        # if time_tracker.is_sleep_time() and action != Action.NOTHING:
        #     action = Action.NOTHING

        # --> TODO: We assume that our RL agent only can generate the type of action and has no control over timing or portion of the carbs/insulin
        # Step 1: Apply action to simulator (only updates bolus/carb arrays) and update the current state
        self.simulator.apply_action_to_inputs(self.full_current_window, action)
        self.current_state = self.get_state()

        # --> TODO: The predictor currently returns only CGM values; consider multi-output prediction (e.g., insulin need, MA) for future extensions
        # Step 2: Predict CGM (without shifting anything)
        predicted_cgm = self.simulator.predict_next_cgm()

        # Step 3: Compute reward based on this outcome
        reward = self.compute_reward(predicted_cgm)

        # Step 4: Now shift data (append predicted_cgm + inputs)
        self.simulator.commit_next_input(predicted_cgm)

        # Step 5: Update state
        self.current_state = self.get_state()
        self.prev_action = action

        return self.current_state, reward, False, {}

    def reset(self):
        self.simulator.reset()
        self.full_current_window = self.get_window()
        self.current_state = self.get_state()
        self.prev_action = Action.NOTHING
        return self.current_state

    # --> TODO: Add reward logging (hypo/hyper breakdown) per step for debugging and performance visualization
    @staticmethod
    def compute_reward(predicted_cgm):
        hypo = predicted_cgm < Threshold.HYPOGLYCEMIA
        hyper = predicted_cgm > Threshold.HYPERGLYCEMIA
        normal = ~hypo & ~hyper

        # Extract weights
        w_normal = PredictorConfig.WEIGHTS[0]
        w_hypo = PredictorConfig.WEIGHTS[1]
        w_hyper = PredictorConfig.WEIGHTS[2]

        # Piecewise reward components
        reward = np.zeros_like(predicted_cgm)
        reward[hypo] = -w_hypo * (Threshold.HYPOGLYCEMIA - predicted_cgm[hypo])
        reward[hyper] = -w_hyper * (predicted_cgm[hyper] - Threshold.HYPERGLYCEMIA)
        reward[normal] = w_normal * (predicted_cgm[normal] - Threshold.HYPOGLYCEMIA) / (Threshold.HYPERGLYCEMIA - Threshold.HYPOGLYCEMIA) * 10

        total_reward = np.sum(reward)
        return total_reward


def main():
    patient_id = "563"
    env = Environment(patient_id)
    device = torch.device("cpu")
    agent = TD3_BC(state_dim=360, action_dim=1, max_action=2.0, device=device)
    buffer = ReplayBuffer()

    print("Filling initial replay buffer...")
    for _ in range(RLConfig.MAX_EPISODES):
        state = env.reset()
        for _ in range(RLConfig.MAX_STEPS_PER_EPISODE):
            action = np.random.randint(0, 3)
            next_state, reward, done, _ = env.step(action)
            buffer.add(state, np.array([action], dtype=np.float32), reward, next_state, done)
            state = next_state

    print("Replay buffer filled.")

    print("Starting TD3-BC training...")
    for step in range(RLConfig.TRAINING_STEPS):
        agent.train(buffer, batch_size=256)

    # --> TODO: Add per-episode and rolling average reward summaries for evaluation
    # --> TODO: Save agent actions and CGM predictions to visualize behavioral patterns across time
    print("Evaluating policy...")
    state = env.reset()
    total_reward = 0
    rewards = []
    predicted_cgms = []

    for _ in range(RLConfig.TESTING_STEPS):
        raw_action = agent.select_action(state)
        mapped = int(np.clip(np.round(raw_action[0]), 0, 2))
        print(f"Generated action: Type={mapped}")

        state, reward, done, _ = env.step(mapped)
        total_reward += reward

        rewards.append(reward)
        predicted_cgms.extend(env.simulator.predict_next_cgm())

    print(f"Evaluation reward: {total_reward:.2f}")

    plot_rewards(rewards, title="Evaluation Reward per Step")
    plot_cgm_levels(predicted_cgms, title="Predicted CGM with Ranges")


if __name__ == "__main__":
    main()
