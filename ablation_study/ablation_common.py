import os

import numpy as np
import torch

from config import Action, CQLConfig, DataConfig, EnvConfig, OfflineBufferConfig
from environment import Environment
from model.cql_bc.cql_agent import CQL
from replay_buffer import ReplayBuffer
from utils import cal_coefficient_of_variation, cal_time_above_range, cal_time_below_range, cal_time_in_range

FIXED_TIMING = "fixed_timing"
CARBOHYDRATE_ONLY = "carbohydrate_only"
INSULIN_ONLY = "insulin_only"

SUPPORTED_CASES = {FIXED_TIMING, CARBOHYDRATE_ONLY, INSULIN_ONLY}


class RestrictedActionSpace:
    """Defines one GUIDE action-space ablation while preserving CQL-BC."""

    def __init__(self, case, fixed_time_index=0):
        if case not in SUPPORTED_CASES:
            raise ValueError(f"Unsupported action-space ablation: {case}")

        if not 0 <= fixed_time_index <= 11:
            raise ValueError("fixed_time_index must be between 0 and 11")

        self.case = case
        self.fixed_time_index = fixed_time_index

        if case == FIXED_TIMING:
            # [nothing score, eat score, inject score, carbohydrate, insulin]
            self.action_low = np.array([0, 0, 0, CQLConfig.CARB_RANGE[0], CQLConfig.INSULIN_RANGE[0]], dtype=np.float32, )
            self.action_high = np.array([1, 1, 1, CQLConfig.CARB_RANGE[1], CQLConfig.INSULIN_RANGE[1]], dtype=np.float32, )

        elif case == CARBOHYDRATE_ONLY:
            # [nothing score, eat score, carbohydrate, time index]
            self.action_low = np.array([0, 0, CQLConfig.CARB_RANGE[0], 0], dtype=np.float32)
            self.action_high = np.array([1, 1, CQLConfig.CARB_RANGE[1], 11], dtype=np.float32)

        else:
            # [nothing score, inject score, insulin, time index]
            self.action_low = np.array([0, 0, CQLConfig.INSULIN_RANGE[0], 0], dtype=np.float32)
            self.action_high = np.array([1, 1, CQLConfig.INSULIN_RANGE[1], 11], dtype=np.float32)

    @property
    def action_dim(self):
        return len(self.action_high)

    def sample_buffer_action(self):
        """Sample one behavior-policy action within this restricted space."""
        if self.case == FIXED_TIMING:
            scores = np.random.dirichlet(np.ones(3))
            action_type = int(np.argmax(scores))
            carbohydrate = np.random.uniform(*OfflineBufferConfig.CARB_RANGE)
            insulin = np.random.uniform(*OfflineBufferConfig.INSULIN_RANGE)

            action_vector = np.array([scores[0], scores[1], scores[2], carbohydrate, insulin], dtype=np.float32, )
            action_value = (carbohydrate if action_type == Action.EAT else insulin if action_type == Action.INJECT else 0.0)
            environment_action = (action_type, action_value, self.fixed_time_index,)

            return action_vector, environment_action

        scores = np.random.dirichlet(np.ones(2))
        local_action_type = int(np.argmax(scores))
        time_index = int(np.random.randint(0, 12))

        if self.case == CARBOHYDRATE_ONLY:
            carbohydrate = np.random.uniform(*OfflineBufferConfig.CARB_RANGE)
            action_vector = np.array([scores[0], scores[1], carbohydrate, time_index], dtype=np.float32, )
            action_type = Action.EAT if local_action_type == 1 else Action.NOTHING
            action_value = carbohydrate if action_type == Action.EAT else 0.0

        else:
            insulin = np.random.uniform(*OfflineBufferConfig.INSULIN_RANGE)
            action_vector = np.array([scores[0], scores[1], insulin, time_index], dtype=np.float32, )
            action_type = Action.INJECT if local_action_type == 1 else Action.NOTHING
            action_value = insulin if action_type == Action.INJECT else 0.0

        environment_action = (action_type, action_value, time_index)
        return action_vector, environment_action

    def decode_policy_action(self, raw_action):
        """Map a restricted policy vector to the Environment action tuple."""
        clipped_action = np.clip(raw_action, self.action_low, self.action_high)

        if self.case == FIXED_TIMING:
            action_type = int(np.argmax(clipped_action[:3]))
            carbohydrate = clipped_action[3]
            insulin = clipped_action[4]
            action_value = (carbohydrate if action_type == Action.EAT else insulin if action_type == Action.INJECT else 0.0)

            return action_type, action_value, self.fixed_time_index

        local_action_type = int(np.argmax(clipped_action[:2]))
        action_value = clipped_action[2] if local_action_type == 1 else 0.0
        time_index = int(np.clip(np.round(clipped_action[3]), 0, 11))

        if self.case == CARBOHYDRATE_ONLY:
            action_type = Action.EAT if local_action_type == 1 else Action.NOTHING

        else:
            action_type = Action.INJECT if local_action_type == 1 else Action.NOTHING

        return action_type, action_value, time_index


class RestrictedReplayBuffer(ReplayBuffer):
    """The standard replay buffer filled within one restricted action space."""

    def __init__(self, action_space, max_size=1_000_000):
        super().__init__(max_size=max_size)
        self.action_space = action_space

    def fill_replay_buffer(self, env, seed):
        print(f"Filling {self.action_space.case} offline replay buffer...")
        np.random.seed(seed)

        for state_index in range(OfflineBufferConfig.NUM_TRAIN_INIT_STATE):
            for _ in range(OfflineBufferConfig.MAX_EPISODES):
                state = env.reset(state_index=state_index, is_testing=False)
                episode_transitions = []

                for step in range(OfflineBufferConfig.MAX_STEPS_PER_EPISODE):
                    action_vector, environment_action = (self.action_space.sample_buffer_action())

                    next_state, _, _, reward, _, _ = env.step(step, environment_action)
                    episode_transitions.append((state, action_vector, reward, next_state))
                    state = next_state

                episode_end_reward = env.compute_episode_reward()
                reward_adjustment = episode_end_reward / len(episode_transitions)

                for state, action, reward, next_state in episode_transitions:
                    self.add(state, action, reward + reward_adjustment, next_state, False, )

        print(f"{self.action_space.case} replay buffer filled with "
              f"{len(self)} transitions.")


def train_cql(agent, buffer):
    print("Starting CQL-BC training...")

    for step in range(CQLConfig.TRAINING_STEPS):
        agent.train(buffer, batch_size=CQLConfig.BATCH_SIZE)

        if step % 200 == 0:
            print(f"[CQL-BC Train] step {step}/{CQLConfig.TRAINING_STEPS}")

    print("CQL-BC training finished.")


def _calculate_metrics(predicted_cgms):
    return {
        "tir": cal_time_in_range(predicted_cgms),
        "tbr": cal_time_below_range(predicted_cgms),
        "tar": cal_time_above_range(predicted_cgms),
        "cv": cal_coefficient_of_variation(predicted_cgms),
    }


def _format_test_metrics(test_index, metrics):
    return (
        f"\n--------------------- Results for Test {test_index} "
        f"----------------------\n"
        f"Time-in-Range (TIR)          : {metrics['tir']:.2f}%\n"
        f"Time-below-Range (TBR)       : {metrics['tbr']:.2f}%\n"
        f"Time-above-Range (TAR)       : {metrics['tar']:.2f}%\n"
        f"Coefficient of Variation (CV): {metrics['cv']:.2f}%\n"
    )


def _format_summary(all_metrics):
    labels = {
        "tir": "Time-in-Range (TIR)",
        "tbr": "Time-below-Range (TBR)",
        "tar": "Time-above-Range (TAR)",
        "cv": "Coefficient of Variation (CV)",
    }

    lines = ["\n----------------- Final Results ------------------"]
    for metric_name in ("tir", "tbr", "tar", "cv"):
        values = np.asarray(all_metrics[metric_name], dtype=np.float32)
        lines.append(
            f"Average {labels[metric_name]:<29}: "
            f"{np.mean(values):.2f}% (± {np.std(values):.2f}%), "
            f"Median: {np.median(values):.2f}%"
        )
    return "\n".join(lines) + "\n"


def test_cql(env, agent, action_space, folder_path):
    print(f"Evaluating {action_space.case} policy...")

    log_path = os.path.join(folder_path, "eval_results.txt")
    if os.path.isfile(log_path):
        os.remove(log_path)

    all_metrics = {"tir": [], "tbr": [], "tar": [], "cv": []}

    for test_index in range(CQLConfig.NUM_TEST_INIT_STATE):
        state = env.reset(state_index=test_index, is_testing=True)
        predicted_cgms = []

        for step in range(CQLConfig.TESTING_STEPS):
            raw_action = agent.select_action(state)
            action = action_space.decode_policy_action(raw_action)
            state, predicted_cgm, _, _, _, _ = env.step(step, action)
            predicted_cgms.extend(predicted_cgm)

        metrics = _calculate_metrics(predicted_cgms)
        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)

        test_output = _format_test_metrics(test_index + 1, metrics)
        print(test_output)

        with open(log_path, "a") as log_file:
            log_file.write(test_output)

    summary_output = _format_summary(all_metrics)
    print(summary_output)

    with open(log_path, "a") as log_file:
        log_file.write(summary_output)


def run_ablation(case, dataset_name, patient_id, seed):
    device = torch.device("cpu")
    folder_path = f"./ablation_study/tests/{case}/{dataset_name}/{dataset_name}_patient_{patient_id}/seed_{seed}/"

    os.makedirs(folder_path, exist_ok=True)

    action_space = RestrictedActionSpace(case=case)
    environment = Environment(dataset_name=dataset_name, patient_id=patient_id)
    agent = CQL(state_dim=EnvConfig.STATE_DIM,
                action_dim=action_space.action_dim,
                max_action=action_space.action_high,
                device=device)

    buffer = RestrictedReplayBuffer(action_space=action_space)
    buffer.fill_replay_buffer(environment, seed)

    train_cql(agent, buffer)
    test_cql(environment, agent, action_space, folder_path)


def run_from_config(case):
    run_ablation(case=case, dataset_name=DataConfig.DATASET, patient_id=str(DataConfig.PATIENT_ID), seed=DataConfig.SEEDS[0])
